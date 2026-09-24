"""Why the visual CAM still loses to a random ordering on predicted-real clips, and what fixes it.

The sign fix left the predicted-fake half untouched and only halved the predicted-real gap, so
something else is wrong. Two families of cause, separated in one pass:

  the metric is unfair
      A CAM-ordered mask is one blob; a pixel-random mask is speckle, and speckle adds blend
      seams, which this detector was trained to read as manipulation. Control: the same map
      rolled half an image away - identical value histogram, identical blob shape, wrong
      location. If the roll scores like the map rather than like pixel-random, the gap is
      geometry rather than evidence, and pixel-random is the wrong control. Seam length is
      measured directly. A second fill (per-channel mean) separates "blur removes the
      high-frequency evidence" from "the edit introduces an edge".

  the map is wrong
      Grad-CAM weights each channel by its mean gradient, which assumes the channel helps
      everywhere it fires. That assumption fails when the target is the negative class, where
      evidence is the absence of a cue rather than its presence. HiResCAM and LayerCAM weight
      per location instead, from the same single backward pass, so all three are compared for
      the price of one.

    python tools/xai_fidelity/diagnose_real_half.py --data_dir <root> --split test \
        --ckpt_dir checkpoints --out /content/diagnose.json
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from agents.visual_xception import XceptionDeepfakeDetector  # noqa: E402
from xai_utils import find_target_layer  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEPS = 20
SIDE = 299
MAPS = ("gradcam", "hirescam", "layercam")


def auc(vals):
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(vals, dx=1.0 / (len(vals) - 1)))


def mask_at(order, frac):
    m = np.zeros(SIDE * SIDE, dtype=bool)
    m[order[: int(round(frac * SIDE * SIDE))]] = True
    return m.reshape(SIDE, SIDE)


def perimeter(mask2d):
    """Seam length of the deleted region: how much blur/sharp boundary the edit introduces."""
    m = mask2d.astype(np.uint8)
    return int(np.abs(np.diff(m, axis=0)).sum() + np.abs(np.diff(m, axis=1)).sum())


def concentration(sal):
    """Share of the map's mass in its top decile; 0.1 is uniform, 1.0 is a point."""
    v = np.sort(sal.flatten())[::-1]
    return float(v[: max(1, len(v) // 10)].sum() / (v.sum() + 1e-9))


def norm_resize(h):
    h = F.relu(h).squeeze().detach().cpu().numpy().astype(np.float32)
    if h.max() > 0:
        h = h / h.max()
    return cv2.resize(h, (SIDE, SIDE))


def all_maps(model, layer, x):
    """One forward and one backward on the predicted class; three weightings of the result."""
    acts, grads = {}, {}
    h = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    try:
        model.zero_grad()
        with torch.enable_grad():
            out = model(x.unsqueeze(0).to(DEV))
        a = acts["a"]
        a.register_hook(lambda g: grads.__setitem__("g", g))
        logit = out.squeeze()
        logit.backward(torch.ones_like(logit) * (1.0 if logit.item() >= 0 else -1.0))
    finally:
        h.remove()
    a, g = acts["a"].detach(), grads["g"].detach()
    return {"gradcam": norm_resize((a * g.mean(dim=(2, 3), keepdim=True)).sum(1)),
            "hirescam": norm_resize((a * g).sum(1)),
            "layercam": norm_resize((a * F.relu(g)).sum(1))}


def curve(score_fn, x, fill, order, track):
    out = []
    for k in range(STEPS + 1):
        m = torch.from_numpy(mask_at(order, k / STEPS)).unsqueeze(0)
        out.append(track(score_fn(torch.where(m, fill, x))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt_dir", default=os.path.join(REPO, "checkpoints"))
    ap.add_argument("--files", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips_released.txt"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    d = os.path.join(a.data_dir, a.split)
    pick = [ln.strip() for ln in open(a.files) if ln.strip()][: a.n]

    model = XceptionDeepfakeDetector(num_classes=1, dropout_rate=0.3, pretrained=False).to(DEV)
    ck = torch.load(sorted(glob.glob(os.path.join(a.ckpt_dir, "xception", "*.pth")))[0],
                    map_location=DEV, weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    layer = find_target_layer(model)

    tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((SIDE, SIDE)),
                             transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    rows = []
    for i, f in enumerate(pick, 1):
        z = np.load(os.path.join(d, f), allow_pickle=True)
        faces = z["faces"]
        if not len(faces):
            continue
        face = faces[len(faces) // 2]
        face = face.astype(np.uint8) if face.dtype != np.uint8 else face
        x = tf(face)

        def score(t):
            with torch.no_grad():
                return float(torch.sigmoid(model(t.unsqueeze(0).to(DEV))).item())

        base = score(x)
        pos = base >= 0.5
        track = (lambda s: s) if pos else (lambda s: 1 - s)
        maps = all_maps(model, layer, x)

        blur = torch.tensor(cv2.GaussianBlur(x.permute(1, 2, 0).numpy(), (0, 0), 12)).permute(2, 0, 1)
        mean = x.mean(dim=(1, 2), keepdim=True).expand_as(x).clone()
        rand_order = rng.permutation(SIDE * SIDE)

        rec = {"file": f, "base_score": base, "predicted": "fake" if pos else "real",
               "del_blur_random": auc(curve(score, x, blur, rand_order, track)),
               "del_mean_random": auc(curve(score, x, mean, rand_order, track)),
               "perimeter_50_random": perimeter(mask_at(rand_order, 0.5)),
               "fully_blurred_tracked": track(score(blur)),
               "fully_mean_tracked": track(score(mean))}
        for name, sal in maps.items():
            order = np.argsort(-sal.flatten())
            rolled = np.argsort(-np.roll(sal, (SIDE // 2, SIDE // 2), axis=(0, 1)).flatten())
            rec[f"del_blur_{name}"] = auc(curve(score, x, blur, order, track))
            rec[f"del_blur_{name}_rolled"] = auc(curve(score, x, blur, rolled, track))
            rec[f"del_mean_{name}"] = auc(curve(score, x, mean, order, track))
            rec[f"perimeter_50_{name}"] = perimeter(mask_at(order, 0.5))
            rec[f"mass_{name}"] = float(sal.mean())
            rec[f"conc_{name}"] = concentration(sal)
            rec[f"zero_{name}"] = float((sal <= 1e-6).mean())
        rows.append(rec)
        print(f"[{i}/{len(pick)}] {f} {rec['predicted']}", flush=True)

    json.dump({"n": len(rows), "seed": a.seed, "per_clip": rows}, open(a.out, "w"), indent=1)
    report(rows)


def report(rows):
    avg = lambda sub, k: float(np.mean([r[k] for r in sub]))  # noqa: E731
    for cls in ("fake", "real"):
        sub = [r for r in rows if r["predicted"] == cls]
        if not sub:
            continue
        print(f"\n### predicted {cls}  (n={len(sub)})   lower deletion AUC is better")
        print(f"  {'map':<10}{'blur':>8}{'rolled':>9}{'gap':>8}{'roll gap':>10}{'mean fill':>11}{'gap':>8}{'seam@50%':>10}{'top-dec':>9}")
        for m in MAPS:
            gb, gr = avg(sub, f"del_blur_{m}"), avg(sub, f"del_blur_{m}_rolled")
            gm = avg(sub, f"del_mean_{m}")
            rb, rm = avg(sub, "del_blur_random"), avg(sub, "del_mean_random")
            print(f"  {m:<10}{gb:>8.3f}{gr:>9.3f}{rb - gb:>+8.3f}{rb - gr:>+10.3f}"
                  f"{gm:>11.3f}{rm - gm:>+8.3f}{avg(sub, f'perimeter_50_{m}'):>10.0f}{avg(sub, f'conc_{m}'):>9.3f}")
        print(f"  {'random':<10}{avg(sub, 'del_blur_random'):>8.3f}{'':>9}{'':>8}{'':>10}"
              f"{avg(sub, 'del_mean_random'):>11.3f}{'':>8}{avg(sub, 'perimeter_50_random'):>10.0f}")
        print(f"  tracked score once fully replaced: blur {avg(sub, 'fully_blurred_tracked'):.3f} | "
              f"mean {avg(sub, 'fully_mean_tracked'):.3f}   (base {avg(sub, 'base_score'):.3f})")


if __name__ == "__main__":
    main()
