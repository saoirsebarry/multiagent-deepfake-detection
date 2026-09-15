"""Quantitative checks on the explanations: fidelity by deletion/insertion, stability under perturbation.

    python tools/xai_fidelity/run_fidelity.py --data_dir <root> --split test --ckpt_dir checkpoints \
        --n 60 --out paper_artifacts/xai_fidelity.json

Visual (GradCAM on XceptionNet) and audio (GradCAM on FreqNet's Mel spectrogram): the
saliency map orders pixels; deletion removes the most salient fraction first and records the
score of the predicted class (a faithful map drives the score down fast, low area under the
curve), insertion restores them onto a blurred input (high area). Both are compared with a
random ordering of the same input. Stability: the map is recomputed under Gaussian noise and
JPEG re-encoding (visual) or additive 30 dB SNR noise (audio) and compared by Pearson
correlation and top-10% overlap. Cross-modal attention: the frames the attention weights rank
highest are replaced by the clip's mean frame and the score change is compared with removing
the same number of random or lowest-attention frames; stability compares attention rankings
under horizontal flip and noise. The ECAPA SHAP values are feature-level and are not scored.
"""
import argparse
import glob
import io
import json
import os
import sys

import cv2
import librosa
import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from torchvision import transforms

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from agents.audio_freqnet import FreqNet  # noqa: E402
from agents.cross_modal_lipsync import CrossModal_CNN_LSTM  # noqa: E402
from agents.visual_xception import XceptionDeepfakeDetector  # noqa: E402
from xai_utils import GradCAM, find_target_layer  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEPS = 20


def first(pattern):
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(f"nothing matches {pattern}")
    return hits[0]


def to_uint8(face):
    face = face.cpu().numpy() if torch.is_tensor(face) else face
    if face.dtype != np.uint8:
        face = (face * 255).astype(np.uint8) if face.max() <= 1.0 else face.astype(np.uint8)
    return face


def curve_auc(vals):
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(vals, dx=1.0 / (len(vals) - 1)))


def deletion_insertion(score_fn, x, sal, blur, rng, order=None):
    """x: (C,H,W) float tensor; sal: (H,W) in [0,1]; blur: (C,H,W) baseline for insertion."""
    H, W = sal.shape
    flat = sal.flatten()
    idx = np.argsort(-flat) if order is None else order
    n = len(idx)
    base = score_fn(x)
    pos = base >= 0.5  # track the predicted class
    f = (lambda s: s) if pos else (lambda s: 1 - s)
    dele, inse = [], []
    for k in range(STEPS + 1):
        m = torch.zeros(n, dtype=torch.bool); m[idx[: int(round(k / STEPS * n))]] = True
        m = m.view(1, H, W)
        dele.append(f(score_fn(torch.where(m, blur, x))))
        inse.append(f(score_fn(torch.where(m, x, blur))))
    return {"deletion_auc": curve_auc(dele), "insertion_auc": curve_auc(inse), "base_score": float(base)}


def sal_stats(a, b):
    a, b = a.flatten(), b.flatten()
    k = max(1, int(0.1 * len(a)))
    ta, tb = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    return {"pearson": float(pearsonr(a, b)[0]) if a.std() > 0 and b.std() > 0 else 0.0,
            "spearman": float(spearmanr(a, b)[0]) if a.std() > 0 and b.std() > 0 else 0.0,
            "top10_overlap": len(ta & tb) / k}


def visual_block(model, faces, rng):
    tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((299, 299)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    face = to_uint8(faces[len(faces) // 2])
    x = tf(face)
    cam = GradCAM(model, find_target_layer(model))

    def score(t):
        with torch.no_grad():
            return float(torch.sigmoid(model(t.unsqueeze(0).to(DEV))).item())

    def heat(t):
        h = cam(t.unsqueeze(0).to(DEV))
        return cv2.resize(h.astype(np.float32), (299, 299))

    sal = heat(x)
    blur = torch.tensor(cv2.GaussianBlur(x.permute(1, 2, 0).numpy(), (0, 0), 12)).permute(2, 0, 1)
    fid = deletion_insertion(score, x, sal, blur, rng)
    rand = deletion_insertion(score, x, sal, blur, rng, order=rng.permutation(299 * 299))
    noisy = tf(np.clip(face.astype(np.float32) + rng.normal(0, 8, face.shape), 0, 255).astype(np.uint8))
    buf = io.BytesIO(); Image.fromarray(face).save(buf, format="JPEG", quality=60); jpeg = tf(np.array(Image.open(io.BytesIO(buf.getvalue()))))
    out = {**fid, "random_deletion_auc": rand["deletion_auc"], "random_insertion_auc": rand["insertion_auc"],
           "stability_noise": {**sal_stats(sal, heat(noisy)), "score_shift": abs(score(noisy) - fid["base_score"])},
           "stability_jpeg60": {**sal_stats(sal, heat(jpeg)), "score_shift": abs(score(jpeg) - fid["base_score"])}}
    cam.remove_hooks()
    return out


def mel_input(w):
    sr, L = 16000, 5 * 16000
    y = w[:L] if len(w) > L else np.pad(w, (0, L - len(w)))
    m = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=224), ref=np.max)
    m = (m - m.min()) / (m.max() - m.min()) if m.max() > m.min() else np.zeros_like(m)
    return torch.tensor(m, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)


def audio_block(model, w, rng):
    w = w.astype(np.float32)
    x = mel_input(w)
    cam = GradCAM(model, find_target_layer(model))

    def score(t):
        with torch.no_grad():
            return float(torch.sigmoid(model(t.unsqueeze(0).to(DEV))).item())

    def heat(t):
        h = cam(t.unsqueeze(0).to(DEV))
        return cv2.resize(h.astype(np.float32), (t.shape[2], t.shape[1]))

    sal = heat(x)
    blur = torch.tensor(cv2.GaussianBlur(x[0].numpy(), (0, 0), 6)).unsqueeze(0).repeat(3, 1, 1)
    fid = deletion_insertion(score, x, sal, blur, rng)
    rand = deletion_insertion(score, x, sal, blur, rng, order=rng.permutation(x.shape[1] * x.shape[2]))
    snr = 30.0; noise = rng.normal(0, 1, len(w)).astype(np.float32)
    noise *= np.sqrt((w ** 2).mean() / (10 ** (snr / 10))) / (noise.std() + 1e-9)
    xn = mel_input(w + noise); xg = mel_input(0.7 * w)
    out = {**fid, "random_deletion_auc": rand["deletion_auc"], "random_insertion_auc": rand["insertion_auc"],
           "stability_noise30dB": {**sal_stats(sal, heat(xn)), "score_shift": abs(score(xn) - fid["base_score"])},
           "stability_gain0.7": {**sal_stats(sal, heat(xg)), "score_shift": abs(score(xg) - fid["base_score"])}}
    cam.remove_hooks()
    return out


def crossmodal_block(model, faces, w, rng):
    vt = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224), antialias=True), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    n = min(len(faces), 20)
    vis = torch.zeros((20, 3, 224, 224))
    for i in range(n):
        vis[i] = vt(to_uint8(faces[i]))
    m = librosa.power_to_db(librosa.feature.melspectrogram(y=w.astype(np.float32), sr=16000, n_fft=2048, hop_length=512, n_mels=128), ref=np.max)
    mel = np.zeros((128, 313), dtype=np.float32)
    mel[:, : min(313, m.shape[1])] = m[:, :313]
    aud = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(DEV)

    def run(v):
        with torch.no_grad():
            out, att = model(v.unsqueeze(0).to(DEV), aud)
            return float(torch.softmax(out, 1)[0, 1].item()), att[0, :n].cpu().numpy()

    base, att = run(vis)
    pos = base >= 0.5
    f = (lambda s: s) if pos else (lambda s: 1 - s)
    mean_frame = vis[:n].mean(0, keepdim=True)
    order = np.argsort(-att)
    drops = {}
    for k in (1, 2, 3, 5):
        if k > n:
            break

        def drop(ids):
            v = vis.clone(); v[list(ids)] = mean_frame
            return f(base) - f(run(v)[0])

        drops[f"top{k}"] = drop(order[:k])
        drops[f"bottom{k}"] = drop(order[-k:])
        drops[f"random{k}"] = float(np.mean([drop(rng.choice(n, k, replace=False)) for _ in range(5)]))
    flipped = torch.flip(vis, dims=[3])
    noisy = vis + torch.randn_like(vis) * 0.05
    _, att_f = run(flipped); _, att_n = run(noisy)
    return {"base_score": base, "n_frames": int(n), "attention_entropy": float(-(att * np.log(att + 1e-9)).sum() / np.log(n)) if n > 1 else 0.0,
            "attention_max": float(att.max()), "top_frame": int(order[0]), "score_drop": drops,
            "stability_flip": {"spearman": float(spearmanr(att, att_f)[0]) if n > 2 else 1.0, "same_top_frame": bool(order[0] == np.argmax(att_f))},
            "stability_noise": {"spearman": float(spearmanr(att, att_n)[0]) if n > 2 else 1.0, "same_top_frame": bool(order[0] == np.argmax(att_n))}}


def summarise(rows):
    keys = {}
    for r in rows:
        for k, v in r.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, (int, float)):
                        keys.setdefault(f"{k}.{k2}", []).append(float(v2))
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.setdefault(k, []).append(float(v))
    return {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)} for k, v in keys.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt_dir", default=os.path.join(REPO, "checkpoints"))
    ap.add_argument("--n", type=int, default=60); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    d = os.path.join(a.data_dir, a.split)
    files = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
    reals = [f for f in files if "_label_real" in f]; fakes = [f for f in files if "_label_fake" in f]
    pick = list(rng.choice(reals, min(a.n // 2, len(reals)), replace=False)) + list(rng.choice(fakes, min(a.n - a.n // 2, len(fakes)), replace=False))
    vis = XceptionDeepfakeDetector(num_classes=1, dropout_rate=0.3, pretrained=False).to(DEV)
    ck = torch.load(first(os.path.join(a.ckpt_dir, "xception", "*.pth")), map_location=DEV, weights_only=False)
    vis.load_state_dict(ck.get("model_state_dict", ck)); vis.eval()
    fq = FreqNet(num_classes=1).to(DEV); fq.load_state_dict(torch.load(first(os.path.join(a.ckpt_dir, "freqnet", "*.pth")), map_location=DEV, weights_only=False)); fq.eval()
    cm = CrossModal_CNN_LSTM().to(DEV); cm.load_state_dict(torch.load(first(os.path.join(a.ckpt_dir, "cross_modal", "*.pth")), map_location=DEV, weights_only=False)); cm.eval()
    rows = {"visual": [], "audio": [], "crossmodal": []}
    for i, f in enumerate(pick, 1):
        z = np.load(os.path.join(d, f), allow_pickle=True)
        faces, w = z["faces"], z["waveform"]
        rec = {"file": f, "label": "fake" if "_label_fake" in f else "real"}
        if len(faces):
            rows["visual"].append({**rec, **visual_block(vis, faces, rng)})
        if w.size > 4096:
            rows["audio"].append({**rec, **audio_block(fq, w, rng)})
        if len(faces) and w.size > 4096:
            rows["crossmodal"].append({**rec, **crossmodal_block(cm, faces, w, rng)})
        print(f"[{i}/{len(pick)}] {f}", flush=True)
    out = {"split": a.split, "n_clips": len(pick), "seed": a.seed, "steps": STEPS,
           "summary": {k: summarise(v) for k, v in rows.items()}, "per_clip": rows}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
