"""Train one additional visual agent on staged benchmark clips, leaving the released five untouched.

    python tools/external_benchmark/train_added_agent.py --train <staged_train>/test --val <staged_val>/test --out checkpoints/added_agent

An EfficientNet-B0 face-crop classifier on the same MTCNN crops every other agent reads. The clip score is the mean sigmoid over
the clip's faces with horizontal-flip averaging, as for the released visual agent. The epoch is chosen on the validation clips.
Face crops are cached once as a uint8 memmap and augmented on the accelerator, so no DataLoader workers are forked.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torchvision import models

SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_cache(clip_dir, cache):
    if os.path.exists(cache + ".json"):
        return json.load(open(cache + ".json"))
    files = sorted(glob.glob(os.path.join(clip_dir, "*.npz")))
    counts = []
    for f in files:
        with np.load(f) as z:
            counts.append(len(z["faces"]))
    total = int(sum(counts))
    arr = np.lib.format.open_memmap(cache + ".npy", mode="w+", dtype=np.uint8, shape=(total, SIZE, SIZE, 3))
    labels, clip_of, pos = np.zeros(total, np.uint8), np.zeros(total, np.int32), 0
    for ci, f in enumerate(files):
        with np.load(f) as z:
            faces = z["faces"]
        if len(faces):
            t = torch.from_numpy(faces).permute(0, 3, 1, 2).float()
            t = F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False)
            arr[pos:pos + len(faces)] = t.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
            labels[pos:pos + len(faces)] = int(f.endswith("_label_fake.npz"))
            clip_of[pos:pos + len(faces)] = ci
            pos += len(faces)
    arr.flush()
    np.save(cache + "_labels.npy", labels); np.save(cache + "_clip.npy", clip_of)
    meta = {"files": [os.path.basename(f) for f in files], "n_faces": total}
    json.dump(meta, open(cache + ".json", "w"))
    return meta


def load_cache(cache):
    return (np.load(cache + ".npy", mmap_mode="r"), np.load(cache + "_labels.npy"), np.load(cache + "_clip.npy"),
            json.load(open(cache + ".json")))


def to_input(batch_uint8, device, augment):
    x = torch.from_numpy(np.ascontiguousarray(batch_uint8)).to(device).permute(0, 3, 1, 2).float() / 255.0
    if augment:
        flip = torch.rand(len(x), device=device) < 0.5
        x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
        gain = 1 + 0.2 * (torch.rand(len(x), 1, 1, 1, device=device) - 0.5)
        bias = 0.1 * (torch.rand(len(x), 1, 1, 1, device=device) - 0.5)
        x = (x * gain + bias).clamp(0, 1)
        if torch.rand(1).item() < 0.5:  # resolution loss, a stand-in for re-compression
            s = int(SIZE * (0.4 + 0.5 * torch.rand(1).item()))
            x = F.interpolate(F.interpolate(x, size=(s, s), mode="bilinear", antialias=True), size=(SIZE, SIZE), mode="bilinear")
    return (x - MEAN.to(device)) / STD.to(device)


def clip_scores(model, faces, clip_of, n_clips, device, bs=128):
    model.eval()
    sums, cnt = np.zeros(n_clips), np.zeros(n_clips)
    with torch.no_grad():
        for i in range(0, len(faces), bs):
            x = to_input(faces[i:i + bs], device, False)
            p = (torch.sigmoid(model(x)) + torch.sigmoid(model(x.flip(3)))).squeeze(1).cpu().numpy() / 2
            np.add.at(sums, clip_of[i:i + bs], p); np.add.at(cnt, clip_of[i:i + bs], 1)
    return sums / np.maximum(cnt, 1), cnt > 0


def make_model():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--bs", type=int, default=48); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--one_epoch_per_process", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    build_cache(a.train, os.path.join(a.out, "cache_train")); build_cache(a.val, os.path.join(a.out, "cache_val"))
    Xtr, ytr, _, mtr = load_cache(os.path.join(a.out, "cache_train"))
    Xva, yva, cva, mva = load_cache(os.path.join(a.out, "cache_val"))
    yva_clip = np.array([int(f.endswith("_label_fake.npz")) for f in mva["files"]])
    print(f"train faces {len(Xtr)} ({int(ytr.sum())} fake) from {len(mtr['files'])} clips; val {len(Xva)} faces, {len(mva['files'])} clips", flush=True)

    model = make_model().to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(Xtr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    pos_weight = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], device=a.device, dtype=torch.float32)
    state_path, log_path = os.path.join(a.out, "last.pth"), os.path.join(a.out, "train_log.json")
    start, log, best = 0, [], -1.0
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location=a.device)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, log, best = st["epoch"] + 1, st["log"], st["best"]
        rng = np.random.default_rng(a.seed + start)
        print(f"resumed after epoch {start}", flush=True)

    for epoch in range(start, a.epochs):
        model.train(); t0, losses = time.time(), []
        order = rng.permutation(len(Xtr))
        for i in range(0, len(order) - a.bs + 1, a.bs):
            idx = np.sort(order[i:i + a.bs])
            x = to_input(Xtr[idx], a.device, True)
            y = torch.from_numpy(ytr[idx].astype(np.float32)).to(a.device).view(-1, 1)
            loss = F.binary_cross_entropy_with_logits(model(x), y, pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); losses.append(loss.item())
            if a.device == "mps" and (i // a.bs) % 25 == 0:
                torch.mps.empty_cache()  # the MPS allocator grows without bound over a long run
        scores, ok = clip_scores(model, Xva, cva, len(mva["files"]), a.device)
        auc = float(roc_auc_score(yva_clip[ok], scores[ok]))
        log.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_clip_auc": auc, "seconds": round(time.time() - t0)})
        print(json.dumps(log[-1]), flush=True)
        if auc > best:
            best = auc
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val_clip_auc": auc}, os.path.join(a.out, "best.pth"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch, "log": log, "best": best}, state_path)
        json.dump(log, open(log_path, "w"), indent=1)
        if a.one_epoch_per_process and epoch + 1 < a.epochs:
            raise SystemExit(3)  # caller relaunches; a fresh process starts with an empty MPS heap
    print(f"DONE best val clip AUC {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
