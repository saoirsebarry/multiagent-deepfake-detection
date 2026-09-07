"""Robustness fine-tune of the visual XceptionNet agent from its released checkpoint.
Released inference kept: every stored frame (stored channel order) -> PIL -> 299 ->
ToTensor -> Normalize(0.5), mean over frames and horizontal flips, sigmoid. Fixes:
deployment-style frame corruption, per-clip random-frame sampling, label smoothing
0.05, selection by validation log-loss."""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import common as C  # noqa: E402
from agents.visual_xception import XceptionDeepfakeDetector  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--frames_per_clip", type=int, default=2)
A = ap.parse_args(); os.makedirs(A.out_dir, exist_ok=True); C.seed_all()
device = C.pick_device()
TF = transforms.Compose([transforms.ToPILImage(), transforms.Resize((299, 299)), transforms.ToTensor(),
                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
TRAIN_GEO = transforms.Compose([transforms.ToPILImage(), transforms.Resize((319, 319)), transforms.RandomCrop(299),
                                transforms.RandomRotation(10), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


class FrameSet(Dataset):
    def __init__(self, root):
        self.dir, self.files = C.split_files(root, "train")

    def __len__(self):
        return len(self.files) * A.frames_per_clip

    def __getitem__(self, i):
        d = np.load(os.path.join(self.dir, self.files[i % len(self.files)]), allow_pickle=True)
        faces = d["faces"]; y = C.label_of(d)
        face = C.to_uint8(faces[random.randrange(len(faces))]) if len(faces) else np.zeros((299, 299, 3), np.uint8)
        return TRAIN_GEO(C.robust_image_aug(face)), torch.tensor(y, dtype=torch.float32)


def score_split(model, root, split):
    model.eval(); d, files = C.split_files(root, split); out = {}
    with torch.no_grad():
        for f in files:
            faces = np.load(os.path.join(d, f), allow_pickle=True)["faces"]
            if len(faces) == 0:
                out[f] = 0.5; continue
            ts = []
            for face in faces:
                t = TF(C.to_uint8(face)); ts.append(t); ts.append(torch.flip(t, dims=[2]))
            out[f] = float(torch.sigmoid(model(torch.stack(ts).to(device))).mean().item())
    return out


model = XceptionDeepfakeDetector(num_classes=1).to(device)
ck = torch.load(os.path.join(C.REPO, "checkpoints/xception/polyglotfake_xception_best_unbal_all_faceaug.pth"), map_location=device)
model.load_state_dict(ck.get("model_state_dict", ck))
labels_of = lambda ss: {f: 1.0 if "_label_fake" in f else 0.0 for f in ss}
vs = score_split(model, A.data_dir, "val"); C.fidelity(vs, C.COLUMNS["visual"])
hist = [{"epoch": 0, **C.evaluate(vs, labels_of(vs))}]; print("epoch 00", hist[-1], flush=True)
best = dict(hist[-1]); best_path = os.path.join(A.out_dir, "best_model.pth"); torch.save({"epoch": 0, "model_state_dict": model.state_dict()}, best_path)
dl = DataLoader(FrameSet(A.data_dir), batch_size=A.batch, shuffle=True, num_workers=C.NUM_WORKERS, drop_last=True)
opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=A.epochs)
crit = nn.BCEWithLogitsLoss(); eps = 0.05
for ep in range(1, A.epochs + 1):
    model.train(); tl = []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(); loss = crit(model(x).squeeze(1), y * (1 - 2 * eps) + eps); loss.backward(); opt.step(); tl.append(loss.item())
    sched.step()
    vs = score_split(model, A.data_dir, "val"); m = C.evaluate(vs, labels_of(vs))
    hist.append({"epoch": ep, "train_loss": float(np.mean(tl)), **m})
    print(f"epoch {ep:02d} train {np.mean(tl):.4f} val logloss {m['logloss']:.4f} auc {m['auc']:.5f} acc {m['acc']:.4f}", flush=True)
    if m["logloss"] < best["logloss"] - 1e-4:
        best = {"epoch": ep, **m}; torch.save({"epoch": ep, "model_state_dict": model.state_dict()}, best_path); print("  -> new best", flush=True)
json.dump(hist, open(os.path.join(A.out_dir, "history.json"), "w"), indent=1)
C.decide("visual", C.COLUMNS["visual"], best, A.out_dir)
model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
for split in ("val", "test"):
    if os.path.isdir(os.path.join(A.data_dir, split)):
        ss = score_split(model, A.data_dir, split)
        C.write_scores(os.path.join(A.out_dir, f"visual_{split}_scores.csv"), split, ss, labels_of(ss)); print(split, C.evaluate(ss, labels_of(ss)), flush=True)
print("XCEPTION-DONE", flush=True)
