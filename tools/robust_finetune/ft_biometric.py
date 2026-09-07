"""Robustness fine-tune of the Biometric-Quality agent from its released checkpoint."""
import argparse
import json
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
from train_biometric import FaceQualityNet, stack5, score_split, IMAGE_SIZE  # noqa: E402
import common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--batch", type=int, default=32)
A = ap.parse_args(); os.makedirs(A.out_dir, exist_ok=True); C.seed_all()
device = C.pick_device()
print("device", device, flush=True)


class TrainSet(Dataset):
    def __init__(self, root):
        self.dir, self.files = C.split_files(root, "train")
        self.pil_aug = transforms.Compose([transforms.RandomRotation(degrees=15),
                                           transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                           transforms.RandomApply([transforms.GaussianBlur(5)], p=0.3)])
        self.erase = transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(os.path.join(self.dir, self.files[i]), allow_pickle=True)
        faces = d["faces"]; y = C.label_of(d)
        if len(faces) == 0:
            return torch.zeros((5, IMAGE_SIZE, IMAGE_SIZE)), torch.tensor(y)
        face = cv2.cvtColor(C.to_uint8(faces[random.randrange(len(faces))]), cv2.COLOR_BGR2RGB)
        face = C.robust_image_aug(face)
        img = self.pil_aug(Image.fromarray(face).resize((IMAGE_SIZE, IMAGE_SIZE)))
        return self.erase(stack5(np.array(img))), torch.tensor(y)


model = FaceQualityNet().to(device)
model.load_state_dict(torch.load(os.path.join(C.REPO, "checkpoints/biometric/best_model.pth"), map_location=device)["model_state_dict"])
val_dir, val_files = C.split_files(A.data_dir, "val")
val_labels = {f: 1.0 if "_label_fake" in f else 0.0 for f in val_files}
vs = score_split(model, A.data_dir, "val", device)
C.fidelity(vs, C.COLUMNS["biometric"])
hist = [{"epoch": 0, **C.evaluate(vs, val_labels)}]
print("epoch 00", hist[-1], flush=True)
best = dict(hist[-1]); best_path = os.path.join(A.out_dir, "best_model.pth")
torch.save({"epoch": 0, "model_state_dict": model.state_dict()}, best_path)

dl = DataLoader(TrainSet(A.data_dir), batch_size=A.batch, shuffle=True, num_workers=C.NUM_WORKERS, drop_last=True)
opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=0.05)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=A.epochs)
bce = nn.BCELoss(); eps = 0.05
for ep in range(1, A.epochs + 1):
    model.train(); tl = []
    for x, y in dl:
        x, y = x.to(device), y.to(device).unsqueeze(1)
        opt.zero_grad(); loss = bce(model(x).clamp(1e-6, 1 - 1e-6), y * (1 - 2 * eps) + eps)
        loss.backward(); opt.step(); tl.append(loss.item())
    sched.step()
    vs = score_split(model, A.data_dir, "val", device)
    m = C.evaluate(vs, val_labels); hist.append({"epoch": ep, "train_loss": float(np.mean(tl)), **m})
    print(f"epoch {ep:02d} train {np.mean(tl):.4f} val logloss {m['logloss']:.4f} auc {m['auc']:.5f} acc {m['acc']:.4f}", flush=True)
    if m["logloss"] < best["logloss"] - 1e-4:
        best = {"epoch": ep, **m}; torch.save({"epoch": ep, "model_state_dict": model.state_dict()}, best_path)
        print("  -> new best", flush=True)
json.dump(hist, open(os.path.join(A.out_dir, "history.json"), "w"), indent=1)
adopted = C.decide("biometric", C.COLUMNS["biometric"], best, A.out_dir)
model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
for split in ("val", "test"):
    if os.path.isdir(os.path.join(A.data_dir, split)):
        ss = score_split(model, A.data_dir, split, device)
        C.write_scores(os.path.join(A.out_dir, f"biometric_{split}_scores.csv"), split, ss, {f: 1.0 if "_label_fake" in f else 0.0 for f in ss})
        print(split, C.evaluate(ss, {f: 1.0 if "_label_fake" in f else 0.0 for f in ss}), flush=True)
print("BIOMETRIC-DONE", flush=True)
