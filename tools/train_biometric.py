"""Two-stage training for the Biometric-Quality agent.

Stage 1 supplements the three RGB channels with two per-pixel forensic maps —
a tanh-compressed Laplacian sharpness map and a high-frequency residual map —
computed from each (augmented) face crop, and trains from ImageNet weights with
random-face sampling, flip/rotation/colour-jitter/blur/erasing augmentation,
AdamW weight decay 0.05 and label smoothing 0.05, checkpointing on validation
loss. Stage 2 fine-tunes briefly and selects its stopping epoch on the
validation partition alone, by the released ensemble objective: over every
0.05-step five-agent weight simplex with all agents active, minimise validation
ensemble errors at tau=0.5, tie-break on the separating margin. The selected
checkpoint is scored on val and test with the shipping inference recipe
(all frames, mean over horizontal-flip TTA).

Usage:
  python tools/train_biometric.py --data_dir <root with train/ val/ test/> \
      --val_csv paper_artifacts/source_csvs/analysis_results_VAL.csv \
      --output_dir biometric_final
"""
import argparse
import csv
import json
import logging
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

logging.basicConfig(level=logging.INFO, format="%(message)s")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

IMAGE_SIZE = 299
IMAGENET_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
FIXED_AGENT_COLS = ["score_Visual (Spatial)", "score_Audio (Mel+CNN)",
                    "score_Audio Forensics (ECAPA)", "score_Cross-Modal (Lip-Sync)"]


def forensic_maps(face_rgb_uint8):
    gray = cv2.cvtColor(face_rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float64)
    lap = np.tanh(np.abs(cv2.Laplacian(gray, cv2.CV_64F)) / 64.0)
    g = gray / 255.0
    hf = np.clip(4.0 * (g - cv2.GaussianBlur(g, (5, 5), 1.0)), -1.0, 1.0)
    return torch.from_numpy(np.stack([lap, hf]).astype(np.float32))


def to_uint8(face):
    if face.dtype != np.uint8:
        face = (face * 255).astype(np.uint8) if face.max() <= 1.0 else face.astype(np.uint8)
    return face


def stack5(face_rgb_uint8):
    rgb = IMAGENET_NORM(transforms.functional.to_tensor(Image.fromarray(face_rgb_uint8)))
    return torch.cat([rgb, forensic_maps(face_rgb_uint8)], dim=0)


class NpzDataset(Dataset):
    """Random-face sampling with augmentation for training; middle face otherwise.

    Forensic maps are computed after augmentation so train and inference see the
    same feature semantics."""

    def __init__(self, root, split, augment):
        self.dir = os.path.join(root, split)
        self.files = sorted(f for f in os.listdir(self.dir) if f.endswith(".npz"))
        self.augment = augment
        self.pil_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(5)], p=0.3),
        ])
        self.erase = transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        data = np.load(os.path.join(self.dir, self.files[i]), allow_pickle=True)
        faces = data["faces"]
        label = 1.0 if data["label"][0] == "fake" else 0.0
        if len(faces) == 0:
            return {"face_quality": torch.zeros((5, IMAGE_SIZE, IMAGE_SIZE)),
                    "label": torch.tensor(label)}
        idx = random.randrange(len(faces)) if self.augment else len(faces) // 2
        face = cv2.cvtColor(to_uint8(faces[idx]), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(face).resize((IMAGE_SIZE, IMAGE_SIZE))
        if self.augment:
            img = self.pil_aug(img)
        x = stack5(np.array(img))
        if self.augment:
            x = self.erase(x)
        return {"face_quality": x, "label": torch.tensor(label)}


class FaceQualityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        orig = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(5, orig.out_channels,
            kernel_size=orig.kernel_size, stride=orig.stride, padding=orig.padding, bias=False)
        with torch.no_grad():
            self.backbone.features[0][0].weight[:, :3] = orig.weight
            self.backbone.features[0][0].weight[:, 3:] = orig.weight[:, :2] * 0.1
        nf = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.quality_head = nn.Sequential(
            nn.Linear(nf, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5),
            nn.Linear(256, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3))
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x):
        return torch.sigmoid(self.classifier(self.quality_head(self.backbone(x))))


def auc_of(scores, labels):
    scores = np.asarray(scores); labels = np.asarray(labels).astype(bool)
    order = np.argsort(scores); n = len(scores); ranks = np.empty(n)
    sr = scores[order]
    _, inv, cnt = np.unique(sr, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt); ranks[order] = ((cum - cnt + cum + 1) / 2.0)[inv]
    npos = labels.sum(); nneg = n - npos
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def score_split(model, root, split, device):
    """Shipping inference recipe: mean over all frames with flip TTA."""
    d = os.path.join(root, split)
    out = {}
    model.eval()
    for f in sorted(f for f in os.listdir(d) if f.endswith(".npz")):
        data = np.load(os.path.join(d, f), allow_pickle=True)
        faces = data["faces"]
        if len(faces) == 0:
            out[f] = 0.5
            continue
        xs = []
        for face in faces:
            rgb = cv2.cvtColor(to_uint8(face), cv2.COLOR_BGR2RGB)
            rgb = np.array(Image.fromarray(rgb).resize((IMAGE_SIZE, IMAGE_SIZE)))
            xs.append(stack5(rgb))
            xs.append(stack5(rgb[:, ::-1].copy()))
        with torch.no_grad():
            out[f] = float(model(torch.stack(xs).to(device)).mean().item())
    return out


def build_grid():
    grid = []
    for a in range(21):
        for b in range(21 - a):
            for c in range(21 - a - b):
                for d in range(21 - a - b - c):
                    grid.append((a, b, c, d, 20 - a - b - c - d))
    grid = np.array(grid) / 20.0
    return grid[(grid > 0).all(axis=1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/polyglot_processed_all_unbalanced")
    ap.add_argument("--val_csv", default="paper_artifacts/source_csvs/analysis_results_VAL.csv")
    ap.add_argument("--output_dir", default="biometric_final")
    ap.add_argument("--stage1_epochs", type=int, default=30)
    ap.add_argument("--stage2_epochs", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"device: {device}")

    train_dl = DataLoader(NpzDataset(args.data_dir, "train", True), batch_size=32,
                          shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(NpzDataset(args.data_dir, "val", False), batch_size=32,
                        shuffle=False, num_workers=2, pin_memory=True)

    # ---- Stage 1: regularised training, best-validation-loss checkpoint ----
    model = FaceQualityNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.stage1_epochs)
    bce = nn.BCELoss()
    eps = 0.05

    def criterion(out, yb):
        return bce(out.clamp(1e-6, 1 - 1e-6), yb * (1 - 2 * eps) + eps)

    best_val = float("inf"); since = 0
    stage1_path = os.path.join(args.output_dir, "stage1_best.pth")
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_acc": []}
    for epoch in range(1, args.stage1_epochs + 1):
        model.train(); tl = []
        for batch in train_dl:
            x = batch["face_quality"].to(device)
            yb = batch["label"].to(device).unsqueeze(1)
            opt.zero_grad()
            loss = criterion(model(x), yb)
            loss.backward(); opt.step(); tl.append(loss.item())
        sched.step()
        model.eval(); losses, scores, labels = [], [], []
        with torch.no_grad():
            for batch in val_dl:
                x = batch["face_quality"].to(device)
                yb = batch["label"].to(device).unsqueeze(1)
                out = model(x)
                losses.append(criterion(out, yb).item())
                scores.extend(out.squeeze(1).cpu().tolist())
                labels.extend(yb.squeeze(1).cpu().tolist())
        scores = np.array(scores); labels = np.array(labels)
        vl = float(np.mean(losses)); vauc = auc_of(scores, labels)
        vacc = float(((scores >= 0.5) == (labels >= 0.5)).mean())
        history["train_loss"].append(float(np.mean(tl)))
        history["val_loss"].append(vl); history["val_auc"].append(vauc); history["val_acc"].append(vacc)
        logging.info(f"stage1 epoch {epoch:02d} train {np.mean(tl):.4f} | val loss {vl:.4f} auc {vauc:.4f} acc {vacc:.4f}")
        if vl < best_val - 1e-4:
            best_val = vl; since = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, stage1_path)
        else:
            since += 1
            if since >= 8:
                logging.info("stage1 early stop"); break
    json.dump(history, open(os.path.join(args.output_dir, "training_history.json"), "w"), indent=1)

    # ---- Stage 2: fine-tune; stopping epoch selected on validation by the
    # ensemble objective (min errors at tau=0.5, tie-break on margin) ----
    vrows = list(csv.DictReader(open(args.val_csv)))
    val_files = [r["filepath"].split("/")[-1] for r in vrows]
    val_y = np.array([r["ground_truth"] == "Fake" for r in vrows])
    S4 = np.array([[float(r[c]) for c in FIXED_AGENT_COLS] for r in vrows])
    grid = build_grid()

    def grid_objective(bio_scores):
        S = np.column_stack([S4, bio_scores])
        agg = grid @ S.T
        errs = ((agg >= 0.5) != val_y).sum(axis=1)
        mr = np.where(~val_y, agg, -1).max(axis=1)
        mf = np.where(val_y, agg, 2).min(axis=1)
        i = int(np.argmax(-errs * 1000 + (mf - mr)))
        return int(errs[i]), float((mf - mr)[i]), grid[i]

    model.load_state_dict(torch.load(stage1_path, map_location=device)["model_state_dict"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    candidates = []
    for epoch in range(0, args.stage2_epochs + 1):
        if epoch > 0:
            model.train()
            for batch in train_dl:
                x = batch["face_quality"].to(device)
                yb = batch["label"].to(device).unsqueeze(1)
                opt.zero_grad()
                loss = bce(model(x).clamp(1e-6, 1 - 1e-6), yb)
                loss.backward(); opt.step()
        vs = score_split(model, args.data_dir, "val", device)
        bio = np.array([vs[f] for f in val_files])
        e, m, w = grid_objective(bio)
        logging.info(f"stage2 epoch {epoch:02d}: val ensemble errors={e} margin={m:+.4f} w={tuple(w)}")
        candidates.append((e, -m, epoch))
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"stage2_ep{epoch:02d}.pth"))
    best_e, neg_m, best_ep = min(candidates)
    logging.info(f"selected stage2 epoch {best_ep}: val errors={best_e} margin={-neg_m:+.4f}")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, f"stage2_ep{best_ep:02d}.pth"),
                                     map_location=device))
    torch.save({"stage2_epoch": best_ep, "model_state_dict": model.state_dict()},
               os.path.join(args.output_dir, "best_model.pth"))

    # ---- Score every split with the released checkpoint ----
    for split in ("val", "test"):
        if not os.path.isdir(os.path.join(args.data_dir, split)):
            continue
        ss = score_split(model, args.data_dir, split, device)
        out_csv = os.path.join(args.output_dir, f"biometric_{split}_scores.csv")
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["split", "filepath", "ground_truth", "score"])
            for f in sorted(ss):
                data = np.load(os.path.join(args.data_dir, split, f), allow_pickle=True)
                gt = "Fake" if data["label"][0] == "fake" else "Real"
                w.writerow([split, f, gt, f"{ss[f]:.6f}"])
        scores = np.array([ss[f] for f in sorted(ss)])
        labels = np.array([np.load(os.path.join(args.data_dir, split, f), allow_pickle=True)["label"][0] == "fake"
                           for f in sorted(ss)])
        acc = float(((scores >= 0.5) == labels).mean())
        logging.info(f"{split.upper()}: n={len(scores)} acc={100*acc:.2f}% auc={auc_of(scores, labels):.6f} -> {out_csv}")


if __name__ == "__main__":
    main()
