"""Retrain the Biometric-Quality agent with regularisation aimed at its
distribution-general overfit (val AUC 1.000 vs test 0.988 for the v1 model).

Changes against src/agents/biometric_quality.py, all fixed before any test
evaluation: a random face is sampled per clip each training epoch (the v1
model saw only the middle face), augmentation adds saturation/hue jitter,
Gaussian blur and random erasing, weight decay rises to 0.05, the loss uses
label smoothing 0.05, and the released checkpoint is selected by best
validation LOSS (validation AUC saturates at 1.0 and cannot rank candidates).
Single stage from ImageNet weights. Inference preprocessing is unchanged, so
the checkpoint drops into the orchestrator as-is.

Usage:
  python tools/train_biometric_v2.py --data_dir <root with train/ val/ [test/]>
Scores every split it finds with the best checkpoint (middle face, the
orchestrator's inference prep) and writes biometric_v2_scores.csv next to the
checkpoint.
"""
import argparse
import csv
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agents.biometric_quality import FaceQualityNet, PolyglotFakeDataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class AugmentedDataset(PolyglotFakeDataset):
    """Random-face sampling and stronger augmentation for training."""

    def _build_transforms(self, normalize):
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
        ]
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                       saturation=0.3, hue=0.1),
                transforms.RandomApply([transforms.GaussianBlur(5)], p=0.3),
            ])
        transform_list.append(transforms.ToTensor())
        if normalize:
            transform_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        if self.augment:
            transform_list.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)))
        return transforms.Compose(transform_list)

    def _prepare_face_quality_data(self, faces):
        if len(faces) == 0:
            return torch.zeros((5, self.image_size, self.image_size))
        idx = random.randrange(len(faces)) if self.augment else len(faces) // 2
        face = faces[idx]
        face_tensor = self.transform(face)
        quality_features = self._extract_quality_metrics(face)
        return torch.cat([face_tensor, quality_features], dim=0)


def auc_of(scores, labels):
    scores = np.asarray(scores); labels = np.asarray(labels).astype(bool)
    order = np.argsort(scores); n = len(scores); ranks = np.empty(n)
    sr = scores[order]
    _, inv, cnt = np.unique(sr, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt); ranks[order] = ((cum - cnt + cum + 1) / 2.0)[inv]
    npos = labels.sum(); nneg = n - npos
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def evaluate(model, loader, device, criterion):
    model.eval()
    losses, scores, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["face_quality"].to(device)
            yb = batch["label"].to(device).unsqueeze(1)
            out = model(x)
            losses.append(criterion(out, yb).item())
            scores.extend(out.squeeze(1).cpu().tolist())
            labels.extend(yb.squeeze(1).cpu().tolist())
    scores = np.array(scores); labels = np.array(labels)
    acc = float(((scores >= 0.5) == (labels >= 0.5)).mean())
    return float(np.mean(losses)), auc_of(scores, labels), acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/polyglot_processed_all_unbalanced")
    parser.add_argument("--output_dir", default="biometric_v2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds = AugmentedDataset(args.data_dir, "train", augment=True, normalize=True)
    val_ds = AugmentedDataset(args.data_dir, "val", augment=False, normalize=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    model = FaceQualityNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    eps = args.label_smoothing
    bce = nn.BCELoss()

    def criterion(out, yb):
        return bce(out.clamp(1e-6, 1 - 1e-6), yb * (1 - 2 * eps) + eps)

    best_val_loss = float("inf"); best_path = os.path.join(args.output_dir, "best_model.pth")
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_acc": []}
    since_best = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl = []
        for batch in train_dl:
            x = batch["face_quality"].to(device)
            yb = batch["label"].to(device).unsqueeze(1)
            opt.zero_grad()
            loss = criterion(model(x), yb)
            loss.backward(); opt.step()
            tl.append(loss.item())
        sched.step()
        vl, vauc, vacc = evaluate(model, val_dl, device, criterion)
        history["train_loss"].append(float(np.mean(tl)))
        history["val_loss"].append(vl); history["val_auc"].append(vauc); history["val_acc"].append(vacc)
        logging.info(f"epoch {epoch:02d} train {np.mean(tl):.4f} | val loss {vl:.4f} auc {vauc:.4f} acc {vacc:.4f}")
        if vl < best_val_loss - 1e-4:
            best_val_loss = vl; since_best = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_metrics": {"loss": vl, "auc": vauc, "accuracy": vacc},
                        "history": history}, best_path)
            logging.info(f"  -> new best (val loss {vl:.4f})")
        else:
            since_best += 1
            if since_best >= args.patience:
                logging.info("early stop"); break

    # Score every split with the best checkpoint using the orchestrator's prep.
    model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
    model.eval()
    prep = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((299, 299)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    import cv2
    out_csv = os.path.join(args.output_dir, "biometric_v2_scores.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["split", "filepath", "ground_truth", "score"])
        for split in ("val", "test", "train"):
            d = os.path.join(args.data_dir, split)
            if not os.path.isdir(d):
                continue
            files = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
            scores, labels = [], []
            for f in files:
                data = np.load(os.path.join(d, f), allow_pickle=True)
                faces = data["faces"]; label = 1 if data["label"][0] == "fake" else 0
                if len(faces) == 0:
                    s = 0.5
                else:
                    face = faces[len(faces) // 2 if len(faces) > 1 else 0]
                    if face.dtype != np.uint8:
                        face = (face * 255).astype(np.uint8) if face.max() <= 1.0 else face.astype(np.uint8)
                    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    blur = torch.full((1, 299, 299), cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0)
                    expo = torch.full((1, 299, 299), float(np.mean(gray)) / 255.0)
                    x = torch.cat([prep(face), blur, expo], dim=0).unsqueeze(0).float().to(device)
                    with torch.no_grad():
                        s = float(model(x).item())
                w.writerow([split, f, "Fake" if label else "Real", f"{s:.6f}"])
                scores.append(s); labels.append(label)
            scores = np.array(scores); labels = np.array(labels, dtype=bool)
            acc = float(((scores >= 0.5) == labels).mean())
            spec = float((scores[~labels] < 0.5).mean()) if (~labels).any() else float("nan")
            rec = float((scores[labels] >= 0.5).mean())
            logging.info(f"{split.upper()}: n={len(scores)} acc={100*acc:.2f}% recall={100*rec:.2f}% "
                         f"specificity={100*spec:.2f}% auc={auc_of(scores, labels):.4f}")
    json.dump(history, open(os.path.join(args.output_dir, "history.json"), "w"), indent=1)
    logging.info(f"scores -> {out_csv}")


if __name__ == "__main__":
    main()
