"""Retrain an 8-channel visual XceptionNet (RGB plus error-level-analysis map, log-magnitude
spectrum, local-binary-pattern code and Cb/Cr chroma, src/agents/visual_xception_ch.py) from
ImageNet weights with the released two-stage recipe (head-only epochs at 1e-3, then unfreeze from block 11 for OneCycle epochs at 1e-5
with Mixup, label smoothing 0.1 and balanced class weights) plus three fixes fixed before
the run: a deployment-corruption augmentation block applied to every training crop, which
breaks the sharpness shortcut in the stored crops; clip-level validation selection under
the released inference recipe; per-epoch checkpoints for the corrupted-validation rule.
Every score written uses the released inference recipe unchanged."""
import argparse
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import common as C  # noqa: E402
from agents.visual_xception import XceptionDeepfakeDetector  # noqa: E402
from agents.visual_xception_ch import XceptionChannelsDetector, to_input  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--stage1_epochs", type=int, default=5); ap.add_argument("--stage2_epochs", type=int, default=20)
ap.add_argument("--batch", type=int, default=32); ap.add_argument("--patience", type=int, default=7)
ap.add_argument("--smoke", action="store_true", help="few batches per epoch, for a pipeline check")
A = ap.parse_args(); os.makedirs(A.out_dir, exist_ok=True); C.seed_all()
device = C.pick_device(); AMP = device.type == "cuda"
RELEASED = os.path.join(C.REPO, "checkpoints/xception/polyglotfake_xception_best_unbal_all_faceaug.pth")
INITIAL_LR, FINE_TUNE_LR, WEIGHT_DECAY, LABEL_SMOOTHING, MIXUP_ALPHA, FINE_TUNE_AT_BLOCK = 1e-3, 1e-5, 1e-4, 0.1, 0.2, 11
TF = transforms.Compose([transforms.ToPILImage(), transforms.Resize((299, 299)), transforms.ToTensor(),
                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def preload(split):
    d, files = C.split_files(A.data_dir, split)
    if A.smoke:
        files = files[:24]
    faces, labels = [], []
    for f in files:
        z = np.load(os.path.join(d, f), allow_pickle=True)
        faces.append(np.ascontiguousarray(C.to_uint8(z["faces"]))); labels.append(C.label_of(z))
    log(f"{split}: {len(files)} clips, {sum(len(x) for x in faces)} faces in memory")
    return files, faces, np.array(labels, dtype=np.float32)


def sharpness(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def clip8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def cutout(img):
    h, w = img.shape[:2]; img = img.copy()
    for _ in range(2):
        hh, ww = random.randint(1, 49), random.randint(1, 49)
        if random.random() < 0.8:
            y = random.randint(max(0, h // 4), min(h, 3 * h // 4) - hh); x = random.randint(max(0, w // 4), min(w, 3 * w // 4) - ww)
        else:
            y = random.randint(0, h - hh); x = random.randint(0, w - ww)
        img[y:y + hh, x:x + ww] = 0
    return img


def grid_distort(img, steps=5, limit=0.3):
    h, w = img.shape[:2]
    xs = np.linspace(0, w, steps + 1); ys = np.linspace(0, h, steps + 1)
    dx = np.cumsum(np.diff(xs) * (1 + np.random.uniform(-limit, limit, steps))); dy = np.cumsum(np.diff(ys) * (1 + np.random.uniform(-limit, limit, steps)))
    xs2 = np.concatenate([[0], dx]) * (w / dx[-1]); ys2 = np.concatenate([[0], dy]) * (h / dy[-1])
    mx = np.interp(np.arange(w), xs2, xs).astype(np.float32); my = np.interp(np.arange(h), ys2, ys).astype(np.float32)
    return cv2.remap(img, np.tile(mx, (h, 1)), np.tile(my[:, None], (1, w)), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def released_family_aug(img):
    """The released training augmentation family, re-implemented with OpenCV at the same
    magnitudes: resize 319 + random crop 299, flip, one photometric op, one noise/blur op,
    face cutout, shift-scale-rotate, grid distortion, gamma, JPEG 70-100."""
    img = cv2.resize(img, (319, 319), interpolation=cv2.INTER_LINEAR)
    y, x = random.randint(0, 20), random.randint(0, 20); img = img[y:y + 299, x:x + 299]
    if random.random() < 0.5:
        img = img[:, ::-1]
    img = np.ascontiguousarray(img)
    if random.random() < 0.8:
        k = random.randrange(3)
        if k == 0:
            img = clip8(img.astype(np.float32) * (1 + random.uniform(-0.3, 0.3)) + random.uniform(-0.3, 0.3) * 255)
        elif k == 1:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
            hsv[..., 0] = (hsv[..., 0] + random.randint(-20, 20)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-30, 30), 0, 255); hsv[..., 2] = np.clip(hsv[..., 2] + random.randint(-20, 20), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        else:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB); lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if random.random() < 0.5:
        k = random.randrange(3)
        if k == 0:
            img = clip8(img.astype(np.float32) + np.random.normal(0, np.sqrt(random.uniform(10, 50)), img.shape))
        elif k == 1:
            ks = random.choice([3, 5, 7]); img = cv2.GaussianBlur(img, (ks, ks), 0)
        else:
            img = cv2.medianBlur(img, random.choice([3, 5]))
    if random.random() < 0.5:
        img = cutout(img)
    if random.random() < 0.6:
        M = cv2.getRotationMatrix2D((149.5, 149.5), random.uniform(-15, 15), 1 + random.uniform(-0.15, 0.15))
        M[:, 2] += (random.uniform(-0.1, 0.1) * 299, random.uniform(-0.1, 0.1) * 299)
        img = cv2.warpAffine(img, M, (299, 299), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if random.random() < 0.3:
        img = grid_distort(img)
    if random.random() < 0.3:
        g = random.uniform(0.8, 1.2); img = cv2.LUT(img, clip8(((np.arange(256) / 255.0) ** g) * 255))
    if random.random() < 0.4:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(70, 100)])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def corruption_block(img):
    p = C.sample_image_params(); p["flip"] = False
    return C.apply_image_params(img, p)


def to_tensor(img):
    return torch.from_numpy(((img.astype(np.float32) / 255.0) - 0.5) / 0.5).permute(2, 0, 1)


class FaceSet(Dataset):
    def __init__(self, faces, labels):
        self.faces, self.labels = faces, labels
        self.index = [(ci, fi) for ci, arr in enumerate(faces) for fi in range(len(arr))]
        if A.smoke:
            self.index = self.index[:6 * A.batch]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ci, fi = self.index[i]
        img = released_family_aug(corruption_block(self.faces[ci][fi]))
        return to_input(img), torch.tensor(self.labels[ci])


def score_faces(model, faces_list, keys, corrupt=False):
    seven = isinstance(model, XceptionChannelsDetector); model.eval(); out = {}
    with torch.no_grad():
        for f, faces in zip(keys, faces_list):
            if len(faces) == 0:
                out[f] = 0.5; continue
            fs = C.corrupt_faces(faces, f) if corrupt else faces
            ts = []
            for face in fs:
                u = C.to_uint8(face)
                if seven:
                    ts.append(to_input(u)); ts.append(to_input(np.ascontiguousarray(u[:, ::-1])))
                else:
                    t = TF(u); ts.append(t); ts.append(torch.flip(t, dims=[2]))
            out[f] = float(torch.sigmoid(model(torch.stack(ts).to(device))).mean().item())
    return out


def score_split_lazy(model, split):
    d, files = C.split_files(A.data_dir, split); out = {}
    for f in files:
        faces = np.load(os.path.join(d, f), allow_pickle=True)["faces"]
        out.update(score_faces(model, [faces], [f]))
    return out


def val_metrics(model):
    clean = C.evaluate(score_faces(model, val_faces, val_files), LABELS_V)
    corr = C.evaluate(score_faces(model, val_faces, val_files, corrupt=True), LABELS_V)
    return {**clean, **{"corr_" + k: v for k, v in corr.items()}}


def run_epoch(model, dl, opt, class_w, ep, sched=None, use_mixup=False):
    model.train(); crit = nn.BCEWithLogitsLoss(reduction="none"); tl = []; correct = 0; total = 0
    for x, y in dl:
        x, y = x.to(device, non_blocking=True), y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            out = model(x).squeeze(1)
        out = out.float()
        if use_mixup and random.random() > 0.5:
            lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)); idx = torch.randperm(x.size(0), device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
                out = model(lam * x + (1 - lam) * x[idx]).squeeze(1)
            out = out.float(); ya, yb = y, y[idx]
            sa = ya * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING; sb = yb * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
            la = (crit(out, sa) * torch.where(ya == 1, class_w[1], class_w[0])).mean(); lb = (crit(out, sb) * torch.where(yb == 1, class_w[1], class_w[0])).mean()
            loss = lam * la + (1 - lam) * lb
        else:
            s = y * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
            loss = (crit(out, s) * torch.where(y == 1, class_w[1], class_w[0])).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if sched is not None:
            sched.step()
        tl.append(loss.item()); correct += int(((out > 0).float() == y).sum().item()); total += y.numel()
    return float(np.mean(tl)), correct / max(total, 1)


train_files, train_faces, train_y = preload("train"); val_files, val_faces, val_y = preload("val")
LABELS_V = dict(zip(val_files, val_y.tolist()))

# sharpness diagnostic: the stored crops carry a class-correlated sharpness cue; the corruption block should remove it
diag = {}
for cls, name in ((0.0, "real"), (1.0, "fake")):
    idx = [i for i, y in enumerate(train_y) if y == cls]; random.Random(0).shuffle(idx); before, after = [], []
    for ci in idx[:100]:
        face = train_faces[ci][len(train_faces[ci]) // 2]; before.append(sharpness(face)); after.append(sharpness(corruption_block(face)))
    diag[name] = {"n": len(before), "median_laplacian_var_stored": float(np.median(before)), "median_laplacian_var_after_corruption": float(np.median(after))}
json.dump(diag, open(os.path.join(A.out_dir, "sharpness_diagnostic.json"), "w"), indent=1); log("sharpness diagnostic", diag)

# reference row: the released checkpoint scored on validation, clean and corrupted (fidelity-gated)
ref = XceptionDeepfakeDetector(num_classes=1, pretrained=False).to(device)
ck = torch.load(RELEASED, map_location=device, weights_only=False); ref.load_state_dict(ck.get("model_state_dict", ck))
vs = score_faces(ref, val_faces, val_files)
if not A.smoke:
    C.fidelity(vs, C.COLUMNS["visual"])
hist = [{"epoch": 0, "stage": "released", **C.evaluate(vs, LABELS_V), **{"corr_" + k: v for k, v in C.evaluate(score_faces(ref, val_faces, val_files, corrupt=True), LABELS_V).items()}}]
log("released reference", hist[0]); del ref, ck

model = XceptionChannelsDetector(num_classes=1, dropout_rate=0.5, pretrained=True).to(device)
pos = float(train_y.mean()); class_w = torch.tensor([0.5 / (1 - pos), 0.5 / pos], device=device); log("class weights", class_w.tolist())
dl = DataLoader(FaceSet(train_faces, train_y), batch_size=A.batch, shuffle=True, num_workers=C.NUM_WORKERS, pin_memory=AMP, drop_last=True,
                persistent_workers=C.NUM_WORKERS > 0)
best_path, robust_path = os.path.join(A.out_dir, "best_model.pth"), os.path.join(A.out_dir, "best_robust.pth")
best_ll, best_rob, bad = float("inf"), float("inf"), 0
robust = lambda m: 0.5 * (m["logloss"] + m["corr_logloss"])


def after_epoch(ep, stage, tr_loss, tr_acc):
    global best_ll, best_rob, bad
    m = val_metrics(model); hist.append({"epoch": ep, "stage": stage, "train_loss": tr_loss, "train_acc": tr_acc, **m})
    torch.save({"epoch": ep, "model_state_dict": model.state_dict()}, os.path.join(A.out_dir, f"epoch{ep:02d}.pth"))
    if m["logloss"] < best_ll - 1e-4:
        best_ll = m["logloss"]; bad = 0; torch.save({"epoch": ep, "model_state_dict": model.state_dict()}, best_path)
    else:
        bad += 1
    if robust(m) < best_rob - 1e-4:
        best_rob = robust(m); torch.save({"epoch": ep, "model_state_dict": model.state_dict()}, robust_path)
    log(f"epoch {ep:02d} [{stage}] train {tr_loss:.4f}/{tr_acc:.3f} | val logloss {m['logloss']:.4f} auc {m['auc']:.5f} acc {m['acc']:.4f} | corrupted logloss {m['corr_logloss']:.4f} auc {m['corr_auc']:.5f}")
    json.dump(hist, open(os.path.join(A.out_dir, "history.json"), "w"), indent=1)
    return m


log("stage 1: head only")
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=INITIAL_LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
plateau = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, min_lr=1e-7)
ep = 0
for _ in range(A.stage1_epochs):
    ep += 1; tr = run_epoch(model, dl, opt, class_w, ep, use_mixup=False); m = after_epoch(ep, "head", *tr); plateau.step(m["logloss"])

log("stage 2: unfreeze from block", FINE_TUNE_AT_BLOCK)
model.unfreeze_from_block(FINE_TUNE_AT_BLOCK)
groups = {"base": [], "attention": [], "fc": []}
for name, p in model.named_parameters():
    if p.requires_grad:
        groups["fc" if "fc" in name else "attention" if "attention" in name else "base"].append(p)
opt = torch.optim.AdamW([{"params": groups["base"], "lr": FINE_TUNE_LR}, {"params": groups["attention"], "lr": FINE_TUNE_LR * 5},
                         {"params": groups["fc"], "lr": FINE_TUNE_LR * 10}], weight_decay=WEIGHT_DECAY)
sched = OneCycleLR(opt, max_lr=[FINE_TUNE_LR, FINE_TUNE_LR * 5, FINE_TUNE_LR * 10], total_steps=A.stage2_epochs * len(dl), pct_start=0.3, anneal_strategy="cos")
bad = 0
for k in range(A.stage2_epochs):
    ep += 1
    if k == 4:
        model.unfreeze_from_block(FINE_TUNE_AT_BLOCK - 1)  # as coded in the released script: these parameters are not in the optimiser
    tr = run_epoch(model, dl, opt, class_w, ep, sched=sched, use_mixup=True); after_epoch(ep, "finetune", *tr)
    if bad >= A.patience:
        log(f"early stop after {A.patience} epochs without validation log-loss improvement"); break

summary = C.decide("visual", C.COLUMNS["visual"], hist, A.out_dir); summary["variant"] = "ch"
json.dump(summary, open(os.path.join(A.out_dir, "decision.json"), "w"), indent=1)
final_path = robust_path if summary["adopted_rule"] == "rule2" else best_path
model.load_state_dict(torch.load(final_path, map_location=device, weights_only=False)["model_state_dict"])
for split in ("val", "test"):
    if os.path.isdir(os.path.join(A.data_dir, split)):
        ss = score_split_lazy(model, split) if split == "test" else score_faces(model, val_faces, val_files)
        labels = {f: 1.0 if "_label_fake" in f else 0.0 for f in ss}
        C.write_scores(os.path.join(A.out_dir, f"visual_{split}_scores.csv"), split, ss, labels); log(split, C.evaluate(ss, labels))
log("XCEPTION-CH-DONE")
