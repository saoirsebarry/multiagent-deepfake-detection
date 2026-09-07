"""Robustness fine-tune of the Cross-Modal agent from its released checkpoint.
Released data contract kept: 20 frames (stored channel order) at 224 with ImageNet
normalisation, 128x313 log-mel, softmax P(fake)."""
import argparse
import json
import os
import random
import sys

import cv2
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import common as C  # noqa: E402
from agents.cross_modal_lipsync import CrossModal_CNN_LSTM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--batch", type=int, default=8)
A = ap.parse_args(); os.makedirs(A.out_dir, exist_ok=True); C.seed_all()
device = C.pick_device()
IMAGE_SIZE, MAX_FACES = 224, 20
SR, N_FFT, HOP, N_MELS, AUDIO_LEN = 16000, 2048, 512, 128, 313
NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def mel_of(waveform):
    out = np.zeros((N_MELS, AUDIO_LEN), dtype=np.float32)
    if waveform.size > 0:
        m = librosa.power_to_db(librosa.feature.melspectrogram(y=waveform.astype("float32"), sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS), ref=np.max)
        if m.shape[1] > AUDIO_LEN:
            out = m[:, :AUDIO_LEN].astype(np.float32)
        else:
            out[:, :m.shape[1]] = m
    return out


def frames_tensor(faces, aug=False):
    n = min(len(faces), MAX_FACES)
    params = C.sample_image_params() if aug else None
    angle = random.uniform(-10, 10) if aug and random.random() < 0.5 else 0.0
    drop = set(random.sample(range(n), k=random.randint(1, 3))) if (aug and n > 4 and random.random() < 0.3) else set()
    out = torch.zeros((MAX_FACES, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
    for i in range(n):
        if i in drop:
            continue
        face = C.to_uint8(faces[i])
        if aug:
            face = C.apply_image_params(face, params)
            if angle:
                h, w = face.shape[:2]
                face = cv2.warpAffine(face, cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0), (w, h))
        t = torch.from_numpy(np.ascontiguousarray(face)).permute(2, 0, 1).float() / 255.0
        out[i] = NORM(transforms.functional.resize(t, (IMAGE_SIZE, IMAGE_SIZE), antialias=True))
    return out


class CMSet(Dataset):
    def __init__(self, root, split, aug):
        self.dir, self.files = C.split_files(root, split); self.aug = aug

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(os.path.join(self.dir, self.files[i]), allow_pickle=True)
        w = d["waveform"]
        if self.aug:
            w = C.robust_audio_aug(w)
        mel = mel_of(w)
        if self.aug and random.random() < 0.5:
            f0 = random.randint(0, N_MELS - 17); mel[f0:f0 + random.randint(4, 16), :] = mel.min()
            t0 = random.randint(0, AUDIO_LEN - 31); mel[:, t0:t0 + random.randint(8, 30)] = mel.min()
        return frames_tensor(d["faces"], self.aug), torch.from_numpy(mel).unsqueeze(0), torch.tensor(int(C.label_of(d)))


def score_split(model, root, split):
    model.eval(); d, files = C.split_files(root, split); out = {}
    with torch.no_grad():
        for f in files:
            npz = np.load(os.path.join(d, f), allow_pickle=True)
            vis = frames_tensor(npz["faces"]).unsqueeze(0).to(device)
            mel = torch.from_numpy(mel_of(npz["waveform"])).unsqueeze(0).unsqueeze(0).to(device)
            logits, _ = model(vis, mel); out[f] = float(torch.softmax(logits, dim=1)[0, 1].item())
    return out


model = CrossModal_CNN_LSTM().to(device)
model.load_state_dict(torch.load(os.path.join(C.REPO, "checkpoints/cross_modal/lip_sync_model_crossattention.pth"), map_location=device))
labels_of = lambda ss: {f: 1.0 if "_label_fake" in f else 0.0 for f in ss}
vs = score_split(model, A.data_dir, "val"); C.fidelity(vs, C.COLUMNS["crossmodal"])
hist = [{"epoch": 0, **C.evaluate(vs, labels_of(vs))}]; print("epoch 00", hist[-1], flush=True)
best = dict(hist[-1]); best_path = os.path.join(A.out_dir, "best_model.pth"); torch.save(model.state_dict(), best_path)
for p in model.cnn_base[0][15:].parameters():
    p.requires_grad = True
ft = list(model.cnn_base[0][15:].parameters()); ft_ids = {id(p) for p in ft}
base = [p for p in model.parameters() if id(p) not in ft_ids and p.requires_grad]
opt = torch.optim.AdamW([{"params": base}, {"params": ft, "lr": A.lr / 4}], lr=A.lr, weight_decay=0.05)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=A.epochs)
crit = nn.CrossEntropyLoss(label_smoothing=0.05)
dl = DataLoader(CMSet(A.data_dir, "train", True), batch_size=A.batch, shuffle=True, num_workers=C.NUM_WORKERS, drop_last=True)
for ep in range(1, A.epochs + 1):
    model.train(); tl = []
    for vis, mel, y in dl:
        vis, mel, y = vis.to(device), mel.to(device), y.to(device)
        opt.zero_grad(); logits, _ = model(vis, mel); loss = crit(logits, y); loss.backward(); opt.step(); tl.append(loss.item())
    sched.step()
    vs = score_split(model, A.data_dir, "val"); m = C.evaluate(vs, labels_of(vs))
    hist.append({"epoch": ep, "train_loss": float(np.mean(tl)), **m})
    print(f"epoch {ep:02d} train {np.mean(tl):.4f} val logloss {m['logloss']:.4f} auc {m['auc']:.5f} acc {m['acc']:.4f}", flush=True)
    if m["logloss"] < best["logloss"] - 1e-4:
        best = {"epoch": ep, **m}; torch.save(model.state_dict(), best_path); print("  -> new best", flush=True)
json.dump(hist, open(os.path.join(A.out_dir, "history.json"), "w"), indent=1)
C.decide("crossmodal", C.COLUMNS["crossmodal"], best, A.out_dir)
model.load_state_dict(torch.load(best_path, map_location=device))
for split in ("val", "test"):
    if os.path.isdir(os.path.join(A.data_dir, split)):
        ss = score_split(model, A.data_dir, split)
        C.write_scores(os.path.join(A.out_dir, f"crossmodal_{split}_scores.csv"), split, ss, labels_of(ss)); print(split, C.evaluate(ss, labels_of(ss)), flush=True)
print("CROSSMODAL-DONE", flush=True)
