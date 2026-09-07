"""Robustness fine-tune of the FreqNet audio agent from its released checkpoint.
Released inference kept: 5 s waveform, 224-band log-mel (hop 512), min-max normalised,
3-channel, sigmoid. Fixes: waveform corruption augmentation, label smoothing 0.05,
selection by validation log-loss instead of accuracy."""
import argparse
import json
import os
import sys

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "src", "agents"))
import common as C  # noqa: E402
from audio_freqnet import FreqNet  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch", type=int, default=16)
A = ap.parse_args(); os.makedirs(A.out_dir, exist_ok=True); C.seed_all()
device = C.pick_device(allow_mps=False)  # FFT layers
SR, N_MELS, HOP, LEN = 16000, 224, 512, 5 * 16000
try:
    import torchaudio.transforms as T
    SPEC_AUG = nn.Sequential(T.FrequencyMasking(freq_mask_param=30), T.TimeMasking(time_mask_param=50))
except Exception:
    SPEC_AUG = None


def spec_of(w):
    w = w[:LEN] if len(w) > LEN else np.pad(w, (0, LEN - len(w)))
    m = librosa.power_to_db(librosa.feature.melspectrogram(y=w.astype(np.float32), sr=SR, n_mels=N_MELS, hop_length=HOP), ref=np.max)
    m = (m - m.min()) / (m.max() - m.min() + 1e-6)
    return torch.tensor(m, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)


class WavSet(Dataset):
    def __init__(self, root, split, aug):
        self.dir, self.files = C.split_files(root, split); self.aug = aug

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(os.path.join(self.dir, self.files[i]), allow_pickle=True)
        w = d["waveform"].astype(np.float32)
        if w.size == 0:
            w = np.zeros(LEN, dtype=np.float32)
        if self.aug:
            w = C.robust_audio_aug(w)
        x = spec_of(w)
        if self.aug and SPEC_AUG is not None:
            x = SPEC_AUG(x)
        return x, torch.tensor(C.label_of(d), dtype=torch.float32)


def score_split(model, root, split):
    model.eval(); d, files = C.split_files(root, split); out = {}
    with torch.no_grad():
        for f in files:
            npz = np.load(os.path.join(d, f), allow_pickle=True); w = npz["waveform"].astype(np.float32)
            if w.size < 400:
                out[f] = 0.5; continue
            out[f] = float(torch.sigmoid(model(spec_of(w).unsqueeze(0).to(device))).item())
    return out


model = FreqNet(num_classes=1).to(device)
model.load_state_dict(torch.load(os.path.join(C.REPO, "checkpoints/freqnet/freqnet_model_all_unbalanced_improved.pth"), map_location=device))
labels_of = lambda ss: {f: 1.0 if "_label_fake" in f else 0.0 for f in ss}
vs = score_split(model, A.data_dir, "val"); C.fidelity(vs, C.COLUMNS["freqnet"])
hist = [{"epoch": 0, **C.evaluate(vs, labels_of(vs))}]; print("epoch 00", hist[-1], flush=True)
best = dict(hist[-1]); best_path = os.path.join(A.out_dir, "best_model.pth"); torch.save(model.state_dict(), best_path)
dl = DataLoader(WavSet(A.data_dir, "train", True), batch_size=A.batch, shuffle=True, num_workers=C.NUM_WORKERS, drop_last=True)
opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=A.epochs)
crit = nn.BCEWithLogitsLoss(); eps = 0.05
for ep in range(1, A.epochs + 1):
    model.train(); tl = []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(); loss = crit(model(x), y * (1 - 2 * eps) + eps); loss.backward(); opt.step(); tl.append(loss.item())
    sched.step()
    vs = score_split(model, A.data_dir, "val"); m = C.evaluate(vs, labels_of(vs))
    hist.append({"epoch": ep, "train_loss": float(np.mean(tl)), **m})
    print(f"epoch {ep:02d} train {np.mean(tl):.4f} val logloss {m['logloss']:.4f} auc {m['auc']:.5f} acc {m['acc']:.4f}", flush=True)
    if m["logloss"] < best["logloss"] - 1e-4:
        best = {"epoch": ep, **m}; torch.save(model.state_dict(), best_path); print("  -> new best", flush=True)
json.dump(hist, open(os.path.join(A.out_dir, "history.json"), "w"), indent=1)
C.decide("freqnet", C.COLUMNS["freqnet"], best, A.out_dir)
model.load_state_dict(torch.load(best_path, map_location=device))
for split in ("val", "test"):
    if os.path.isdir(os.path.join(A.data_dir, split)):
        ss = score_split(model, A.data_dir, split)
        C.write_scores(os.path.join(A.out_dir, f"freqnet_{split}_scores.csv"), split, ss, labels_of(ss)); print(split, C.evaluate(ss, labels_of(ss)), flush=True)
print("FREQNET-DONE", flush=True)
