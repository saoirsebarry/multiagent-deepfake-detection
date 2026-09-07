"""ECAPA-TDNN head retrain with augmented-embedding training data.
The frozen SpeechBrain backbone encodes each training clip clean and K corrupted times
(gain, noise, band-limit, shift); the released 43k-parameter head is retrained on the
union with the released optimiser, then selected on clean validation log-loss.
Feature extraction mirrors OptimizedAudioDataset._extract_features exactly."""
import argparse
import json
import os
import sys

import librosa
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "src", "agents"))
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
import common as C  # noqa: E402
from audio_forensics_ecapa import CONFIG, FastAudioFeatureExtractor, OptimizedLightweightForensics  # noqa: E402
from speechbrain.inference import EncoderClassifier  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--aug_copies", type=int, default=2); ap.add_argument("--epochs", type=int, default=60)
A = ap.parse_args(); os.makedirs(A.out_dir, exist_ok=True); C.seed_all()
device = C.pick_device(allow_mps=False)
enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                     savedir=os.path.join(C.REPO, "checkpoints/speechbrain_cache"),
                                     run_opts={"device": str(device)}).eval()
fx = FastAudioFeatureExtractor(); SR = CONFIG["audio"]["sample_rate"]; DUR = SR * CONFIG["audio"]["duration"]
WIN = int(CONFIG["audio"]["window_size"] * SR); DIM = 192 + 11


def features(w):
    if w.size < 400:
        return np.zeros(DIM, np.float32)
    w = w[:DUR] if len(w) > DUR else np.pad(w, (0, DUR - len(w)))
    with torch.no_grad():
        emb = enc.encode_batch(torch.tensor(w).unsqueeze(0).to(device)).squeeze().cpu().numpy()
        pos = [0, len(w) // 2 - WIN // 2, len(w) - WIN]
        embs = np.array([enc.encode_batch(torch.tensor(w[p:p + WIN]).unsqueeze(0).to(device)).squeeze().cpu().numpy() for p in pos if p >= 0 and p + WIN <= len(w)])
    if len(embs) >= 2:
        dist = np.linalg.norm(embs[:-1] - embs[1:], axis=1); temporal = np.array([dist.mean(), dist.std()]); var = embs.std(axis=0).mean()
    else:
        temporal = np.zeros(2); var = 0.0
    return np.concatenate([emb, fx.extract_fast_prosody(w, SR), fx.extract_fast_artifacts(w, SR), temporal, [var]]).astype(np.float32)


def build(split, copies):
    cache = os.path.join(A.out_dir, f"features_{split}_k{copies}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True); return z["X"], z["y"], list(z["files"])
    d, files = C.split_files(A.data_dir, split); X, y, names = [], [], []
    for i, f in enumerate(files, 1):
        npz = np.load(os.path.join(d, f), allow_pickle=True); w = npz["waveform"].astype(np.float32); lab = C.label_of(npz)
        X.append(features(w)); y.append(lab); names.append(f)
        for _ in range(copies):
            X.append(features(C.robust_audio_aug(w))); y.append(lab); names.append(f)
        if i % 100 == 0:
            print(f"[{split}] {i}/{len(files)}", flush=True)
    X = np.stack(X); y = np.array(y, np.float32); np.savez_compressed(cache, X=X, y=y, files=np.array(names))
    return X, y, names


def score(model, mean, std, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor((X - mean) / (std + 1e-6), dtype=torch.float32).to(device)).squeeze(1)).cpu().numpy()


Xv, yv, fv = build("val", 0)
labels_v = {f: float(l) for f, l in zip(fv, yv)}
rel_stats = np.load(os.path.join(C.REPO, "checkpoints/ecapa_forensic_head/training_stats.npz"))
rel = OptimizedLightweightForensics(embedding_dim=192, num_forensic_features=11).to(device)
rel.load_state_dict(torch.load(os.path.join(C.REPO, "checkpoints/ecapa_forensic_head/audio_forensics_model_finetuned_best.pth"), map_location=device, weights_only=False))
vs = dict(zip(fv, score(rel, rel_stats["mean"], rel_stats["std"], Xv))); C.fidelity(vs, C.COLUMNS["ecapa"], tol=0.05)
released_eval = C.evaluate(vs, labels_v); print("released head on recomputed val features", released_eval, flush=True)

Xt, yt, _ = build("train", A.aug_copies)
mean, std = Xt.mean(axis=0), Xt.std(axis=0)
np.savez(os.path.join(A.out_dir, "training_stats.npz"), mean=mean, std=std)
Xn = torch.tensor((Xt - mean) / (std + 1e-6), dtype=torch.float32); Yn = torch.tensor(yt)
model = OptimizedLightweightForensics(embedding_dim=192, num_forensic_features=11).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
steps = int(np.ceil(len(Xn) / 16))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, epochs=A.epochs, steps_per_epoch=steps)
crit = nn.BCEWithLogitsLoss(); best = {"logloss": float("inf")}; hist = []
best_path = os.path.join(A.out_dir, "best_model.pth")
for ep in range(1, A.epochs + 1):
    model.train(); perm = torch.randperm(len(Xn)); tl = []
    for b in range(steps):
        idx = perm[b * 16:(b + 1) * 16]
        if len(idx) < 2:
            continue
        x, y = Xn[idx].to(device), Yn[idx].to(device)
        opt.zero_grad(); loss = crit(model(x).squeeze(1), y * 0.9 + 0.05); loss.backward(); opt.step(); sched.step(); tl.append(loss.item())
    vs = dict(zip(fv, score(model, mean, std, Xv))); m = C.evaluate(vs, labels_v)
    hist.append({"epoch": ep, "train_loss": float(np.mean(tl)), **m})
    if m["logloss"] < best["logloss"] - 1e-4:
        best = {"epoch": ep, **m}; torch.save(model.state_dict(), best_path)
    if ep % 5 == 0:
        print(f"epoch {ep:02d} train {np.mean(tl):.4f} val logloss {m['logloss']:.4f} auc {m['auc']:.5f} acc {m['acc']:.4f}", flush=True)
json.dump(hist, open(os.path.join(A.out_dir, "history.json"), "w"), indent=1)
C.decide("ecapa", C.COLUMNS["ecapa"], best, A.out_dir)
model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
for split in ("val", "test"):
    if os.path.isdir(os.path.join(A.data_dir, split)):
        X, y, files = build(split, 0); ss = dict(zip(files, score(model, mean, std, X)))
        labels = {f: float(l) for f, l in zip(files, y)}
        C.write_scores(os.path.join(A.out_dir, f"ecapa_{split}_scores.csv"), split, ss, labels); print(split, C.evaluate(ss, labels), flush=True)
print("ECAPA-DONE", flush=True)
