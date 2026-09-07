"""Cross-modal agent retrain with the anti-overfit recipe.

Same architecture, data contract and score derivation as the released agent
(20 BGR frames at 224, ImageNet norm, 128x313 log-mel, softmax over two
logits). Changes are regularisation only: clip-consistent geometric
augmentation (one flip/rotation decision per clip rather than per frame),
random frame dropout, SpecAugment-style masks on the mel, AdamW weight decay
0.05, label smoothing 0.05, cosine schedule, checkpoint on best validation
LOSS. Mels are cached once per split. Fidelity gate first: the scoring path
must reproduce the released cross-modal column from the released checkpoint.
"""
import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import cv2  # noqa: E402
import librosa  # noqa: E402
from agents.cross_modal_lipsync import CrossModal_CNN_LSTM  # noqa: E402

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

IMAGE_SIZE, MAX_FACES = 224, 20
SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, AUDIO_LEN = 16000, 2048, 512, 128, 313
ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="data/polyglot_processed_all_unbalanced")
ap.add_argument("--val_csv", default="paper_artifacts/source_csvs/analysis_results_VAL.csv")
ap.add_argument("--output_dir", default="crossmodal_trained")
ARGS = ap.parse_args()
DATA = ARGS.data_dir
OUT = ARGS.output_dir
os.makedirs(OUT, exist_ok=True)
MEL_CACHE = os.path.join(OUT, "mel_cache")
os.makedirs(MEL_CACHE, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def mel_of(split, fname, waveform):
    cache = os.path.join(MEL_CACHE, split + "__" + fname + ".npy")
    if os.path.exists(cache):
        return np.load(cache)
    out = np.zeros((N_MELS, AUDIO_LEN), dtype=np.float32)
    if waveform.size > 0:
        m = librosa.feature.melspectrogram(y=waveform.astype("float32"), sr=SAMPLE_RATE,
                                           n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        m = librosa.power_to_db(m, ref=np.max)
        if m.shape[1] > AUDIO_LEN:
            out = m[:, :AUDIO_LEN].astype(np.float32)
        else:
            out[:, :m.shape[1]] = m
    np.save(cache, out)
    return out


def frames_tensor(faces, aug=False):
    """20x3x224x224 with released prep; clip-consistent augmentation if aug."""
    n = min(len(faces), MAX_FACES)
    do_flip = aug and random.random() < 0.5
    angle = random.uniform(-10, 10) if aug and random.random() < 0.5 else 0.0
    bright = random.uniform(0.8, 1.2) if aug else 1.0
    drop = set(random.sample(range(n), k=random.randint(1, 3))) if (aug and n > 4 and random.random() < 0.3) else set()
    out = torch.zeros((MAX_FACES, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
    for i in range(n):
        if i in drop:
            continue
        face = faces[i]
        if face.dtype != np.uint8:
            face = (face * 255).astype(np.uint8) if face.max() <= 1.0 else face.astype(np.uint8)
        if do_flip:
            face = face[:, ::-1]
        if angle:
            h, w = face.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            face = cv2.warpAffine(face, M, (w, h))
        t = torch.from_numpy(np.ascontiguousarray(face)).permute(2, 0, 1).float() / 255.0
        t = transforms.functional.resize(t, (IMAGE_SIZE, IMAGE_SIZE), antialias=True)
        if bright != 1.0:
            t = (t * bright).clamp(0, 1)
        out[i] = NORM(t)
    return out


class CMDataset(Dataset):
    def __init__(self, split, aug):
        self.split = split
        self.dir = os.path.join(DATA, split)
        self.files = sorted(f for f in os.listdir(self.dir) if f.endswith(".npz"))
        self.aug = aug

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        d = np.load(os.path.join(self.dir, f), allow_pickle=True)
        faces = d["faces"]
        mel = mel_of(self.split, f, d["waveform"]).copy()
        if self.aug and random.random() < 0.5:
            f0 = random.randint(0, N_MELS - 17)
            mel[f0:f0 + random.randint(4, 16), :] = mel.min()
            t0 = random.randint(0, AUDIO_LEN - 31)
            mel[:, t0:t0 + random.randint(8, 30)] = mel.min()
        label = 1 if d["label"][0] == "fake" else 0
        return frames_tensor(faces, self.aug), torch.from_numpy(mel).unsqueeze(0), torch.tensor(label)


def score_split(model, split_dir, files=None):
    """Released scoring path: softmax P(fake)."""
    model.eval()
    files = files or sorted(f for f in os.listdir(split_dir) if f.endswith(".npz"))
    out = {}
    split = os.path.basename(split_dir.rstrip("/"))
    for f in files:
        d = np.load(os.path.join(split_dir, f), allow_pickle=True)
        vis = frames_tensor(d["faces"], aug=False).unsqueeze(0).to(device)
        mel = torch.from_numpy(mel_of(split, f, d["waveform"])).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(vis, mel)
            out[f] = float(torch.softmax(logits, dim=1)[0, 1].item())
    return out


def main():
    # ---- fidelity gate on the released checkpoint ----
    released = {r["filepath"].split("/")[-1]: float(r["score_Cross-Modal (Lip-Sync)"])
                for r in csv.DictReader(open(ARGS.val_csv))}
    model = CrossModal_CNN_LSTM().to(device)
    model.load_state_dict(torch.load(os.path.join(HERE, "..", "checkpoints/cross_modal/lip_sync_model_crossattention.pth"),
                                     map_location=device))
    val_files = sorted(f for f in os.listdir(os.path.join(DATA, "val")) if f.endswith(".npz"))
    gate = val_files[:: max(1, len(val_files) // 25)][:25]
    got = score_split(model, os.path.join(DATA, "val"), gate)
    diff = max(abs(got[f] - released[f]) for f in gate)
    print(f"fidelity gate on {len(gate)} clips: max |diff| = {diff:.6f}", flush=True)
    if diff > 0.01:
        raise SystemExit("FIDELITY GATE FAILED")

    # ---- retrain from ImageNet init ----
    model = CrossModal_CNN_LSTM().to(device)
    for p in model.cnn_base[0][15:].parameters():
        p.requires_grad = True
    ft = list(model.cnn_base[0][15:].parameters())
    ft_ids = {id(p) for p in ft}
    base = [p for p in model.parameters() if id(p) not in ft_ids and p.requires_grad]
    opt = torch.optim.AdamW([{"params": base}, {"params": ft, "lr": 1e-5}],
                            lr=1e-4, weight_decay=0.05)
    EPOCHS = 40
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    train_dl = DataLoader(CMDataset("train", True), batch_size=8, shuffle=True,
                          num_workers=4, pin_memory=False, persistent_workers=True)
    val_dl = DataLoader(CMDataset("val", False), batch_size=8, shuffle=False,
                        num_workers=2, pin_memory=False, persistent_workers=True)

    best_val = float("inf"); since = 0
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    for ep in range(1, EPOCHS + 1):
        model.train(); tl = []
        for vis, mel, y in train_dl:
            vis, mel, y = vis.to(device), mel.to(device), y.to(device)
            opt.zero_grad()
            logits, _ = model(vis, mel)
            loss = crit(logits, y)
            loss.backward(); opt.step(); tl.append(loss.item())
        sched.step()
        model.eval(); vl = []; correct = 0; total = 0
        with torch.no_grad():
            for vis, mel, y in val_dl:
                vis, mel, y = vis.to(device), mel.to(device), y.to(device)
                logits, _ = model(vis, mel)
                vl.append(crit(logits, y).item() * len(y))
                correct += int((logits.argmax(1) == y).sum()); total += len(y)
        vloss = sum(vl) / total
        hist["train_loss"].append(float(np.mean(tl)))
        hist["val_loss"].append(vloss); hist["val_acc"].append(correct / total)
        print(f"epoch {ep:02d} train {np.mean(tl):.4f} | val loss {vloss:.4f} acc {correct/total:.4f}", flush=True)
        if vloss < best_val - 1e-4:
            best_val = vloss; since = 0
            torch.save(model.state_dict(), os.path.join(OUT, "crossmodal_best.pth"))
            print("  -> new best", flush=True)
        else:
            since += 1
            if since >= 10:
                print("early stop", flush=True)
                break
    import json
    json.dump(hist, open(os.path.join(OUT, "history.json"), "w"), indent=1)

    # ---- score all splits with the released prep ----
    model.load_state_dict(torch.load(os.path.join(OUT, "crossmodal_best.pth"), map_location=device))
    for split_dir, name in [(os.path.join(DATA, "val"), "val"), (os.path.join(DATA, "test"), "test")]:
        ss = score_split(model, split_dir)
        with open(os.path.join(OUT, f"crossmodal_{name}_scores.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["filepath", "score"])
            for f in sorted(ss):
                w.writerow([f, f"{ss[f]:.6f}"])
        print(f"{name}: {len(ss)} clips scored", flush=True)
    print("CM-RETRAIN-DONE", flush=True)


if __name__ == "__main__":
    main()
