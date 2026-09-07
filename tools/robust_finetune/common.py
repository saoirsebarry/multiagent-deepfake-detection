"""Shared pieces for the robustness fine-tunes: device, augmentation, metrics, I/O.

Protocol (fixed before any run): gradient updates on the training split only; every
selection on the clean validation split with the released inference recipe; a retrained
agent replaces the released one only if its validation log-loss is lower and its
validation AUC-ROC is within 0.002 of the released agent's; the test split and the frozen
YouTube set are read once after all selections.
"""
import csv
import os
import random
import sys

import cv2
import numpy as np
import torch

cv2.setNumThreads(0)
# forked DataLoader workers crash on macOS with OpenCV + Metal; Linux/CUDA is fine
NUM_WORKERS = int(os.environ.get("FT_WORKERS", 0 if sys.platform == "darwin" else 4))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VAL_CSV = os.path.join(REPO, "paper_artifacts/source_csvs/analysis_results_VAL.csv")
SEED = 42
COLUMNS = {"visual": "score_Visual (Spatial)", "freqnet": "score_Audio (Mel+CNN)",
           "ecapa": "score_Audio Forensics (ECAPA)", "crossmodal": "score_Cross-Modal (Lip-Sync)",
           "biometric": "score_Facial Biometric (Quality)"}


def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(allow_mps=True):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_uint8(face):
    if face.dtype != np.uint8:
        face = (face * 255).astype(np.uint8) if face.max() <= 1.0 else face.astype(np.uint8)
    return face


def sample_image_params():
    return {"down": random.uniform(0.3, 0.9) if random.random() < 0.7 else None,
            "jpeg": random.randint(30, 90) if random.random() < 0.7 else None,
            "blur": random.uniform(0.5, 1.5) if random.random() < 0.3 else None,
            "noise": random.uniform(2, 8) if random.random() < 0.3 else None,
            "gain": (random.uniform(0.8, 1.2), random.uniform(-20, 20)) if random.random() < 0.5 else None,
            "flip": random.random() < 0.5}


def apply_image_params(img, p):
    """Deployment-style corruption of a uint8 HxWx3 frame (any channel order)."""
    h, w = img.shape[:2]
    if p["down"]:
        small = cv2.resize(img, (max(8, int(w * p["down"])), max(8, int(h * p["down"]))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if p["jpeg"]:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), p["jpeg"]])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if p["blur"]:
        img = cv2.GaussianBlur(img, (0, 0), p["blur"])
    if p["noise"]:
        img = np.clip(img.astype(np.float32) + np.random.normal(0, p["noise"], img.shape), 0, 255).astype(np.uint8)
    if p["gain"]:
        a, b = p["gain"]
        img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    if p["flip"]:
        img = img[:, ::-1]
    return np.ascontiguousarray(img)


def robust_image_aug(img):
    return apply_image_params(img, sample_image_params())


def robust_audio_aug(w, sr=16000):
    """Deployment-style corruption of a float32 waveform (length preserved)."""
    w = np.asarray(w, dtype=np.float32).copy()
    n = len(w)
    if n == 0:
        return w
    if random.random() < 0.5:
        w *= 10 ** (random.uniform(-6, 6) / 20)
    if random.random() < 0.5:
        shift = int(random.uniform(-0.3, 0.3) * sr)
        w = np.roll(w, shift)
        if shift > 0:
            w[:shift] = 0
        elif shift < 0:
            w[shift:] = 0
    if random.random() < 0.3:
        import librosa
        w = librosa.resample(librosa.resample(w, orig_sr=sr, target_sr=8000), orig_sr=8000, target_sr=sr)
        w = w[:n] if len(w) >= n else np.pad(w, (0, n - len(w)))
    if random.random() < 0.5:
        snr = random.uniform(15, 35)
        p = float(np.mean(w ** 2)) + 1e-9
        w = w + np.random.normal(0, np.sqrt(p / (10 ** (snr / 10))), w.shape).astype(np.float32)
    return np.clip(w, -1.0, 1.0).astype(np.float32)


def auc_of(scores, labels):
    scores = np.asarray(scores, dtype=np.float64); labels = np.asarray(labels).astype(bool)
    order = np.argsort(scores); n = len(scores); ranks = np.empty(n)
    _, inv, cnt = np.unique(scores[order], return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt); ranks[order] = ((cum - cnt + cum + 1) / 2.0)[inv]
    npos = labels.sum(); nneg = n - npos
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def logloss_of(scores, labels):
    s = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1 - 1e-6); y = np.asarray(labels).astype(float)
    return float(-np.mean(y * np.log(s) + (1 - y) * np.log(1 - s)))


def released_val(column):
    rows = list(csv.DictReader(open(VAL_CSV)))
    return {r["filepath"].split("/")[-1]: float(r[column]) for r in rows}


def split_files(data_dir, split):
    d = os.path.join(data_dir, split)
    return d, sorted(f for f in os.listdir(d) if f.endswith(".npz"))


def label_of(npz):
    return 1.0 if str(npz["label"][0]).lower() == "fake" else 0.0


def evaluate(scores, labels):
    s = np.array([scores[f] for f in sorted(scores)]); y = np.array([labels[f] for f in sorted(scores)])
    return {"auc": auc_of(s, y), "logloss": logloss_of(s, y), "acc": float(((s >= 0.5) == (y >= 0.5)).mean())}


def write_scores(path, split, scores, labels):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["split", "filepath", "ground_truth", "score"])
        for f in sorted(scores):
            w.writerow([split, f, "Fake" if labels[f] else "Real", f"{scores[f]:.6f}"])


def fidelity(scores, column, tol=0.02):
    rel = released_val(column)
    diff = max(abs(scores[f] - rel[f]) for f in scores if f in rel)
    print(f"fidelity vs released validation column: max |diff| = {diff:.6f}", flush=True)
    if diff > tol:
        raise SystemExit("FIDELITY GATE FAILED: scoring path does not reproduce the released agent")
    return diff


def decide(agent, column, best, out_dir):
    """Apply the pre-registered adoption rule and record the decision."""
    import json
    rel = released_val(column)
    labels = {f: 1.0 if "_label_fake" in f else 0.0 for f in rel}
    rm = evaluate(rel, labels)
    adopted = best["logloss"] < rm["logloss"] and best["auc"] >= rm["auc"] - 0.002
    summary = {"agent": agent, "released_val": rm, "best_val": best, "adopted": adopted,
               "rule": "adopt iff val logloss lower and val AUC >= released - 0.002"}
    json.dump(summary, open(os.path.join(out_dir, "decision.json"), "w"), indent=1)
    print("DECISION", json.dumps(summary), flush=True)
    return adopted
