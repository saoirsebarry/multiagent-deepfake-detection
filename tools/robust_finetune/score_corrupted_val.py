"""Score the validation split clean (k = 0) and under K deployment-style corruptions
with all five agents, using the released inference recipes, so ensemble weights can be
selected on validation under the corruptions the training fine-tunes target.
    python tools/robust_finetune/score_corrupted_val.py --data_dir <root> --out_csv val_corrupted.csv --k 2 [--ckpt_json overrides.json]
overrides.json: {"biometric": path, "crossmodal": path, "freqnet": path, "visual": path,
                 "ecapa": path, "ecapa_stats": path}  (missing keys -> released checkpoints)
"""
import argparse
import csv
import json
import os
import random
import sys

import cv2
import librosa
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src", "agents"))
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
import common as C  # noqa: E402
from train_biometric import FaceQualityNet, stack5, IMAGE_SIZE as BIO_SIZE  # noqa: E402
from agents.cross_modal_lipsync import CrossModal_CNN_LSTM  # noqa: E402
from agents.visual_xception import XceptionDeepfakeDetector  # noqa: E402
from agents.visual_xception_ch import XceptionChannelsDetector, to_input  # noqa: E402
from audio_freqnet import FreqNet  # noqa: E402
from audio_forensics_ecapa import CONFIG as ECFG, FastAudioFeatureExtractor, OptimizedLightweightForensics  # noqa: E402
from speechbrain.inference import EncoderClassifier  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True); ap.add_argument("--out_csv", required=True)
ap.add_argument("--k", type=int, default=2); ap.add_argument("--ckpt_json", default=None); ap.add_argument("--split", default="val")
A = ap.parse_args(); C.seed_all()
ovr = json.load(open(A.ckpt_json)) if A.ckpt_json else {}
dev = C.pick_device(); cpu = torch.device("cpu")
R = C.REPO
def ck(name, default): return ovr.get(name, os.path.join(R, default))

bio = FaceQualityNet().to(dev); bio.load_state_dict(torch.load(ck("biometric", "checkpoints/biometric/best_model.pth"), map_location=dev, weights_only=False)["model_state_dict"]); bio.eval()
cm = CrossModal_CNN_LSTM().to(dev); cm.load_state_dict(torch.load(ck("crossmodal", "checkpoints/cross_modal/lip_sync_model_crossattention.pth"), map_location=dev, weights_only=False)); cm.eval()
VIS_CH = ovr.get("visual_variant") == "ch"
vis = (XceptionChannelsDetector(num_classes=1, pretrained=False) if VIS_CH else XceptionDeepfakeDetector(num_classes=1)).to(dev)
_c = torch.load(ck("visual", "checkpoints/xception/polyglotfake_xception_best_unbal_all_faceaug.pth"), map_location=dev, weights_only=False); vis.load_state_dict(_c.get("model_state_dict", _c)); vis.eval()
fq_dev = dev if dev.type == "cuda" else cpu
fq = FreqNet(num_classes=1).to(fq_dev); fq.load_state_dict(torch.load(ck("freqnet", "checkpoints/freqnet/freqnet_model_all_unbalanced_improved.pth"), map_location=fq_dev, weights_only=False)); fq.eval()
ec = OptimizedLightweightForensics(embedding_dim=192, num_forensic_features=11).to(dev); ec.load_state_dict(torch.load(ck("ecapa", "checkpoints/ecapa_forensic_head/audio_forensics_model_finetuned_best.pth"), map_location=dev, weights_only=False)); ec.eval()
stats = np.load(ck("ecapa_stats", "checkpoints/ecapa_forensic_head/training_stats.npz"))
enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=os.path.join(R, "checkpoints/speechbrain_cache"), run_opts={"device": str(dev)}).eval()
fx = FastAudioFeatureExtractor()

TF_VIS = transforms.Compose([transforms.ToPILImage(), transforms.Resize((299, 299)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
SR = 16000


def s_visual(faces):
    ts = []
    for f in faces:
        t = TF_VIS(C.to_uint8(f)); ts.append(t); ts.append(torch.flip(t, dims=[2]))
    with torch.no_grad():
        return float(torch.sigmoid(vis(torch.stack(ts).to(dev))).mean().item())


def s_bio(faces):
    xs = []
    for f in faces:
        rgb = cv2.cvtColor(C.to_uint8(f), cv2.COLOR_BGR2RGB); rgb = np.array(Image.fromarray(rgb).resize((BIO_SIZE, BIO_SIZE)))
        xs.append(stack5(rgb)); xs.append(stack5(rgb[:, ::-1].copy()))
    with torch.no_grad():
        return float(bio(torch.stack(xs).to(dev)).mean().item())


def s_cm(faces, w):
    n = min(len(faces), 20); out = torch.zeros((20, 3, 224, 224))
    for i in range(n):
        t = torch.from_numpy(np.ascontiguousarray(C.to_uint8(faces[i]))).permute(2, 0, 1).float() / 255.0
        out[i] = NORM(transforms.functional.resize(t, (224, 224), antialias=True))
    mel = np.zeros((128, 313), dtype=np.float32)
    if w.size > 0:
        m = librosa.power_to_db(librosa.feature.melspectrogram(y=w.astype("float32"), sr=SR, n_fft=2048, hop_length=512, n_mels=128), ref=np.max)
        mel[:, :min(313, m.shape[1])] = m[:, :313]
    with torch.no_grad():
        logits, _ = cm(out.unsqueeze(0).to(dev), torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(dev))
        return float(torch.softmax(logits, dim=1)[0, 1].item())


def s_freq(w):
    L = 5 * SR; w = w[:L] if len(w) > L else np.pad(w, (0, L - len(w)))
    m = librosa.power_to_db(librosa.feature.melspectrogram(y=w.astype(np.float32), sr=SR, n_mels=224, hop_length=512), ref=np.max)
    m = (m - m.min()) / (m.max() - m.min() + 1e-6)
    with torch.no_grad():
        return float(torch.sigmoid(fq(torch.tensor(m, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(fq_dev))).item())


def s_ecapa(w):
    if w.size < 400:
        return 0.5
    D = SR * 6; WIN = int(2.0 * SR); w = w[:D] if len(w) > D else np.pad(w, (0, D - len(w)))
    with torch.no_grad():
        emb = enc.encode_batch(torch.tensor(w).unsqueeze(0).to(dev)).squeeze().cpu().numpy()
        pos = [0, len(w) // 2 - WIN // 2, len(w) - WIN]
        embs = np.array([enc.encode_batch(torch.tensor(w[p:p + WIN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy() for p in pos if p >= 0 and p + WIN <= len(w)])
    if len(embs) >= 2:
        d = np.linalg.norm(embs[:-1] - embs[1:], axis=1); temporal = np.array([d.mean(), d.std()]); var = embs.std(axis=0).mean()
    else:
        temporal = np.zeros(2); var = 0.0
    feat = np.concatenate([emb, fx.extract_fast_prosody(w, SR), fx.extract_fast_artifacts(w, SR), temporal, [var]]).astype(np.float32)
    x = torch.tensor((feat - stats["mean"]) / (stats["std"] + 1e-6), dtype=torch.float32).unsqueeze(0).to(dev)
    with torch.no_grad():
        return float(torch.sigmoid(ec(x)).item())


d, files = C.split_files(A.data_dir, A.split)
with open(A.out_csv, "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["filepath", "ground_truth", "k"] + [C.COLUMNS[a] for a in ["visual", "freqnet", "ecapa", "crossmodal", "biometric"]])
    for i, f in enumerate(files, 1):
        npz = np.load(os.path.join(d, f), allow_pickle=True); faces0 = npz["faces"]; w0 = npz["waveform"].astype(np.float32)
        gt = "Fake" if C.label_of(npz) else "Real"
        for k in range(A.k + 1):
            C._seed(f, f"k{k}")
            if k == 0:
                faces, w = faces0, w0
            else:
                p = C.sample_image_params(); p["flip"] = False
                faces = [C.apply_image_params(C.to_uint8(x), p) for x in faces0]; w = C.robust_audio_aug(w0)
            row = [f, gt, k, s_visual(faces) if len(faces) else 0.5, s_freq(w) if w.size else 0.5, s_ecapa(w), s_cm(faces, w), s_bio(faces) if len(faces) else 0.5]
            wr.writerow([row[0], row[1], row[2]] + [f"{v:.6f}" for v in row[3:]]); fh.flush()
        if i % 25 == 0:
            print(f"[{i}/{len(files)}]", flush=True)
print("CORRUPTED-VAL-DONE", A.out_csv, flush=True)
