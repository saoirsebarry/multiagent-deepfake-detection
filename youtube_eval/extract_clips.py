"""Strict YouTube clip extraction: single-identity, sharp, face-only sequences.

Rebuilds the evaluation clips from the source videos listed in
youtube_eval/manifest.json. Download the videos first (yt-dlp,
bestvideo+bestaudio merged to mp4), then run this script; clip-level labels
for scoring are the ground_truth column of
paper_artifacts/source_csvs/analysis_results_youtube.csv.

Same windowing, audio and npz format as the repo extractor
(dataset_youtube.FaceAudioExtractor.extract_sequences_to_npz), with per-frame
gates the original lacks:
  det prob >= 0.95, face box min side >= 100 source px, crop Laplacian
  variance >= 60, and facenet identity continuity (cos >= 0.55 vs previous
  frame, >= 0.50 vs the sequence's first frame).
A window fails as a whole if any frame fails, mirroring the original scan.
Detection uses facenet-pytorch MTCNN (detection is dataset prep, not the
system under test).
"""
import glob
import json
import os
import re

import cv2
import librosa
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--videos_dir", required=True, help="mp4s named *_<youtube_id>.mp4")
ap.add_argument("--labels_json", required=True, help="{youtube_id: real|fake} video-level labels")
ap.add_argument("--out_dir", required=True)
ARGS = ap.parse_args()
OUT = ARGS.out_dir
os.makedirs(OUT, exist_ok=True)

SAMPLE_RATE = 16000
FRAME_COUNT, FRAME_STRIDE, IMAGE_SIZE = 10, 10, 299
AUDIO_SECONDS = 5
MAX_SEQ = 5
MIN_PROB, MIN_SIDE, MIN_SHARP = 0.95, 100, 60.0
MIN_SIM_PREV, MIN_SIM_FIRST = 0.55, 0.50

device = torch.device("cpu")
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


def embed(crop_rgb):
    t = (torch.from_numpy(cv2.resize(crop_rgb, (160, 160))).permute(2, 0, 1).float() - 127.5) / 128.0
    with torch.no_grad():
        e = resnet(t.unsqueeze(0))
    return (e / e.norm())[0]


def best_face(frame_bgr):
    """Largest-prob face; returns (crop299_bgr, crop_rgb, reason)."""
    h, w = frame_bgr.shape[:2]
    scale = 720.0 / w if w > 720 else 1.0
    small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else frame_bgr
    boxes, probs = mtcnn.detect(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
    if boxes is None:
        return None, None, "no-face"
    i = int(np.argmax(probs))
    if probs[i] < MIN_PROB:
        return None, None, "low-prob"
    x1, y1, x2, y2 = (boxes[i] / scale).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if min(x2 - x1, y2 - y1) < MIN_SIDE:
        return None, None, "too-small"
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None, "empty"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < MIN_SHARP:
        return None, None, "blurry"
    crop299 = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))
    return crop299, cv2.cvtColor(crop299, cv2.COLOR_BGR2RGB), None


def try_window(cap, start_frame):
    faces, idxs = [], []
    first_emb, prev_emb = None, None
    fail_counts = {}
    for i in range(FRAME_COUNT):
        idx = start_frame + i * FRAME_STRIDE
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            return None, {"read": 1}
        crop, crop_rgb, why = best_face(frame)
        if crop is None:
            fail_counts[why] = fail_counts.get(why, 0) + 1
            return None, fail_counts
        e = embed(crop_rgb)
        if prev_emb is not None:
            if float((e * prev_emb).sum()) < MIN_SIM_PREV or float((e * first_emb).sum()) < MIN_SIM_FIRST:
                return None, {"identity-cut": 1}
        else:
            first_emb = e
        prev_emb = e
        faces.append(crop)
        idxs.append(idx)
    return {"faces": faces, "frame_indices": idxs}, None


labels = json.load(open(ARGS.labels_json))
frame_span = (FRAME_COUNT - 1) * FRAME_STRIDE + 1
total_saved = 0
for v in sorted(glob.glob(os.path.join(ARGS.videos_dir, "*.mp4"))):
    vid = re.search(r"_([A-Za-z0-9_-]{11})\.mp4$", v).group(1)
    label = labels[vid]
    base = os.path.splitext(os.path.basename(v))[0]
    cap = cv2.VideoCapture(v)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    waveform, _ = librosa.load(v, sr=SAMPLE_RATE, mono=True)
    total_audio = len(waveform) / SAMPLE_RATE

    cur, found, reject_stats = 0, 0, {}
    while cur <= total_frames - frame_span and found < MAX_SEQ:
        seq, fails = try_window(cap, cur)
        if seq:
            mid_t = (cur + frame_span / 2) / fps
            a0 = max(0, mid_t - AUDIO_SECONDS / 2)
            a1 = min(total_audio, a0 + AUDIO_SECONDS)
            audio = waveform[int(a0 * SAMPLE_RATE):int(a1 * SAMPLE_RATE)]
            target = AUDIO_SECONDS * SAMPLE_RATE
            if len(audio) < target:
                audio = np.pad(audio, (0, target - len(audio)))
            out = os.path.join(OUT, f"{base}_seq{found:03d}_label_{label}.npz")
            np.savez_compressed(out, faces=np.array(seq["faces"]), waveform=audio,
                                label=np.array([label]),
                                metadata={"frame_indices": seq["frame_indices"]})
            found += 1
            cur += frame_span + 30
        else:
            for k, n in (fails or {}).items():
                reject_stats[k] = reject_stats.get(k, 0) + n
            cur += 30
    cap.release()
    total_saved += found
    print(f"{vid} ({label}): {found} strict sequences | window rejections: {reject_stats}", flush=True)
print("STRICT-TOTAL:", total_saved)
