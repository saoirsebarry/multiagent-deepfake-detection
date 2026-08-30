"""Precompute the ECAPA agent's feature vectors once so training reads them from disk.

OptimizedAudioDataset._extract_features runs four SpeechBrain encodes plus librosa.pyin
on every __getitem__. The backbone is frozen and nothing is augmented, so the vector for
a clip is identical in every epoch: recomputing it 50 times over is the dominant cost of
this stage. This writes each split once and train_audio_forensics_ecapa reads the cache.
"""
import argparse
import os
import sys
import time

# Must precede any librosa import: the wheel ships a numba cache compiled against a
# different numpy ABI, and loading it segfaults inside the _parabolic_interpolation gufunc.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "agents"))

_EXTRACTOR = None


def _init_worker():
    global _EXTRACTOR
    import torch

    torch.set_num_threads(1)
    from audio_forensics_ecapa import CONFIG, FastAudioFeatureExtractor
    from speechbrain.inference import EncoderClassifier

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(REPO_ROOT, "pretrained_models", "spkrec-ecapa-voxceleb"),
        run_opts={"device": "cpu"},
    )
    encoder.eval()
    _EXTRACTOR = (encoder, FastAudioFeatureExtractor(), CONFIG)


def _feature_dim(config):
    return config["model_params"]["embedding_dim"] + config["model_params"]["num_forensic_features"]


def _extract(path):
    """Mirror of OptimizedAudioDataset._extract_features, kept numerically identical."""
    import torch

    encoder, forensics, config = _EXTRACTOR
    sr = config["audio"]["sample_rate"]

    data = np.load(path, allow_pickle=True)
    waveform = data["waveform"].astype(np.float32)
    label = 1.0 if data["label"][0] == "fake" else 0.0

    if waveform.size < 400:
        return np.zeros(_feature_dim(config), dtype=np.float32), label, True

    target_length = sr * config["audio"]["duration"]
    waveform = (
        waveform[:target_length]
        if len(waveform) > target_length
        else np.pad(waveform, (0, target_length - len(waveform)))
    )

    with torch.no_grad():
        embedding = encoder.encode_batch(torch.tensor(waveform).unsqueeze(0)).squeeze().numpy()

        window_size = int(config["audio"]["window_size"] * sr)
        positions = [0, len(waveform) // 2 - window_size // 2, len(waveform) - window_size]
        embeddings = [
            encoder.encode_batch(torch.tensor(waveform[p : p + window_size]).unsqueeze(0)).squeeze().numpy()
            for p in positions
            if p >= 0 and p + window_size <= len(waveform)
        ]

    embeddings_arr = np.array(embeddings)
    if len(embeddings_arr) >= 2:
        distances = np.linalg.norm(embeddings_arr[:-1] - embeddings_arr[1:], axis=1)
        temporal_features = np.array([np.mean(distances), np.std(distances)])
        embedding_var = np.std(embeddings_arr, axis=0).mean()
    else:
        temporal_features = np.zeros(2)
        embedding_var = 0.0

    prosody = forensics.extract_fast_prosody(waveform, sr)
    artifacts = forensics.extract_fast_artifacts(waveform, sr)

    features = np.concatenate([embedding, prosody, artifacts, temporal_features, [embedding_var]])
    return features.astype(np.float32), label, False


def _extract_safe(path):
    try:
        features, label, degenerate = _extract(path)
        return os.path.basename(path), features, label, degenerate, ""
    except Exception as exc:  # a single unreadable clip must not abort the whole split
        return os.path.basename(path), None, 0.0, True, f"{type(exc).__name__}: {exc}"


def build_split(data_dir, split, out_dir, workers):
    import multiprocessing as mp

    split_dir = os.path.join(data_dir, split)
    files = sorted(f for f in os.listdir(split_dir) if f.endswith(".npz"))
    paths = [os.path.join(split_dir, f) for f in files]
    print(f"[{split}] {len(paths)} clips, {workers} workers", flush=True)

    rows, started, failures = [], time.time(), []
    # Spawn, not fork: SpeechBrain and torchaudio hold state that is not fork-safe.
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        for i, row in enumerate(pool.imap(_extract_safe, paths, chunksize=4), 1):
            rows.append(row)
            if row[4]:
                failures.append((row[0], row[4]))
            if i % 25 == 0 or i == len(paths):
                rate = i / max(time.time() - started, 1e-9)
                eta = (len(paths) - i) / max(rate, 1e-9)
                print(
                    f"[{split}] {i}/{len(paths)}  {rate:.2f} clip/s  eta {eta/60:.1f} min"
                    f"  degenerate={sum(1 for r in rows if r[3])}",
                    flush=True,
                )

    dim = max(len(r[1]) for r in rows if r[1] is not None)
    features = np.stack([r[1] if r[1] is not None else np.zeros(dim, dtype=np.float32) for r in rows])
    labels = np.array([r[2] for r in rows], dtype=np.float32)
    names = np.array([r[0] for r in rows])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split}.npz")
    # savez_compressed appends .npz unless the name already ends in it, so the staged
    # name must carry the suffix or the rename below chases a file that was never written.
    tmp_path = out_path + ".partial.npz"
    np.savez_compressed(tmp_path, features=features, labels=labels, files=names)
    os.replace(tmp_path, out_path)

    print(
        f"[{split}] wrote {out_path}  features={features.shape}  fake={int(labels.sum())}"
        f"  real={int((1 - labels).sum())}  degenerate={sum(1 for r in rows if r[3])}"
        f"  in {(time.time()-started)/60:.1f} min",
        flush=True,
    )
    for name, err in failures[:10]:
        print(f"    FAILED {name}: {err}", flush=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--limit", type=int, default=0, help="trial run over the first N clips")
    args = parser.parse_args()

    if args.limit:
        _init_worker()
        split_dir = os.path.join(args.data_dir, args.splits[0])
        files = sorted(f for f in os.listdir(split_dir) if f.endswith(".npz"))[: args.limit]
        started = time.time()
        for f in files:
            _extract_safe(os.path.join(split_dir, f))
        elapsed = time.time() - started
        print(f"trial: {len(files)} clips in {elapsed:.1f}s = {elapsed/len(files):.2f}s/clip single-worker")
        return

    for split in args.splits:
        build_split(args.data_dir, split, args.out_dir, args.workers)


if __name__ == "__main__":
    main()
