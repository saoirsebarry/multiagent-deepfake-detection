"""Turn a video list into the clip .npz format the agents read, keeping benchmark metadata.

    python tools/external_benchmark/stage_videos.py --manifest videos.csv --out_dir <dir> \
        [--crf 23 40] [--limit N]

`videos.csv` needs `path` and `label` (real/fake or 0/1); every other column (generator,
language, source_video, identity, split ...) is carried into `<out_dir>/metadata.csv` and
joined back onto the scores by report_by_group.py. Clips are written with the released
preprocessing (MTCNN at 0.95, frame stride 10, 20 faces at 299 px, 16 kHz audio) into
`<out_dir>/test/`. With `--crf`, each video is first re-encoded with x264 at those constant
rate factors and staged again under `<out_dir>/crf<k>/test/`, the FaceForensics++ convention
for compression tiers.
"""
import argparse
import csv
import os
import subprocess
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src", "data_preprocessing"))


def label_of(v):
    v = str(v).strip().lower()
    return 1 if v in ("1", "fake", "true", "manipulated") else 0


def reencode(src, dst, crf):
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-c:v", "libx264", "-crf", str(crf),
                    "-preset", "fast", "-c:a", "aac", "-b:a", "96k", dst], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--crf", type=int, nargs="*", default=[]); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default=None, help="k/n: process only rows with index %% n == k, so several processes can stage in parallel")
    ap.add_argument("--max_faces", type=int, default=20); ap.add_argument("--frame_stride", type=int, default=10)
    ap.add_argument("--image_size", type=int, default=299); ap.add_argument("--sample_rate", type=int, default=16000)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.manifest)))
    if a.limit:
        rows = rows[: a.limit]
    all_rows = rows
    if a.shard:
        k, n = (int(x) for x in a.shard.split("/"))
        rows = [r for i, r in enumerate(rows) if i % n == k]
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    import preprocess
    from mtcnn.mtcnn import MTCNN
    detector = MTCNN()
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "metadata.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip"] + list(rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            base = os.path.basename(r["path"]).rsplit(".", 1)[0]
            w.writerow({"clip": f"{base}_label_{'fake' if label_of(r['label']) else 'real'}.npz", **r})
    tiers = [("", None)] + [(f"crf{k}", k) for k in a.crf]
    for name, crf in tiers:
        file_list = []
        for r in rows:
            src = r["path"]
            if crf is not None:
                dst = os.path.join(a.out_dir, name, "videos", os.path.basename(src).rsplit(".", 1)[0] + ".mp4")
                reencode(src, dst, crf); src = dst
            file_list.append((src, label_of(r["label"])))
        out = os.path.join(a.out_dir, name, "test") if name else os.path.join(a.out_dir, "test")
        args = types.SimpleNamespace(max_faces=a.max_faces, frame_stride=a.frame_stride, image_size=a.image_size, sample_rate=a.sample_rate)
        no_audio = preprocess.process_and_save_files(file_list, out, detector, args)
        print(f"{name or 'original'}: {len(os.listdir(out))} clips staged, {len(no_audio)} without audio", flush=True)


if __name__ == "__main__":
    main()
