"""Stage the preprocessed splits from Drive to local disk, file by file with retries.
Drive's FUSE mount fails intermittently under bulk copies; this resumes on re-run
(files already present with the same size are skipped).
    python tools/robust_finetune/stage_data.py --src <drive processed dir> --dst /content/pgf [--splits train val test]
"""
import argparse
import os
import shutil
import time

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
ap.add_argument("--splits", nargs="+", default=["train", "val", "test"]); ap.add_argument("--retries", type=int, default=5)
A = ap.parse_args()
for split in A.splits:
    src_dir, dst_dir = os.path.join(A.src, split), os.path.join(A.dst, split)
    if not os.path.isdir(src_dir):
        print(f"[{split}] missing in source, skipped", flush=True); continue
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".npz")); t0 = time.time(); copied = failed = 0
    for i, f in enumerate(files, 1):
        s, d = os.path.join(src_dir, f), os.path.join(dst_dir, f)
        try:
            size = os.path.getsize(s)
        except OSError:
            size = -1
        if os.path.exists(d) and size >= 0 and os.path.getsize(d) == size:
            continue
        for attempt in range(A.retries):
            try:
                shutil.copyfile(s, d + ".part"); os.replace(d + ".part", d); copied += 1; break
            except OSError as e:
                time.sleep(2 * (attempt + 1))
                if attempt == A.retries - 1:
                    failed += 1; print(f"[{split}] FAILED {f}: {e}", flush=True)
        if i % 200 == 0:
            print(f"[{split}] {i}/{len(files)} ({time.time() - t0:.0f}s)", flush=True)
    n = len([f for f in os.listdir(dst_dir) if f.endswith(".npz")])
    print(f"[{split}] done: {n}/{len(files)} present, {copied} copied now, {failed} failed, {time.time() - t0:.0f}s", flush=True)
