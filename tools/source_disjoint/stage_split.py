"""Copy only the clips a split manifest names from the Drive copy of the released partition.

    python tools/source_disjoint/stage_split.py --manifest <split_manifest.json> \
        --processed <drive root with train/ val/ test/> --dst /content/pgf_sd

Writes real files (not symlinks) into <dst>/{train,val,test}/ so a fresh runtime needs
about half the copy time of staging the whole released partition. Resumable: clips already
present with the right size are skipped. The manifest records where each clip came from
only by name, so every released split directory is searched.
"""
import argparse
import json
import os
import shutil
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--processed", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--retries", type=int, default=5)
    a = ap.parse_args()
    m = json.load(open(a.manifest))
    where = {}
    for split in ("train", "val", "test"):
        d = os.path.join(a.processed, split)
        if os.path.isdir(d):
            for f in os.listdir(d):
                where[f] = os.path.join(d, f)
    t0 = time.time()
    for split, names in m["clips"].items():
        out = os.path.join(a.dst, split); os.makedirs(out, exist_ok=True)
        copied = present = failed = 0
        for i, f in enumerate(names, 1):
            src = where.get(f)
            if src is None:
                print(f"[{split}] MISSING in processed tree: {f}", flush=True); failed += 1; continue
            dst = os.path.join(out, f)
            if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
                present += 1; continue
            for attempt in range(a.retries):
                try:
                    shutil.copyfile(src, dst + ".partial"); os.replace(dst + ".partial", dst); copied += 1; break
                except OSError as e:
                    time.sleep(2 * (attempt + 1))
                    if attempt == a.retries - 1:
                        print(f"[{split}] FAILED {f}: {e}", flush=True); failed += 1
            if i % 200 == 0:
                print(f"[{split}] {i}/{len(names)} ({time.time() - t0:.0f}s)", flush=True)
        print(f"[{split}] done: {len(names)} listed, {present} present, {copied} copied now, {failed} failed, {time.time() - t0:.0f}s", flush=True)
    print("STAGE-SPLIT-DONE", flush=True)


if __name__ == "__main__":
    main()
