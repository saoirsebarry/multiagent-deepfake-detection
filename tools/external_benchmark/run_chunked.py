"""Stage and score a benchmark manifest in chunks, checkpointing each chunk to durable storage.

    python tools/external_benchmark/run_chunked.py --out_dir <drive dir> [--chunk 192] [--shards 2]

A Colab VM can disappear at any time and takes its local disk with it, so staging the whole
manifest before scoring risks losing hours of work. This driver cuts the manifest into chunks
and, for each one, stages the clips on local disk, scores them, appends the rows to
`<out_dir>/<tier>/scores.csv` and records the chunk in `<out_dir>/<tier>/progress.json`, then
deletes the staged clips. A crash costs at most the chunk in flight; re-running resumes from
the progress file. `--pool` reuses clips staged by an earlier run.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TIERS = [("original", None), ("crf23", 23), ("crf40", 40)]
PY = sys.executable or "python3"


def label_of(v):
    return 1 if str(v).strip().lower() in ("1", "fake", "true", "manipulated") else 0


def clip_name(row):
    # preprocess.py strips every ".mp4", so MAVOS's "<name>.mp4.mp4" loses both
    base = os.path.basename(row["path"]).replace(".mp4", "")
    return f"{base}_label_{'fake' if label_of(row['label']) else 'real'}.npz"


def write_metadata(rows, path):
    if os.path.exists(path):
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip"] + list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({"clip": clip_name(r), **r})


def append_rows(src, dst):
    rows = list(csv.DictReader(open(src)))
    if not rows:
        return 0
    new = not os.path.exists(dst)
    with open(dst, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    return len(rows)


def stage_and_score(chunk, tier, crf, args, work):
    manifest = os.path.join(work, "chunk.csv")
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(chunk[0].keys()))
        w.writeheader()
        w.writerows(chunk)

    clips = os.path.join(work, "clips")
    shutil.rmtree(clips, ignore_errors=True)
    data_dir = clips if crf is None else os.path.join(clips, f"crf{crf}")
    os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)

    if crf is None and args.pool and os.path.isdir(args.pool):
        for r in chunk:
            src = os.path.join(args.pool, clip_name(r))
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(data_dir, "test"))

    procs = []
    for k in range(args.shards):
        cmd = [PY, "-u", os.path.join(REPO, "tools", "external_benchmark", "stage_videos.py"),
               "--manifest", manifest, "--out_dir", clips, "--shard", f"{k}/{args.shards}"]
        if crf is not None:
            cmd += ["--crf", str(crf), "--skip_original"]
        log = open(os.path.join(work, f"stage_{k}.log"), "w")
        procs.append((subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT), log))
    for proc, log in procs:
        proc.wait()
        log.close()
    staged = len(os.listdir(os.path.join(data_dir, "test")))
    if not staged:
        raise RuntimeError(f"no clips staged for {tier}; see {work}/stage_*.log")

    scores = os.path.join(work, "chunk_scores.csv")
    score_log = os.path.join(work, "score.log")
    cmd = [PY, "-u", os.path.join(REPO, "tools", "source_disjoint", "score_split.py"),
           "--data_dir", data_dir, "--split", "test", "--ckpt_dir", args.ckpt_dir,
           "--out", scores, "--device", args.device, "--workers", str(args.workers)]
    with open(score_log, "w") as log:
        rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not os.path.exists(scores):
        raise RuntimeError(f"scoring failed (rc={rc}); see {score_log}")
    return staged, scores, clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--manifest", default=None, help="defaults to <out_dir>/videos.csv")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--chunk", type=int, default=192)
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--pool", default=None, help="directory of clips staged by an earlier run")
    ap.add_argument("--work", default="/content/mavos_work")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--tiers", nargs="+", default=[t for t, _ in TIERS])
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.manifest or os.path.join(a.out_dir, "videos.csv"))))
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.work, exist_ok=True)
    write_metadata(rows, os.path.join(a.out_dir, "metadata.csv"))
    chunks = [rows[i:i + a.chunk] for i in range(0, len(rows), a.chunk)]
    print(f"{len(rows)} clips in {len(chunks)} chunks of {a.chunk}", flush=True)

    tiers = [(t, c) for t, c in TIERS if t in a.tiers]
    for tier, crf in tiers:
        tier_dir = os.path.join(a.out_dir, tier)
        os.makedirs(tier_dir, exist_ok=True)
        progress_path = os.path.join(tier_dir, "progress.json")
        progress = json.load(open(progress_path)) if os.path.exists(progress_path) else {"done": []}
        scores_path = os.path.join(tier_dir, "scores.csv")

        for i, chunk in enumerate(chunks):
            if i in progress["done"]:
                continue
            t0 = time.time()
            for attempt in range(1, a.retries + 1):
                try:
                    staged, scores, clips = stage_and_score(chunk, tier, crf, a, a.work)
                    break
                except RuntimeError as exc:
                    if attempt == a.retries:
                        raise
                    # the MPS backend faults intermittently; a chunk is cheap to redo
                    print(f"{tier} chunk {i + 1}: attempt {attempt} failed ({exc}); retrying", flush=True)
                    shutil.rmtree(os.path.join(a.work, "clips"), ignore_errors=True)
                    time.sleep(15)
            n = append_rows(scores, scores_path)
            progress["done"].append(i)
            json.dump(progress, open(progress_path, "w"))
            shutil.rmtree(clips, ignore_errors=True)
            print(f"{tier} chunk {i + 1}/{len(chunks)}: {staged} staged, {n} scored, "
                  f"{round(time.time() - t0)}s", flush=True)
        print(f"TIER-DONE {tier}", flush=True)

    report = os.path.join(a.out_dir, "report.json")
    tier_scores = " ".join(f'"{os.path.join(a.out_dir, t, "scores.csv")}"' for t, _ in tiers)
    subprocess.run(
        f"cd {REPO} && {PY} -u tools/external_benchmark/report_by_group.py --scores {tier_scores} "
        f'--metadata "{os.path.join(a.out_dir, "metadata.csv")}" '
        f"--group_by generator language source_video --cluster source_video --tau 0.35 "
        f'--out "{report}"',
        shell=True, check=False,
    )
    print("MAVOS-ALL-DONE", flush=True)


if __name__ == "__main__":
    main()
