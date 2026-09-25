"""Select and download a stratified MAVOS-DD test sample and write the manifest stage_videos.py reads.

    python tools/external_benchmark/mavos_manifest.py --snapshot <dir> --out videos.csv \
        --cap 40 --real_cap 40 [--languages english spanish ...] [--seed 42]

`--snapshot` is a local copy of the metadata files of huggingface.co/datasets/unibuc-cs/MAVOS-DD
(state.json, dataset_info.json, data-*.arrow), obtained with `huggingface_hub.snapshot_download`
after accepting the dataset's terms; the videos themselves are fetched here, one file at a
time, only for the rows selected. Rows are the official test split; at most `--cap` fakes per
(language, generative method) cell and `--real_cap` reals per language, seeded. The MAVOS-DD
file name `<portrait>.png_<youtube id>_<start>_<n>.mp4.mp4` gives the portrait identity and
the driving source video, both carried into the manifest for clustered reporting.
"""
import argparse
import csv
import os
import re

import numpy as np

NAME = re.compile(r"^(?:(?P<portrait>[^/]+?\.(?:png|jpg|jpeg))_)?(?P<yt>[A-Za-z0-9_-]{11})_(?P<start>\d+)(?:_(?P<n>\d+))?\.mp4(?:\.mp4)?$")


def parse_name(path):
    base = os.path.basename(path)
    m = NAME.match(base)
    if not m:
        return {"identity": base.split("_")[0], "source_video": base.rsplit(".", 1)[0]}
    return {"identity": m.group("portrait") or m.group("yt"), "source_video": m.group("yt")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="unibuc-cs/MAVOS-DD"); ap.add_argument("--video_dir", default=None)
    ap.add_argument("--cap", type=int, default=40); ap.add_argument("--real_cap", type=int, default=40)
    ap.add_argument("--languages", nargs="*", default=None); ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--no_download", action="store_true")
    a = ap.parse_args()
    from datasets import load_from_disk
    ds = load_from_disk(a.snapshot)
    df = ds.to_pandas()
    df = df[df["split"] == a.split]
    if a.languages:
        df = df[df["language"].isin(a.languages)]
    df["is_fake"] = df["label"].astype(str).str.lower().eq("fake")
    rng = np.random.default_rng(a.seed)
    keep = []
    for (lang, method), g in df[df.is_fake].groupby(["language", "generative_method"]):
        keep.extend(rng.choice(g.index.values, min(a.cap, len(g)), replace=False).tolist())
    for lang, g in df[~df.is_fake].groupby("language"):
        keep.extend(rng.choice(g.index.values, min(a.real_cap, len(g)), replace=False).tolist())
    sel = df.loc[sorted(keep)].copy()
    video_dir = a.video_dir or os.path.join(a.snapshot, "videos")
    os.makedirs(video_dir, exist_ok=True)
    if not a.no_download:
        from huggingface_hub import hf_hub_download
        for i, p in enumerate(sel["video_path"], 1):
            hf_hub_download(a.repo, p, repo_type="dataset", local_dir=video_dir)
            if i % 50 == 0:
                print(f"downloaded {i}/{len(sel)}", flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "label", "generator", "language", "identity", "source_video", "open_set_model", "open_set_language"])
        for _, r in sel.iterrows():
            info = parse_name(r["video_path"])
            w.writerow([os.path.join(video_dir, r["video_path"]), "fake" if r["is_fake"] else "real", r["generative_method"], r["language"],
                        info["identity"], info["source_video"], r.get("open_set_model", ""), r.get("open_set_language", "")])
    print(f"{len(sel)} rows -> {a.out}; fakes {int(sel.is_fake.sum())}, reals {int((~sel.is_fake).sum())}")
    print(sel.groupby(["language", "generative_method"]).size().to_string())


if __name__ == "__main__":
    main()
