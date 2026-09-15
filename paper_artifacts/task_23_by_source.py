"""Task 23: results by source video, synthesis method and language rather than by clip.

    python paper_artifacts/task_23_by_source.py [--tau 0.35] [--train_dir <released train/ npz dir>]

Also records how the released clip-level partition relates to source videos: how many test
sources have their authentic clip (or a sibling forgery) in the training or validation
partition. Without `--train_dir` the training location is inferred: a test source whose
authentic clip is in neither test nor validation had it in training.
"""
from __future__ import annotations

import argparse
import os
import re

import pandas as pd

from _common import ALL_AGENT_COLS, CSV_DIR, OUT, save_json

PAT = re.compile(r"^([a-z]{2})_(\d+)(?:_to_([a-z]{2})_([A-Za-z0-9]+))?_label_(real|fake)\.npz$")


def annotate(df):
    m = df.filepath.str.extract(PAT)
    df["source"] = m[0] + "_" + m[1]; df["lang"] = m[0]; df["target_lang"] = m[2]; df["method"] = m[3]
    df["y"] = (df.ground_truth == "Fake").astype(int)
    df["agg"] = df[ALL_AGENT_COLS].mean(axis=1)
    return df


def table(df, key, tau):
    rows = []
    for k, g in df.groupby(key, dropna=False):
        pred = (g["agg"] >= tau).astype(int)
        rows.append({key: k if isinstance(k, str) else "n/a", "n": int(len(g)), "real": int((g.y == 0).sum()), "fake": int((g.y == 1).sum()),
                     "accuracy": float((pred == g.y).mean()),
                     "recall": float(pred[g.y == 1].mean()) if (g.y == 1).any() else None,
                     "specificity": float((pred[g.y == 0] == 0).mean()) if (g.y == 0).any() else None,
                     "min_fake_score": float(g.loc[g.y == 1, "agg"].min()) if (g.y == 1).any() else None,
                     "max_real_score": float(g.loc[g.y == 0, "agg"].max()) if (g.y == 0).any() else None})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--train_dir", default=None)
    a = ap.parse_args()
    test = annotate(pd.read_csv(CSV_DIR / "analysis_results_with_5_agents.csv"))
    val = annotate(pd.read_csv(CSV_DIR / "analysis_results_VAL.csv"))
    test_src = set(test.source); val_src = set(val.source)
    real_loc = {s: "test" for s in test.loc[test.y == 0, "source"]}
    real_loc.update({s: "val" for s in val.loc[val.y == 0, "source"]})
    if a.train_dir:
        train_src = {PAT.match(f).group(1) + "_" + PAT.match(f).group(2) for f in os.listdir(a.train_dir) if PAT.match(f)}
        for f in os.listdir(a.train_dir):
            m = PAT.match(f)
            if m and m.group(5) == "real":
                real_loc[f"{m.group(1)}_{m.group(2)}"] = "train"
    else:
        train_src = None
    fake_src = set(test.loc[test.y == 1, "source"])
    loc = {"test": 0, "val": 0, "train": 0}
    for s in fake_src:
        loc[real_loc.get(s, "train")] += 1
    partition = {
        "test_clips": int(len(test)), "test_sources": len(test_src), "val_sources": len(val_src),
        "test_sources_also_in_val": len(test_src & val_src),
        "test_sources_also_in_train": len(test_src & train_src) if train_src is not None else "train list not supplied; see inferred count",
        "test_fake_sources": len(fake_src),
        "test_fake_sources_whose_real_clip_is_in": loc,
        "train_location_inferred": train_src is None,
        "fakes_per_test_source": {"mean": float(test[test.y == 1].groupby("source").size().mean()), "max": int(test[test.y == 1].groupby("source").size().max())},
    }
    per_source = table(test, "source", a.tau)
    worst = sorted(per_source, key=lambda r: r["accuracy"])[:10]
    yt = pd.read_csv(CSV_DIR / "analysis_results_youtube.csv")
    yt = yt[yt.ground_truth.notna()].copy()
    yt["y"] = (yt.ground_truth == "Fake").astype(int); yt["agg"] = yt[ALL_AGENT_COLS].mean(axis=1)
    yt["label_note"] = yt.get("label_note", pd.Series([""] * len(yt))).fillna("")
    out = {"tau": a.tau, "weights": "equal",
           "released_partition_vs_sources": partition,
           "by_method": table(test, "method", a.tau), "by_source_language": table(test, "lang", a.tau),
           "by_target_language": table(test, "target_lang", a.tau),
           "sources_with_errors": [r for r in per_source if r["accuracy"] < 1.0],
           "lowest_accuracy_sources": worst,
           "youtube_by_video": [{**r, "ground_truth": str(yt.loc[yt.video_id == r["video_id"], "ground_truth"].iloc[0])} for r in table(yt, "video_id", a.tau)]}
    save_json(out, OUT / "results_by_source.json")
    pd.DataFrame(out["by_method"]).to_csv(OUT / "results_by_method.csv", index=False)
    pd.DataFrame(out["youtube_by_video"]).to_csv(OUT / "youtube_by_video.csv", index=False)
    print(pd.DataFrame(out["by_method"]).to_string(index=False))
    print(pd.DataFrame(out["youtube_by_video"]).to_string(index=False))
    print("partition:", partition)


if __name__ == "__main__":
    main()
