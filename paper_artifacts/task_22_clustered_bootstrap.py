"""Task 22: source-clustered uncertainty for the released per-clip scores.

Clips that derive from one source video are not independent, so clip-level resampling can
understate the interval. This resamples source videos (PolyGlotFake) and source videos
(YouTube) with replacement and reports both intervals side by side, at the paper's equal
weights and operating threshold.

    python paper_artifacts/task_22_clustered_bootstrap.py [--tau 0.35] [--boot 10000]
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import average_precision_score, roc_auc_score

from _common import ALL_AGENT_COLS, CSV_DIR, OUT, RANDOM_SEED, save_json

PAT = re.compile(r"^([a-z]{2})_(\d+)(?:_to_[a-z]{2}_[A-Za-z0-9]+)?_label_(real|fake)\.npz$")


def source_of(name: str) -> str:
    m = PAT.match(name.split("/")[-1])
    return f"{m.group(1)}_{m.group(2)}" if m else name


def intervals(df, score, y, pred, cluster, B, rng):
    groups = {k: v.index.values for k, v in df.groupby(cluster)}
    keys = np.array(list(groups))
    clip, clus = {"acc": [], "auc": [], "ap": [], "spec": [], "sens": []}, {"acc": [], "auc": [], "ap": [], "spec": [], "sens": []}
    for _ in range(B):
        for store, idx in ((clip, rng.integers(0, len(df), len(df))),
                           (clus, np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)]))):
            yb, sb, pb = y[idx], score[idx], pred[idx]
            store["acc"].append((pb == yb).mean())
            store["spec"].append((pb[yb == 0] == 0).mean() if (yb == 0).any() else np.nan)
            store["sens"].append((pb[yb == 1] == 1).mean() if (yb == 1).any() else np.nan)
            two = yb.min() != yb.max()
            store["auc"].append(roc_auc_score(yb, sb) if two else np.nan)
            store["ap"].append(average_precision_score(yb, sb) if two else np.nan)
    pct = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return {"clip_resampled": {k: pct(v) for k, v in clip.items()},
            "cluster_resampled": {k: pct(v) for k, v in clus.items()},
            "n_clusters": int(len(keys)),
            "clusters_per_split_note": "cluster = source video (PolyGlotFake clip name prefix) or video_id (YouTube)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--boot", type=int, default=10_000)
    a = ap.parse_args()
    rng = np.random.default_rng(RANDOM_SEED)
    out = {"tau": a.tau, "weights": "equal (0.20 each)", "bootstrap_resamples": a.boot, "seed": RANDOM_SEED}

    df = pd.read_csv(CSV_DIR / "analysis_results_with_5_agents.csv")
    df["source"] = df.filepath.map(source_of)
    y = (df.ground_truth == "Fake").astype(int).to_numpy()
    score = df[ALL_AGENT_COLS].mean(axis=1).to_numpy()
    pred = (score >= a.tau).astype(int)
    errors = int((pred != y).sum()); n = len(y)
    cp = [float(sps.beta.ppf(0.025, n - errors, errors + 1)), float(sps.beta.ppf(0.975, n - errors + 1, errors)) if errors else 1.0]
    per_source = df.assign(err=(pred != y)).groupby("source").err.sum()
    out["polyglotfake_test"] = {
        "n": n, "errors": errors, "accuracy": float((pred == y).mean()),
        "clopper_pearson_accuracy": cp,
        "n_sources": int(df.source.nunique()),
        "sources_with_an_error": int((per_source > 0).sum()),
        "max_errors_in_one_source": int(per_source.max()),
        **intervals(df, score, y, pred, "source", a.boot, rng),
    }

    yt = pd.read_csv(CSV_DIR / "analysis_results_youtube.csv")
    yt = yt[yt.ground_truth.notna()].reset_index(drop=True)
    yy = (yt.ground_truth == "Fake").astype(int).to_numpy()
    ys = yt[ALL_AGENT_COLS].mean(axis=1).to_numpy()
    yp = (ys >= a.tau).astype(int)
    out["youtube"] = {"n": int(len(yt)), "n_videos": int(yt.video_id.nunique()), "errors": int((yp != yy).sum()),
                      "accuracy": float((yp == yy).mean()), **intervals(yt, ys, yy, yp, "video_id", a.boot, rng)}
    save_json(out, OUT / "clustered_bootstrap.json")
    for k in ("polyglotfake_test", "youtube"):
        r = out[k]
        print(f"{k}: n={r['n']} clusters={r['n_clusters']} acc={100*r['accuracy']:.2f}% | "
              f"acc CI clip {r['clip_resampled']['acc']} cluster {r['cluster_resampled']['acc']} | "
              f"AUC CI clip {r['clip_resampled']['auc']} cluster {r['cluster_resampled']['auc']}")


if __name__ == "__main__":
    main()
