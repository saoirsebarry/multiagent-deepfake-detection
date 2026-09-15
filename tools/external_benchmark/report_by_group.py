"""Report scored benchmark clips by generator, language, source video and compression tier.

    python tools/external_benchmark/report_by_group.py --scores <dir>/scores.csv [--scores <dir>/crf23/scores.csv ...] \
        --metadata <dir>/metadata.csv --group_by generator language source_video --cluster source_video \
        --tau 0.35 --out report.json

Scores come from tools/source_disjoint/score_split.py (one row per clip with the five agent
columns). Metrics per group: n, accuracy, recall or specificity, AUC where both classes are
present, mean aggregate. Intervals are bootstraps over `--cluster` units (source video by
default) so clips from one video do not count as independent.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

COLS = ["score_Visual (Spatial)", "score_Audio (Mel+CNN)", "score_Audio Forensics (ECAPA)",
        "score_Cross-Modal (Lip-Sync)", "score_Facial Biometric (Quality)"]


def boot(df, tau, cluster, B, rng):
    groups = {k: v.index.values for k, v in df.groupby(cluster)} if cluster in df else None
    keys = np.array(list(groups)) if groups else None
    acc, auc = [], []
    for _ in range(B):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)]) if groups else rng.integers(0, len(df), len(df))
        d = df.iloc[idx]
        acc.append(((d.sys >= tau).astype(int) == d.y).mean())
        auc.append(roc_auc_score(d.y, d.sys) if d.y.nunique() == 2 else np.nan)
    p = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return {"accuracy": p(acc), "auc": p(auc), "n_clusters": int(len(keys)) if groups else None}


def group_table(df, key, tau):
    rows = []
    for k, g in df.groupby(key):
        pred = (g.sys >= tau).astype(int)
        rows.append({key: str(k), "n": int(len(g)), "fake": int(g.y.sum()), "accuracy": float((pred == g.y).mean()),
                     "recall": float(pred[g.y == 1].mean()) if (g.y == 1).any() else None,
                     "specificity": float((pred[g.y == 0] == 0).mean()) if (g.y == 0).any() else None,
                     "auc": float(roc_auc_score(g.y, g.sys)) if g.y.nunique() == 2 else None,
                     "mean_score": float(g.sys.mean()),
                     "per_agent_auc": {c: float(roc_auc_score(g.y, g[c])) for c in COLS} if g.y.nunique() == 2 else None})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True); ap.add_argument("--metadata", required=True)
    ap.add_argument("--group_by", nargs="+", default=["generator", "language"]); ap.add_argument("--cluster", default="source_video")
    ap.add_argument("--tau", type=float, default=0.35); ap.add_argument("--boot", type=int, default=5000); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = pd.read_csv(a.metadata)
    rng = np.random.default_rng(42)
    out = {"tau": a.tau, "weights": "equal", "cluster": a.cluster, "tiers": {}}
    for path in a.scores:
        df = pd.read_csv(path)
        df["clip"] = df.filepath.map(os.path.basename)
        df = df.merge(meta, on="clip", how="inner")
        df["y"] = (df.ground_truth == "Fake").astype(int); df["sys"] = df[COLS].mean(axis=1)
        pred = (df.sys >= a.tau).astype(int)
        tier = os.path.basename(os.path.dirname(path)) or "original"
        out["tiers"][tier] = {
            "n": int(len(df)), "n_fake": int(df.y.sum()), "accuracy": float((pred == df.y).mean()),
            "fp": int(((pred == 1) & (df.y == 0)).sum()), "fn": int(((pred == 0) & (df.y == 1)).sum()),
            "auc": float(roc_auc_score(df.y, df.sys)) if df.y.nunique() == 2 else None,
            "per_agent_auc": {c: float(roc_auc_score(df.y, df[c])) for c in COLS} if df.y.nunique() == 2 else None,
            "ci_clip": boot(df, a.tau, None, a.boot, rng), "ci_cluster": boot(df, a.tau, a.cluster, a.boot, rng),
            "escalation_rate": float((((df[COLS[2:]] >= 0.5).any(axis=1) & ~(df[COLS[2:]] >= 0.5).all(axis=1)) | (df[COLS[2:]].std(axis=1) >= 0.30)).mean()),
            "by": {k: group_table(df, k, a.tau) for k in a.group_by if k in df},
        }
        t = out["tiers"][tier]
        print(f"{tier}: n={t['n']} acc={100*t['accuracy']:.1f}% fp={t['fp']} fn={t['fn']} auc={t['auc']} cluster-CI acc {t['ci_cluster']['accuracy']}")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
