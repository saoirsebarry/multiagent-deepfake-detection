"""Read out one or more seeded runs on a source-disjoint partition.

    python tools/source_disjoint/analyse.py --runs <run_dir> [<run_dir> ...] --out readout.json [--tau 0.35]

For every run: the threshold is set on that run's validation scores by the protocol's rule
(equal weights; the lowest-error band with no missed fake, taking its midpoint), or held at
--tau and checked against that band, the test partition is read once at that threshold, and every interval is a source-clustered
bootstrap (sources resampled with replacement) beside the clip-level one. Across runs the
mean and standard deviation of each metric are reported, plus the pooled per-clip agreement.
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import average_precision_score, roc_auc_score

COLS = ["score_Visual (Spatial)", "score_Audio (Mel+CNN)", "score_Audio Forensics (ECAPA)",
        "score_Cross-Modal (Lip-Sync)", "score_Facial Biometric (Quality)"]
TRIO = ["score_Audio Forensics (ECAPA)", "score_Cross-Modal (Lip-Sync)", "score_Facial Biometric (Quality)"]
PAT = re.compile(r"^([a-z]{2})_(\d+)(?:_to_([a-z]{2})_([A-Za-z0-9]+))?_label_(real|fake)\.npz$")


def load(path):
    df = pd.read_csv(path)
    df["filepath"] = df.filepath.map(os.path.basename)
    m = df.filepath.str.extract(PAT)
    df["source"] = m[0] + "_" + m[1]
    df["method"] = m[3]
    df["y"] = (df.ground_truth == "Fake").astype(int)
    df["sys"] = df[COLS].mean(axis=1)
    df["trio"] = df[TRIO].mean(axis=1)
    return df


def select_tau(val):
    """Midpoint of the widest interval that leaves no validation fake missed at minimum error."""
    s = np.sort(np.unique(np.concatenate([val["sys"].values, [0.0, 1.0]])))
    cands = (s[:-1] + s[1:]) / 2
    fn = np.array([((val["sys"] < t) & (val.y == 1)).sum() for t in cands])
    err = np.array([((val["sys"] >= t) != (val.y == 1)).sum() for t in cands])
    ok = fn == 0
    best = err[ok].min()
    idx = np.where(ok & (err == best))[0]
    # widest contiguous run of admissible cut-points
    runs, start = [], idx[0]
    for i, j in zip(idx, idx[1:]):
        if j != i + 1:
            runs.append((start, i)); start = j
    runs.append((start, idx[-1]))
    lo_i, hi_i = max(runs, key=lambda r: s[r[1] + 1] - s[r[0]])
    lo, hi = float(s[lo_i]), float(s[hi_i + 1])
    return {"tau": round((lo + hi) / 2, 4), "band": [lo, hi], "val_errors": int(best), "val_fn": 0}


def bootstrap(df, tau, B, seed, cluster):
    rng = np.random.default_rng(seed)
    pred = (df["sys"] >= tau).astype(int).values
    y = df.y.values; s = df["sys"].values
    groups = {k: v.index.values for k, v in df.groupby(cluster)} if cluster else None
    keys = np.array(list(groups)) if groups else None
    acc, auc = [], []
    for _ in range(B):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)]) if groups else rng.integers(0, len(df), len(df))
        yb = y[idx]
        acc.append((pred[idx] == yb).mean())
        auc.append(roc_auc_score(yb, s[idx]) if yb.min() != yb.max() else np.nan)
    pct = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return {"accuracy": pct(acc), "auc": pct(auc), "n_clusters": int(len(keys)) if groups else None}


def mcnemar(pred_a, pred_b, y):
    b = int(((pred_a == y) & (pred_b != y)).sum()); c = int(((pred_a != y) & (pred_b == y)).sum())
    p = 1.0 if b + c == 0 else float(min(1.0, 2 * sps.binom.cdf(min(b, c), b + c, 0.5)))
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_exact": p}


def readout(run, B, seed, fixed_tau=None):
    val = load(os.path.join(run, "scores_val.csv")); test = load(os.path.join(run, "scores_test.csv"))
    sel = select_tau(val)
    if fixed_tau is not None:
        vp = (val["sys"] >= fixed_tau).astype(int)
        sel = {"tau": fixed_tau, "rule": "fixed", "validation_band": sel["band"], "band_midpoint": sel["tau"],
               "inside_band": bool(sel["band"][0] <= fixed_tau <= sel["band"][1]),
               "val_errors": int((vp != val.y).sum()), "val_fn": int(((vp == 0) & (val.y == 1)).sum())}
    tau = sel["tau"]
    pred = (test["sys"] >= tau).astype(int); y = test.y
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    out = {
        "run": run, "tau_selection": sel, "n_test": int(len(test)), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
        "n_test_sources": int(test.source.nunique()),
        "accuracy": float((pred == y).mean()), "fp": fp, "fn": fn,
        "auc": float(roc_auc_score(y, test["sys"])), "ap": float(average_precision_score(y, test["sys"])),
        "max_real": float(test.loc[y == 0, "sys"].max()), "min_fake": float(test.loc[y == 1, "sys"].min()),
        "per_agent_auc": {c: float(roc_auc_score(y, test[c])) for c in COLS},
        "per_agent_acc_at_0.5": {c: float(((test[c] >= 0.5).astype(int) == y).mean()) for c in COLS},
        "recall_by_method": {m: float(((g["sys"] >= tau).mean())) for m, g in test[y == 1].groupby("method")},
        "ci_clip": bootstrap(test, tau, B, seed, None),
        "ci_source": bootstrap(test, tau, B, seed, "source"),
        "phase1_trio_errors": int((((test.trio >= tau).astype(int)) != y).sum()),
        "mcnemar_five_vs_trio": mcnemar(pred.values, (test.trio >= tau).astype(int).values, y.values),
        "leave_one_out": {},
    }
    for c in COLS:
        rest = [k for k in COLS if k != c]
        p2 = (test[rest].mean(axis=1) >= tau).astype(int)
        out["leave_one_out"][c] = {"errors": int((p2 != y).sum()), "auc": float(roc_auc_score(y, test[rest].mean(axis=1))), "mcnemar": mcnemar(pred.values, p2.values, y.values)}
    return out, test.assign(pred=pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau", type=float, default=None, help="hold the threshold fixed instead of selecting it per run")
    a = ap.parse_args()
    runs, frames = [], []
    for r in a.runs:
        o, f = readout(r, a.bootstrap, a.seed, a.tau); runs.append(o); frames.append(f)
        print(f"{r}: tau {o['tau_selection']['tau']} acc {100*o['accuracy']:.2f}% fp {o['fp']} fn {o['fn']} auc {o['auc']:.5f} "
              f"src-CI acc {o['ci_source']['accuracy']} auc {o['ci_source']['auc']}", flush=True)
    summary = {}
    for k in ("accuracy", "auc", "ap", "fp", "fn", "phase1_trio_errors"):
        v = np.array([r[k] for r in runs], dtype=float)
        summary[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "values": v.tolist()}
    summary["per_agent_auc"] = {c: {"mean": float(np.mean([r["per_agent_auc"][c] for r in runs])), "sd": float(np.std([r["per_agent_auc"][c] for r in runs], ddof=1)) if len(runs) > 1 else 0.0} for c in COLS}
    if len(frames) > 1:
        merged = frames[0][["filepath", "y", "pred"]].rename(columns={"pred": "pred0"})
        for i, f in enumerate(frames[1:], 1):
            merged = merged.merge(f[["filepath", "pred"]].rename(columns={"pred": f"pred{i}"}), on="filepath")
        preds = merged[[c for c in merged if c.startswith("pred")]].values
        summary["clips_misclassified_by_every_seed"] = int((preds != merged.y.values[:, None]).all(axis=1).sum())
        summary["clips_misclassified_by_any_seed"] = int((preds != merged.y.values[:, None]).any(axis=1).sum())
    json.dump({"runs": runs, "summary": summary}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
