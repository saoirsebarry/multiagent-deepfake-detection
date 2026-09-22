"""Diagnostic: how much of a benchmark's transfer loss do the released agents recover if only the fusion is re-fitted?

    python tools/external_benchmark/fusion_recalibration.py --scores <dir>/original/scores.csv --metadata <dir>/metadata.csv

Uses the benchmark's labels, so it is not a transfer result. Two protocols, both out-of-fold: logistic fusion of the five
agent scores with folds disjoint in source video, and leave-one-forgery-class-out, where the weights never see the class
they are tested on (authentic clips are split in half between fitting and testing).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True); ap.add_argument("--metadata", required=True)
    ap.add_argument("--cluster", default="source_video"); ap.add_argument("--forgery_class", default="generator")
    ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    s = pd.read_csv(a.scores); s["clip"] = s.filepath.map(os.path.basename)
    j = s.drop_duplicates("clip").merge(pd.read_csv(a.metadata).drop_duplicates("clip"), on="clip").reset_index(drop=True)
    cols = [c for c in j.columns if c.startswith("score_")]
    X, y, g = j[cols].values, (j.ground_truth == "Fake").astype(int).values, j[a.cluster].values
    cls = j[a.forgery_class].values
    rng = np.random.default_rng(a.seed)
    members = {k: v.index.values for k, v in j.groupby(a.cluster)}
    keys = np.array(list(members))
    boots = [np.concatenate([members[k] for k in rng.choice(keys, len(keys), replace=True)]) for _ in range(a.boot)]

    def auc_ci(score):
        v = [roc_auc_score(y[b], score[b]) for b in boots]
        return [float(roc_auc_score(y, score)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    oof, weights = np.zeros(len(j)), []
    for tr, te in GroupKFold(5).split(X, y, g):
        lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X[tr], y[tr])
        oof[te] = lr.decision_function(X[te]); weights.append(lr.coef_[0])
    out = {"released": auc_ci(j.final_score.values), "refit_source_disjoint": auc_ci(oof),
           "refit_weights": dict(zip(cols, np.mean(weights, axis=0).round(3).tolist())), "leave_one_class_out": {}}

    halves = np.array_split(rng.permutation(np.where(y == 0)[0]), 2)
    for c in sorted(set(cls[y == 1])):
        pair = []
        for fit_half, test_half in [(0, 1), (1, 0)]:
            tr = np.concatenate([np.where((y == 1) & (cls != c))[0], halves[fit_half]])
            te = np.concatenate([np.where((y == 1) & (cls == c))[0], halves[test_half]])
            lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X[tr], y[tr])
            pair.append((roc_auc_score(y[te], j.final_score.values[te]), roc_auc_score(y[te], lr.decision_function(X[te]))))
        released, refit = np.mean(pair, axis=0)
        out["leave_one_class_out"][str(c)] = {"released": float(released), "refit": float(refit)}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
