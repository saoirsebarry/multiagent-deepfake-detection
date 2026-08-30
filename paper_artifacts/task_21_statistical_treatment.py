"""Statistical treatment of the tau = 0.37 operating point.

A zero-error count on a finite test set still carries sampling uncertainty; the
correct interval for it is Clopper-Pearson, since a bootstrap of a zero-error
sample degenerates to [1, 1]. Alongside it: bootstrap intervals for the
threshold-free metrics, and McNemar tests between the five-agent system and each
ablated configuration, which are computable exactly because every configuration
is a deterministic re-aggregation of the same stored per-agent scores.
"""
import json

import numpy as np
from scipy import stats as sps
from sklearn.metrics import roc_auc_score, average_precision_score

from _common import AGENT_WEIGHTS, CSV_DIR, OUT
import pandas as pd

TAU = 0.37
SEED = 42
N_BOOT = 10_000

df = pd.read_csv(CSV_DIR / "analysis_results_with_5_agents.csv")
y = (df["ground_truth"] == "Fake").astype(int).to_numpy()
cols = list(AGENT_WEIGHTS)
w = np.array([AGENT_WEIGHTS[c] for c in cols])
S = df[cols].to_numpy()
score = S @ (w / w.sum())
pred = (score >= TAU).astype(int)
n = len(y)
errors = int((pred != y).sum())

acc_lo, acc_hi = sps.beta.ppf(0.025, n - errors, errors + 1), sps.beta.ppf(0.975, n - errors + 1, errors)
if errors == 0:
    acc_lo, acc_hi = sps.beta.ppf(0.025, n, 1), 1.0

rng = np.random.default_rng(SEED)
boot = {"auc_roc": [], "average_precision": [], "accuracy": []}
for _ in range(N_BOOT):
    idx = rng.integers(0, n, n)
    yb, sb, pb = y[idx], score[idx], pred[idx]
    if yb.min() == yb.max():
        continue
    boot["auc_roc"].append(roc_auc_score(yb, sb))
    boot["average_precision"].append(average_precision_score(yb, sb))
    boot["accuracy"].append(float((pb == yb).mean()))
ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}

def config_preds(drop=None, renorm_cols=None):
    use = renorm_cols or [c for c in cols if c != drop]
    wu = np.array([AGENT_WEIGHTS[c] for c in use])
    sc = df[use].to_numpy() @ (wu / wu.sum())
    return (sc >= TAU).astype(int)

mcnemar = {}
for drop in cols:
    alt = config_preds(drop=drop)
    b = int(((pred == y) & (alt != y)).sum())   # system right, ablation wrong
    c = int(((pred != y) & (alt == y)).sum())   # system wrong, ablation right
    if b + c == 0:
        p = 1.0
    else:
        p = float(sps.binomtest(min(b, c), b + c, 0.5).pvalue)  # exact, small-count regime
    mcnemar[drop.replace("score_", "")] = {"b_sys_only_right": b, "c_abl_only_right": c, "exact_p": p}

yt_path = CSV_DIR.parent / "youtube_metrics_tau037.json"
youtube = None
if yt_path.exists():
    yt = json.loads(yt_path.read_text())
    cm = yt["confusion_matrix"]
    k, m = cm["TP"] + cm["TN"], yt["n_rows_parseable"]
    youtube = {
        "n": m, "correct": k,
        "accuracy_clopper_pearson_95": [float(sps.beta.ppf(0.025, k, m - k + 1)),
                                         float(sps.beta.ppf(0.975, k + 1, m - k))],
    }

result = {
    "tau": TAU, "n_test": n, "errors_at_tau": errors, "seed": SEED, "n_boot": N_BOOT,
    "accuracy_clopper_pearson_95": [float(acc_lo), float(acc_hi)],
    "bootstrap_95": ci,
    "mcnemar_vs_leave_one_out": mcnemar,
    "youtube_accuracy_interval": youtube,
    "note": "Baseline-model McNemar tests are not computable: baseline per-sample "
            "predictions were not retained (see mcnemar_tests.json).",
}
(OUT / "statistical_treatment.json").write_text(json.dumps(result, indent=1))
print(json.dumps(result, indent=1))
