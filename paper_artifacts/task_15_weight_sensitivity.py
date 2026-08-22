"""Task 15: agent-weight sensitivity.

Answers the reviewer request for a sensitivity analysis over the five agent weights
(0.20, 0.15, 0.20, 0.25, 0.20). The aggregate score is a weighted linear combination of
five stored per-agent sigmoid outputs, so any alternative weight vector can be evaluated
exactly on the same saved test predictions without re-running inference.

Four analyses:
  a  exhaustive enumeration of the 0.05 simplex grid (10,626 vectors)
  b  one-at-a-time perturbation of each weight
  c  local perturbation, every weight within +/-0.05 at 0.01 resolution (161,051 vectors)
  d  Dirichlet sampling, concentrated on the selected vector and over the whole simplex

Writes weight_sensitivity.json plus the two CSVs behind it.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
CSV = HERE / "source_csvs" / "analysis_results_with_5_agents.csv"
COLS = ["score_Visual (Spatial)", "score_Audio (Mel+CNN)",
        "score_Audio Forensics (ECAPA)", "score_Cross-Modal (Lip-Sync)",
        "score_Facial Biometric (Quality)"]
NAMES = ["Visual", "FreqNet", "ECAPA", "CrossModal", "Biometric"]
W0 = np.array([0.20, 0.15, 0.20, 0.25, 0.20])
TAU = 0.37
RNG = np.random.default_rng(42)

df = pd.read_csv(CSV)
S = df[COLS].to_numpy(np.float64)
y = (df.ground_truth == "Fake").astype(int).to_numpy()
n1, n0 = int(y.sum()), int((y == 0).sum())

# the stored aggregate must be reproducible from the per-agent scores, or nothing below holds
assert np.abs(S @ W0 - df.final_score.to_numpy()).max() < 1e-12, \
    "weighted mean does not reproduce final_score"


def metrics(w):
    a = S @ w
    real, fake = a[y == 0], a[y == 1]
    pred = (a > TAU).astype(int)
    return dict(auc=roc_auc_score(y, a), ap=average_precision_score(y, a),
                acc_tau=100.0 * (pred == y).mean(), err_tau=int((pred != y).sum()),
                margin=float(fake.min() - real.max()))


def batch(W, chunk=4000):
    """Vectorised AUC / separation margin / error count for many weight vectors."""
    auc = np.empty(len(W)); mar = np.empty(len(W)); err = np.empty(len(W), int)
    for i in range(0, len(W), chunk):
        A = S @ W[i:i + chunk].T
        r = np.apply_along_axis(rankdata, 0, A)
        auc[i:i + chunk] = (r[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n0 * n1)
        mar[i:i + chunk] = A[y == 1].min(axis=0) - A[y == 0].max(axis=0)
        err[i:i + chunk] = ((A > TAU).astype(int) != y[:, None]).sum(axis=0)
    return auc, mar, err


out = {"selected_weights": W0.tolist(), "tau": TAU, "n_test": int(len(df)),
       "baseline": {k: float(v) for k, v in metrics(W0).items()}}
print("selected vector:", json.dumps(out["baseline"], indent=2))

# ---- (a) exhaustive 0.05 grid ----------------------------------------------
grid = np.array([np.array(c) / 20.0
                 for c in itertools.product(range(21), repeat=5) if sum(c) == 20])
auc, mar, err = batch(grid)
G = pd.DataFrame(grid, columns=NAMES).assign(auc=auc, margin=mar, err_tau=err,
                                             acc_tau=100.0 * (1 - err / len(y)))
G.to_csv(HERE / "weight_grid_full.csv", index=False)
all5 = (grid > 0).all(axis=1)
rank = int((mar > out["baseline"]["margin"]).sum()) + 1
uni = metrics(np.full(5, 0.2))
print(f"\ngrid: {len(G)} vectors | AUC>=0.999 {100*(auc>=0.999).mean():.1f}% | "
      f"separable {int((mar>0).sum())} | zero-error {int((err==0).sum())} | "
      f"selected-vector margin rank {rank}/{len(G)}")
out["grid"] = dict(n=int(len(G)), n_auc_ge_999=int((auc >= 0.999).sum()),
                   pct_auc_ge_999=float(100 * (auc >= 0.999).mean()),
                   auc_min=float(auc.min()), auc_median=float(np.median(auc)),
                   n_separable=int((mar > 0).sum()), n_zero_error=int((err == 0).sum()),
                   n_all_five_present=int(all5.sum()),
                   all_five_auc_min=float(auc[all5].min()),
                   selected_margin_rank=rank,
                   uniform_weights={k: float(v) for k, v in uni.items()})

# ---- (b) one-at-a-time ------------------------------------------------------
oat = []
for i, n in enumerate(NAMES):
    for d in (-0.10, -0.05, 0.05, 0.10):
        w = W0.copy(); w[i] = max(0.0, w[i] + d); w /= w.sum()
        oat.append({"agent": n, "delta": d,
                    **{k: float(v) for k, v in metrics(w).items()}})
pd.DataFrame(oat).to_csv(HERE / "weight_oat.csv", index=False)
out["one_at_a_time"] = oat

# ---- (c) local box, 0.01 resolution ----------------------------------------
steps = np.arange(-0.05, 0.0501, 0.01)
D = np.array(list(itertools.product(steps, repeat=5)))
W = W0 + D
keep = (W >= 0).all(axis=1)
W, D = W[keep], D[keep]
W = W / W.sum(axis=1, keepdims=True)
linf = np.abs(D).max(axis=1)
auc, mar, err = batch(W)
print(f"local +/-0.05: {len(W)} vectors | AUC min {auc.min():.5f} | "
      f"worst {err.max()} errors ({100*(1-err.max()/len(y)):.2f}% acc) | "
      f"zero-error {100*(err==0).mean():.1f}%")
out["local_box"] = dict(
    n=int(len(W)), auc_min=float(auc.min()), auc_p1=float(np.percentile(auc, 1)),
    pct_auc_1=float(100 * (auc >= 0.99999).mean()),
    pct_zero_error=float(100 * (err == 0).mean()),
    worst_err=int(err.max()), worst_acc=float(100 * (1 - err.max() / len(y))),
    by_linf={f"{b:.2f}": dict(n=int(np.isclose(linf, b).sum()),
                              auc_min=float(auc[np.isclose(linf, b)].min()),
                              worst_err=int(err[np.isclose(linf, b)].max()),
                              pct_zero_error=float(100 * (err[np.isclose(linf, b)] == 0).mean()))
             for b in (0.01, 0.02, 0.03, 0.04, 0.05)})

# ---- (d) Dirichlet ----------------------------------------------------------
out["dirichlet"] = {}
for tag, alpha in [("concentrated", 100 * W0), ("diffuse", 10 * W0),
                   ("uniform_simplex", np.ones(5))]:
    Wd = RNG.dirichlet(alpha, size=20000)
    auc, mar, err = batch(Wd)
    acc = 100.0 * (1 - err / len(y))
    print(f"dirichlet {tag:16s} AUC p1 {np.percentile(auc,1):.5f} | acc p1 {np.percentile(acc,1):.2f}%")
    out["dirichlet"][tag] = dict(n=int(len(Wd)), auc_min=float(auc.min()),
                                 auc_p1=float(np.percentile(auc, 1)),
                                 auc_median=float(np.median(auc)),
                                 pct_auc_1=float(100 * (auc >= 0.99999).mean()),
                                 pct_zero_error=float(100 * (err == 0).mean()),
                                 acc_p1=float(np.percentile(acc, 1)),
                                 acc_median=float(np.median(acc)))

(HERE / "weight_sensitivity.json").write_text(json.dumps(out, indent=2))
print("\nwrote weight_sensitivity.json, weight_grid_full.csv, weight_oat.csv")
