"""Task 17: re-derive every numeric claim the paper rests on, from the released CSVs.

Each check prints PASS or FAIL and the script exits non-zero if any fails, so a claim that
drifts out of agreement with the data is caught before resubmission rather than by a reviewer.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "source_csvs"
C5 = ["score_Visual (Spatial)", "score_Audio (Mel+CNN)", "score_Audio Forensics (ECAPA)",
      "score_Cross-Modal (Lip-Sync)", "score_Facial Biometric (Quality)"]
W5 = np.array([0.05, 0.20, 0.30, 0.05, 0.40]); TAU = 0.5
fails = []

def check(name, got, want, tol=5e-4):
    ok = abs(got - want) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:56s} got {got:.5g}  expected {want:.5g}")
    if not ok: fails.append(name)

df = pd.read_csv(SRC / "analysis_results_with_5_agents.csv")
S = df[C5].to_numpy(); y = (df.ground_truth == "Fake").astype(int).to_numpy(); agg = S @ W5
err = (np.abs(S - y[:, None]) > 0.5).astype(float)

print("five-agent headline (tau = 0.5)")
check("aggregate reproduces stored final_score (max |diff|)", float(np.abs(agg - df.final_score).max()), 0.0, 1e-6)
check("errors at tau=0.5", float(((agg >= TAU).astype(int) != y).sum()), 3.0, 0)
check("false positives at tau=0.5", float(((agg >= TAU) & (y == 0)).sum()), 0.0, 0)
check("separation margin (min fake - max real)", float(agg[y == 1].min() - agg[y == 0].max()), 0.0349, 5e-4)
from sklearn.metrics import roc_auc_score, average_precision_score
check("AUC-ROC is exactly 1 (perfect ranking)", float(roc_auc_score(y, agg)), 1.0, 1e-9)
check("average precision is exactly 1", float(average_precision_score(y, agg)), 1.0, 1e-9)

print("\ncorrelations (section 4.3)")
Cs, Ce = np.corrcoef(S.T), np.corrcoef(err.T)
off = lambda C: float(np.max(C - np.eye(len(C))))
check("max SCORE correlation (cross-modal / biometric)", off(Cs), 0.940, 1e-3)
check("max ERROR correlation (cross-modal / biometric)", off(Ce), 0.576, 1e-3)
check("cross-modal / ECAPA error correlation", float(Ce[3, 2]), 0.016, 1e-3)

print("\nthree-agent Phase-1 configuration (sections 4.1, 4.4)")
w3 = np.array([W5[2], W5[3], W5[4]]); w3 = w3 / w3.sum()
S3 = S[:, [2, 3, 4]]
p1 = S3 @ w3
check("renormalised Phase-1 errors (Tables 9, 11, 13)", float(((p1 >= TAU).astype(int) != y).sum()), 2.0, 0)
check("renormalised Phase-1 false negatives", float(((p1 < TAU) & (y == 1)).sum()), 0.0, 0)

print("\nfigure 10 worked example, released orchestrator")
r = df[df.filepath.str.contains("en_37_to_ru_Xtts", na=False)].iloc[0]
trio = np.array([r[C5[2]], r[C5[3]], r[C5[4]]])
check("Phase-1 aggregate for the example clip", float(trio @ w3), 0.995, 1e-3)
check("Phase-1 score spread (std)", float(trio.std()), 0.004, 1e-3)
verdicts = (trio >= TAU).astype(int).tolist()
print(f"  note: Phase-1 verdicts at tau={TAU} are {verdicts} -> "
      f"{'split, escalation forced' if len(set(verdicts)) > 1 else 'unanimous, no escalation'}")

print("\nweight sensitivity (section 4.5.3)")
uni = S @ np.full(5, 0.2)
check("uniform weighting errors at tau=0.5", float(((uni >= TAU).astype(int) != y).sum()), 4.0, 0)
check("uniform weighting AUC (not exactly 1)", float(roc_auc_score(y, uni)), 0.999983, 5e-6)

print("\nvalidation provenance of the released weights (section 3.5)")
dv = pd.read_csv(SRC / "analysis_results_v2_VAL.csv")
Sv = dv[C5].to_numpy(); yv = (dv.ground_truth == "Fake").astype(int).to_numpy()
aggv = Sv @ W5
check("validation errors at tau=0.5", float(((aggv >= TAU).astype(int) != yv).sum()), 0.0, 0)
check("validation margin", float(aggv[yv == 1].min() - aggv[yv == 0].max()), 0.1727, 5e-4)
check("validation max real (band low side)", float(aggv[yv == 0].max()), 0.3345, 5e-4)
check("validation min fake (band high side)", float(aggv[yv == 1].min()), 0.5072, 5e-4)

print("\nYouTube evaluation (section 4.11)")
dy = pd.read_csv(SRC / "analysis_results_youtube.csv")
Sy = dy[C5].to_numpy(); yy = (dy.ground_truth == "Fake").astype(int).to_numpy()
aggy = Sy @ W5
check("YouTube clips", float(len(dy)), 37.0, 0)
check("YouTube errors at tau=0.5", float(((aggy >= TAU).astype(int) != yy).sum()), 8.0, 0)
check("YouTube false positives", float(((aggy >= TAU) & (yy == 0)).sum()), 0.0, 0)
check("YouTube AUC", float(roc_auc_score(yy, aggy)), 0.979, 1e-3)

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
