"""Task 17: re-derive every numeric claim the revision rests on, from the released CSVs.

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
W5 = np.array([0.20, 0.15, 0.20, 0.25, 0.20]); TAU = 0.37
fails = []

def check(name, got, want, tol=5e-4):
    ok = abs(got - want) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:56s} got {got:.5g}  expected {want:.5g}")
    if not ok: fails.append(name)

df = pd.read_csv(SRC / "analysis_results_with_5_agents.csv")
S = df[C5].to_numpy(); y = (df.ground_truth == "Fake").astype(int).to_numpy(); agg = S @ W5
err = np.abs(S - y[:, None])

print("five-agent headline")
check("aggregate reproduces stored final_score (max |diff|)", float(np.abs(agg - df.final_score).max()), 0.0, 1e-12)
check("errors at tau=0.37", float(((agg > TAU).astype(int) != y).sum()), 0.0, 0)
check("separation margin", float(agg[y == 1].min() - agg[y == 0].max()), 0.0307, 5e-4)

print("\ncorrelations (section 4.3)")
Cs, Ce = np.corrcoef(S.T), np.corrcoef(err.T)
off = lambda C: float(np.max(C - np.eye(len(C))))
check("max SCORE correlation", off(Cs), 0.695)
check("max ERROR correlation", off(Ce), 0.279)
check("cross-modal / ECAPA error correlation", float(Ce[3, 2]), 0.006, 1e-3)

print("\nthree-agent configurations (sections 4.1, 4.4)")
d3 = pd.read_csv(SRC / "analysis_results_with_3_agents.csv")
C3 = ["score_Audio Forensics (ECAPA)", "score_Cross-Modal (Lip-Sync)", "score_Facial Biometric (Quality)"]
S3 = d3[C3].to_numpy(); y3 = (d3.ground_truth == "Fake").astype(int).to_numpy()
check("standalone 3-agent uses weights (0.35,0.40,0.25)",
      float(np.abs(S3 @ np.array([.35, .40, .25]) - d3.final_score).max()), 0.0, 1e-9)
check("standalone 3-agent errors (Table 9)", float(((d3.final_score.to_numpy() > TAU).astype(int) != y3).sum()), 5.0, 0)
wn = np.array([.20, .25, .20]); wn = wn / wn.sum()
check("renormalised Phase-1 errors (Tables 11, 13)", float(((S3 @ wn > TAU).astype(int) != y3).sum()), 4.0, 0)

print("\nfigure 9 worked example, released orchestrator")
r = df[df.filepath.str.contains("en_37_to_ru_Xtts", na=False)].iloc[0]
p1 = np.array([r[C5[4]], r[C5[2]], r[C5[3]]])
check("aggregate for the example clip", float(r.final_score), 0.7519, 1e-3)
check("Phase-1 score spread", float(p1.std()), 0.3906, 1e-3)
print(f"  note: Phase-1 verdicts at tau={TAU} are {(p1 > TAU).astype(int).tolist()} -> "
      f"{'split, escalation forced' if len(set((p1 > TAU).astype(int))) > 1 else 'unanimous'}")

print("\nweight sensitivity (section 4.5.3)")
uni = S @ np.full(5, 0.2)
check("uniform weighting errors at tau=0.37", float(((uni > TAU).astype(int) != y).sum()), 1.0, 0)
from sklearn.metrics import roc_auc_score
check("uniform weighting AUC (not exactly 1: no full separation)",
      float(roc_auc_score(y, uni)), 0.999996, 5e-6)
check("uniform weighting separation margin is negative",
      float(uni[y == 1].min() - uni[y == 0].max()), -0.00634, 5e-4)

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
