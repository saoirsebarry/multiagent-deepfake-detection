"""Task 16: inter-agent correlation, separating the two quantities that get conflated.

Raw correlation between agent SCORES is dominated by the label signal every agent tracks.
If two agents each correlate strongly with the ground-truth label, they are forced to
correlate with each other: for point-biserial correlations rho_A and rho_B the Cauchy-Schwarz
bound gives corr(A,B) >= rho_A*rho_B - sqrt((1-rho_A^2)(1-rho_B^2)), and in the strongly
discriminative regime the product term dominates. A "near-zero score correlation" between two
accurate agents is therefore not achievable, and reporting one indicates a measurement error.

What ensemble error reduction actually depends on is the correlation of the agents' ERRORS.
This script reports both, so the distinction cannot be lost again.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "source_csvs"
COLS = {
    "Visual (XceptionNet)": "score_Visual (Spatial)",
    "Audio (FreqNet)": "score_Audio (Mel+CNN)",
    "Audio Forensics (ECAPA)": "score_Audio Forensics (ECAPA)",
    "Cross-Modal (Lip-Sync)": "score_Cross-Modal (Lip-Sync)",
    "Biometric-Quality": "score_Facial Biometric (Quality)",
}
NAMES = list(COLS)


def matrices(df: pd.DataFrame):
    S = df[list(COLS.values())].to_numpy(float)
    y = (df.ground_truth.str.lower() == "fake").astype(int).to_numpy()
    err = np.abs(S - y[:, None])
    return S, y, np.corrcoef(S.T), np.corrcoef(err.T)


def show(C, title):
    print(f"\n{title}")
    print("                        " + "".join(f"{n.split(' (')[0][:9]:>11s}" for n in NAMES))
    for i, n in enumerate(NAMES):
        print(f"  {n:22s}" + "".join(f"{C[i, j]:11.3f}" for j in range(len(NAMES))))
    off = C - np.eye(len(NAMES))
    i, j = np.unravel_index(np.argmax(off), off.shape)
    print(f"  max off-diagonal {off[i, j]:.3f}  ({NAMES[i]} / {NAMES[j]})")
    return float(off[i, j]), (NAMES[i], NAMES[j])


df = pd.read_csv(SRC / "analysis_results_with_5_agents.csv")
S, y, Cs, Ce = matrices(df)
rho = {n: float(np.corrcoef(S[:, i], y)[0, 1]) for i, n in enumerate(NAMES)}

print(f"PolyGlotFake test set, n={len(df)}")
max_s, pair_s = show(Cs, "Correlation between agent SCORES")
max_e, pair_e = show(Ce, "Correlation between agent ERRORS  |score - label|")

print("\nPoint-biserial correlation of each agent's score with the label:")
for n, v in rho.items():
    print(f"  {n:22s} {v:6.3f}")

cm, ec = NAMES.index("Cross-Modal (Lip-Sync)"), NAMES.index("Audio Forensics (ECAPA)")
lower = rho[NAMES[cm]] * rho[NAMES[ec]] - np.sqrt(
    (1 - rho[NAMES[cm]] ** 2) * (1 - rho[NAMES[ec]] ** 2))
print(f"\nCauchy-Schwarz bound check for the two most accurate agents:")
print(f"  corr(Cross-Modal, ECAPA) >= {lower:.3f};  observed {Cs[cm, ec]:.3f}  "
      f"({'consistent' if Cs[cm, ec] >= lower - 1e-9 else 'VIOLATION'})")
print(f"  their error correlation is {Ce[cm, ec]:.3f}, which is the quantity ensembles depend on")

out = {
    "n": int(len(df)),
    "agents": NAMES,
    "score_correlation": Cs.round(4).tolist(),
    "error_correlation": Ce.round(4).tolist(),
    "point_biserial_with_label": {k: round(v, 4) for k, v in rho.items()},
    "max_score_correlation": {"value": round(max_s, 4), "pair": list(pair_s)},
    "max_error_correlation": {"value": round(max_e, 4), "pair": list(pair_e)},
    "cross_modal_ecapa": {"score": round(float(Cs[cm, ec]), 4),
                          "error": round(float(Ce[cm, ec]), 4),
                          "cauchy_schwarz_lower_bound_on_score": round(float(lower), 4)},
}
(HERE / "agent_correlation.json").write_text(json.dumps(out, indent=2))
pd.DataFrame(Ce, index=NAMES, columns=NAMES).round(4).to_csv(HERE / "agent_error_correlation.csv")
pd.DataFrame(Cs, index=NAMES, columns=NAMES).round(4).to_csv(HERE / "agent_score_correlation.csv")
print("\nwrote agent_correlation.json and the two CSVs")
