"""Task 0: the released weight-selection procedure, auditable by execution.

Reruns the exact validation-only search that produced the released weight
vector: over every vector on the 0.05-step five-agent simplex with all agents
active (3,876 vectors), minimise validation ensemble errors at tau = 0.5 and
break ties by the separating margin (lowest-scoring fake minus highest-scoring
real). Asserts that the argmax equals the released vector, records the band the
conventional threshold must sit inside, then reports the single frozen test
read-out.

Outputs: paper_artifacts/operating_point_provenance.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "source_csvs"
C5 = ["score_Visual (Spatial)", "score_Audio (Mel+CNN)", "score_Audio Forensics (ECAPA)",
      "score_Cross-Modal (Lip-Sync)", "score_Facial Biometric (Quality)"]
RELEASED = np.array([0.05, 0.20, 0.30, 0.05, 0.40])
TAU = 0.5


def main() -> None:
    dv = pd.read_csv(SRC / "analysis_results_v2_VAL.csv")
    Sv = dv[C5].to_numpy()
    yv = (dv.ground_truth == "Fake").to_numpy()

    grid = []
    for a in range(21):
        for b in range(21 - a):
            for c in range(21 - a - b):
                for d in range(21 - a - b - c):
                    grid.append((a, b, c, d, 20 - a - b - c - d))
    grid = np.array(grid) / 20.0
    active = grid[(grid > 0).all(axis=1)]

    agg = active @ Sv.T
    errs = ((agg >= TAU) != yv).sum(axis=1)
    max_real = np.where(~yv, agg, -1).max(axis=1)
    min_fake = np.where(yv, agg, 2).min(axis=1)
    margin = min_fake - max_real
    i_best = int(np.argmax(-errs * 1000 + margin))
    w_star = active[i_best]

    if not np.allclose(w_star, RELEASED):
        raise SystemExit(f"FAIL: selection reproduces {w_star.tolist()}, released is {RELEASED.tolist()}")

    dt = pd.read_csv(SRC / "analysis_results_with_5_agents.csv")
    St = dt[C5].to_numpy()
    yt = (dt.ground_truth == "Fake").to_numpy()
    aggt = St @ w_star
    fp = int(((aggt >= TAU) & ~yt).sum())
    fn = int(((aggt < TAU) & yt).sum())

    out = {
        "procedure": "0.05-step simplex grid, all five agents active; minimise validation "
                     "errors at tau=0.5, tie-break on separating margin",
        "grid_vectors_active": int(len(active)),
        "vectors_with_zero_validation_errors": int((errs == 0).sum()),
        "selected_weights": {c: float(w) for c, w in zip(C5, w_star)},
        "validation": {
            "n": int(len(dv)),
            "errors_at_tau_0.5": int(errs[i_best]),
            "separating_margin": round(float(margin[i_best]), 4),
            "band": [round(float(max_real[i_best]), 4), round(float(min_fake[i_best]), 4)],
            "tau_0.5_inside_band": bool(max_real[i_best] < TAU < min_fake[i_best]),
        },
        "frozen_test_readout": {
            "n": int(len(dt)),
            "accuracy": round(float(1 - (fp + fn) / len(dt)), 6),
            "false_positives": fp,
            "false_negatives": fn,
        },
    }
    with open(HERE / "operating_point_provenance.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
