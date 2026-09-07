"""Task 6: three-agent Phase-1 configuration at tau = 0.5.

The three-agent rows of Tables 9, 11 and 13 are the released five-agent
weights renormalised over the Phase-1 trio (ECAPA-TDNN, Cross-Modal,
Biometric-Quality), applied to the released per-agent scores.
"""
from __future__ import annotations

import numpy as np

from _common import (
    AGENT_WEIGHTS, OUT, TAU, confusion_counts, fmt_pct,
    load_five_agent, metrics_from_counts, predict_at_tau, save_json,
)

TRIO = ["score_Audio Forensics (ECAPA)", "score_Cross-Modal (Lip-Sync)",
        "score_Facial Biometric (Quality)"]


def main() -> None:
    df = load_five_agent()
    w = np.array([AGENT_WEIGHTS[c] for c in TRIO])
    w = w / w.sum()
    score = df[TRIO].to_numpy() @ w
    y_true = df["y_true"].values
    pred = predict_at_tau(score, TAU)
    c = confusion_counts(y_true, pred)
    m = metrics_from_counts(c)

    out = {
        "tau": TAU,
        "configuration": "Phase-1 trio, five-agent weights renormalised",
        "weights_renormalised": {k: round(float(v), 4) for k, v in zip(TRIO, w)},
        "n_samples": int(len(df)),
        "confusion_matrix": {"TP": c["tp"], "TN": c["tn"], "FP": c["fp"], "FN": c["fn"]},
        **m,
    }
    save_json(out, OUT / "three_agent_metrics.json")
    acc = fmt_pct(m["accuracy"])
    print(f"[task06] Phase-1 trio at tau={TAU}: acc={acc} "
          f"errors={c['fp'] + c['fn']} (fp={c['fp']}, fn={c['fn']})")


if __name__ == "__main__":
    main()
