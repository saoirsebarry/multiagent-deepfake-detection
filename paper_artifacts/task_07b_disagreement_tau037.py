"""Task 7b: Disagreement-threshold sweep re-run at tau = 0.37.

Same escalation logic as task_07 (phase-1 std, verdict-split floor of 0.3),
but:
  - verdict-split detection and the final decision both use tau = 0.37
  - aggregate is weighted mean as in the CSV
Escalation rate depends on the verdict-split detection threshold, so when
we change tau from 0.5 to 0.37 the escalation rates shift very slightly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import (
    AGENT_COLS, ALL_AGENT_COLS, OUT,
    confusion_counts, load_five_agent, metrics_from_counts, save_json,
    weighted_mean,
)

TAU = 0.37
PHASE1_COLS = [
    AGENT_COLS["Biometric"],
    AGENT_COLS["ECAPA"],
    AGENT_COLS["CrossModal"],
]
TAU_DISAGREE_SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def compute_disagreement(df):
    s = df[PHASE1_COLS].values
    std3 = s.std(axis=1)
    v = (s >= TAU).astype(int)  # verdict at tau = 0.37 per phase-1 agent
    split = v.min(axis=1) != v.max(axis=1)
    effective_d = np.where(split, np.maximum(std3, 0.3), std3)
    return effective_d, split


def main() -> None:
    df = load_five_agent()
    y_true = df["y_true"].values
    effective_d, split = compute_disagreement(df)

    rows = []
    for td in TAU_DISAGREE_SWEEP:
        escalated = effective_d > td
        p1 = weighted_mean(df, PHASE1_COLS)
        p12 = weighted_mean(df, ALL_AGENT_COLS)
        agg = np.where(escalated, p12, p1)
        pred = (agg >= TAU).astype(int)
        c = confusion_counts(y_true, pred)
        m = metrics_from_counts(c)
        esc_rate = escalated.mean()
        rows.append({
            "tau_disagree": td,
            "escalation_rate": esc_rate,
            "escalation_rate_pct": esc_rate * 100,
            "accuracy": m["accuracy"],
            "accuracy_pct": m["accuracy"] * 100,
            "avg_agents_per_sample": 3 + 2 * esc_rate,
            "miscount": m["miscount"],
        })

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "disagreement_sweep_tau037.csv", index=False, float_format="%.6f")

    lines = [
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"$\tau_d$ & Escalation rate (\%) & Accuracy (\%) & Avg. agents / sample \\",
        r"\midrule",
    ]
    for r in rows:
        bold = abs(r["tau_disagree"] - 0.30) < 1e-9
        cells = [
            f"{r['tau_disagree']:.2f}",
            f"{r['escalation_rate_pct']:.2f}",
            f"{r['accuracy_pct']:.3f}",
            f"{r['avg_agents_per_sample']:.2f}",
        ]
        if bold:
            cells = [r"\textbf{" + c + "}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "disagreement_sweep_tau037.tex").write_text("\n".join(lines) + "\n")

    op = next(r for r in rows if abs(r["tau_disagree"] - 0.30) < 1e-9)
    print(f"[task07b] tau = 0.37, operating point tau_d = 0.30: "
          f"escalation={op['escalation_rate_pct']:.2f}% "
          f"acc={op['accuracy_pct']:.3f}%  miscount={op['miscount']} "
          f"avg_agents={op['avg_agents_per_sample']:.2f}")
    for r in rows:
        print(f"  tau_d={r['tau_disagree']:.2f}  esc={r['escalation_rate_pct']:5.2f}%  "
              f"acc={r['accuracy_pct']:.3f}%  miss={r['miscount']}  "
              f"avg_agents={r['avg_agents_per_sample']:.2f}")


if __name__ == "__main__":
    main()
