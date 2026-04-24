"""Task 7: Disagreement-threshold sweep on the 5-agent CSV.

Uses Phase-1 per-agent scores to decide escalation, then replays the
aggregation according to whether Phase 2 is triggered:
  - escalated rows  -> weighted mean over all 5 agents (same weights as
                       the orchestrator used when writing the CSV)
  - non-escalated   -> weighted mean over only the 3 Phase-1 agents,
                       weights re-normalised to sum to 1

Decision at tau = 0.5.

The disagreement metric mirrors the code:
    std_3 = std(phase_1_scores)
    if verdicts_differ_at_0.5_among_3: d = max(std_3, 0.3)
    else:                              d = std_3
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import (
    AGENT_COLS, AGENT_WEIGHTS, ALL_AGENT_COLS, OUT, TAU,
    confusion_counts, load_five_agent, metrics_from_counts, save_json,
    weighted_mean,
)

PHASE1_COLS = [
    AGENT_COLS["Biometric"],
    AGENT_COLS["ECAPA"],
    AGENT_COLS["CrossModal"],
]

TAU_DISAGREE_SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def compute_disagreement(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (effective_d, verdict_split_flag) per row."""
    s = df[PHASE1_COLS].values
    std3 = s.std(axis=1)
    # verdict per phase-1 agent at tau=0.5
    v = (s >= TAU).astype(int)
    # "split" = any agent disagrees with any other
    split = v.min(axis=1) != v.max(axis=1)
    effective_d = np.where(split, np.maximum(std3, 0.3), std3)
    return effective_d, split


def aggregate_under_escalation(df: pd.DataFrame, escalated: np.ndarray) -> np.ndarray:
    p1 = weighted_mean(df, PHASE1_COLS)      # phase-1 only
    p12 = weighted_mean(df, ALL_AGENT_COLS)  # all 5 agents
    return np.where(escalated, p12, p1)


def main() -> None:
    df = load_five_agent()
    y_true = df["y_true"].values
    effective_d, split = compute_disagreement(df)

    rows = []
    for td in TAU_DISAGREE_SWEEP:
        escalated = effective_d > td
        agg = aggregate_under_escalation(df, escalated)
        pred = (agg >= TAU).astype(int)
        c = confusion_counts(y_true, pred)
        m = metrics_from_counts(c)
        esc_rate = escalated.mean()
        avg_agents = 3 + 2 * esc_rate
        rows.append({
            "tau_disagree": td,
            "escalation_rate": esc_rate,
            "escalation_rate_pct": esc_rate * 100,
            "accuracy": m["accuracy"],
            "accuracy_pct": m["accuracy"] * 100,
            "avg_agents_per_sample": avg_agents,
            "miscount": m["miscount"],
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "disagreement_sweep.csv", index=False, float_format="%.6f")

    lines = [
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"$\tau_d$ & Escalation rate (\%) & Accuracy (\%) & Avg. agents / sample \\",
        r"\midrule",
    ]
    for r in rows:
        bold = abs(r["tau_disagree"] - 0.30) < 1e-9  # operating point
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
    (OUT / "disagreement_sweep.tex").write_text("\n".join(lines) + "\n")

    # Diagnostic at operating point tau_disagree = 0.3
    op_td = 0.30
    escalated_op = effective_d > op_td
    diag = {
        "tau_disagree": op_td,
        "tau_decision": TAU,
        "weights_used": AGENT_WEIGHTS,
        "escalation_rate": float(escalated_op.mean()),
        "n_escalated": int(escalated_op.sum()),
        "n_not_escalated": int((~escalated_op).sum()),
        "per_sample": [
            {
                "filepath": fp,
                "std_phase1": float(std_v),
                "verdict_split": bool(sp),
                "effective_d": float(ed),
                "escalated": bool(esc),
            }
            for fp, std_v, sp, ed, esc in zip(
                df["filepath"].values,
                df[PHASE1_COLS].std(axis=1).values,
                split,
                effective_d,
                escalated_op,
            )
        ],
    }
    save_json(diag, OUT / "escalation_at_tau_disagree_0.3.json")

    op_row = next(r for r in rows if abs(r["tau_disagree"] - 0.30) < 1e-9)
    print(
        f"[task07] operating point tau_d=0.30: "
        f"escalation={op_row['escalation_rate_pct']:.2f}% "
        f"acc={op_row['accuracy_pct']:.3f}% "
        f"avg_agents={op_row['avg_agents_per_sample']:.2f}"
    )


if __name__ == "__main__":
    main()
