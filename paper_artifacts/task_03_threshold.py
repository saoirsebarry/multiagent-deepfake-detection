"""Task 3: Supplementary threshold-robustness table.

Purpose: robustness check, NOT threshold selection. The decision boundary
remains tau = 0.5 throughout the paper.
"""
from __future__ import annotations

import pandas as pd

from _common import (
    OUT, confusion_counts, load_five_agent, metrics_from_counts, predict_at_tau,
)

TAU_SWEEP = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def main() -> None:
    df = load_five_agent()
    y_true = df["y_true"].values
    score = df["final_score"].values

    rows = []
    for tau in TAU_SWEEP:
        pred = predict_at_tau(score, tau)
        c = confusion_counts(y_true, pred)
        m = metrics_from_counts(c)
        n_real = c["tn"] + c["fp"]
        n_fake = c["tp"] + c["fn"]
        fpr = c["fp"] / n_real if n_real else 0.0
        fnr = c["fn"] / n_fake if n_fake else 0.0
        rows.append({
            "tau": tau,
            "accuracy": m["accuracy"],
            "fpr": fpr,
            "fnr": fnr,
            "miscount": m["miscount"],
            "fp_count": c["fp"],
            "fn_count": c["fn"],
        })
    table = pd.DataFrame(rows)
    csv_path = OUT / "threshold_robustness.csv"
    table.to_csv(csv_path, index=False, float_format="%.6f")

    # LaTeX version. tau=0.50 row bold.
    lines = [
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"$\tau$ & Accuracy (\%) & FPR (\%) & FNR (\%) & Miscount \\",
        r"\midrule",
    ]
    for r in rows:
        bold = abs(r["tau"] - 0.50) < 1e-9
        cells = [
            f"{r['tau']:.2f}",
            f"{r['accuracy'] * 100:.3f}",
            f"{r['fpr'] * 100:.3f}",
            f"{r['fnr'] * 100:.3f}",
            f"{r['miscount']}",
        ]
        if bold:
            cells = [r"\textbf{" + c + "}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "threshold_robustness_table.tex").write_text("\n".join(lines) + "\n")

    print(f"[task03] threshold sweep written to {csv_path.name}; "
          f"tau=0.5 acc={table.loc[table['tau'] == 0.5, 'accuracy'].iloc[0] * 100:.3f}%")


if __name__ == "__main__":
    main()
