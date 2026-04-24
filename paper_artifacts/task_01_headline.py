"""Task 1: Headline metrics at tau = 0.5 on the 5-agent CSV.

Expected (from prior audit):
  accuracy = 99.63%, 8 FN, 0 FP, precision = 100%, recall = 99.61%, F1 = 99.80%

If computed values disagree with the expected values by more than
rounding error, STOP and report the discrepancy.
"""
from __future__ import annotations

import sys
from pathlib import Path

from _common import (
    OUT, TAU, confusion_counts, fmt_pct, load_five_agent,
    metrics_from_counts, predict_at_tau, save_json,
)


def main() -> None:
    df = load_five_agent()
    y_pred = predict_at_tau(df["final_score"].values, TAU)
    counts = confusion_counts(df["y_true"].values, y_pred)
    metrics = metrics_from_counts(counts)

    EXPECTED = {"fn": 8, "fp": 0, "accuracy": 0.9963, "precision": 1.0, "recall": 0.9961, "f1": 0.9980}
    for k in ("fn", "fp"):
        if counts[k] != EXPECTED[k]:
            print(f"STOP: headline {k} = {counts[k]}, expected {EXPECTED[k]}", file=sys.stderr)
            sys.exit(2)
    tol = 5e-4
    for k in ("accuracy", "precision", "recall", "f1"):
        if abs(metrics[k] - EXPECTED[k]) > tol:
            print(
                f"STOP: headline {k} = {metrics[k]:.4f}, expected {EXPECTED[k]:.4f} "
                f"(tol={tol})",
                file=sys.stderr,
            )
            sys.exit(2)

    out = {
        "tau": TAU,
        "n_samples": int(len(df)),
        "n_real": int((df["y_true"] == 0).sum()),
        "n_fake": int((df["y_true"] == 1).sum()),
        "confusion_matrix": {
            "TP": counts["tp"], "TN": counts["tn"],
            "FP": counts["fp"], "FN": counts["fn"],
        },
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "miscount": metrics["miscount"],
    }
    save_json(out, OUT / "headline_metrics.json")

    print(
        f"[task01] tau={TAU} | acc={fmt_pct(metrics['accuracy'])} "
        f"prec={fmt_pct(metrics['precision'])} "
        f"rec={fmt_pct(metrics['recall'])} "
        f"f1={fmt_pct(metrics['f1'])} "
        f"errors={metrics['miscount']} (TP={counts['tp']}, TN={counts['tn']}, "
        f"FP={counts['fp']}, FN={counts['fn']})"
    )


if __name__ == "__main__":
    main()
