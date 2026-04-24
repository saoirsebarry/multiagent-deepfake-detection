"""Task 6: Three-agent baseline re-thresholded at tau = 0.5.

Input: multiagent_results_csv_files/analysis_results_with_3_agents.csv

Regardless of the threshold the CSV was originally generated with, we
apply tau = 0.5 to the stored final_score column so the number is
directly comparable to the 5-agent headline.
"""
from __future__ import annotations

from _common import (
    OUT, TAU, confusion_counts, fmt_pct,
    load_three_agent, metrics_from_counts, predict_at_tau, save_json,
)


def main() -> None:
    df = load_three_agent()
    if len(df) != 2162:
        raise SystemExit(f"STOP: expected 2162 rows in 3-agent CSV, got {len(df)}")

    y_true = df["y_true"].values
    score = df["final_score"].values
    pred = predict_at_tau(score, TAU)
    c = confusion_counts(y_true, pred)
    m = metrics_from_counts(c)

    out = {
        "tau": TAU,
        "n_samples": int(len(df)),
        "n_real": int((y_true == 0).sum()),
        "n_fake": int((y_true == 1).sum()),
        "confusion_matrix": {"TP": c["tp"], "TN": c["tn"], "FP": c["fp"], "FN": c["fn"]},
        "accuracy": m["accuracy"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "miscount": m["miscount"],
    }
    save_json(out, OUT / "three_agent_metrics.json")
    print(
        f"[task06] 3-agent @ tau=0.5: acc={fmt_pct(m['accuracy'])} "
        f"prec={fmt_pct(m['precision'])} rec={fmt_pct(m['recall'])} "
        f"f1={fmt_pct(m['f1'])} errors={m['miscount']} "
        f"(TP={c['tp']}, FP={c['fp']}, FN={c['fn']}, TN={c['tn']})"
    )


if __name__ == "__main__":
    main()
