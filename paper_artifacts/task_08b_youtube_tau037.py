"""Task 8b: YouTube evaluation re-thresholded at tau = 0.37.

Escalation rate is threshold-independent and identical to the tau = 0.5 run.
"""
from __future__ import annotations

import pandas as pd

from _common import (
    CSV_DIR, OUT, confusion_counts, fmt_pct, metrics_from_counts,
    predict_at_tau, save_json,
)

TAU = 0.37


def main() -> None:
    path = CSV_DIR / "analysis_results_with_5_agents_orchestration.csv"
    raw = pd.read_csv(path)

    mask = raw["final_score"].notna() & raw["ground_truth"].notna()
    df = raw[mask].copy().reset_index(drop=True)
    df["y_true"] = (df["ground_truth"] == "Fake").astype(int)

    pred = predict_at_tau(df["final_score"].values, TAU)
    c = confusion_counts(df["y_true"].values, pred)
    m = metrics_from_counts(c)

    phase_counts = {}
    escalation_rate = None
    if "phase" in df.columns:
        phase_counts = df["phase"].value_counts().to_dict()
        escalation_rate = float((df["phase"] != "quick").sum()) / len(df) if len(df) else 0.0

    out = {
        "csv_file": path.name,
        "n_rows_raw": int(len(raw)),
        "n_rows_parseable": int(len(df)),
        "n_real": int((df["y_true"] == 0).sum()),
        "n_fake": int((df["y_true"] == 1).sum()),
        "tau": TAU,
        "confusion_matrix": {"TP": c["tp"], "TN": c["tn"], "FP": c["fp"], "FN": c["fn"]},
        "accuracy": m["accuracy"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "miscount": m["miscount"],
        "phase_counts": phase_counts,
        "escalation_rate": escalation_rate,
    }
    save_json(out, OUT / "youtube_metrics_tau037.json")

    cm = pd.DataFrame({
        "": ["Pred Real (0)", "Pred Fake (1)"],
        "True Real (0)": [c["tn"], c["fp"]],
        "True Fake (1)": [c["fn"], c["tp"]],
    })
    cm.to_csv(OUT / "youtube_confusion_matrix_tau037.csv", index=False)

    tex_lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r" & True Real & True Fake \\",
        r"\midrule",
        f"Predicted Real & {c['tn']} & {c['fn']} \\\\",
        f"Predicted Fake & {c['fp']} & {c['tp']} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (OUT / "youtube_confusion_matrix_tau037.tex").write_text("\n".join(tex_lines) + "\n")

    print(
        f"[task08b] YouTube @ tau=0.37  n={len(df)}  "
        f"acc={fmt_pct(m['accuracy'])}  prec={fmt_pct(m['precision'])}  "
        f"rec={fmt_pct(m['recall'])}  f1={fmt_pct(m['f1'])}  "
        f"errors={m['miscount']} (TP={c['tp']}, FP={c['fp']}, FN={c['fn']}, TN={c['tn']})"
    )
    if escalation_rate is not None:
        print(f"[task08b] phases: {phase_counts}; escalation={escalation_rate*100:.2f}% "
              f"(threshold-independent)")


if __name__ == "__main__":
    main()
