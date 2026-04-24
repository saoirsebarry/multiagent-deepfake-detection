"""Task 8: YouTube evaluation reconciliation.

Reports the honest contents of the saved orchestration CSV, computed at
tau = 0.5. Does not invent missing samples.
"""
from __future__ import annotations

import pandas as pd

from _common import (
    CSV_DIR, OUT, TAU, confusion_counts, fmt_pct,
    metrics_from_counts, predict_at_tau, save_json,
)


def main() -> None:
    path = CSV_DIR / "analysis_results_with_5_agents_orchestration.csv"
    raw = pd.read_csv(path)

    total = len(raw)
    parseable_mask = raw["final_score"].notna() & raw["ground_truth"].notna()
    df = raw[parseable_mask].copy().reset_index(drop=True)
    df["y_true"] = (df["ground_truth"] == "Fake").astype(int)

    pred = predict_at_tau(df["final_score"].values, TAU)
    c = confusion_counts(df["y_true"].values, pred)
    m = metrics_from_counts(c)

    phase_counts = {}
    escalation_rate = None
    if "phase" in df.columns:
        phase_counts = df["phase"].value_counts().to_dict()
        # "quick" = no escalation; anything else = escalated
        escalated = (df["phase"] != "quick").sum()
        escalation_rate = float(escalated) / len(df) if len(df) else 0.0

    out = {
        "csv_file": path.name,
        "n_rows_raw": int(total),
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
        "note": (
            "The orchestration CSV records a 'phase' field directly. "
            "'quick' = Phase 1 only (no escalation); 'iterative' and 'strong' "
            "= Phase 2 deployed. Escalation rate = (iterative + strong) / total."
        ),
    }
    save_json(out, OUT / "youtube_metrics.json")

    # Confusion-matrix CSV + LaTeX
    cm = pd.DataFrame({
        "": ["Pred Real (0)", "Pred Fake (1)"],
        "True Real (0)": [c["tn"], c["fp"]],
        "True Fake (1)": [c["fn"], c["tp"]],
    })
    cm.to_csv(OUT / "youtube_confusion_matrix.csv", index=False)

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
    (OUT / "youtube_confusion_matrix.tex").write_text("\n".join(tex_lines) + "\n")

    print(
        f"[task08] YouTube: raw={total}, parseable={len(df)}, "
        f"{(df['y_true'] == 0).sum()} real + {(df['y_true'] == 1).sum()} fake"
    )
    print(
        f"[task08] @ tau=0.5  acc={fmt_pct(m['accuracy'])} "
        f"prec={fmt_pct(m['precision'])} rec={fmt_pct(m['recall'])} "
        f"f1={fmt_pct(m['f1'])} errors={m['miscount']} "
        f"(TP={c['tp']}, FP={c['fp']}, FN={c['fn']}, TN={c['tn']})"
    )
    if escalation_rate is not None:
        print(f"[task08] phases: {phase_counts}; escalation = {escalation_rate * 100:.1f}%")

    if len(df) < 50:
        print(
            f"[task08] NOTE: parseable subset has {len(df)} rows, below the "
            f"50-sample benchmark size claimed in the thesis. Author decision "
            f"required: treat this as the reported YouTube eval, or rerun."
        )


if __name__ == "__main__":
    main()
