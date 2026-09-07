"""Task 8: YouTube evaluation at tau = 0.5.

Reads the frozen 100-clip set (analysis_results_youtube.csv) and reports its
confusion matrix, metrics and escalation rate under Algorithm 1.
"""
from __future__ import annotations

import pandas as pd

from _common import (
    CSV_DIR, OUT, TAU, confusion_counts, fmt_pct,
    metrics_from_counts, predict_at_tau, save_json,
)


def main() -> None:
    path = CSV_DIR / "analysis_results_youtube.csv"
    raw = pd.read_csv(path)

    total = len(raw)
    # A NaN aggregate (an agent returned no score, e.g. FreqNet on a muted audio
    # track) compares False at tau in the released orchestrator: verdict Real.
    parseable_mask = raw["ground_truth"].notna()
    df = raw[parseable_mask].copy().reset_index(drop=True)
    df["y_true"] = (df["ground_truth"] == "Fake").astype(int)

    pred = predict_at_tau(df["final_score"].values, TAU)
    c = confusion_counts(df["y_true"].values, pred)
    m = metrics_from_counts(c)

    trio = df[["score_Audio Forensics (ECAPA)", "score_Cross-Modal (Lip-Sync)",
               "score_Facial Biometric (Quality)"]].to_numpy()
    verdicts = trio >= TAU
    split = verdicts.any(axis=1) & ~verdicts.all(axis=1)
    escalated = split | (trio.std(axis=1) >= 0.30)
    phase_counts = {"phase1_only": int((~escalated).sum()), "escalated": int(escalated.sum())}
    escalation_rate = float(escalated.mean()) if len(df) else 0.0

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
            "Escalation follows Algorithm 1 on the stored Phase-1 scores: a verdict "
            "split among ECAPA-TDNN, Cross-Modal and Biometric-Quality at tau, or a "
            "score standard deviation of at least 0.30, deploys Phase 2."
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


if __name__ == "__main__":
    main()
