"""Task 2: ROC and PR curves for the system aggregate and per-agent overlay."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from _common import (
    AGENT_COLOUR, AGENT_COLS, OKABE_ITO, OUT, TAU,
    load_five_agent, save_both, save_json, set_plot_style,
)


def operating_point(y_true: np.ndarray, score: np.ndarray, tau: float = TAU) -> dict:
    pred = (score >= tau).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 1.0  # when no positives predicted, prec undefined; use 1
    return {"tpr": tpr, "fpr": fpr, "precision": prec, "recall": tpr}


def plot_roc_system(y_true: np.ndarray, score: np.ndarray) -> tuple:
    set_plot_style()
    fpr, tpr, _ = roc_curve(y_true, score)
    auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color=OKABE_ITO["black"], lw=1.6, label=f"5-agent system (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls="--", label="Chance")

    op = operating_point(y_true, score)
    ax.plot(op["fpr"], op["tpr"], marker="*", markersize=12, color=OKABE_ITO["vermil"],
            linestyle="none", label=fr"Operating point ($\tau=0.5$)")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="lower right")
    save_both(fig, OUT / "roc_curve_system")
    return auc_val, op


def plot_pr_system(y_true: np.ndarray, score: np.ndarray) -> tuple:
    set_plot_style()
    prec, rec, _ = precision_recall_curve(y_true, score)
    ap = average_precision_score(y_true, score)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, color=OKABE_ITO["black"], lw=1.6, label=f"5-agent system (AP = {ap:.3f})")

    # baseline = prevalence
    prev = y_true.mean()
    ax.axhline(prev, color="gray", lw=0.8, ls="--", label=f"Prevalence = {prev:.3f}")

    op = operating_point(y_true, score)
    ax.plot(op["recall"], op["precision"], marker="*", markersize=12,
            color=OKABE_ITO["vermil"], linestyle="none",
            label=fr"Operating point ($\tau=0.5$)")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="lower left")
    save_both(fig, OUT / "pr_curve_system")
    return ap, op


def plot_all_roc(y_true, scores_by_name):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls="--")
    for name, (score, colour, style) in scores_by_name.items():
        fpr, tpr, _ = roc_curve(y_true, score)
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colour, lw=1.4, ls=style,
                label=f"{name} (AUC = {a:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="lower right", fontsize=8)
    save_both(fig, OUT / "roc_curve_all_agents_plus_system")


def plot_all_pr(y_true, scores_by_name):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    prev = y_true.mean()
    ax.axhline(prev, color="gray", lw=0.8, ls="--", label=f"Prevalence = {prev:.3f}")
    for name, (score, colour, style) in scores_by_name.items():
        p, r, _ = precision_recall_curve(y_true, score)
        ap = average_precision_score(y_true, score)
        ax.plot(r, p, color=colour, lw=1.4, ls=style,
                label=f"{name} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="lower left", fontsize=8)
    save_both(fig, OUT / "pr_curve_all_agents_plus_system")


def main() -> None:
    df = load_five_agent()
    y_true = df["y_true"].values
    score = df["final_score"].values

    auc_val, op_roc = plot_roc_system(y_true, score)
    ap_val, op_pr = plot_pr_system(y_true, score)

    scores_by_name = {}
    for agent_name, col in AGENT_COLS.items():
        scores_by_name[agent_name] = (df[col].values, AGENT_COLOUR[agent_name], "-")
    scores_by_name["System (mean)"] = (score, AGENT_COLOUR["System"], "-")

    plot_all_roc(y_true, scores_by_name)
    plot_all_pr(y_true, scores_by_name)

    summary = {
        "auc_roc": float(auc_val),
        "average_precision": float(ap_val),
        "operating_tpr_at_tau_0.5": float(op_roc["tpr"]),
        "operating_fpr_at_tau_0.5": float(op_roc["fpr"]),
        "operating_precision_at_tau_0.5": float(op_pr["precision"]),
        "operating_recall_at_tau_0.5": float(op_pr["recall"]),
    }
    save_json(summary, OUT / "roc_pr_summary.json")
    print(f"[task02] AUC-ROC = {auc_val:.4f}  AP = {ap_val:.4f} "
          f"op-point TPR={op_roc['tpr']:.4f} FPR={op_roc['fpr']:.4f}")


if __name__ == "__main__":
    main()
