"""Task 4: Bootstrap 95% confidence intervals at tau = 0.5.

10,000 resamples with replacement, seed 42. Reports point estimate plus
2.5 / 97.5 percentiles for accuracy, precision, recall, F1, AUC-ROC, AP.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from _common import (
    OUT, RANDOM_SEED, TAU, confusion_counts, fmt_pct,
    load_five_agent, metrics_from_counts, predict_at_tau, save_json,
)

N_ITER = 10_000


def compute_metrics(y_true: np.ndarray, score: np.ndarray) -> dict:
    pred = predict_at_tau(score, TAU)
    c = confusion_counts(y_true, pred)
    m = metrics_from_counts(c)
    # AUC/AP require both classes present
    if len(np.unique(y_true)) < 2:
        return {**m, "auc_roc": np.nan, "average_precision": np.nan}
    auc = roc_auc_score(y_true, score)
    ap = average_precision_score(y_true, score)
    return {**m, "auc_roc": auc, "average_precision": ap}


def main() -> None:
    df = load_five_agent()
    y_true = df["y_true"].values
    score = df["final_score"].values
    n = len(df)

    point = compute_metrics(y_true, score)

    rng = np.random.default_rng(RANDOM_SEED)
    acc = np.empty(N_ITER)
    prec = np.empty(N_ITER)
    rec = np.empty(N_ITER)
    f1 = np.empty(N_ITER)
    auc = np.full(N_ITER, np.nan)
    ap = np.full(N_ITER, np.nan)

    for i in range(N_ITER):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        sc = score[idx]
        m = compute_metrics(yt, sc)
        acc[i] = m["accuracy"]
        prec[i] = m["precision"]
        rec[i] = m["recall"]
        f1[i] = m["f1"]
        auc[i] = m["auc_roc"]
        ap[i] = m["average_precision"]

    def ci(arr: np.ndarray) -> tuple[float, float]:
        clean = arr[~np.isnan(arr)]
        return float(np.percentile(clean, 2.5)), float(np.percentile(clean, 97.5))

    low, high = ci(acc)
    acc_ci = {"point": float(point["accuracy"]), "ci_low": low, "ci_high": high}
    low, high = ci(prec); prec_ci = {"point": float(point["precision"]), "ci_low": low, "ci_high": high}
    low, high = ci(rec); rec_ci = {"point": float(point["recall"]), "ci_low": low, "ci_high": high}
    low, high = ci(f1); f1_ci = {"point": float(point["f1"]), "ci_low": low, "ci_high": high}
    low, high = ci(auc); auc_ci = {"point": float(point["auc_roc"]), "ci_low": low, "ci_high": high}
    low, high = ci(ap); ap_ci = {"point": float(point["average_precision"]), "ci_low": low, "ci_high": high}

    out = {
        "n_iterations": N_ITER,
        "seed": RANDOM_SEED,
        "accuracy": acc_ci,
        "precision": prec_ci,
        "recall": rec_ci,
        "f1": f1_ci,
        "auc_roc": auc_ci,
        "average_precision": ap_ci,
    }
    save_json(out, OUT / "bootstrap_cis.json")

    abstract = (
        f"{acc_ci['point'] * 100:.2f}% accuracy "
        f"(95% CI [{acc_ci['ci_low'] * 100:.2f}%, {acc_ci['ci_high'] * 100:.2f}%])"
    )
    print(f"[task04] {abstract}")
    print(f"[task04] AUC = {auc_ci['point']:.4f} [{auc_ci['ci_low']:.4f}, {auc_ci['ci_high']:.4f}]")
    print(f"[task04] AP  = {ap_ci['point']:.4f} [{ap_ci['ci_low']:.4f}, {ap_ci['ci_high']:.4f}]")


if __name__ == "__main__":
    main()
