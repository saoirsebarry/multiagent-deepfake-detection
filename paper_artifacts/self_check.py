"""Self-check: load all task JSON outputs and verify cross-task consistency.

Fails loudly (non-zero exit) on any numeric disagreement beyond tolerance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent

TOL = 1e-4  # floating-point tolerance for cross-task metric agreement


def load(name: str):
    p = OUT / name
    if not p.exists():
        raise SystemExit(f"FAIL self-check: {name} not found")
    with open(p) as f:
        return json.load(f)


def approx_eq(a: float, b: float, tol: float = TOL) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def main() -> None:
    errors: list[str] = []

    headline = load("headline_metrics.json")
    bootstrap = load("bootstrap_cis.json")
    roc_pr = load("roc_pr_summary.json")
    ablation_first_row = None
    import csv
    with open(OUT / "ablation_table.csv") as f:
        ablation_first_row = next(csv.DictReader(f))
    three = load("three_agent_metrics.json")

    # 1. headline accuracy == bootstrap point estimate
    if not approx_eq(headline["accuracy"], bootstrap["accuracy"]["point"]):
        errors.append(
            f"accuracy mismatch: headline={headline['accuracy']:.6f} vs "
            f"bootstrap point={bootstrap['accuracy']['point']:.6f}"
        )
    # 2. headline precision/recall/f1 == bootstrap
    for m in ("precision", "recall", "f1"):
        if not approx_eq(headline[m], bootstrap[m]["point"]):
            errors.append(
                f"{m} mismatch: headline={headline[m]:.6f} vs "
                f"bootstrap point={bootstrap[m]['point']:.6f}"
            )
    # 3. ROC/PR operating-point precision/recall == headline
    if not approx_eq(roc_pr["operating_precision_at_tau_0.5"], headline["precision"]):
        errors.append(
            f"operating precision mismatch: roc_pr="
            f"{roc_pr['operating_precision_at_tau_0.5']:.6f} vs headline="
            f"{headline['precision']:.6f}"
        )
    if not approx_eq(roc_pr["operating_recall_at_tau_0.5"], headline["recall"]):
        errors.append(
            f"operating recall mismatch: roc_pr="
            f"{roc_pr['operating_recall_at_tau_0.5']:.6f} vs headline="
            f"{headline['recall']:.6f}"
        )
    # 4. Ablation baseline accuracy == headline accuracy
    ab_acc = float(ablation_first_row["accuracy"])
    if not approx_eq(ab_acc, headline["accuracy"]):
        errors.append(
            f"ablation baseline accuracy {ab_acc:.6f} != "
            f"headline accuracy {headline['accuracy']:.6f}"
        )
    # 5. Ablation baseline miscount == headline miscount
    if int(ablation_first_row["miscount"]) != int(headline["miscount"]):
        errors.append(
            f"ablation baseline miscount {ablation_first_row['miscount']} != "
            f"headline miscount {headline['miscount']}"
        )
    # 6. Three-agent sanity: accuracy between 0 and 1, miscount in range
    if not 0.0 <= three["accuracy"] <= 1.0:
        errors.append(f"three-agent accuracy out of range: {three['accuracy']}")

    # 7. Bootstrap CIs contain the point estimate
    for m in ("accuracy", "precision", "recall", "f1", "auc_roc", "average_precision"):
        e = bootstrap[m]
        p, lo, hi = e["point"], e["ci_low"], e["ci_high"]
        # Precision has edge case when all resamples are 100%
        if not (lo - TOL <= p <= hi + TOL):
            errors.append(
                f"bootstrap {m}: point {p:.4f} outside CI [{lo:.4f}, {hi:.4f}]"
            )

    # 8. AUC/AP: roc_pr_summary == bootstrap point
    if not approx_eq(roc_pr["auc_roc"], bootstrap["auc_roc"]["point"]):
        errors.append(
            f"AUC mismatch: roc_pr={roc_pr['auc_roc']:.6f} vs "
            f"bootstrap={bootstrap['auc_roc']['point']:.6f}"
        )
    if not approx_eq(roc_pr["average_precision"], bootstrap["average_precision"]["point"]):
        errors.append(
            f"AP mismatch: roc_pr={roc_pr['average_precision']:.6f} vs "
            f"bootstrap={bootstrap['average_precision']['point']:.6f}"
        )

    # Sanity table
    print("=== Self-check sanity table ===")
    print(f"  headline accuracy      = {headline['accuracy'] * 100:.4f}%  "
          f"(CI [{bootstrap['accuracy']['ci_low'] * 100:.2f}%, "
          f"{bootstrap['accuracy']['ci_high'] * 100:.2f}%])")
    print(f"  headline precision     = {headline['precision'] * 100:.4f}%")
    print(f"  headline recall        = {headline['recall'] * 100:.4f}%")
    print(f"  headline F1            = {headline['f1'] * 100:.4f}%")
    print(f"  AUC-ROC                = {roc_pr['auc_roc']:.4f}")
    print(f"  AP                     = {roc_pr['average_precision']:.4f}")
    print(f"  ablation baseline acc  = {ab_acc * 100:.4f}%  "
          f"(miscount={ablation_first_row['miscount']})")
    print(f"  three-agent accuracy   = {three['accuracy'] * 100:.4f}%  "
          f"(miscount={three['miscount']})")

    if errors:
        print("\n*** SELF-CHECK FAILED ***")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("\nself-check OK")


if __name__ == "__main__":
    main()
