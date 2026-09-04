"""Task 0: Derive a decision threshold on a VALIDATION score file, freeze it, and
apply it to the TEST split.

This script codifies a val -> freeze -> test selection procedure: tau and
(optionally) the weights are selected only from the validation CSV, then frozen and
evaluated once on `analysis_results_with_5_agents.csv`. NOTE: no faithful validation
score file currently exists — re-scoring from the released checkpoints does not
reproduce the released FreqNet and Cross-Modal scores (see docs/REPRODUCE.md,
"Known limitation"), so this script cannot presently audit the paper's operating
point, and the paper makes no validation-provenance claim for it.

Inputs (stdlib only — no numpy/pandas needed):
  --val   path to the validation per-agent score CSV
          (generate with: python src/orchestrator.py --split val \
             --output_file paper_artifacts/source_csvs/analysis_results_with_5_agents_VAL.csv)
  --test  path to the test per-agent score CSV (already shipped)

Outputs:
  paper_artifacts/operating_point_provenance.json

The script is deliberately honest: if the validation-derived threshold is NOT 0.37,
or the validation-best weights are NOT the paper's hand-picked vector, it says so
loudly rather than hiding the discrepancy.
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Per-agent score columns, in a fixed order. Same labels the orchestrator writes.
AGENT_COLS = [
    "score_Visual (Spatial)",
    "score_Audio (Mel+CNN)",
    "score_Audio Forensics (ECAPA)",
    "score_Cross-Modal (Lip-Sync)",
    "score_Facial Biometric (Quality)",
]

# The weight vector the paper reports. NOT an optimiser output — a design choice
# this script validates against the data, rather than assumes.
PAPER_WEIGHTS = {
    "score_Visual (Spatial)": 0.20,
    "score_Audio (Mel+CNN)": 0.15,
    "score_Audio Forensics (ECAPA)": 0.20,
    "score_Cross-Modal (Lip-Sync)": 0.25,
    "score_Facial Biometric (Quality)": 0.20,
}
PAPER_TAU = 0.37


# --------------------------------------------------------------------------- #
# Data + metric helpers (pure stdlib)
# --------------------------------------------------------------------------- #
def load(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"STOP: {path} is empty")
    return rows


def labels(rows: list[dict]) -> list[int]:
    return [1 if r["ground_truth"].strip().lower().startswith("fake") else 0 for r in rows]


def weighted_scores(rows: list[dict], weights: dict[str, float]) -> list[float]:
    """Per-row weighted mean of the agent columns, renormalised over the weights given."""
    total = sum(weights[c] for c in AGENT_COLS)
    out = []
    for r in rows:
        s = sum(float(r[c]) * weights[c] for c in AGENT_COLS) / total
        out.append(s)
    return out


def auc(scores: list[float], y: list[int]) -> float:
    """ROC-AUC via the Mann-Whitney U statistic with average-rank tie handling."""
    paired = sorted(zip(scores, y), key=lambda t: t[0])
    n = len(paired)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and paired[j][0] == paired[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    npos = sum(1 for _, lab in paired if lab == 1)
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    rank_pos = sum(r for r, (_, lab) in zip(ranks, paired) if lab == 1)
    return (rank_pos - npos * (npos + 1) / 2.0) / (npos * nneg)


def metrics_at(scores: list[float], y: list[int], tau: float) -> dict:
    tp = tn = fp = fn = 0
    for s, lab in zip(scores, y):
        pred = 1 if s >= tau else 0
        if pred == 1 and lab == 1:
            tp += 1
        elif pred == 0 and lab == 0:
            tn += 1
        elif pred == 1 and lab == 0:
            fp += 1
        else:
            fn += 1
    n = tp + tn + fp + fn
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "tau": tau,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / n if n else 0.0,
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "precision": prec,
        "recall": tpr,
        "f1": (2 * prec * tpr / (prec + tpr)) if (prec + tpr) else 0.0,
        "miscount": fp + fn,
    }


def separation(scores: list[float], y: list[int]) -> dict:
    real = [s for s, lab in zip(scores, y) if lab == 0]
    fake = [s for s, lab in zip(scores, y) if lab == 1]
    max_real = max(real) if real else None
    min_fake = min(fake) if fake else None
    sep = (max_real is not None and min_fake is not None and max_real < min_fake)
    return {
        "max_real": max_real,
        "min_fake": min_fake,
        "separable": sep,
        "gap": (min_fake - max_real) if sep else None,
    }


def select_tau_on_val(scores: list[float], y: list[int]) -> dict:
    """Pick tau that maximises validation balanced accuracy; among the tied plateau
    take the midpoint (maximum margin). Robust whether or not val is separable."""
    cands = sorted(set(scores))
    grid = [cands[0] - 1e-6]
    grid += [(cands[i] + cands[i + 1]) / 2.0 for i in range(len(cands) - 1)]
    grid += [cands[-1] + 1e-6]
    scored = [(t, metrics_at(scores, y, t)["balanced_accuracy"]) for t in grid]
    best = max(b for _, b in scored)
    plateau = [t for t, b in scored if best - b <= 1e-12]
    lo, hi = min(plateau), max(plateau)
    return {
        "rule": "argmax val balanced accuracy; midpoint of the tied plateau (max margin)",
        "best_balanced_accuracy": best,
        "plateau_low": lo,
        "plateau_high": hi,
        "tau_star": (lo + hi) / 2.0,
    }


def search_weights_on_val(rows: list[dict], y: list[int], step: float = 0.05) -> dict:
    """Coarse simplex search (step 0.05) for the weights that maximise the validation
    real/fake score margin. Reported for transparency — it is NOT what the paper used,
    but it shows whether the paper's hand-picked vector is near the val optimum."""
    units = round(1.0 / step)
    best = None
    for combo in product(range(units + 1), repeat=len(AGENT_COLS) - 1):
        if sum(combo) > units:
            continue
        last = units - sum(combo)
        raw = list(combo) + [last]
        w = {c: raw[i] * step for i, c in enumerate(AGENT_COLS)}
        s = weighted_scores(rows, w)
        sep = separation(s, y)
        margin = sep["gap"] if sep["separable"] else (
            (sep["min_fake"] - sep["max_real"]) if sep["min_fake"] is not None else -1.0
        )
        if best is None or margin > best["margin"]:
            best = {"weights": w, "margin": margin, "separable": sep["separable"]}
    return best


# --------------------------------------------------------------------------- #
def describe_weights(rows: list[dict], y: list[int], weights: dict) -> dict:
    s = weighted_scores(rows, weights)
    sep = separation(s, y)
    return {
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "val_auc": auc(s, y),
        "val_separation": sep,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", type=Path,
                    default=HERE / "source_csvs" / "analysis_results_with_5_agents_VAL.csv")
    ap.add_argument("--test", type=Path,
                    default=HERE / "source_csvs" / "analysis_results_with_5_agents.csv")
    ap.add_argument("--weights", choices=["paper", "auc", "search"], default="paper",
                    help="Which weight vector to freeze for the test evaluation (default: paper).")
    ap.add_argument("--search-step", type=float, default=0.05,
                    help="Granularity of the transparency weight search (default: 0.05).")
    ap.add_argument("--no-search", action="store_true",
                    help="Skip the simplex weight search (faster). Required if --weights search is not used.")
    ap.add_argument("--out", type=Path, default=HERE / "operating_point_provenance.json")
    args = ap.parse_args()
    if args.weights == "search" and args.no_search:
        ap.error("--weights search is incompatible with --no-search")

    if not args.val.exists():
        raise SystemExit(
            f"STOP: validation CSV not found at {args.val}\n"
            "Generate it first with:\n"
            "  python src/orchestrator.py --split val \\\n"
            "    --output_file paper_artifacts/source_csvs/analysis_results_with_5_agents_VAL.csv"
        )

    val = load(args.val)
    test = load(args.test)
    yv, yt = labels(val), labels(test)

    # 1. Per-agent discrimination on validation (the empirical basis the paper narrates).
    per_agent = {}
    for c in AGENT_COLS:
        sc = [float(r[c]) for r in val]
        per_agent[c] = {"val_auc": auc(sc, yv), "val_separation": separation(sc, yv)}

    # 2. Three weight vectors, each described on validation.
    auc_raw = {c: max(per_agent[c]["val_auc"] - 0.5, 0.0) for c in AGENT_COLS}
    tot = sum(auc_raw.values()) or 1.0
    auc_weights = {c: auc_raw[c] / tot for c in AGENT_COLS}

    weight_options = {
        "paper": describe_weights(val, yv, PAPER_WEIGHTS),
        "auc": describe_weights(val, yv, auc_weights),
    }
    search_best = None
    if not args.no_search or args.weights == "search":
        search_best = search_weights_on_val(val, yv, step=args.search_step)
        weight_options["search"] = describe_weights(val, yv, search_best["weights"])

    chosen_map = {"paper": PAPER_WEIGHTS, "auc": auc_weights}
    if search_best is not None:
        chosen_map["search"] = search_best["weights"]
    chosen_weights = chosen_map[args.weights]

    # 3. Derive tau on VALIDATION using the chosen weights.
    val_scores = weighted_scores(val, chosen_weights)
    tau_sel = select_tau_on_val(val_scores, yv)
    tau_star = tau_sel["tau_star"]
    val_sep = separation(val_scores, yv)

    # 4. FREEZE (weights, tau_star); evaluate ONCE on test. Also report the paper's tau for reference.
    test_scores = weighted_scores(test, chosen_weights)
    test_at_tau_star = metrics_at(test_scores, yt, tau_star)
    test_at_paper_tau = metrics_at(test_scores, yt, PAPER_TAU)

    reproduces_tau = abs(tau_star - PAPER_TAU) <= 0.02
    weights_match_paper = (args.weights == "paper") or all(
        abs(chosen_weights[c] - PAPER_WEIGHTS[c]) <= 1e-9 for c in AGENT_COLS
    )

    provenance = {
        "inputs": {"val_csv": str(args.val), "test_csv": str(args.test),
                   "n_val": len(val), "n_test": len(test),
                   "weights_source": args.weights},
        "paper_claim": {"tau": PAPER_TAU, "weights": {k: round(v, 4) for k, v in PAPER_WEIGHTS.items()}},
        "val_per_agent": per_agent,
        "weight_options_on_val": weight_options,
        "chosen_weights": {k: round(v, 4) for k, v in chosen_weights.items()},
        "tau_selection_on_val": tau_sel,
        "val_separation_at_chosen_weights": val_sep,
        "val_metrics_at_tau_star": metrics_at(val_scores, yv, tau_star),
        "test_metrics_at_tau_star_FROZEN": test_at_tau_star,
        "test_metrics_at_paper_tau_0_37": test_at_paper_tau,
        "reproduces_paper_tau": reproduces_tau,
        "weights_match_paper": weights_match_paper,
    }
    with open(args.out, "w") as f:
        json.dump(provenance, f, indent=2)

    # ----- Human-readable summary ----- #
    print("=" * 70)
    print("OPERATING-POINT PROVENANCE  (val -> freeze -> test)")
    print("=" * 70)
    print(f"weights source: {args.weights}   n_val={len(val)}  n_test={len(test)}")
    print("\nPer-agent validation AUC:")
    for c in AGENT_COLS:
        print(f"  {c:34s} AUC={per_agent[c]['val_auc']:.4f}")
    print(f"\nValidation-derived threshold  tau* = {tau_star:.4f}")
    print(f"  rule: {tau_sel['rule']}")
    print(f"  val plateau: [{tau_sel['plateau_low']:.4f}, {tau_sel['plateau_high']:.4f}]")
    if val_sep["separable"]:
        print(f"  val separable: max_real={val_sep['max_real']:.4f} < min_fake={val_sep['min_fake']:.4f}")
    print(f"\nFROZEN on test  (weights={args.weights}, tau={tau_star:.4f}):")
    m = test_at_tau_star
    print(f"  accuracy={m['accuracy']*100:.4f}%  FP={m['fp']}  FN={m['fn']}  (test reference at paper tau=0.37: "
          f"{test_at_paper_tau['accuracy']*100:.4f}%)")

    print("\n" + "-" * 70)
    if reproduces_tau:
        print(f"[OK] Validation-derived tau*={tau_star:.4f} reproduces the paper's 0.37 (within 0.02).")
    else:
        print(f"[PROVENANCE NOTE] Validation-derived tau*={tau_star:.4f} does NOT match the paper's 0.37.")
        print("     Either adopt the validation-derived value in the paper, or investigate the discrepancy.")
    if args.weights == "paper":
        po = weight_options["paper"]["val_separation"]
        print(f"[INFO] Paper weights on val: AUC={weight_options['paper']['val_auc']:.4f}, "
              f"separable={po['separable']}.")
        if search_best is not None:
            so = weight_options["search"]
            print("       Best val-margin search weights: "
                  f"{ {k.split('(')[-1].rstrip(')'): round(v,2) for k,v in so['weights'].items()} }.")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
