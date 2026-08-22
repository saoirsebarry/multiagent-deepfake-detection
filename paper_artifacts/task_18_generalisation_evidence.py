"""Task 18: per-agent generalisation evidence that does not require retraining.

Reviewer comment 4 asks for training curves "to demonstrate convergence and avoid
overfitting concerns". Per-epoch histories survive for one agent only. This script
supplies the second half of that question - over-fitting - directly, for all five
agents, from the released test predictions.

Two measurements:

  1. Development-to-test gap. Each agent's development accuracy, as reported in the
     methodology, against its measured accuracy on the held-out test set. A large
     drop is the signature of over-fitting to the development split.

  2. Per-manipulation-method recall. PolyGlotFake filenames encode which synthesis
     method produced each fake, so recall can be split by method. An agent that has
     over-fitted to particular artifacts shows a wide spread across methods; one that
     has learned a general cue shows a narrow spread. This speaks to the dataset's
     "known and finite" manipulation set directly, which loss curves cannot.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "source_csvs" / "analysis_results_with_5_agents.csv"
TAU = 0.37

AGENTS = {
    "Visual (XceptionNet)": "score_Visual (Spatial)",
    "Audio (FreqNet)": "score_Audio (Mel+CNN)",
    "Audio Forensics (ECAPA-TDNN)": "score_Audio Forensics (ECAPA)",
    "Cross-Modal (Lip-Sync)": "score_Cross-Modal (Lip-Sync)",
    "Biometric-Quality": "score_Facial Biometric (Quality)",
}

# Development-set accuracy as stated in the methodology, with its source, so each
# number in the gap column is traceable. None where the manuscript reports none.
DEV = {
    "Visual (XceptionNet)": (0.940, "§3.3.1, Stage 2 validation accuracy"),
    "Audio (FreqNet)": (0.980, "§3.3.2, reported accuracy for the final audio agent"),
    "Audio Forensics (ECAPA-TDNN)": (None, "no development accuracy reported"),
    "Cross-Modal (Lip-Sync)": (0.990, "§3.3.3, cross-attention validation accuracy"),
    "Biometric-Quality": (0.9958, "released checkpoint val_metrics.accuracy at the best epoch"),
}

df = pd.read_csv(CSV)
y = (df.ground_truth.str.lower() == "fake").astype(int).to_numpy()
method = df.filepath.str.extract(r"_to_[a-z]{2}_(.+?)_label_fake")[0]

print(f"PolyGlotFake test set: {len(df)} clips ({(y == 0).sum()} real, {y.sum()} fake)\n")

# ---- 1. development-to-test gap -------------------------------------------
print("1. DEVELOPMENT-TO-TEST GAP")
print(f"   {'agent':30s} {'dev':>7s} {'test':>7s} {'gap':>8s}   source")
rows = []
for name, col in AGENTS.items():
    pred = (df[col].to_numpy() > TAU).astype(int)
    test_acc = float((pred == y).mean())
    dev, src = DEV[name]
    gap = None if dev is None else dev - test_acc
    rows.append({"agent": name, "dev_accuracy": dev, "test_accuracy": round(test_acc, 4),
                 "gap": None if gap is None else round(gap, 4), "dev_source": src})
    d = f"{dev*100:6.1f}%" if dev is not None else "     -"
    g = f"{gap*100:+7.1f}pp" if gap is not None else "       -"
    print(f"   {name:30s} {d} {test_acc*100:6.1f}% {g}   {src}")

# ---- 2. per-method recall --------------------------------------------------
print("\n2. RECALL BY MANIPULATION METHOD (fake clips only)")
methods = [m for m in method.dropna().unique()]
counts = {m: int((method == m).sum()) for m in methods}
order = sorted(methods, key=lambda m: -counts[m])
print(f"   {'agent':30s} " + "".join(f"{m[:9]:>10s}" for m in order) + f"{'spread':>9s}")
per_method = {}
for name, col in AGENTS.items():
    s = df[col].to_numpy()
    recalls = {}
    for m in order:
        sel = (method == m).to_numpy() & (y == 1)
        recalls[m] = float(((s > TAU)[sel] == 1).mean()) if sel.sum() else float("nan")
    spread = max(recalls.values()) - min(recalls.values())
    per_method[name] = {"recall": {k: round(v, 4) for k, v in recalls.items()},
                        "spread": round(spread, 4)}
    print(f"   {name:30s} " + "".join(f"{recalls[m]*100:9.1f}%" for m in order)
          + f"{spread*100:8.1f}pp")
print(f"\n   clips per method: " + ", ".join(f"{m} {counts[m]}" for m in order))

# ---- summary ---------------------------------------------------------------
gaps = [r["gap"] for r in rows if r["gap"] is not None]
spreads = [v["spread"] for v in per_method.values()]
print("\nSUMMARY")
print(f"   largest development-to-test gap : {max(gaps)*100:.1f}pp "
      f"({[r['agent'] for r in rows if r['gap'] == max(gaps)][0]})")
print(f"   smallest                        : {min(gaps)*100:.1f}pp "
      f"({[r['agent'] for r in rows if r['gap'] == min(gaps)][0]})")
print(f"   widest per-method recall spread : {max(spreads)*100:.1f}pp "
      f"({[k for k, v in per_method.items() if v['spread'] == max(spreads)][0]})")
print(f"   narrowest                       : {min(spreads)*100:.1f}pp "
      f"({[k for k, v in per_method.items() if v['spread'] == min(spreads)][0]})")

(HERE / "generalisation_evidence.json").write_text(json.dumps(
    {"tau": TAU, "n": int(len(df)), "development_to_test": rows,
     "per_method_recall": per_method, "method_counts": counts}, indent=2))
pd.DataFrame(rows).to_csv(HERE / "generalisation_gap.csv", index=False)
print("\nwrote generalisation_evidence.json and generalisation_gap.csv")
