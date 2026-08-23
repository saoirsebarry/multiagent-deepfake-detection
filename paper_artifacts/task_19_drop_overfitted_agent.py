"""Task 19: does dropping the over-fitted agent help?

Task 18 showed FreqNet is the one agent that has fitted to particular vocoders: its
recall runs from 99.5% on MicroTTS to 6.3% on Vall-E-X. The natural question is
whether the ensemble is better off without it. This re-aggregates the stored
per-agent scores for every configuration, so no retraining is involved.

Reported per configuration: threshold-free AUC-ROC and average precision, accuracy
and errors at tau = 0.37, the separation margin, and recall on each manipulation
method - because a method-level view is the only way to see whether an over-fitted
member drags the ensemble down where it is weak.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
CSV = HERE / "source_csvs" / "analysis_results_with_5_agents.csv"
TAU = 0.37
AGENTS = {
    "Visual": ("score_Visual (Spatial)", 0.20),
    "FreqNet": ("score_Audio (Mel+CNN)", 0.15),
    "ECAPA": ("score_Audio Forensics (ECAPA)", 0.20),
    "CrossModal": ("score_Cross-Modal (Lip-Sync)", 0.25),
    "Biometric": ("score_Facial Biometric (Quality)", 0.20),
}

df = pd.read_csv(CSV)
y = (df.ground_truth.str.lower() == "fake").astype(int).to_numpy()
method = df.filepath.str.extract(r"_to_[a-z]{2}_(.+?)_label_fake")[0]
methods = list(method.dropna().value_counts().index)


def evaluate(members):
    cols = [AGENTS[m][0] for m in members]
    w = np.array([AGENTS[m][1] for m in members], float)
    w = w / w.sum()                       # renormalise, as Phase 1 does
    agg = df[cols].to_numpy() @ w
    pred = (agg > TAU).astype(int)
    rec = {}
    for mth in methods:
        sel = (method == mth).to_numpy() & (y == 1)
        rec[mth] = float(pred[sel].mean()) if sel.sum() else float("nan")
    return {
        "members": list(members),
        "auc": float(roc_auc_score(y, agg)),
        "ap": float(average_precision_score(y, agg)),
        "accuracy": float((pred == y).mean()),
        "errors": int((pred != y).sum()),
        "false_pos": int(((pred == 1) & (y == 0)).sum()),
        "false_neg": int(((pred == 0) & (y == 1)).sum()),
        "margin": float(agg[y == 1].min() - agg[y == 0].max()),
        "recall_by_method": {k: round(v, 4) for k, v in rec.items()},
        "worst_method_recall": float(min(rec.values())),
        "recall_spread": float(max(rec.values()) - min(rec.values())),
    }


ALL = list(AGENTS)
NO_FREQ = [m for m in ALL if m != "FreqNet"]
configs = {"all five": ALL, "without FreqNet": NO_FREQ}
# also drop each other agent once, so "without FreqNet" has something to be compared against
for m in ALL:
    if m != "FreqNet":
        configs[f"without {m}"] = [x for x in ALL if x != m]

res = {name: evaluate(mem) for name, mem in configs.items()}

print(f"PolyGlotFake test set, n={len(df)}, tau={TAU}\n")
hdr = f"{'configuration':18s}{'AUC':>9s}{'AP':>8s}{'acc %':>8s}{'err':>5s}{'FP':>4s}{'FN':>4s}{'margin':>9s}"
print(hdr); print("-" * len(hdr))
for name, r in res.items():
    print(f"{name:18s}{r['auc']:9.5f}{r['ap']:8.4f}{100*r['accuracy']:8.3f}"
          f"{r['errors']:5d}{r['false_pos']:4d}{r['false_neg']:4d}{r['margin']:9.4f}")

print(f"\nENSEMBLE RECALL BY MANIPULATION METHOD")
print(f"{'configuration':18s}" + "".join(f"{m[:9]:>11s}" for m in methods) + f"{'worst':>9s}")
for name in ("all five", "without FreqNet"):
    r = res[name]
    print(f"{name:18s}" + "".join(f"{100*r['recall_by_method'][m]:10.2f}%" for m in methods)
          + f"{100*r['worst_method_recall']:8.2f}%")

a, b = res["all five"], res["without FreqNet"]
print("\nVERDICT")
print(f"   dropping FreqNet changes AUC by {b['auc']-a['auc']:+.5f} and accuracy by "
      f"{100*(b['accuracy']-a['accuracy']):+.3f} pp ({a['errors']} -> {b['errors']} errors)")
worst_m = min(a['recall_by_method'], key=lambda k: a['recall_by_method'][k])
print(f"   on {worst_m}, the ensemble's weakest method, recall goes "
      f"{100*a['recall_by_method'][worst_m]:.2f}% -> {100*b['recall_by_method'][worst_m]:.2f}%")
print(f"   FreqNet's own recall there is 6.3% (Table 17), yet the ensemble "
      f"{'still benefits from' if b['errors'] > a['errors'] else 'is better without'} it")

(HERE / "drop_overfitted_agent.json").write_text(json.dumps(res, indent=2))
pd.DataFrame([{k: v for k, v in r.items() if k != "recall_by_method"} for r in res.values()]
             ).to_csv(HERE / "drop_overfitted_agent.csv", index=False)
print("\nwrote drop_overfitted_agent.json / .csv")
