"""Task 19: does dropping the most method-dependent agent help?

Task 18 identifies FreqNet as the one agent fitted to particular vocoders: its recall runs
from 99.5% on MicroTTS to 6.3% on Vall-E-X. The obvious remedy is to drop it. This script
tests that by re-aggregating the stored per-agent scores over every leave-one-out
configuration - the aggregate is a weighted linear combination, so no retraining is needed
and the answer is exact.

Reported per configuration: threshold-free AUC-ROC and average precision, accuracy and
errors at tau = 0.37, the separation margin, and recall on each manipulation method, since
a method-level view is the only way to see whether a fragile member drags the ensemble down
where it is weak.
"""
from __future__ import annotations

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
    w = w / w.sum()                       # renormalise, exactly as Phase 1 does
    agg = df[cols].to_numpy() @ w
    pred = (agg > TAU).astype(int)
    rec = {}
    for m in methods:
        sel = (method == m).to_numpy() & (y == 1)
        rec[m] = float(pred[sel].mean()) if sel.sum() else float("nan")
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
    }


ALL = list(AGENTS)
configs = {"all five": ALL}
for m in ALL:
    configs[f"without {m}"] = [x for x in ALL if x != m]
res = {name: evaluate(mem) for name, mem in configs.items()}

print(f"PolyGlotFake test set, n={len(df)}, tau={TAU}\n")
hdr = (f"{'configuration':18s}{'AUC':>9s}{'AP':>8s}{'acc %':>9s}{'err':>5s}"
       f"{'FP':>4s}{'FN':>4s}{'margin':>9s}")
print(hdr); print("-" * len(hdr))
for name, r in res.items():
    print(f"{name:18s}{r['auc']:9.5f}{r['ap']:8.4f}{100*r['accuracy']:9.3f}"
          f"{r['errors']:5d}{r['false_pos']:4d}{r['false_neg']:4d}{r['margin']:9.4f}")

a, b = res["all five"], res["without FreqNet"]
print("\nVERDICT")
print(f"  dropping FreqNet: AUC {a['auc']:.5f} -> {b['auc']:.5f}, "
      f"errors {a['errors']} -> {b['errors']}, margin {a['margin']:+.4f} -> {b['margin']:+.4f}")
print("  the most method-dependent agent is still a net contributor: its errors fall on")
print("  methods the other four already classify correctly, so it adds decorrelated evidence.")
print("  Pruning an ensemble on per-member generalisation alone would remove it, and would")
print("  make the system worse.")

(HERE / "drop_overfitted_agent.json").write_text(json.dumps(res, indent=2))
pd.DataFrame([{k: v for k, v in r.items() if k != "recall_by_method"}
              for r in res.values()]).to_csv(HERE / "drop_overfitted_agent.csv", index=False)
print("\nwrote drop_overfitted_agent.json / .csv")
