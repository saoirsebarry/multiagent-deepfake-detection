"""Task 5b: Ablation table regenerated at tau = 0.37 (the code's default).

Same procedure as task_05_ablation.py but at tau = 0.37 instead of 0.5.
At tau = 0.37 the full 5-agent baseline has 0 errors (100 % accuracy), so
any "Remove X" row that introduces errors represents a real contribution
from the removed agent — not an artefact of averaging under a
conservative decision boundary.
"""
from __future__ import annotations

import pandas as pd

from _common import (
    AGENT_COLS, AGENT_WEIGHTS, ALL_AGENT_COLS, OUT,
    confusion_counts, load_five_agent, metrics_from_counts, weighted_mean,
)

TAU = 0.37


def metrics_for(df, cols):
    score = weighted_mean(df, cols)
    pred = (score >= TAU).astype(int)
    c = confusion_counts(df["y_true"].values, pred)
    m = metrics_from_counts(c)
    return {**m, **c}


def main():
    df = load_five_agent()
    base = metrics_for(df, ALL_AGENT_COLS)
    base_acc = base["accuracy"]
    base_miss = base["miscount"]

    removal_label = {
        "XceptionNet": "Visual (XceptionNet)",
        "FreqNet": "Audio (FreqNet)",
        "ECAPA": "Audio Forensics (ECAPA)",
        "CrossModal": "Cross-Modal (Lip-Sync)",
        "Biometric": "Facial Biometric (Quality)",
    }

    rows = [{
        "configuration": "Full 5-agent ensemble (baseline)",
        "n_agents": 5,
        "accuracy": base["accuracy"],
        "miscount": base["miscount"],
        "delta_pp": 0.0,
        "fp": base["fp"], "fn": base["fn"],
        "anomaly": "",
    }]
    for key, col in AGENT_COLS.items():
        remaining = [c for c in ALL_AGENT_COLS if c != col]
        m = metrics_for(df, remaining)
        delta_pp = (m["accuracy"] - base_acc) * 100
        anomaly = ""
        if m["miscount"] < base_miss:
            anomaly = (
                f"miscount {m['miscount']} < baseline {base_miss}: "
                "removed agent was net-harmful at this threshold (inspect)"
            )
        rows.append({
            "configuration": f"Remove {removal_label[key]}",
            "n_agents": 4,
            "accuracy": m["accuracy"],
            "miscount": m["miscount"],
            "delta_pp": delta_pp,
            "fp": m["fp"], "fn": m["fn"],
            "anomaly": anomaly,
        })
    # Top-3, Audio-only, Visual-only, Single best (Cross-Modal)
    for label, cols, n in [
        ("Top-3 (Biometric + ECAPA + Cross-Modal)",
         [AGENT_COLS["Biometric"], AGENT_COLS["ECAPA"], AGENT_COLS["CrossModal"]], 3),
        ("Audio only (FreqNet + ECAPA)",
         [AGENT_COLS["FreqNet"], AGENT_COLS["ECAPA"]], 2),
        ("Visual only (XceptionNet + Biometric)",
         [AGENT_COLS["XceptionNet"], AGENT_COLS["Biometric"]], 2),
        ("Single best agent (Cross-Modal)",
         [AGENT_COLS["CrossModal"]], 1),
    ]:
        m = metrics_for(df, cols)
        rows.append({
            "configuration": label,
            "n_agents": n,
            "accuracy": m["accuracy"],
            "miscount": m["miscount"],
            "delta_pp": (m["accuracy"] - base_acc) * 100,
            "fp": m["fp"], "fn": m["fn"],
            "anomaly": "",
        })

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "ablation_table_tau037.csv", index=False, float_format="%.6f")

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Configuration & \# agents & Accuracy (\%) & Miscount & $\Delta$ (pp) \\",
        r"\midrule",
    ]
    b = rows[0]
    lines.append(
        f"{b['configuration']} & {b['n_agents']} & "
        f"{b['accuracy']*100:.3f} & {b['miscount']} & --- \\\\"
    )
    lines.append(r"\midrule")
    for r in rows[1:]:
        delta = r["delta_pp"]
        sign = "+" if delta > 0 else ("$-$" if delta < 0 else "")
        delta_str = f"{sign}{abs(delta):.3f}"
        mark = r" \dag" if r["anomaly"] else ""
        lines.append(
            f"{r['configuration']}{mark} & {r['n_agents']} & "
            f"{r['accuracy']*100:.3f} & {r['miscount']} & {delta_str} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "ablation_table_tau037.tex").write_text("\n".join(lines) + "\n")

    anomalies = [r for r in rows if r["anomaly"]]
    print(f"[task05b] tau=0.37 baseline miscount={base_miss}, "
          f"acc={base_acc*100:.3f}% | rows={len(rows)} | anomalies={len(anomalies)}")
    for r in rows:
        print(f"  {r['configuration']:<45s} acc={r['accuracy']*100:6.3f}%  "
              f"miss={r['miscount']:<3d} delta={r['delta_pp']:+6.3f}pp "
              f"(FP={r['fp']}, FN={r['fn']})")
    for a in anomalies:
        print(f"  ANOMALY: {a['configuration']}: {a['anomaly']}")


if __name__ == "__main__":
    main()
