"""Task 5: Ablation table regenerated from saved per-agent scores.

Important finding during preparation: the `final_score` column in
analysis_results_with_5_agents.csv is a WEIGHTED mean, not unweighted.
Reconstruction from the per-agent columns using the weights hard-coded
in multiagent_langchain_additional_agents.py
  { Visual 0.20, FreqNet 0.15, ECAPA 0.20, CrossModal 0.25, Biometric 0.20 }
reproduces final_score exactly (max abs diff < 1e-15 over all 2162 rows).

To keep the ablation baseline consistent with the headline (8 errors at
tau = 0.5) the ablation uses the SAME weighted aggregation rule, with the
remaining agents' weights re-normalised to sum to 1 when one is removed.

Sanity check: any "Remove X" row with miscount < baseline miscount is
flagged as an anomaly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import (
    AGENT_COLS, AGENT_WEIGHTS, ALL_AGENT_COLS, OUT, TAU,
    confusion_counts, load_five_agent, metrics_from_counts, predict_at_tau,
    weighted_mean,
)


def aggregate(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return weighted_mean(df, cols)


def metrics_for(df: pd.DataFrame, cols: list[str]) -> dict:
    score = aggregate(df, cols)
    pred = predict_at_tau(score, TAU)
    c = confusion_counts(df["y_true"].values, pred)
    m = metrics_from_counts(c)
    return {**m, **c}


def main() -> None:
    df = load_five_agent()

    # Row 1 — baseline
    base = metrics_for(df, ALL_AGENT_COLS)
    baseline_miscount = base["miscount"]
    baseline_acc = base["accuracy"]

    # Readable labels matching the paper
    removal_label = {
        "XceptionNet": "Visual (XceptionNet)",
        "FreqNet": "Audio (FreqNet)",
        "ECAPA": "Audio Forensics (ECAPA)",
        "CrossModal": "Cross-Modal (Lip-Sync)",
        "Biometric": "Facial Biometric (Quality)",
    }

    rows = []
    rows.append({
        "configuration": "Full 5-agent ensemble (baseline)",
        "n_agents": 5,
        "accuracy": base["accuracy"],
        "miscount": base["miscount"],
        "delta_pp": 0.0,
        "fp": base["fp"], "fn": base["fn"],
        "anomaly": "",
    })

    for agent_key, col in AGENT_COLS.items():
        remaining = [c for c in ALL_AGENT_COLS if c != col]
        m = metrics_for(df, remaining)
        delta_pp = (m["accuracy"] - baseline_acc) * 100
        anomaly = ""
        if m["miscount"] < baseline_miscount:
            anomaly = (
                f"miscount {m['miscount']} < baseline {baseline_miscount}: "
                "removed agent was net-harmful at this threshold (inspect)"
            )
        rows.append({
            "configuration": f"Remove {removal_label[agent_key]}",
            "n_agents": 4,
            "accuracy": m["accuracy"],
            "miscount": m["miscount"],
            "delta_pp": delta_pp,
            "fp": m["fp"], "fn": m["fn"],
            "anomaly": anomaly,
        })

    # Row 7 — top 3 (Biometric + ECAPA + Cross-Modal)
    top3 = [AGENT_COLS["Biometric"], AGENT_COLS["ECAPA"], AGENT_COLS["CrossModal"]]
    m = metrics_for(df, top3)
    rows.append({
        "configuration": "Top-3 (Biometric + ECAPA + Cross-Modal)",
        "n_agents": 3,
        "accuracy": m["accuracy"],
        "miscount": m["miscount"],
        "delta_pp": (m["accuracy"] - baseline_acc) * 100,
        "fp": m["fp"], "fn": m["fn"],
        "anomaly": "",
    })

    # Row 8 — audio only
    audio = [AGENT_COLS["FreqNet"], AGENT_COLS["ECAPA"]]
    m = metrics_for(df, audio)
    rows.append({
        "configuration": "Audio only (FreqNet + ECAPA)",
        "n_agents": 2,
        "accuracy": m["accuracy"],
        "miscount": m["miscount"],
        "delta_pp": (m["accuracy"] - baseline_acc) * 100,
        "fp": m["fp"], "fn": m["fn"],
        "anomaly": "",
    })

    # Row 9 — visual only
    visual = [AGENT_COLS["XceptionNet"], AGENT_COLS["Biometric"]]
    m = metrics_for(df, visual)
    rows.append({
        "configuration": "Visual only (XceptionNet + Biometric)",
        "n_agents": 2,
        "accuracy": m["accuracy"],
        "miscount": m["miscount"],
        "delta_pp": (m["accuracy"] - baseline_acc) * 100,
        "fp": m["fp"], "fn": m["fn"],
        "anomaly": "",
    })

    # Row 10 — single best agent (Cross-Modal)
    single = [AGENT_COLS["CrossModal"]]
    m = metrics_for(df, single)
    rows.append({
        "configuration": "Single best agent (Cross-Modal)",
        "n_agents": 1,
        "accuracy": m["accuracy"],
        "miscount": m["miscount"],
        "delta_pp": (m["accuracy"] - baseline_acc) * 100,
        "fp": m["fp"], "fn": m["fn"],
        "anomaly": "",
    })

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "ablation_table.csv", index=False, float_format="%.6f")

    # LaTeX
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Configuration & \# agents & Accuracy (\%) & Miscount & $\Delta$ (pp) \\",
        r"\midrule",
    ]
    # baseline separated
    b = rows[0]
    lines.append(
        f"{b['configuration']} & {b['n_agents']} & "
        f"{b['accuracy'] * 100:.3f} & {b['miscount']} & --- \\\\"
    )
    lines.append(r"\midrule")
    for r in rows[1:]:
        delta = r["delta_pp"]
        sign = "+" if delta > 0 else ("$-$" if delta < 0 else "")
        delta_str = f"{sign}{abs(delta):.3f}"
        mark = r" \dag" if r["anomaly"] else ""
        lines.append(
            f"{r['configuration']}{mark} & {r['n_agents']} & "
            f"{r['accuracy'] * 100:.3f} & {r['miscount']} & {delta_str} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "ablation_table.tex").write_text("\n".join(lines) + "\n")

    anomalies = [r for r in rows if r["anomaly"]]
    print(f"[task05] baseline miscount={baseline_miscount}, acc={baseline_acc * 100:.3f}% "
          f"| rows={len(rows)} | anomalies={len(anomalies)}")
    for a in anomalies:
        print(f"[task05] ANOMALY: {a['configuration']}: {a['anomaly']}")


if __name__ == "__main__":
    main()
