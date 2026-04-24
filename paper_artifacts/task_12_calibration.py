"""Task 12: Reliability diagram and ECE on the 5-agent final_score."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import (
    OKABE_ITO, OUT, load_five_agent, save_both, save_json, set_plot_style,
)

N_BINS = 10


def main() -> None:
    df = load_five_agent()
    y_true = df["y_true"].values.astype(float)
    score = df["final_score"].values.astype(float)

    bin_edges = np.linspace(0.0, 1.0, N_BINS + 1)
    # assign bin index, excluding the last bin's upper edge from overflow
    bin_idx = np.digitize(score, bin_edges[1:-1])  # 0 .. N_BINS-1

    per_bin = []
    total_n = len(score)
    ece = 0.0
    for i in range(N_BINS):
        mask = bin_idx == i
        n_in = int(mask.sum())
        if n_in == 0:
            per_bin.append({
                "bin": i,
                "edge_low": float(bin_edges[i]),
                "edge_high": float(bin_edges[i + 1]),
                "bin_centre": float(0.5 * (bin_edges[i] + bin_edges[i + 1])),
                "count": 0,
                "mean_score": None,
                "empirical_positive_rate": None,
            })
            continue
        mean_score = float(score[mask].mean())
        emp_rate = float(y_true[mask].mean())
        per_bin.append({
            "bin": i,
            "edge_low": float(bin_edges[i]),
            "edge_high": float(bin_edges[i + 1]),
            "bin_centre": float(0.5 * (bin_edges[i] + bin_edges[i + 1])),
            "count": n_in,
            "mean_score": mean_score,
            "empirical_positive_rate": emp_rate,
        })
        ece += (n_in / total_n) * abs(mean_score - emp_rate)

    save_json({"ece_10bins": float(ece), "per_bin": per_bin},
              OUT / "calibration_metrics.json")

    # plot
    set_plot_style()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    xs, ys, ns = [], [], []
    for pb in per_bin:
        if pb["count"] == 0:
            continue
        xs.append(pb["bin_centre"])
        ys.append(pb["empirical_positive_rate"])
        ns.append(pb["count"])
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls="--", label="Perfect calibration")
    ax.scatter(
        xs, ys,
        s=np.clip(np.sqrt(ns) * 3, 10, 250),
        color=OKABE_ITO["blue"],
        edgecolor="none",
        alpha=0.85,
        label=f"Empirical (ECE = {ece:.3f})",
    )
    ax.plot(xs, ys, color=OKABE_ITO["blue"], lw=1.0, alpha=0.6)
    ax.set_xlabel("Predicted probability P(fake)")
    ax.set_ylabel("Empirical positive rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left")
    save_both(fig, OUT / "reliability_diagram")

    print(f"[task12] ECE (10 bins) = {ece:.4f}; "
          f"non-empty bins = {sum(1 for p in per_bin if p['count'])}")


if __name__ == "__main__":
    main()
