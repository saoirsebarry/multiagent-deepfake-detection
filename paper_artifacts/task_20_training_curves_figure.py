"""Render the per-epoch train/validation loss curves for all five detection agents.

Reviewer comment 4 asked for convergence evidence covering every agent, which the
manuscript previously carried only as tabulated endpoints. Sources differ by agent and
are labelled on each panel, because provenance is the part a reader needs to judge.
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [
    ("XceptionNet", "Visual (XceptionNet)"),
    ("FreqNet", "Audio (FreqNet)"),
    ("ECAPA-TDNN", "Audio Forensics (ECAPA-TDNN)"),
    ("Cross-Modal", "Cross-Modal (Lip-Sync)"),
    ("Biometric-Quality", "Biometric-Quality"),
]


def _series(curve):
    """Pull (epochs, train_loss, val_loss) out of a curve dict, tolerating key aliases."""
    def pick(*names):
        for n in names:
            if curve.get(n):
                return list(curve[n])
        return []

    train = pick("train_loss", "train_losses", "loss")
    val = pick("val_loss", "val_losses", "validation_loss")
    span = max(len(train), len(val))
    raw = curve.get("epochs") or curve.get("epoch")
    # recovered_curves.json stores a count; the biometric history stores the axis itself.
    if isinstance(raw, int):
        epochs = list(range(1, raw + 1))
    elif raw:
        epochs = list(raw)
    else:
        epochs = list(range(1, span + 1))
    return epochs, train, val


def load_ecapa_csv(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return {
        "epochs": [int(float(r["epoch"])) for r in rows],
        "train_loss": [float(r["train_loss"]) for r in rows],
        "val_loss": [float(r["val_loss"]) for r in rows],
        "val_auc": [float(r["val_auc"]) for r in rows],
        "val_acc": [float(r["val_acc"]) for r in rows],
        "source": "retrained on the regenerated split",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recovered", required=True, help="recovered_curves.json")
    ap.add_argument("--biometric", required=True, help="biometric_training_history.json")
    ap.add_argument("--ecapa_csv", required=True, help="ECAPA training_log.csv")
    ap.add_argument("--out", default="paper_artifacts/training_curves_all_agents")
    args = ap.parse_args()

    curves = {}
    with open(args.recovered) as fh:
        blob = json.load(fh)
    payload = blob.get("agents", blob)
    for key, curve in payload.items():
        if isinstance(curve, dict):
            curves[key] = curve

    with open(args.biometric) as fh:
        curves["Biometric-Quality"] = json.load(fh)
    curves["ECAPA-TDNN"] = load_ecapa_csv(args.ecapa_csv)

    def match(name):
        for k in curves:
            if k.lower().replace("_", "-").startswith(name.lower().split("-")[0][:5]):
                return curves[k]
        return None

    fig, grid = plt.subplots(2, 3, figsize=(11.0, 6.0))
    axes = grid.flatten()
    grid[1][2].axis("off")
    missing = []
    for ax, (key, title) in zip(axes, PANELS):
        curve = curves.get(key) or match(key)
        if not curve:
            missing.append(key)
            ax.text(0.5, 0.5, "no per-epoch\nhistory", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=9)
            continue
        epochs, train, val = _series(curve)
        if train:
            ax.plot(epochs[: len(train)], train, color="#1f77b4", lw=1.6, label="Train")
        if val:
            ax.plot(epochs[: len(val)], val, color="#d62728", lw=1.6, label="Validation")
            best = min(range(len(val)), key=lambda i: val[i])
            ax.axvline(epochs[best], color="#666666", ls=":", lw=1.1)
            ax.plot([epochs[best]], [val[best]], "o", color="#d62728", ms=4.5)
            # Anchor the label inboard of the marker, or a best epoch near the right-hand
            # edge pushes its text outside the panel.
            late = epochs[best] > epochs[0] + 0.6 * (epochs[-1] - epochs[0])
            ax.annotate(
                f"best {val[best]:.3f}\nep. {epochs[best]}",
                xy=(epochs[best], val[best]),
                xytext=(-6, 11) if late else (6, 11),
                textcoords="offset points",
                ha="right" if late else "left",
                fontsize=7,
                color="#333333",
            )
        # Log scale: the agents' losses span two orders of magnitude, and on a linear
        # axis ECAPA's tail - the part the over-fitting claim rests on - is invisible.
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss (log scale)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5, which="both")
        ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = f"{args.out}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print("wrote", path)
    if missing:
        print("MISSING per-epoch history for:", ", ".join(missing))
    else:
        print("all five agents plotted")


if __name__ == "__main__":
    main()
