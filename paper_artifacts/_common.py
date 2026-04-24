"""Shared helpers for paper_artifacts scripts.

All numeric work assumes:
  - ground_truth column encodes strings 'Fake' / 'Real' (the CSVs use title case)
  - 1 = fake, 0 = real
  - final_score and per-agent score_* columns are P(fake)
  - decision threshold tau is fixed at 0.5 unless a task is explicitly a sweep
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
# In the publication release the source CSVs live under paper_artifacts/source_csvs/;
# in the development repo they live under multiagent_results_csv_files/.
# Pick whichever exists.
CSV_DIR = Path(__file__).resolve().parent / "source_csvs"
if not CSV_DIR.exists():
    CSV_DIR = REPO / "multiagent_results_csv_files"
OUT = Path(__file__).resolve().parent

TAU = 0.5
RANDOM_SEED = 42

AGENT_COLS = {
    "XceptionNet": "score_Visual (Spatial)",
    "FreqNet": "score_Audio (Mel+CNN)",
    "ECAPA": "score_Audio Forensics (ECAPA)",
    "CrossModal": "score_Cross-Modal (Lip-Sync)",
    "Biometric": "score_Facial Biometric (Quality)",
}
ALL_AGENT_COLS = list(AGENT_COLS.values())

# Weights used to generate the stored `final_score` column in
# analysis_results_with_5_agents.csv, discovered by reproducing the column
# from the per-agent score columns. Source:
#   multiagent_langchain_additional_agents.py  (CONFIG.decision_engine.weights,
#   line 82). This is the orchestrator that actually wrote the CSV.
# These weights sum to 1.0; keys match AGENT_COLS labels.
AGENT_WEIGHTS = {
    "score_Visual (Spatial)":          0.20,
    "score_Audio (Mel+CNN)":           0.15,
    "score_Audio Forensics (ECAPA)":   0.20,
    "score_Cross-Modal (Lip-Sync)":    0.25,
    "score_Facial Biometric (Quality)":0.20,
}


def weighted_mean(df, cols):
    """Weighted mean of `cols`, using AGENT_WEIGHTS, re-normalised over the
    subset passed in. Used by the ablation task so that "Remove X" rows
    reaggregate the remaining four agents the same way the full 5-agent
    baseline was aggregated (just without the removed column).
    """
    import numpy as np
    w = np.array([AGENT_WEIGHTS[c] for c in cols], dtype=float)
    w = w / w.sum()
    return (df[cols].values * w).sum(axis=1)

# Okabe-Ito colour-blind-safe palette
OKABE_ITO = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermil":  "#D55E00",
    "purple":  "#CC79A7",
}

AGENT_COLOUR = {
    "XceptionNet": OKABE_ITO["orange"],
    "FreqNet":     OKABE_ITO["skyblue"],
    "ECAPA":       OKABE_ITO["green"],
    "CrossModal":  OKABE_ITO["blue"],
    "Biometric":   OKABE_ITO["purple"],
    "System":      OKABE_ITO["black"],
}


def set_plot_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def load_five_agent() -> pd.DataFrame:
    path = CSV_DIR / "analysis_results_with_5_agents.csv"
    df = pd.read_csv(path)
    if len(df) != 2162:
        raise SystemExit(f"STOP: expected 2162 rows in {path.name}, got {len(df)}")
    gt = df["ground_truth"].value_counts().to_dict()
    if gt.get("Real") != 118 or gt.get("Fake") != 2044:
        raise SystemExit(
            f"STOP: expected 118 Real + 2044 Fake in {path.name}, got {gt}"
        )
    df["y_true"] = (df["ground_truth"] == "Fake").astype(int)
    return df


def load_three_agent() -> pd.DataFrame:
    path = CSV_DIR / "analysis_results_with_3_agents.csv"
    df = pd.read_csv(path)
    df["y_true"] = (df["ground_truth"] == "Fake").astype(int)
    return df


def load_youtube() -> pd.DataFrame:
    path = CSV_DIR / "analysis_results_with_5_agents_orchestration.csv"
    df = pd.read_csv(path)
    return df


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return dict with tp, tn, fp, fn counts. Positive class = 1 = fake."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_from_counts(c: dict) -> dict:
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "miscount": fp + fn}


def predict_at_tau(score: np.ndarray, tau: float = TAU) -> np.ndarray:
    return (np.asarray(score) >= tau).astype(int)


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def save_both(fig, stem: Path) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(stem) + ".pdf")
    fig.savefig(str(stem) + ".png")
    plt.close(fig)


def fmt_pct(x: float, nd: int = 2) -> str:
    return f"{x * 100:.{nd}f}%"
