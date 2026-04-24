"""Task 11: McNemar's test against transformer baselines.

Requires per-sample predictions from each baseline on the exact same
2,162-sample PolyGlotFake test set. We search the repo for candidate
prediction CSVs aligned to the test set identifiers used in
analysis_results_with_5_agents.csv (column 'filepath').

If no aligned predictions exist we do NOT blindly rerun inference —
we report 'predictions not saved; rerun needed' per the task spec.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

from _common import CSV_DIR, OUT, REPO, TAU, load_five_agent, save_json

BASELINES = [
    {"key": "GenConViT_AE",
     "search_patterns": ["*genconvit*ae*", "*genconvit_ae*"],
     "scan_dirs": [REPO / "transformer_comparison", REPO / "proposed transformer models"]},
    {"key": "GenConViT_VAE",
     "search_patterns": ["*genconvit*vae*", "*genconvit_vae*"],
     "scan_dirs": [REPO / "transformer_comparison", REPO / "proposed transformer models"]},
    {"key": "LIPINC_V2",
     "search_patterns": ["*lipinc*"],
     "scan_dirs": [REPO / "transformer_comparison"]},
    {"key": "Custom_ViT_LateFusion",
     "search_patterns": ["*vit*predict*", "*vit_polyglot*predict*"],
     "scan_dirs": [REPO / "proposed transformer models"]},
    {"key": "Hybrid_CNN_Transformer",
     "search_patterns": ["*cnn_transformer*predict*", "*hybrid*predict*"],
     "scan_dirs": [REPO / "proposed transformer models"]},
]


def find_predictions(baseline: dict) -> list[Path]:
    hits = []
    for d in baseline["scan_dirs"]:
        if not d.exists():
            continue
        for pat in baseline["search_patterns"]:
            for f in d.rglob(pat):
                if f.suffix.lower() in (".csv", ".pkl", ".json", ".parquet"):
                    hits.append(f)
    return hits


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar via two-sided binomial test on b vs (b+c)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(b, n, p=0.5, alternative="two-sided").pvalue)


def main() -> None:
    df = load_five_agent()
    ours_correct = (
        (df["final_score"] >= TAU).astype(int).values == df["y_true"].values
    )
    n_test = len(df)

    out = {
        "tau": TAU,
        "n_test": int(n_test),
        "baselines": {},
        "general_note": (
            "McNemar contingency is [[a b][c d]] where "
            "a = both correct, b = ours correct & baseline wrong, "
            "c = ours wrong & baseline correct, d = both wrong."
        ),
    }

    for baseline in BASELINES:
        hits = find_predictions(baseline)
        entry: dict = {
            "search_patterns": baseline["search_patterns"],
            "scan_dirs": [str(d) for d in baseline["scan_dirs"]],
            "files_found": [str(h) for h in hits],
        }
        if not hits:
            entry["status"] = "predictions not saved; rerun needed"
            entry["p_value"] = None
            out["baselines"][baseline["key"]] = entry
            continue

        # Try to load the first CSV and see if it has a filepath column that
        # aligns with ours.
        loaded = False
        for h in hits:
            if h.suffix.lower() != ".csv":
                continue
            try:
                bd = pd.read_csv(h)
            except Exception as e:
                entry[f"load_error:{h.name}"] = str(e)
                continue
            # Heuristic: require a filepath or filename column AND a prediction
            # column.
            fp_col = next((c for c in bd.columns if c.lower() in ("filepath", "filename", "file")), None)
            pred_col = next(
                (c for c in bd.columns if c.lower() in ("prediction", "pred", "verdict", "score")),
                None,
            )
            if fp_col is None or pred_col is None:
                entry["reject_reason"] = (
                    f"{h.name}: no (filepath, prediction) columns found "
                    f"(columns: {list(bd.columns)})"
                )
                continue

            # align
            merged = df.merge(bd[[fp_col, pred_col]], left_on="filepath", right_on=fp_col, how="left")
            if merged[pred_col].isna().sum() > 0:
                entry["reject_reason"] = (
                    f"{h.name}: could not align {merged[pred_col].isna().sum()}"
                    f" samples to our test set"
                )
                continue

            # convert prediction to binary
            vals = merged[pred_col]
            if vals.dtype.kind in "if":
                base_correct = ((vals >= TAU).astype(int).values == df["y_true"].values)
            else:
                base_correct = (
                    vals.astype(str).str.lower().isin(["fake", "deepfake", "1", "true"]).astype(int)
                    == df["y_true"].values
                )
            a = int(((ours_correct) & (base_correct)).sum())
            b = int(((ours_correct) & (~base_correct)).sum())
            c = int(((~ours_correct) & (base_correct)).sum())
            d = int(((~ours_correct) & (~base_correct)).sum())
            p = mcnemar_exact(b, c)
            entry["used_file"] = str(h)
            entry["contingency"] = {"a": a, "b": b, "c": c, "d": d}
            entry["p_value"] = p
            entry["status"] = "ok"
            loaded = True
            break

        if not loaded:
            entry["status"] = "predictions not saved; rerun needed"
            entry["p_value"] = None
        out["baselines"][baseline["key"]] = entry

    save_json(out, OUT / "mcnemar_tests.json")

    n_ok = sum(1 for v in out["baselines"].values() if v.get("status") == "ok")
    n_missing = len(out["baselines"]) - n_ok
    print(f"[task11] baselines with aligned predictions: {n_ok} / {len(out['baselines'])}")
    if n_missing:
        missing = [k for k, v in out["baselines"].items() if v.get("status") != "ok"]
        print(f"[task11] MISSING (rerun needed): {', '.join(missing)}")


if __name__ == "__main__":
    main()
