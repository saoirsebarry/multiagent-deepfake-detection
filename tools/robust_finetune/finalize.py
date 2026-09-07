"""Assemble the candidate system from the adopted agents and select its weights on
validation; read the test split once.
    finalize.py --ft_dir <robust_ft> --out_dir <dir> --stage overrides
        writes overrides.json (adopted checkpoints) for score_corrupted_val.py; no test read
    finalize.py --ft_dir <robust_ft> --out_dir <dir> --stage final [--val_corrupted <csv>]
        clean weight rule, the corrupted-validation rule when the CSV is given (the final
        vector, per PROTOCOL.md), then ONE pass over the test split; writes candidate CSVs in
        the released column layout plus readout.json
"""
import argparse
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE)); SRC = os.path.join(REPO, "paper_artifacts", "source_csvs")
ORDER = ["visual", "freqnet", "ecapa", "crossmodal", "biometric"]
COL = {"visual": "score_Visual (Spatial)", "freqnet": "score_Audio (Mel+CNN)", "ecapa": "score_Audio Forensics (ECAPA)",
       "crossmodal": "score_Cross-Modal (Lip-Sync)", "biometric": "score_Facial Biometric (Quality)"}
DIR = {"visual": "xception_retrain", "freqnet": "freqnet", "ecapa": "ecapa", "crossmodal": "crossmodal", "biometric": "biometric"}
ap = argparse.ArgumentParser()
ap.add_argument("--ft_dir", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--stage", choices=["overrides", "final"], required=True); ap.add_argument("--val_corrupted", default=None)
A = ap.parse_args(); os.makedirs(A.out_dir, exist_ok=True)

decisions, adopted = {}, {}
for a in ORDER:
    p = os.path.join(A.ft_dir, DIR[a], "decision.json")
    if os.path.exists(p):
        d = json.load(open(p)); decisions[a] = d; adopted[a] = bool(d["adopted"])
        print(f"{a:10s} adopted={d['adopted']!s:5} rule={d.get('adopted_rule')} epoch={d.get('adopted_epoch')} | released val logloss {d['released_val']['logloss']:.4f} auc {d['released_val']['auc']:.5f}"
              + (f" | best val logloss {d[d['adopted_rule']]['best']['logloss']:.4f} auc {d[d['adopted_rule']]['best']['auc']:.5f}" if d["adopted"] else ""), flush=True)
    else:
        adopted[a] = False; print(f"{a:10s} no decision.json -> released agent kept", flush=True)


def ckpt_of(a):
    return os.path.join(A.ft_dir, DIR[a], "best_robust.pth" if decisions[a].get("adopted_rule") == "rule2" else "best_model.pth")


ovr = {a: ckpt_of(a) for a in ORDER if adopted[a]}
if adopted.get("ecapa"):
    ovr["ecapa_stats"] = os.path.join(A.ft_dir, DIR["ecapa"], "training_stats.npz")
json.dump(ovr, open(os.path.join(A.out_dir, "overrides.json"), "w"), indent=1); print("overrides", json.dumps(ovr), flush=True)
if A.stage == "overrides":
    raise SystemExit(0)


def load_scores(a, split):
    rows = csv.DictReader(open(os.path.join(A.ft_dir, DIR[a], f"{a}_{split}_scores.csv")))
    return {r["filepath"].split("/")[-1]: float(r["score"]) for r in rows}


def assemble(split, base_csv):
    rows = list(csv.DictReader(open(os.path.join(SRC, base_csv))))
    for a in ORDER:
        if adopted[a]:
            new = load_scores(a, split)
            for r in rows:
                r[COL[a]] = f"{new[r['filepath'].split('/')[-1]]:.6f}"
    return rows


def matrix(rows):
    return np.array([[float(r[COL[a]]) for a in ORDER] for r in rows]), np.array([r["ground_truth"] == "Fake" for r in rows])


def tiered(S, w):
    trio = S[:, [2, 3, 4]]; verd = trio >= 0.5
    esc = (verd.any(axis=1) & ~verd.all(axis=1)) | (trio.std(axis=1) >= 0.3)
    w3 = w[[2, 3, 4]] / w[[2, 3, 4]].sum()
    return np.where(esc, S @ w, trio @ w3), esc


def auc_of(s, y):
    order = np.argsort(s); n = len(s); ranks = np.empty(n)
    _, inv, cnt = np.unique(s[order], return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt); ranks[order] = ((cum - cnt + cum + 1) / 2.0)[inv]
    npos = y.sum(); nneg = n - npos
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


GRID = []
for a in range(21):
    for b in range(21 - a):
        for c in range(21 - a - b):
            for d in range(21 - a - b - c):
                GRID.append((a, b, c, d, 20 - a - b - c - d))
GRID = np.array(GRID) / 20.0; GRID = GRID[(GRID > 0).all(axis=1)]


def select(S_err, y_err, S_clear, y_clear):
    """Released rule: all-active 0.05 simplex, minimise tiered errors at tau 0.5 over
    (S_err, y_err), tie-break on two-sided clearance over the clean rows."""
    best = None
    for w in GRID:
        t, _ = tiered(S_err, w); errs = int(((t >= 0.5) != y_err).sum())
        tc, _ = tiered(S_clear, w); clear = float(min(tc[y_clear].min() - 0.5, 0.5 - tc[~y_clear].max()))
        if best is None or (errs, -clear) < (best[0], -best[1]):
            best = (errs, clear, w)
    return {"weights": dict(zip(ORDER, map(float, best[2]))), "errors": best[0], "clearance": best[1], "n_rows": int(len(y_err))}


val_rows = assemble("val", "analysis_results_VAL.csv"); Sv, yv = matrix(val_rows)
clean = select(Sv, yv, Sv, yv); print("clean-validation rule:", json.dumps(clean), flush=True)
final_rule, final = "clean", clean
if A.val_corrupted:
    corr = [r for r in csv.DictReader(open(A.val_corrupted)) if int(r["k"]) >= 1]
    Sc, yc = matrix(corr)
    robust = select(np.vstack([Sv, Sc]), np.concatenate([yv, yc]), Sv, yv); print("clean+corrupted rule (fix 6):", json.dumps(robust), flush=True)
    final_rule, final = "clean+corrupted", robust
w = np.array([final["weights"][a] for a in ORDER])
tv, _ = tiered(Sv, w)
val_read = {"tiered_errors": int(((tv >= 0.5) != yv).sum()), "auc_full": auc_of(Sv @ w, yv), "band": {"max_real": float((Sv @ w)[~yv].max()), "min_fake": float((Sv @ w)[yv].min())}}

# the single test read
test_rows = assemble("test", "analysis_results_with_5_agents.csv"); St, yt = matrix(test_rows)
tt, esc = tiered(St, w); full = St @ w
fp = int(((tt >= 0.5) & ~yt).sum()); fn = int(((tt < 0.5) & yt).sum())
test_read = {"n": int(len(yt)), "acc": 100 * (1 - (fp + fn) / len(yt)), "fp": fp, "fn": fn, "auc_full": auc_of(full, yt), "auc_tiered": auc_of(tt, yt),
             "esc_pct": 100 * float(esc.mean()), "band_full": {"max_real": float(full[~yt].max()), "min_fake": float(full[yt].min())},
             "per_agent": {a: {"auc": auc_of(St[:, i], yt), "acc": 100 * float(((St[:, i] >= 0.5) == yt).mean()),
                               "fp": int(((St[:, i] >= 0.5) & ~yt).sum()), "fn": int(((St[:, i] < 0.5) & yt).sum())} for i, a in enumerate(ORDER)}}
for rows, S in ((val_rows, Sv), (test_rows, St)):
    fs = S @ w
    for r, v in zip(rows, fs):
        r["final_score"] = f"{v:.6f}"; r["system_verdict"] = "Deepfake" if v >= 0.5 else "Real"
for name, rows in (("analysis_results_VAL_candidate.csv", val_rows), ("analysis_results_with_5_agents_candidate.csv", test_rows)):
    with open(os.path.join(A.out_dir, name), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
readout = {"adopted": adopted, "adopted_rule": {a: decisions[a].get("adopted_rule") for a in decisions}, "overrides": ovr,
           "weights_clean_rule": clean, "weights_final_rule": final_rule, "weights_final": final, "validation": val_read, "test": test_read}
json.dump(readout, open(os.path.join(A.out_dir, "readout.json"), "w"), indent=1)
print("READOUT", json.dumps(readout), flush=True)
