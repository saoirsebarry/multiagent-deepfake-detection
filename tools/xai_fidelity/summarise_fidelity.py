"""Split a fidelity results file by predicted class, and diff two of them.

    python tools/xai_fidelity/summarise_fidelity.py paper_artifacts/xai_fidelity_released.json
    python tools/xai_fidelity/summarise_fidelity.py before.json after.json

Needs only the standard library plus numpy, so it runs on the released artifacts without the
model checkpoints. A pooled deletion AUC near its random control can mean the map is
uninformative, or that it is informative for both classes while the metric tracks one of them;
only the split tells the two apart.
"""
import json
import sys

import numpy as np

COLS = ("n", "deletion_auc", "random_deletion_auc", "deletion_gap",
        "insertion_auc", "random_insertion_auc", "insertion_gap", "deletion_beats_random_pct")


def split(rows):
    if not rows or "deletion_auc" not in rows[0]:
        return {}
    out = {}
    for name, keep in (("all", lambda r: True),
                       ("predicted_fake", lambda r: r["base_score"] >= 0.5),
                       ("predicted_real", lambda r: r["base_score"] < 0.5)):
        sub = [r for r in rows if keep(r)]
        if not sub:
            continue
        de = np.array([r["deletion_auc"] for r in sub])
        rd = np.array([r["random_deletion_auc"] for r in sub])
        ins = np.array([r["insertion_auc"] for r in sub])
        ri = np.array([r["random_insertion_auc"] for r in sub])
        out[name] = {"n": len(sub), "deletion_auc": de.mean(), "random_deletion_auc": rd.mean(),
                     "deletion_gap": (rd - de).mean(), "insertion_auc": ins.mean(),
                     "random_insertion_auc": ri.mean(), "insertion_gap": (ins - ri).mean(),
                     "deletion_beats_random_pct": 100 * (de < rd).mean()}
    return out


def row(label, s, ref=None):
    def cell(k):
        v = s[k]
        if k == "n":
            return f"{int(v):>4}"
        d = "" if ref is None else f" ({v - ref[k]:+.3f})"
        return f"{v:>10.3f}{d}" if not k.endswith("pct") else f"{v:>9.0f}%{'' if ref is None else f' ({v - ref[k]:+.0f})'}"
    return f"  {label:<16}" + "".join(cell(k) for k in COLS)


def report(path, ref_path=None):
    d = json.load(open(path))
    ref = json.load(open(ref_path)) if ref_path else None
    print(f"\n=== {path}" + (f"   vs {ref_path}" if ref_path else ""))
    if d.get("config"):
        print(f"    config: {d['config']}")
    for block in ("visual", "audio"):
        rows = d["per_clip"].get(block, [])
        s = split(rows)
        if not s:
            continue
        r = split(ref["per_clip"].get(block, [])) if ref else None
        print(f"\n  ## {block}")
        print(f"  {'subset':<16}{'n':>4}{'deletion':>10}{'rnd':>10}{'gap':>10}{'insert':>10}{'rnd':>10}{'gap':>10}{'beats':>10}")
        for k in ("all", "predicted_fake", "predicted_real"):
            if k in s:
                print(row(k, s[k], (r or {}).get(k)))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    report(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
