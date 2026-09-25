"""Audit the source-disjoint partition: clip accounting, composition, and the identity clustering.

    python tools/source_disjoint/audit_partition.py \
        --manifest paper_artifacts/source_disjoint/split_manifest.json \
        --groups paper_artifacts/source_disjoint/identity_groups.json \
        --runs paper_artifacts/source_disjoint/seed42 paper_artifacts/source_disjoint/seed43 paper_artifacts/source_disjoint/seed44 \
        --out paper_artifacts/source_disjoint/partition_audit.json

The identity groups are inferred from face embeddings, so they have no ground truth. PolyGlotFake
numbers the clips cut from one recording consecutively within a language, which gives an
independent check: the clustering should merge adjacent IDs far more often than distant ones,
and a test source whose adjacent-ID neighbour sits in training but in another group is a
candidate identity overlap. The conservative read-out drops those test sources, and any source
whose authentic clip was never embedded, and re-reads every run at the released threshold.
"""
import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

PAT = re.compile(r"^([a-z]{2})_(\d+)(?:_to_([a-z]{2})_([A-Za-z0-9]+))?_label_(real|fake)\.npz$")
LANGS = ("ar", "en", "es", "fr", "ja", "ru", "zh")
METHODS = ("Bark", "MicroTts", "Tacotron", "Vall", "Xtts")
SPLITS = ("train", "val", "test")


def composition(manifest):
    out = {}
    for split in SPLITS:
        parsed = [PAT.match(f).groups() for f in manifest["clips"][split]]
        real = Counter(p[0] for p in parsed if p[4] == "real")
        fake = Counter(p[0] for p in parsed if p[4] == "fake")
        out[split] = {
            "clips": len(parsed),
            "sources_with_clips": len({(p[0], p[1]) for p in parsed}),
            "real_by_source_language": {lang: real[lang] for lang in LANGS},
            "fake_by_source_language": {lang: fake[lang] for lang in LANGS},
            "fake_by_target_language": {lang: Counter(p[2] for p in parsed if p[4] == "fake")[lang] for lang in LANGS},
            "fake_by_method": {m: Counter(p[3] for p in parsed if p[4] == "fake")[m] for m in METHODS},
        }
    return out


def merge_rate_by_gap(source_to_group, gaps=(1, 2, 3, 4, 5, 10, 20)):
    ids = defaultdict(set)
    for s in source_to_group:
        lang, n = s.split("_")
        ids[lang].add(int(n))
    rate = {}
    for gap in gaps:
        pairs = [(f"{lang}_{n}", f"{lang}_{n + gap}") for lang, ns in ids.items() for n in ns if n + gap in ns]
        merged = sum(source_to_group[a] == source_to_group[b] for a, b in pairs)
        rate[str(gap)] = {"pairs": len(pairs), "merged": merged}
    far = [(f"{lang}_{a}", f"{lang}_{b}") for lang, ns in ids.items() for a in ns for b in ns if b - a >= 20]
    rate[">=20"] = {"pairs": len(far), "merged": sum(source_to_group[a] == source_to_group[b] for a, b in far)}
    return rate


def flagged_test_sources(manifest, source_to_group):
    split_of = {s: split for split, sources in manifest["sources"].items() for s in sources}
    embedded = set(source_to_group)
    flagged = {s for s in manifest["sources"]["test"] if s not in embedded}
    for s in manifest["sources"]["test"]:
        if s not in embedded:
            continue
        lang, n = s.split("_")
        for m in (int(n) - 1, int(n) + 1):
            other = f"{lang}_{m}"
            if split_of.get(other) in ("train", "val") and source_to_group.get(other) != source_to_group[s]:
                flagged.add(s)
    return sorted(flagged)


def readout(run, keep, tau):
    rows = list(csv.DictReader(open(os.path.join(run, "scores_test.csv"))))
    cols = [c for c in rows[0] if c.startswith("score_")]
    y, s = [], []
    for r in rows:
        name = os.path.basename(r["filepath"])
        source = "_".join(name.split("_")[:2])
        if keep(source):
            y.append(int("_label_fake" in name))
            s.append(float(np.mean([float(r[c]) for c in cols])))
    y, s = np.array(y), np.array(s)
    pred = (s >= tau).astype(int)
    return {"clips": int(len(y)), "real": int((y == 0).sum()), "fake": int((y == 1).sum()),
            "false_positives": int(((pred == 1) & (y == 0)).sum()), "false_negatives": int(((pred == 0) & (y == 1)).sum()),
            "accuracy": float((pred == y).mean()), "auc_roc": float(roc_auc_score(y, s))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    manifest, groups = json.load(open(a.manifest)), json.load(open(a.groups))
    s2g = groups["source_to_group"]
    flagged = flagged_test_sources(manifest, s2g)
    subsets = {"all": lambda s: True, "without_flagged_sources": lambda s: s not in flagged, "flagged_sources_only": lambda s: s in flagged}
    reads = {name: [readout(r, keep, a.tau) for r in a.runs] for name, keep in subsets.items()}
    for name, rs in reads.items():
        acc = np.array([r["accuracy"] for r in rs])
        reads[name] = {"runs": rs, "accuracy_mean": float(acc.mean()), "accuracy_sd": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0}
    out = {
        "tau": a.tau,
        "composition": composition(manifest),
        "sources_listed": {s: len(v) for s, v in manifest["sources"].items()},
        "sources_without_embedded_authentic_clip": {s: split for split, v in manifest["sources"].items() for s in v if s not in s2g},
        "clustering_threshold_sweep": groups["threshold_sweep"],
        "merge_rate_by_id_gap": merge_rate_by_gap(s2g),
        "test_sources": len(manifest["sources"]["test"]),
        "flagged_test_sources": flagged,
        "readout": reads,
    }
    json.dump(out, open(a.out, "w"), indent=1)
    for name, r in reads.items():
        print(f"{name}: clips {r['runs'][0]['clips']} accuracy {[round(100 * x['accuracy'], 2) for x in r['runs']]} "
              f"mean {100 * r['accuracy_mean']:.2f} sd {100 * r['accuracy_sd']:.2f} FP {[x['false_positives'] for x in r['runs']]} "
              f"FN {[x['false_negatives'] for x in r['runs']]} AUC {[round(x['auc_roc'], 5) for x in r['runs']]}")
    print("merge rate by id gap:", {k: f"{v['merged']}/{v['pairs']}" for k, v in out["merge_rate_by_id_gap"].items()})
    print("flagged test sources:", len(flagged), "of", out["test_sources"])


if __name__ == "__main__":
    main()
