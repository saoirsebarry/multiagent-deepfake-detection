"""Build a source-disjoint train/val/test partition of the preprocessed PolyGlotFake clips.

PolyGlotFake derives every manipulated clip from one authentic source video, and the clip
name carries that source: `en_37_label_real.npz` is the source and
`en_37_to_ru_Xtts_label_fake.npz` one of its forgeries. The released clip-level split
scatters a source's clips across train, validation and test, so the test set measures
within-source generalisation. This script groups clips by source, assigns whole sources
to one partition (70/15/15 within each language, seeded), balances train and validation
by downsampling the fake class as the released split does, and leaves test unbalanced.

    python tools/source_disjoint/make_split.py --processed <root with train/ val/ test/> \
        --out_dir <new root> --seed 42 [--audit]

The new root holds symlinks, so no clip is copied. `--audit` reports the released
partition's source overlap before building anything.
"""
import argparse
import hashlib
import json
import os
import random
import re
from collections import defaultdict

PAT = re.compile(r"^([a-z]{2})_(\d+)(?:_to_([a-z]{2})_([A-Za-z0-9]+))?_label_(real|fake)\.npz$")
SPLITS = ("train", "val", "test")


def parse(name):
    m = PAT.match(name)
    if not m:
        raise ValueError(f"unrecognised clip name: {name}")
    lang, sid, tgt, method, label = m.groups()
    return {"source": f"{lang}_{sid}", "lang": lang, "target_lang": tgt, "method": method, "label": label}


def collect(processed):
    clips = {}
    for split in SPLITS:
        d = os.path.join(processed, split)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".npz"):
                if f in clips:
                    raise SystemExit(f"{f} appears in two released splits")
                clips[f] = {"released_split": split, "path": os.path.join(d, f), **parse(f)}
    if not clips:
        raise SystemExit(f"no .npz clips under {processed}")
    return clips


def audit_released(clips):
    by_split = defaultdict(set)
    real_split = {}
    for f, c in clips.items():
        by_split[c["released_split"]].add(c["source"])
        if c["label"] == "real":
            real_split[c["source"]] = c["released_split"]
    test_fake_sources = {c["source"] for c in clips.values() if c["released_split"] == "test" and c["label"] == "fake"}
    where_real = defaultdict(int)
    for s in test_fake_sources:
        where_real[real_split.get(s, "absent")] += 1
    out = {
        "sources_per_split": {s: len(by_split[s]) for s in SPLITS},
        "test_sources_also_in_train": len(by_split["test"] & by_split["train"]),
        "test_sources_also_in_val": len(by_split["test"] & by_split["val"]),
        "test_fake_sources": len(test_fake_sources),
        "test_fake_sources_by_location_of_their_real_clip": dict(where_real),
        "test_sources_disjoint_from_train_and_val": len(by_split["test"] - by_split["train"] - by_split["val"]),
    }
    return out


def assign(clips, seed, groups=None):
    """Split units are identity groups when given (a group spans every source showing one person),
    otherwise single sources; units are shuffled within the language of their first source."""
    rng = random.Random(seed)
    unit_of = {c["source"]: (groups or {}).get(c["source"], c["source"]) for c in clips.values()}
    per_lang, unit_lang = defaultdict(set), {}
    for c in clips.values():
        u = unit_of[c["source"]]
        unit_lang.setdefault(u, c["lang"])
        per_lang[unit_lang[u]].add(u)
    unit_split = {}
    for lang in sorted(per_lang):
        units = sorted(per_lang[lang])
        rng.shuffle(units)
        n = len(units)
        n_test = round(0.15 * n)
        n_val = round(0.15 * n)
        for i, u in enumerate(units):
            unit_split[u] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    source_split = {s: unit_split[u] for s, u in unit_of.items()}
    members = {s: [] for s in SPLITS}
    for f in sorted(clips):
        members[source_split[clips[f]["source"]]].append(f)
    # balance train and val by downsampling fakes to the real count, as the released split does
    for split in ("train", "val"):
        reals = [f for f in members[split] if clips[f]["label"] == "real"]
        fakes = [f for f in members[split] if clips[f]["label"] == "fake"]
        rng.shuffle(fakes)
        members[split] = sorted(reals + fakes[: len(reals)])
    return source_split, members


def build_tree(clips, members, out_dir):
    for split in SPLITS:
        d = os.path.join(out_dir, split)
        os.makedirs(d, exist_ok=True)
        for f in members[split]:
            dst = os.path.join(d, f)
            if os.path.lexists(dst):
                os.unlink(dst)
            os.symlink(os.path.abspath(clips[f]["path"]), dst)


def summarise(clips, members):
    out = {}
    for split in SPLITS:
        fs = members[split]
        out[split] = {
            "clips": len(fs),
            "real": sum(clips[f]["label"] == "real" for f in fs),
            "fake": sum(clips[f]["label"] == "fake" for f in fs),
            "sources": len({clips[f]["source"] for f in fs}),
            "methods": sorted({clips[f]["method"] for f in fs if clips[f]["method"]}),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True, help="root holding the released train/ val/ test/ npz dirs")
    ap.add_argument("--out_dir", help="root to build the source-disjoint symlink tree under")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest", help="JSON manifest path (default <out_dir>/split_manifest.json)")
    ap.add_argument("--audit", action="store_true", help="report the released partition's source overlap")
    ap.add_argument("--groups", help="identity_groups.json from cluster_identities.py; keeps each identity group in one partition")
    a = ap.parse_args()
    clips = collect(a.processed)
    report = {"n_clips": len(clips), "n_sources": len({c["source"] for c in clips.values()})}
    if a.audit:
        report["released_partition_audit"] = audit_released(clips)
        print(json.dumps(report, indent=1))
    if not a.out_dir:
        return
    groups = json.load(open(a.groups))["source_to_group"] if a.groups else None
    source_split, members = assign(clips, a.seed, groups)
    for s1 in SPLITS:
        for s2 in SPLITS:
            if s1 < s2:
                shared = {clips[f]["source"] for f in members[s1]} & {clips[f]["source"] for f in members[s2]}
                assert not shared, f"sources shared between {s1} and {s2}: {sorted(shared)[:5]}"
    build_tree(clips, members, a.out_dir)
    listing = "\n".join(f"{s}\t{f}" for s in SPLITS for f in members[s])
    manifest = {
        "protocol": "source-disjoint: every clip of a source video sits in one partition (and every source of an identity group, when --groups is given); 70/15/15 by unit within language; train and val fake class downsampled to the real count",
        "seed": a.seed,
        "identity_groups": a.groups,
        "sha256_over_split_and_clip_names": hashlib.sha256(listing.encode()).hexdigest(),
        "summary": summarise(clips, members),
        "released_partition_audit": audit_released(clips),
        "sources": {s: sorted(k for k, v in source_split.items() if v == s) for s in SPLITS},
        "clips": {s: members[s] for s in SPLITS},
    }
    path = a.manifest or os.path.join(a.out_dir, "split_manifest.json")
    json.dump(manifest, open(path, "w"), indent=1)
    print(json.dumps({k: manifest[k] for k in ("seed", "sha256_over_split_and_clip_names", "summary", "released_partition_audit")}, indent=1))
    print("manifest ->", path)


if __name__ == "__main__":
    main()
