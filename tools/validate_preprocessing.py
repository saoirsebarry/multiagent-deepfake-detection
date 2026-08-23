"""Integrity check for a preprocessing run. Safe to run while preprocessing is still going.

Catches the failure modes that would otherwise only surface hours later, in training:
  - truncated or corrupt .npz left behind by a crash mid-write (these would be treated as
    "already done" by the resume logic, so they must be found and deleted)
  - wrong dtype or shape, which trains silently and produces a wrong model
  - clip counts drifting from the manuscript's Table 3 split sizes
  - a split whose contents disagree with the published test-set filenames

Exit code is 0 only if every check passes, so it can gate the training step.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import zipfile
from pathlib import Path

import numpy as np

# Table 3 of the manuscript
EXPECTED = {"train": 1052, "val": 236, "test": 2162}
EXPECTED_KEYS = {"faces", "waveform", "label"}


def check_one(path: Path, deep: bool) -> tuple[bool, str]:
    """Return (ok, reason). A zero-length or non-zip file is a mid-write crash artefact."""
    try:
        if path.stat().st_size == 0:
            return False, "zero bytes"
        if not zipfile.is_zipfile(path):
            return False, "not a valid npz container (truncated write?)"
        with np.load(path, allow_pickle=True) as z:
            keys = set(z.files)
            if not EXPECTED_KEYS <= keys:
                return False, f"missing keys {sorted(EXPECTED_KEYS - keys)}"
            if not deep:
                return True, ""
            faces, wave, label = z["faces"], z["waveform"], z["label"]
            if faces.dtype != np.uint8:
                return False, f"faces dtype {faces.dtype}, expected uint8"
            if faces.ndim != 4 or faces.shape[-1] != 3:
                return False, f"faces shape {faces.shape}, expected (n,H,W,3)"
            # every predicate below is wrapped in bool(): numpy comparisons return
            # np.bool_, which is truthy but fails an `is True` identity test - a trap that
            # produced false FAILs when this ran on Colab.
            if not bool(1 <= faces.shape[0] <= 20):
                return False, f"{faces.shape[0]} faces, expected 1..20"
            if faces.shape[1] != faces.shape[2]:
                return False, f"non-square faces {faces.shape[1:3]}"
            if wave.ndim != 1:
                return False, f"waveform ndim {wave.ndim}"
            if str(label[0]) not in ("real", "fake"):
                return False, f"label {label[0]!r}"
            if str(label[0]) != ("fake" if "_label_fake" in path.name else "real"):
                return False, f"label {label[0]!r} contradicts filename"
        return True, ""
    except Exception as e:  # a corrupt zip raises here rather than returning
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True, help="directory holding train/ val/ test/")
    ap.add_argument("--sample", type=int, default=60,
                    help="deep-validate this many random clips per split (0 = all)")
    ap.add_argument("--published-csv", default="paper_artifacts/source_csvs/"
                                               "analysis_results_with_5_agents.csv",
                    help="used to cross-check the test split when it has been written")
    ap.add_argument("--delete-corrupt", action="store_true",
                    help="remove files that fail the container check so a resume rewrites them")
    ap.add_argument("--expect-splits", default="train,val",
                    help="splits this run is meant to produce")
    args = ap.parse_args()

    root = Path(args.processed)
    wanted = [s.strip() for s in args.expect_splits.split(",") if s.strip()]
    report: dict = {"processed": str(root), "splits": {}}
    problems: list[str] = []
    rng = random.Random(42)

    for split in ("train", "val", "test"):
        d = root / split
        if not d.is_dir():
            if split in wanted:
                problems.append(f"{split}/ missing entirely")
            continue
        files = sorted(d.glob("*.npz"))
        n = len(files)
        target = EXPECTED[split]

        # every file gets the cheap container check; a sample gets the deep check
        deep_set = set(files if args.sample == 0 else rng.sample(files, min(args.sample, n)))
        corrupt: list[tuple[Path, str]] = []
        for f in files:
            ok, why = check_one(f, deep=f in deep_set)
            if not ok:
                corrupt.append((f, why))

        total_bytes = sum(f.stat().st_size for f in files)
        per = total_bytes / n if n else 0
        report["splits"][split] = {
            "clips": n, "target": target, "pct": round(100 * n / target, 1),
            "gib": round(total_bytes / 2**30, 2),
            "mib_per_clip": round(per / 2**20, 2),
            "projected_gib_at_target": round(per * target / 2**30, 2),
            "deep_checked": len(deep_set), "corrupt": len(corrupt),
        }
        print(f"{split:5s} {n:5d}/{target:<5d} ({100*n/target:5.1f}%)  "
              f"{total_bytes/2**30:5.2f} GiB  {per/2**20:4.2f} MiB/clip  "
              f"deep-checked {len(deep_set)}  corrupt {len(corrupt)}")
        for f, why in corrupt[:10]:
            print(f"        CORRUPT {f.name}: {why}")
            if args.delete_corrupt:
                f.unlink(missing_ok=True)
                print("           deleted; a resume will rewrite it")
        if corrupt and not args.delete_corrupt:
            problems.append(f"{split}: {len(corrupt)} corrupt files "
                            f"(re-run with --delete-corrupt, then resume preprocessing)")
        if n > target:
            problems.append(f"{split}: {n} clips exceeds the Table 3 target of {target}")

    # the test split, if present, must match the published filenames exactly
    csv = Path(args.published_csv)
    test_dir = root / "test"
    if test_dir.is_dir() and csv.exists():
        import csv as _csv
        with open(csv) as fh:
            published = {r["filepath"] for r in _csv.DictReader(fh)}
        got = {p.name for p in test_dir.glob("*.npz")}
        missing, extra = published - got, got - published
        report["test_split_vs_published"] = {
            "published": len(published), "present": len(got),
            "missing": len(missing), "extra": len(extra)}
        print(f"\ntest split vs published: {len(got)}/{len(published)} present, "
              f"{len(missing)} missing, {len(extra)} extra")
        if extra:
            problems.append(f"test split has {len(extra)} clips not in the published set - "
                            f"the fold differs from the paper's")

    report["problems"] = problems
    out = root / "_validation.json"
    try:
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}")
    except Exception as e:
        print(f"\n(could not write report: {e})")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
