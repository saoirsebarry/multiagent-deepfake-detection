"""Score one partition with the five agents from an arbitrary checkpoint layout.

    python tools/source_disjoint/score_split.py --data_dir <root> --split test \
        --ckpt_dir <dir> --out results/test.csv [--tau 0.35]

`--ckpt_dir` follows the released `checkpoints/` layout (xception/, freqnet/, cross_modal/,
biometric/, ecapa_forensic_head/); file names inside each agent directory may differ, so the
first *.pth in each directory is taken and the ECAPA statistics from the first *.npz. The
released inference recipes in src/orchestrator.py are reused unchanged; only the CONFIG
paths, the equal weights and the threshold are set here.
"""
import argparse
import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
os.chdir(REPO)


def first(pattern):
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(f"no file matches {pattern}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    import orchestrator as orc

    c = a.ckpt_dir
    orc.CONFIG["data_dir"] = a.data_dir
    orc.CONFIG["dataloader_num_workers"] = a.workers
    orc.CONFIG["model_files"] = {
        "spatial": first(os.path.join(c, "xception", "*.pth")),
        "audio": first(os.path.join(c, "freqnet", "*.pth")),
        "audio_forensics": first(os.path.join(c, "ecapa_forensic_head", "*.pth")),
        "cross_modal": first(os.path.join(c, "cross_modal", "*.pth")),
        "face_quality": first(os.path.join(c, "biometric", "*.pth")),
    }
    orc.CONFIG["audio_forensics_stats_path"] = first(os.path.join(c, "ecapa_forensic_head", "*.npz"))
    orc.CONFIG["decision_engine"] = {"weights": {k: 0.2 for k in orc.CONFIG["decision_engine"]["weights"]}, "threshold": a.tau}
    print("checkpoints:", orc.CONFIG["model_files"], flush=True)
    sys.argv = [sys.argv[0], "--split", a.split, "--output_file", os.path.abspath(a.out)]
    orc.main()


if __name__ == "__main__":
    main()
