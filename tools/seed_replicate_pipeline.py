"""One detached process that produces the seed-replicate test measurement.

Stages, each skipped if its DONE marker exists, so the pipeline resumes after any
interruption: wait for the dataset extraction; preprocess the test split to Drive;
retrain XceptionNet, FreqNet and Cross-Modal with outputs staged to Drive; assemble a
checkpoint set of the four retrained agents plus the released Biometric-Quality model;
score the test split through the released orchestrator; and write the aggregate
metrics at the frozen weights and tau = 0.37 to seed_replicate_test.json.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

DRIVE = "/content/drive/MyDrive/polyglotfake"
REPO = "/content/repo"
WORK = "/content/work"
SEED2 = os.path.join(DRIVE, "seed2")
LOGDIR = os.path.join(DRIVE, "training_logs")
MARKERS = os.path.join(SEED2, "markers")
os.makedirs(MARKERS, exist_ok=True)
os.makedirs(WORK, exist_ok=True)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def done(stage):
    return os.path.exists(os.path.join(MARKERS, stage))


def mark(stage):
    open(os.path.join(MARKERS, stage), "w").write(time.strftime("%F %T"))


def run(cmd, cwd, logname, env_extra=None):
    env = dict(os.environ)
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    if env_extra:
        env.update(env_extra)
    logpath = os.path.join(LOGDIR, logname)
    log(f"run: {' '.join(cmd)} (cwd={cwd}, log={logname})")
    with open(logpath, "ab") as fh:
        rc = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        log(f"FAILED rc={rc}: see {logname}")
        raise SystemExit(rc)


def ensure_link(link, target):
    if os.path.islink(link) or os.path.exists(link):
        if os.path.islink(link) and os.readlink(link) == target:
            return
        if os.path.islink(link):
            os.remove(link)
    os.makedirs(os.path.dirname(link), exist_ok=True)
    os.symlink(target, link)


# ---- stage 0: wait for extraction ----------------------------------------
if not done("0_extracted"):
    log("waiting for unrar to finish")
    while subprocess.run(["pgrep", "-x", "unrar"], capture_output=True).stdout.strip():
        time.sleep(60)
    roots = [d for d in glob.glob("/content/data/*") if os.path.isdir(d)]
    log(f"extraction done; roots: {roots}")
    assert roots, "nothing extracted"
    mark("0_extracted")

# ---- stage 1: preprocess the test split ----------------------------------
if not done("1_test_preprocessed"):
    DATA_ROOT = next(d for d in sorted(glob.glob("/content/data/*")) if os.path.isdir(d))
    run([sys.executable, "-u", "src/data_preprocessing/preprocessed_all_unbalanced.py",
         "--data_dir", DATA_ROOT, "--splits", "test",
         "--output_dir", os.path.join(DRIVE, "processed"),
         "--workers", "6", "--verify_writes"],
        cwd=REPO, logname="seed2_preprocess_test.log")
    n = len([f for f in os.listdir(os.path.join(DRIVE, "processed", "test")) if f.endswith(".npz")])
    log(f"test clips preprocessed: {n}")
    assert n > 2000, n
    mark("1_test_preprocessed")

# ---- stage 1.5: local copy of the processed data --------------------------
# Training reads thousands of small files per epoch; Drive FUSE both throttles that
# and goes stale under load - the first overnight attempt lost a full XceptionNet run
# to silent per-file ENOENTs. Train and score from a VM-local copy instead.
LOCAL = "/content/processed_local"
if not done("1b_local_copy"):
    for split in ("train", "val", "test"):
        src, dst = os.path.join(DRIVE, "processed", split), os.path.join(LOCAL, split)
        os.makedirs(dst, exist_ok=True)
        run(["rsync", "-a", "--info=stats1", src + "/", dst + "/"],
            cwd="/content", logname="seed2_localcopy.log")
        n_src = len(os.listdir(src)); n_dst = len(os.listdir(dst))
        log(f"local copy {split}: {n_dst}/{n_src}")
        assert n_dst == n_src, (split, n_dst, n_src)
    mark("1b_local_copy")

# ---- stages 2-4: retrain the three lost agents ---------------------------
ensure_link(os.path.join(WORK, "polyglot_processed_all_unbalanced"), LOCAL)
for helper in ("src", "checkpoints", "pretrained_models"):
    src = os.path.join(REPO, helper)
    if os.path.exists(src):
        ensure_link(os.path.join(WORK, helper), src)

def assert_readable(split, n_min):
    d = os.path.join(LOCAL, split)
    names = [f for f in os.listdir(d) if f.endswith(".npz")]
    assert len(names) >= n_min, (split, len(names))
    import numpy as np
    np.load(os.path.join(d, names[0]), allow_pickle=True)["label"]
    log(f"data check {split}: {len(names)} clips, first readable")


if not done("2_xception"):
    assert_readable("train", 1000)
    assert_readable("val", 200)
    xdir = os.path.join(SEED2, "xception")
    os.makedirs(xdir, exist_ok=True)
    ensure_link(os.path.join(WORK, "tuning_and_model_output_unbal_all_face_cutout"), xdir)
    run([sys.executable, "-u", os.path.join(REPO, "src/agents/visual_xception.py")],
        cwd=WORK, logname="seed2_xception.log")
    assert os.path.exists(os.path.join(xdir, "polyglotfake_xception_best_pytorch_unbal_all.pth"))
    mark("2_xception")

if not done("3_freqnet"):
    fdir = os.path.join(SEED2, "freqnet")
    os.makedirs(fdir, exist_ok=True)
    run([sys.executable, "-u", os.path.join(REPO, "src/agents/audio_freqnet.py"),
         "--dataroot", LOCAL,
         "--output_model_path", os.path.join(fdir, "freqnet_model_all_unbalanced_improved.pth"),
         "--plot_path", os.path.join(fdir, "training_curves_freqnet_improved.png")],
        cwd=WORK, logname="seed2_freqnet.log")
    assert os.path.exists(os.path.join(fdir, "freqnet_model_all_unbalanced_improved.pth"))
    mark("3_freqnet")

if not done("4_crossmodal"):
    cdir = os.path.join(SEED2, "crossmodal")
    os.makedirs(cdir, exist_ok=True)
    run([sys.executable, "-u", os.path.join(REPO, "src/agents/cross_modal_lipsync.py")],
        cwd=WORK, logname="seed2_crossmodal.log")
    shutil.copy(os.path.join(WORK, "lip_sync_model_crossattention_all_unbalanced.pth"),
                os.path.join(cdir, "lip_sync_model_crossattention_all_unbalanced.pth"))
    mark("4_crossmodal")

# ---- stage 5: assemble checkpoints and score test ------------------------
if not done("5_scored"):
    ck = os.path.join(WORK, "repo_seed2")
    if not os.path.exists(ck):
        subprocess.run(["git", "clone", "-q", "--shared", REPO, ck], check=True)
    subprocess.run(["git", "-C", ck, "checkout", "-q", subprocess.run(
        ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()], check=True)
    pairs = [
        (os.path.join(SEED2, "xception", "polyglotfake_xception_best_pytorch_unbal_all.pth"),
         "checkpoints/xception/polyglotfake_xception_best_unbal_all_faceaug.pth"),
        (os.path.join(SEED2, "freqnet", "freqnet_model_all_unbalanced_improved.pth"),
         "checkpoints/freqnet/freqnet_model_all_unbalanced_improved.pth"),
        (os.path.join(SEED2, "crossmodal", "lip_sync_model_crossattention_all_unbalanced.pth"),
         "checkpoints/cross_modal/lip_sync_model_crossattention.pth"),
        (os.path.join(LOGDIR, "ecapa_v2", "audio_forensics_model_best.pth"),
         "checkpoints/ecapa_forensic_head/audio_forensics_model_finetuned_best.pth"),
        (os.path.join(LOGDIR, "ecapa_v2", "training_stats.npz"),
         "checkpoints/ecapa_forensic_head/training_stats.npz"),
    ]
    for src, rel in pairs:
        dst = os.path.join(ck, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        log(f"staged {rel} <- {os.path.basename(src)}")
    ensure_link(os.path.join(ck, "data", "polyglot_processed_all_unbalanced"), LOCAL)
    out_csv = os.path.join(SEED2, "analysis_results_seed2_test.csv")
    run([sys.executable, "-u", "src/orchestrator.py", "--split", "test", "--output_file", out_csv],
        cwd=ck, logname="seed2_orchestrator_test.log")
    mark("5_scored")

# ---- stage 6: metrics at the frozen operating point ----------------------
import numpy as np
import pandas as pd

W = {"score_Visual (Spatial)": 0.20, "score_Audio (Mel+CNN)": 0.15,
     "score_Audio Forensics (ECAPA)": 0.20, "score_Cross-Modal (Lip-Sync)": 0.25,
     "score_Facial Biometric (Quality)": 0.20}
TAU = 0.37
df = pd.read_csv(os.path.join(SEED2, "analysis_results_seed2_test.csv"))
y = (df["ground_truth"] == "Fake").astype(int).to_numpy()
w = np.array(list(W.values()))
score = df[list(W)].to_numpy() @ (w / w.sum())
pred = (score >= TAU).astype(int)
from sklearn.metrics import roc_auc_score, average_precision_score
per_agent = {c: float(((df[c].to_numpy() >= TAU).astype(int) == y).mean()) for c in W}
result = {
    "composition": "retrained XceptionNet, FreqNet, Cross-Modal, ECAPA-TDNN + released Biometric-Quality",
    "n": int(len(y)), "n_real": int((1 - y).sum()), "n_fake": int(y.sum()),
    "tau": TAU, "weights": {k.replace("score_", ""): v for k, v in W.items()},
    "errors": int((pred != y).sum()),
    "accuracy": float((pred == y).mean()),
    "auc_roc": float(roc_auc_score(y, score)),
    "average_precision": float(average_precision_score(y, score)),
    "margin": float(score[y == 0].max() - 0.0) if False else float(score[y == 1].min() - score[y == 0].max()),
    "max_real_score": float(score[y == 0].max()), "min_fake_score": float(score[y == 1].min()),
    "per_agent_accuracy_at_tau": {k.replace("score_", ""): v for k, v in per_agent.items()},
}
out = os.path.join(SEED2, "seed_replicate_test.json")
json.dump(result, open(out, "w"), indent=1)
log("RESULT " + json.dumps(result))
log(f"written {out}")
