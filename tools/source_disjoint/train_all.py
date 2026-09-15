"""Train the five agents from their pretrained backbones on a partition, one seed per run.

    python tools/source_disjoint/train_all.py --data_dir <root with train/ val/ test/> \
        --out_dir <run dir> --seed 43 [--agents xception,freqnet,crossmodal,biometric,ecapa] \
        [--finetune] [--score]

Each agent runs its released training script (src/agents/*.py, tools/train_*.py) with the
seed, data root and output directory passed through; the recipes are unchanged. Outputs are
collected into <out_dir>/checkpoints/ in the released layout so score_split.py can read
them. `--finetune` then runs the released robustness fine-tunes (tools/robust_finetune/
ft_{freqnet,ecapa,crossmodal}.py) warm-started from this run's own checkpoints, with the
released validation-only adoption rules; `--score` scores val and test with the final set.
Every stage is resumable through DONE markers.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable


def run(cmd, log, env=None, cwd=REPO):
    print(">>", " ".join(cmd), flush=True)
    with open(log, "a") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd, env={**os.environ, **(env or {})})
    if rc != 0:
        raise SystemExit(f"failed ({rc}): {' '.join(cmd)}; see {log}")


def done(marker):
    return os.path.exists(marker)


def mark(marker):
    open(marker, "w").write("ok\n")


def newest(pattern):
    hits = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not hits:
        raise SystemExit(f"nothing matches {pattern}")
    return hits[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--agents", default="xception,freqnet,crossmodal,biometric,ecapa")
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--ecapa_workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args()
    data = os.path.abspath(a.data_dir)
    out = os.path.abspath(a.out_dir)
    ck = os.path.join(out, "checkpoints")
    for d in ("xception", "freqnet", "cross_modal", "biometric", "ecapa_forensic_head"):
        os.makedirs(os.path.join(ck, d), exist_ok=True)
    seed_env = {"PGF_SEED": str(a.seed)}
    agents = a.agents.split(",")

    if "xception" in agents and not done(f"{out}/xception.DONE"):
        w = os.path.join(out, "xception_work")
        run([PY, "-u", "src/agents/visual_xception.py"], f"{out}/xception.log",
            env={**seed_env, "PGF_DATA_DIR": data, "PGF_OUT_DIR": w})
        shutil.copy(os.path.join(w, "polyglotfake_xception_best_pytorch_unbal_all.pth"), os.path.join(ck, "xception", "best.pth"))
        mark(f"{out}/xception.DONE")

    if "freqnet" in agents and not done(f"{out}/freqnet.DONE"):
        run([PY, "-u", "src/agents/audio_freqnet.py", "--dataroot", data, "--seed", str(a.seed),
             "--output_model_path", os.path.join(ck, "freqnet", "best.pth"),
             "--plot_path", os.path.join(out, "freqnet_curves.png")], f"{out}/freqnet.log")
        mark(f"{out}/freqnet.DONE")

    if "crossmodal" in agents and not done(f"{out}/crossmodal.DONE"):
        w = os.path.join(out, "crossmodal_work")
        run([PY, "-u", "tools/train_crossmodal.py", "--data_dir", data, "--output_dir", w,
             "--seed", str(a.seed), "--skip_fidelity"], f"{out}/crossmodal.log")
        shutil.copy(os.path.join(w, "crossmodal_best.pth"), os.path.join(ck, "cross_modal", "best.pth"))
        mark(f"{out}/crossmodal.DONE")

    if "biometric" in agents and not done(f"{out}/biometric.DONE"):
        w = os.path.join(out, "biometric_work")
        run([PY, "-u", "tools/train_biometric.py", "--data_dir", data, "--output_dir", w,
             "--seed", str(a.seed), "--select", "loss"], f"{out}/biometric.log")
        shutil.copy(os.path.join(w, "best_model.pth"), os.path.join(ck, "biometric", "best.pth"))
        mark(f"{out}/biometric.DONE")

    if "ecapa" in agents and not done(f"{out}/ecapa.DONE"):
        feats = os.path.join(out, "ecapa_features")
        run([PY, "-u", "tools/precompute_ecapa_features.py", "--data_dir", data, "--out_dir", feats,
             "--splits", "train", "val", "--workers", str(a.ecapa_workers)], f"{out}/ecapa_features.log")
        w = os.path.join(out, "ecapa_work")
        run([PY, "-u", "src/agents/audio_forensics_ecapa.py", "--data_dir", data, "--output_dir", w,
             "--feature_cache", feats, "--seed", str(a.seed)], f"{out}/ecapa.log")
        shutil.copy(os.path.join(w, "audio_forensics_model_best.pth"), os.path.join(ck, "ecapa_forensic_head", "best.pth"))
        shutil.copy(os.path.join(w, "training_stats.npz"), os.path.join(ck, "ecapa_forensic_head", "training_stats.npz"))
        mark(f"{out}/ecapa.DONE")

    if a.finetune and not done(f"{out}/finetune.DONE"):
        ft = os.path.join(out, "finetune")
        for agent, script, init in [
            ("freqnet", "ft_freqnet.py", os.path.join(ck, "freqnet", "best.pth")),
            ("crossmodal", "ft_crossmodal.py", os.path.join(ck, "cross_modal", "best.pth")),
            ("ecapa", "ft_ecapa.py", os.path.join(ck, "ecapa_forensic_head", "best.pth")),
        ]:
            d = os.path.join(ft, agent)
            if done(f"{d}/DONE"):
                continue
            os.makedirs(d, exist_ok=True)
            cmd = [PY, "-u", f"tools/robust_finetune/{script}", "--data_dir", data, "--out_dir", d,
                   "--init", init, "--skip_fidelity", "--seed", str(a.seed)]
            if agent == "ecapa":
                cmd += ["--init_stats", os.path.join(ck, "ecapa_forensic_head", "training_stats.npz")]
            run(cmd, f"{d}/train.log")
            mark(f"{d}/DONE")
        adopted = {}
        for agent, sub, name in [("freqnet", "freqnet", "best.pth"), ("crossmodal", "cross_modal", "best.pth"), ("ecapa", "ecapa_forensic_head", "best.pth")]:
            dec = json.load(open(os.path.join(ft, agent, "decision.json")))
            adopted[agent] = dec.get("adopted_rule")
            if dec.get("adopted"):
                src = os.path.join(ft, agent, "best_robust.pth" if dec.get("adopted_rule") == "rule2" else "best_model.pth")
                shutil.copy(src, os.path.join(ck, sub, name))
                if agent == "ecapa":
                    shutil.copy(os.path.join(ft, agent, "training_stats.npz"), os.path.join(ck, sub, "training_stats.npz"))
        json.dump(adopted, open(os.path.join(out, "finetune_adoption.json"), "w"), indent=1)
        mark(f"{out}/finetune.DONE")

    if a.score:
        for split in ("val", "test"):
            csv = os.path.join(out, f"scores_{split}.csv")
            if done(csv + ".DONE"):
                continue
            run([PY, "-u", "tools/source_disjoint/score_split.py", "--data_dir", data, "--split", split,
                 "--ckpt_dir", ck, "--out", csv], f"{out}/score_{split}.log")
            mark(csv + ".DONE")
    json.dump({"seed": a.seed, "data_dir": data, "agents": agents, "finetune": a.finetune}, open(os.path.join(out, "run.json"), "w"), indent=1)
    print("TRAIN-ALL-DONE", flush=True)


if __name__ == "__main__":
    main()
