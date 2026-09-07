"""Run every robustness fine-tune in sequence, resumable via DONE markers.
    python tools/robust_finetune/run_all.py --data_dir <root with train/ val/ [test/]> --out_dir <dir>
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--agents", default="biometric,crossmodal,freqnet,ecapa,xception")
A = ap.parse_args()
for agent in A.agents.split(","):
    out = os.path.join(A.out_dir, agent); os.makedirs(out, exist_ok=True)
    marker = os.path.join(out, "DONE")
    if os.path.exists(marker):
        print(f"== {agent}: already done", flush=True); continue
    script = {"biometric": "ft_biometric.py", "crossmodal": "ft_crossmodal.py", "freqnet": "ft_freqnet.py",
              "ecapa": "ft_ecapa.py", "xception": "ft_xception.py", "xception_retrain": "retrain_xception.py"}[agent]
    print(f"== {agent}: running {script}", flush=True)
    log = open(os.path.join(out, "train.log"), "a")
    rc = subprocess.call([sys.executable, "-u", os.path.join(HERE, script), "--data_dir", A.data_dir, "--out_dir", out],
                         stdout=log, stderr=subprocess.STDOUT, cwd=os.path.dirname(os.path.dirname(HERE)))
    log.close()
    print(f"== {agent}: exit {rc}", flush=True)
    if rc == 0:
        open(marker, "w").write("ok\n")
print("RUN-ALL-DONE", flush=True)
