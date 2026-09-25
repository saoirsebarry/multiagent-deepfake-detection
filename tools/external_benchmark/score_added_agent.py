"""Score staged clips with the added agent and write `clip,score_added` rows.

    python tools/external_benchmark/score_added_agent.py --clips <staged>/test --ckpt checkpoints/added_agent/best.pth --out added.csv
"""
import argparse
import csv
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

from train_added_agent import SIZE, make_model, to_input


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    model = make_model().to(a.device)
    model.load_state_dict(torch.load(a.ckpt, map_location=a.device)["model"]); model.eval()
    done = set()
    if os.path.exists(a.out):
        done = {r["clip"] for r in csv.DictReader(open(a.out))}
    with open(a.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if not done:
            w.writerow(["clip", "score_added", "n_faces"])
        for f in sorted(glob.glob(os.path.join(a.clips, "*.npz"))):
            name = os.path.basename(f)
            if name in done:
                continue
            with np.load(f) as z:
                faces = z["faces"]
            if not len(faces):
                w.writerow([name, -1, 0]); continue
            t = torch.from_numpy(faces).permute(0, 3, 1, 2).float()
            t = F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False)
            u8 = t.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
            with torch.no_grad():
                x = to_input(u8, a.device, False)
                p = (torch.sigmoid(model(x)) + torch.sigmoid(model(x.flip(3)))).mean().item() / 2
            w.writerow([name, f"{p:.6f}", len(faces)]); fh.flush()
    print("scored ->", a.out)


if __name__ == "__main__":
    main()
