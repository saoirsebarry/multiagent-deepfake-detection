"""Group PolyGlotFake source videos that show the same person, so a split can be identity-disjoint.

PolyGlotFake publishes no identity labels. This embeds the stored face crops of every
source video's authentic clip with a VGGFace2-pretrained InceptionResnetV1
(facenet-pytorch), averages over frames, and clusters sources by average-linkage agglomeration on
cosine distance. The cut is the loosest value in `--sweep` (0.40 is the usual same-person
operating point for this model) whose largest group stays within `--max_group` sources: the
crops are unaligned, so a loose single cut chains unrelated faces into one group. The sweep
table is written with the groups so the choice is auditable. make_split.py `--groups` then
keeps every group in one partition.

    python tools/source_disjoint/cluster_identities.py --processed <root> --out identity_groups.json
"""
import argparse
import json
import os
import re

import numpy as np
import torch

PAT = re.compile(r"^([a-z]{2})_(\d+)_label_real\.npz$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweep", type=float, nargs="+", default=[0.20, 0.25, 0.30, 0.35, 0.40], help="candidate cosine-distance cuts")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--max_group_frac", type=float, default=0.08, help="abort if any identity group holds more than this fraction of the sources")
    a = ap.parse_args()
    from facenet_pytorch import InceptionResnetV1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    sources, embs = [], []
    for split in ("train", "val", "test"):
        d = os.path.join(a.processed, split)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            m = PAT.match(f)
            if not m:
                continue
            faces = np.load(os.path.join(d, f), allow_pickle=True)["faces"]
            if len(faces) == 0:
                continue
            idx = np.linspace(0, len(faces) - 1, min(a.frames, len(faces))).astype(int)
            x = np.stack([faces[i] for i in idx]).astype(np.float32)
            if x.max() > 1.0:
                x = x / 255.0
            x = torch.tensor(x).permute(0, 3, 1, 2)
            x = torch.nn.functional.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
            x = (x - 0.5) / 0.5
            with torch.no_grad():
                e = net(x.to(device)).mean(0)
            embs.append((e / e.norm()).cpu().numpy()); sources.append(f"{m.group(1)}_{m.group(2)}")
    E = np.stack(embs)
    n = len(sources)
    # average-linkage agglomeration: two groups merge only when their mean pairwise distance is
    # below the threshold, so near-duplicate chains cannot fuse unrelated people into one group
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    d = pdist(E, metric="cosine")
    Z = linkage(d, method="average")
    sweep = {}
    for t in a.sweep:
        lab = fcluster(Z, t=t, criterion="distance")
        sizes = np.bincount(lab)[1:]
        sweep[f"{t:.2f}"] = {"n_groups": int(len(sizes)), "largest_group": int(sizes.max()),
                             "n_groups_with_several_sources": int((sizes > 1).sum()),
                             "n_sources_in_shared_groups": int(sizes[sizes > 1].sum())}
    # the cut is the loosest threshold in the sweep whose largest group stays within the guard;
    # a looser cut chains unrelated faces into one group and empties the other partitions
    max_group = int(a.max_group_frac * n)
    print("threshold sweep:", json.dumps(sweep), flush=True)
    admissible = [t for t in a.sweep if sweep[f"{t:.2f}"]["largest_group"] <= max_group]
    if not admissible:
        raise SystemExit(f"no threshold in {a.sweep} keeps every identity group within {max_group} sources ({a.max_group_frac:.0%} of {n})")
    threshold = max(admissible)
    labels = fcluster(Z, t=threshold, criterion="distance")
    groups = {}
    for s_, lab in zip(sources, labels):
        groups.setdefault(f"g{lab}", []).append(s_)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    D = 1.0 - E @ E.T
    pairs = [(sources[i], sources[j], float(D[i, j])) for i in range(n) for j in range(i + 1, n) if labels[i] == labels[j]]
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    out = {"threshold": threshold, "threshold_sweep": sweep, "max_group": max_group, "max_group_frac": a.max_group_frac, "linkage": "average", "n_sources": n, "n_groups": len(groups),
           "n_groups_with_several_sources": len(multi), "n_sources_in_shared_identity_groups": sum(len(v) for v in multi.values()),
           "largest_group": sizes[0], "group_sizes_top10": sizes[:10],
           "cross_language_links": sum(1 for s_, t, _ in pairs if s_[:2] != t[:2]),
           "source_to_group": {s_: f"g{lab}" for s_, lab in zip(sources, labels)}, "groups": groups, "linked_pairs": pairs}
    json.dump(out, open(a.out, "w"), indent=1)
    print({k: out[k] for k in ("threshold", "threshold_sweep", "n_sources", "n_groups", "n_groups_with_several_sources", "n_sources_in_shared_identity_groups", "largest_group", "group_sizes_top10", "cross_language_links")})


if __name__ == "__main__":
    main()
