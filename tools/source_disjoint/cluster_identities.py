"""Group PolyGlotFake source videos that show the same person, so a split can be identity-disjoint.

PolyGlotFake publishes no identity labels. This embeds the stored face crops of every
source video's authentic clip with a VGGFace2-pretrained InceptionResnetV1
(facenet-pytorch), averages over frames, and links two sources when the cosine distance of
their embeddings is below `--threshold` (0.4 is the usual same-person operating point for
this model). Connected components become identity groups; make_split.py `--groups` then
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
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--frames", type=int, default=5)
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
    dist = 1.0 - E @ E.T
    n = len(sources); parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] < a.threshold:
                parent[find(i)] = find(j); pairs.append((sources[i], sources[j], float(dist[i, j])))
    groups = {}
    for i, s in enumerate(sources):
        groups.setdefault(f"g{find(i)}", []).append(s)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    out = {"threshold": a.threshold, "n_sources": n, "n_groups": len(groups),
           "n_groups_with_several_sources": len(multi), "n_sources_in_shared_identity_groups": sum(len(v) for v in multi.values()),
           "cross_language_links": sum(1 for s, t, _ in pairs if s[:2] != t[:2]),
           "source_to_group": {s: f"g{find(i)}" for i, s in enumerate(sources)}, "groups": groups, "linked_pairs": pairs}
    json.dump(out, open(a.out, "w"), indent=1)
    print({k: out[k] for k in ("n_sources", "n_groups", "n_groups_with_several_sources", "n_sources_in_shared_identity_groups", "cross_language_links")})


if __name__ == "__main__":
    main()
