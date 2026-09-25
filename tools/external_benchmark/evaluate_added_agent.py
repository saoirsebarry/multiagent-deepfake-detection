"""Compare the released five-agent system with the same system plus one added agent, at unchanged equal weights and threshold.

    python tools/external_benchmark/evaluate_added_agent.py --scores <dir>/original/scores.csv --added added_scores.csv \
        --metadata <dir>/metadata.csv --modality modality.csv --seen echomimic memo liveportrait inswapper --out added_agent.json

`--modality` carries the benchmark's own video_fake / audio_fake flags per clip. Intervals are bootstraps over source videos, and
the six-minus-five difference is bootstrapped on the same resamples, so it is a paired comparison.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True); ap.add_argument("--added", required=True); ap.add_argument("--metadata", required=True)
    ap.add_argument("--modality", required=True); ap.add_argument("--seen", nargs="+", required=True)
    ap.add_argument("--seen_languages", nargs="*", default=[]); ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = pd.read_csv(a.scores); s["clip"] = s.filepath.map(os.path.basename)
    j = (s.drop_duplicates("clip").merge(pd.read_csv(a.metadata).drop_duplicates("clip"), on="clip")
          .merge(pd.read_csv(a.added), on="clip").merge(pd.read_csv(a.modality), on="clip").reset_index(drop=True))
    cols = [c for c in j.columns if c.startswith("score_") and c != "score_added"]
    j["y"] = (j.ground_truth == "Fake").astype(int)
    j["five"] = j[cols].mean(axis=1)
    j["six"] = j[cols + ["score_added"]].mean(axis=1)
    rng = np.random.default_rng(42)
    members = {k: v.index.values for k, v in j.groupby("source_video")}
    keys = np.array(list(members))
    boots = [np.concatenate([members[k] for k in rng.choice(keys, len(keys), replace=True)]) for _ in range(a.boot)]
    y = j.y.values

    def auc(col, mask):
        vals, m = j[col].values, mask.values
        point = roc_auc_score(y[m], vals[m]); v = []
        for b in boots:
            bb = b[m[b]]
            if y[bb].min() != y[bb].max():
                v.append(roc_auc_score(y[bb], vals[bb]))
        return [round(float(point), 4), round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

    def paired(mask):
        m, d = mask.values, []
        for b in boots:
            bb = b[m[b]]
            if y[bb].min() != y[bb].max():
                d.append(roc_auc_score(y[bb], j.six.values[bb]) - roc_auc_score(y[bb], j.five.values[bb]))
        return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]

    def block(mask_fake):
        mask = (mask_fake & (j.y == 1)) | (j.y == 0)
        f = j[mask_fake & (j.y == 1)]
        return {"n_forged": int(len(f)), "five": auc("five", mask), "six": auc("six", mask), "added_alone": auc("score_added", mask),
                "six_minus_five_ci": paired(mask), "recall_five": round(float((f.five >= a.tau).mean()), 4),
                "recall_six": round(float((f.six >= a.tau).mean()), 4)}

    real = j[j.y == 0]
    out = {"tau": a.tau, "n": int(len(j)), "specificity_five": round(float((real.five < a.tau).mean()), 4),
           "specificity_six": round(float((real.six < a.tau).mean()), 4), "groups": {}, "classes": {}}
    for name, col in [("five", "five"), ("six", "six")]:
        pred = (j[col] >= a.tau).astype(int)
        out[f"accuracy_{name}"] = round(float((pred == j.y).mean()), 4)
        out[f"balanced_accuracy_{name}"] = round(float(((pred[j.y == 1] == 1).mean() + (pred[j.y == 0] == 0).mean()) / 2), 4)
    vf, af = j.video_fake.astype(bool), j.audio_fake.astype(bool)
    seen = j.generator.isin(a.seen)
    groups = {"all forged": j.y == 1, "video only": vf & ~af, "video and speech": vf & af, "speech only": ~vf & af,
              "video forged, seen generators": vf & seen, "video forged, unseen generators": vf & ~seen,
              "video only, seen generators": vf & ~af & seen, "video only, unseen generators": vf & ~af & ~seen}
    if a.seen_languages:
        sl = j.language.isin(a.seen_languages)
        groups["video forged, unseen languages"] = vf & ~sl
    for name, g in groups.items():
        out["groups"][name] = block(g)
    for gen, grp in j[j.y == 1].groupby("generator"):
        out["classes"][("voice conversion" if gen == "real" else gen) + ("" if gen == "real" else (" (seen)" if gen in a.seen else " (unseen)"))] = block(j.generator == gen)
    json.dump(out, open(a.out, "w"), indent=1)
    fmt = lambda t: f"{t[0]:.3f} [{t[1]:.3f},{t[2]:.3f}]"
    print(f"n={out['n']}  accuracy five {100*out['accuracy_five']:.1f}% -> six {100*out['accuracy_six']:.1f}%   balanced {100*out['balanced_accuracy_five']:.1f}% -> {100*out['balanced_accuracy_six']:.1f}%   specificity {100*out['specificity_five']:.1f}% -> {100*out['specificity_six']:.1f}%\n")
    print(f"{'group':<36}{'n':>6}  {'five-agent':<22}{'six-agent':<22}{'paired diff CI':<19}{'added alone':<22}{'recall 5 -> 6'}")
    for section in ("groups", "classes"):
        for name, b in out[section].items():
            print(f"{name:<36}{b['n_forged']:>6}  {fmt(b['five']):<22}{fmt(b['six']):<22}[{b['six_minus_five_ci'][0]:+.3f},{b['six_minus_five_ci'][1]:+.3f}]    {fmt(b['added_alone']):<22}{100*b['recall_five']:.1f}% -> {100*b['recall_six']:.1f}%")
        print()


if __name__ == "__main__":
    main()
