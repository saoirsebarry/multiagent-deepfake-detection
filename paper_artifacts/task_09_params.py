"""Task 9: Parameter counts for the five agent models.

Instantiates each model, loads the production checkpoint if it can be
applied, and reports total / trainable parameter counts. Parameter
counts do not depend on weight values, so when a checkpoint key
mismatch is unavoidable we fall back to random weights and still
report the architecture's parameter count.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch

from _common import OUT, REPO, save_json

# Find model sources: publication/src/multiagent_models/ in the release,
# or multiagent_models/ at the repo root in the dev tree.
for cand in [REPO / "src" / "agents", REPO / "agents"]:
    if cand.exists():
        sys.path.insert(0, str(cand.parent))
        sys.path.insert(0, str(cand))
        break



def _find_ckpt(primary: str, legacy: str) -> Path:
    # Resolve a checkpoint path against either the publication release layout
    # (publication/checkpoints/...) or the development repo layout.
    p = REPO / "checkpoints" / primary
    if p.exists():
        return p
    return REPO / legacy


def _counts(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def count_xception() -> dict:
    from agents.visual_xception import XceptionDeepfakeDetector
    model = XceptionDeepfakeDetector(pretrained=False)  # load=False so no network
    ckpt_path = _find_ckpt("xception/polyglotfake_xception_best_unbal_all_faceaug.pth", "xceptionnet_models/polyglotfake_xception_best_unbal_all_faceaug.pth")
    c = _counts(model)
    c["checkpoint_applied"] = False
    if ckpt_path.exists():
        try:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            res = model.load_state_dict(sd, strict=False)
            c["checkpoint_applied"] = True
            c["missing_keys"] = len(res.missing_keys)
            c["unexpected_keys"] = len(res.unexpected_keys)
        except Exception as e:
            c["checkpoint_error"] = f"{type(e).__name__}: {e}"
    return c


def count_freqnet() -> dict:
    from agents.audio_freqnet import FreqNet
    # class signature: FreqNet(num_classes=1)
    model = FreqNet(num_classes=1)
    c = _counts(model)
    c["checkpoint_applied"] = False
    ckpt_path = _find_ckpt("freqnet/freqnet_model_all_unbalanced_improved.pth", "freqnet_models/freqnet_model_all_unbalanced_improved.pth")
    if ckpt_path.exists():
        try:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            res = model.load_state_dict(sd, strict=False)
            c["checkpoint_applied"] = True
            c["missing_keys"] = len(res.missing_keys)
            c["unexpected_keys"] = len(res.unexpected_keys)
        except Exception as e:
            c["checkpoint_error"] = f"{type(e).__name__}: {e}"
    return c


def count_crossmodal() -> dict:
    from agents.cross_modal_lipsync import CrossModal_CNN_LSTM
    model = CrossModal_CNN_LSTM()
    c = _counts(model)
    c["checkpoint_applied"] = False
    ckpt_path = _find_ckpt("cross_modal/lip_sync_model_crossattention.pth", "cross_modal_models/lip_sync_model_crossattention.pth")
    if ckpt_path.exists():
        try:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            res = model.load_state_dict(sd, strict=False)
            c["checkpoint_applied"] = True
            c["missing_keys"] = len(res.missing_keys)
            c["unexpected_keys"] = len(res.unexpected_keys)
        except Exception as e:
            c["checkpoint_error"] = f"{type(e).__name__}: {e}"
    return c


def count_biometric() -> dict:
    from agents.biometric_quality import FaceQualityNet
    model = FaceQualityNet()
    c = _counts(model)
    c["checkpoint_applied"] = False
    ckpt_path = _find_ckpt("biometric/fine_tuning/best_model.pth", "biometric_trained_models/fine_tuning/best_model.pth")
    if ckpt_path.exists():
        try:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            res = model.load_state_dict(sd, strict=False)
            c["checkpoint_applied"] = True
            c["missing_keys"] = len(res.missing_keys)
            c["unexpected_keys"] = len(res.unexpected_keys)
        except Exception as e:
            c["checkpoint_error"] = f"{type(e).__name__}: {e}"
    return c


def count_ecapa() -> dict:
    """ECAPA agent has two components:
    (a) ECAPA-TDNN backbone (speechbrain/spkrec-ecapa-voxceleb). Params are
        for the pretrained encoder — we load and count without training.
    (b) OptimizedLightweightForensics classifier head (learned).
    Report both totals separately and a combined total.
    """
    from agents.audio_forensics_ecapa import OptimizedLightweightForensics

    # (b) the classifier head
    classifier = OptimizedLightweightForensics(embedding_dim=192, num_forensic_features=11)
    head = _counts(classifier)

    # (a) try to load ECAPA backbone via speechbrain
    backbone = None
    backbone_info = {"loaded": False}
    try:
        from speechbrain.inference import EncoderClassifier
        backbone = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(REPO / "paper_artifacts" / ".speechbrain_cache"),
            run_opts={"device": "cpu"},
        )
        t = sum(p.numel() for p in backbone.mods.embedding_model.parameters())
        backbone_info = {"loaded": True, "encoder_params": int(t)}
    except Exception as e:
        backbone_info = {"loaded": False, "error": f"{type(e).__name__}: {e}"}

    # Learned classifier: try the production checkpoint
    ckpt = _find_ckpt("ecapa_forensic_head/audio_forensics_model_finetuned_best.pth", "audio_forensic_trained_models/audio_forensics_model_finetuned_best.pth")
    head["checkpoint_applied"] = False
    if ckpt.exists():
        try:
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            res = classifier.load_state_dict(sd, strict=False)
            head["checkpoint_applied"] = True
            head["missing_keys"] = len(res.missing_keys)
            head["unexpected_keys"] = len(res.unexpected_keys)
        except Exception as e:
            head["checkpoint_error"] = f"{type(e).__name__}: {e}"

    combined_total = head["total"] + backbone_info.get("encoder_params", 0)
    combined_trainable = head["trainable"]  # ECAPA backbone is frozen in production
    return {
        "total": int(combined_total),
        "trainable": int(combined_trainable),
        "breakdown": {
            "classifier_head": head,
            "ecapa_backbone": backbone_info,
        },
    }


def main() -> None:
    results = {}
    registry = [
        ("XceptionNet", count_xception),
        ("FreqNet", count_freqnet),
        ("CrossModal_CNN_BiLSTM_CrossAttn", count_crossmodal),
        ("Biometric_5ch_EfficientNet_B0", count_biometric),
        ("ECAPA_TDNN_plus_11features", count_ecapa),
    ]
    for name, fn in registry:
        try:
            results[name] = fn()
        except Exception:
            results[name] = {"error": traceback.format_exc()}

    save_json(results, OUT / "parameter_counts.json")

    for name, r in results.items():
        if "error" in r:
            print(f"[task09] {name}: FAILED — {r['error'].splitlines()[-1]}")
            continue
        total_m = r["total"] / 1e6
        train_m = r["trainable"] / 1e6
        ckpt = "ckpt" if r.get("checkpoint_applied", False) else (
            "ckpt partial" if "missing_keys" in r else "random")
        print(f"[task09] {name}: total={total_m:.2f}M  trainable={train_m:.2f}M  ({ckpt})")


if __name__ == "__main__":
    main()
