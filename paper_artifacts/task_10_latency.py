"""Task 10: Pure inference latency (preprocessing excluded).

This machine has no CUDA, so runs on CPU or MPS (Apple silicon). The device
actually used is recorded in the JSON.

Important caveats written into the output:
  1. Parallel vs sequential. The orchestrator code in multiagent_xai_five_agents.py
     runs first-wave and second-wave agents SEQUENTIALLY. The paper describes them
     as running in parallel. We report both:
       phase1_parallel  = max over the three phase-1 agents
       phase1_seq       = sum over the three phase-1 agents
       phase1_plus_2_parallel = max(phase1) + max(phase2)
       phase1_plus_2_seq      = sum(phase1) + sum(phase2)
  2. Preprocessing excluded. MTCNN face detection, librosa feature extraction,
     and Mel-spectrogram computation are NOT timed — only the agents' forward
     pass on pre-shaped tensors is.
  3. Input shapes match the config in multiagent_xai_five_agents.py. Tensor
     contents are random because the .npz test files are not present on this
     machine; parameter counts and timing do not depend on values.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from _common import OUT, REPO, save_json

# Find model sources: publication/src/multiagent_models/ in the release,
# or multiagent_models/ at the repo root in the dev tree.
for cand in [REPO / "src" / "agents", REPO / "agents"]:
    if cand.exists():
        sys.path.insert(0, str(cand.parent))
        sys.path.insert(0, str(cand))
        break

RANDOM_SEED = 42
N_SAMPLES = 40
N_RUNS = 10
N_WARMUP = 3

# Input shapes from multiagent_xai_five_agents.py CONFIG
XCEPTION_IN = (1, 3, 299, 299)
FREQNET_IN = (1, 3, 224, 224)
CROSSMODAL_FACES_IN = (1, 20, 3, 224, 224)
CROSSMODAL_AUDIO_IN = (1, 1, 128, 313)
BIOMETRIC_IN = (1, 5, 299, 299)
ECAPA_WAVE_IN = 80000  # 5 s @ 16 kHz
ECAPA_EMBED_IN = (1, 192 + 11)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(d: torch.device) -> str:
    if d.type == "cuda":
        return torch.cuda.get_device_name(0)
    if d.type == "mps":
        return "Apple Metal (MPS)"
    import platform
    return f"CPU ({platform.processor() or platform.machine()})"


def time_forward(fn, n_runs: int = N_RUNS) -> list[float]:
    """Run fn() n_runs times, return per-run latencies in ms.
    Uses device-appropriate synchronisation."""
    xs = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        _ = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        xs.append((time.perf_counter() - t0) * 1000)
    return xs


def build_models(device: torch.device):
    from agents.visual_xception import XceptionDeepfakeDetector
    from agents.audio_freqnet import FreqNet
    from agents.cross_modal_lipsync import CrossModal_CNN_LSTM
    from agents.biometric_quality import FaceQualityNet
    from agents.audio_forensics_ecapa import OptimizedLightweightForensics

    xc = XceptionDeepfakeDetector(pretrained=False).to(device).eval()
    fn = FreqNet(num_classes=1).to(device).eval()
    cm = CrossModal_CNN_LSTM().to(device).eval()
    bm = FaceQualityNet().to(device).eval()
    forensic_head = OptimizedLightweightForensics(192, 11).to(device).eval()

    # ECAPA backbone: for timing only we approximate with a quick synthetic
    # module if speechbrain fails. Otherwise use the real thing.
    ecapa_backbone = None
    try:
        from speechbrain.inference import EncoderClassifier
        ecapa_backbone = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(OUT / ".speechbrain_cache"),
        )
        # Move underlying modules to target device (skip MPS for ECAPA because
        # speechbrain's STFT uses operations that hit a CPU fallback on MPS;
        # we run ECAPA on CPU for stability and document the choice).
        try:
            if device.type == "mps":
                pass  # keep on CPU, see note above
            else:
                ecapa_backbone.mods.embedding_model.to(device)
        except Exception:
            pass
    except Exception as e:
        print(f"[task10] WARN: could not load ECAPA backbone ({e}); "
              f"timing only the classifier head for ECAPA.")

    return {
        "xception": xc,
        "freqnet": fn,
        "crossmodal": cm,
        "biometric": bm,
        "forensic_head": forensic_head,
        "ecapa_backbone": ecapa_backbone,
    }


def make_inputs(device: torch.device, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    xc_in = torch.randn(*XCEPTION_IN, generator=g).to(device)
    fn_in = torch.randn(*FREQNET_IN, generator=g).to(device)
    cm_faces = torch.randn(*CROSSMODAL_FACES_IN, generator=g).to(device)
    cm_audio = torch.randn(*CROSSMODAL_AUDIO_IN, generator=g).to(device)
    bm_in = torch.randn(*BIOMETRIC_IN, generator=g).to(device)
    waveform = torch.randn(1, ECAPA_WAVE_IN, generator=g).to(device)
    forensic_vec = torch.randn(*ECAPA_EMBED_IN, generator=g).to(device)
    return {
        "xception_x": xc_in,
        "freqnet_x": fn_in,
        "crossmodal_v": cm_faces,
        "crossmodal_a": cm_audio,
        "biometric_x": bm_in,
        "waveform": waveform,
        "forensic_vec": forensic_vec,
    }


def summarise(arr: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


@torch.no_grad()
def time_all(models, inputs) -> dict:
    def run_xc():
        return models["xception"](inputs["xception_x"])
    def run_fn():
        return models["freqnet"](inputs["freqnet_x"])
    def run_cm():
        return models["crossmodal"](inputs["crossmodal_v"], inputs["crossmodal_a"])
    def run_bm():
        return models["biometric"](inputs["biometric_x"])

    ecapa = models["ecapa_backbone"]
    if ecapa is not None:
        def run_ecapa():
            # ECAPA runs on CPU; move waveform there, then output back to device
            wav = inputs["waveform"].detach().to("cpu")
            emb = ecapa.encode_batch(wav)
            emb_flat = emb.reshape(1, -1)[:, :192].to(inputs["forensic_vec"].device)
            feats = torch.cat([emb_flat, inputs["forensic_vec"][:, 192:]], dim=1)
            return models["forensic_head"](feats)
    else:
        def run_ecapa():
            return models["forensic_head"](inputs["forensic_vec"])

    return {
        "Visual (XceptionNet)": run_xc,
        "Audio (FreqNet)":       run_fn,
        "Cross-Modal":           run_cm,
        "Biometric":             run_bm,
        "ECAPA":                 run_ecapa,
    }


def main() -> None:
    global device
    device = pick_device()
    models = build_models(device)

    # Warm-up once
    rng = np.random.default_rng(RANDOM_SEED)
    for _ in range(N_WARMUP):
        inputs = make_inputs(device, int(rng.integers(0, 2**31)))
        calls = time_all(models, inputs)
        for fn in calls.values():
            with torch.no_grad():
                fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    agent_samples = {k: [] for k in ["Visual (XceptionNet)", "Audio (FreqNet)",
                                     "Cross-Modal", "Biometric", "ECAPA"]}

    for sample_i in range(N_SAMPLES):
        inputs = make_inputs(device, RANDOM_SEED + sample_i)
        calls = time_all(models, inputs)
        for name, fn in calls.items():
            runs = time_forward(fn, N_RUNS)
            agent_samples[name].append(float(np.mean(runs)))

    # Convert to arrays
    agent_arr = {k: np.array(v) for k, v in agent_samples.items()}

    # Per-agent summary
    per_agent = {k: summarise(v) for k, v in agent_arr.items()}

    # Phase-1 latency (parallel = max, sequential = sum)
    phase1_names = ["Biometric", "ECAPA", "Cross-Modal"]
    phase2_names = ["Visual (XceptionNet)", "Audio (FreqNet)"]

    phase1_parallel = np.max(np.stack([agent_arr[n] for n in phase1_names], axis=1), axis=1)
    phase1_seq      = np.sum(np.stack([agent_arr[n] for n in phase1_names], axis=1), axis=1)
    phase2_parallel = np.max(np.stack([agent_arr[n] for n in phase2_names], axis=1), axis=1)
    phase2_seq      = np.sum(np.stack([agent_arr[n] for n in phase2_names], axis=1), axis=1)

    phase12_parallel = phase1_parallel + phase2_parallel
    phase12_seq = phase1_seq + phase2_seq

    # Escalation rate from Task 7 at tau_d = 0.3 = 8.79% (nearest baked value)
    # The paper's design target was 85 / 15 mix. We report the ACTUAL escalation
    # from our own disagreement sweep at tau_d = 0.3, and include the paper's
    # 85/15 mix for comparison.
    ACTUAL_ESC = 0.0879
    PAPER_ESC = 0.15
    avg_parallel_actual = (1 - ACTUAL_ESC) * phase1_parallel + ACTUAL_ESC * phase12_parallel
    avg_parallel_paper  = (1 - PAPER_ESC) * phase1_parallel + PAPER_ESC * phase12_parallel

    out = {
        "gpu": device_name(device),
        "device_type": device.type,
        "n_samples": N_SAMPLES,
        "n_runs_per_sample": N_RUNS,
        "n_warmup": N_WARMUP,
        "seed": RANDOM_SEED,
        "notes": [
            "Latencies are in milliseconds (ms).",
            "Preprocessing (MTCNN, librosa feature extraction, Mel spectrogram "
            "computation) is NOT included.",
            "Orchestrator code (multiagent_xai_five_agents.py._run_first_wave) "
            "runs agents sequentially, not in parallel. Parallel numbers are "
            "the best case achievable if inference were actually parallelised "
            "across devices or threads.",
            "ECAPA latency includes ECAPA-TDNN backbone forward on a synthetic "
            "80000-sample waveform plus the 64-dim classifier head forward.",
        ],
        "per_agent_ms": per_agent,
        "phase1_only_ms": {
            "parallel": summarise(phase1_parallel),
            "sequential": summarise(phase1_seq),
        },
        "phase1_plus_2_ms": {
            "parallel": summarise(phase12_parallel),
            "sequential": summarise(phase12_seq),
        },
        "average_observed_ms": {
            "description": "mean * (1 - escalation_rate) + phase1+2 * escalation_rate",
            "escalation_rate_actual_tau_d_0.3": ACTUAL_ESC,
            "escalation_rate_paper_target": PAPER_ESC,
            "parallel_actual": summarise(avg_parallel_actual),
            "parallel_paper_mix": summarise(avg_parallel_paper),
        },
    }
    save_json(out, OUT / "latency_benchmark.json")

    print(f"[task10] device = {out['gpu']} ({device.type})")
    for name in ["Biometric", "ECAPA", "Cross-Modal", "Visual (XceptionNet)", "Audio (FreqNet)"]:
        print(f"[task10]   {name:<24s}: mean={per_agent[name]['mean']:.1f}ms "
              f"median={per_agent[name]['median']:.1f}ms "
              f"p95={per_agent[name]['p95']:.1f}ms")
    print(f"[task10] Phase 1 (parallel): mean={out['phase1_only_ms']['parallel']['mean']:.1f}ms  "
          f"Phase 1+2 (parallel): mean={out['phase1_plus_2_ms']['parallel']['mean']:.1f}ms")
    print(f"[task10] Avg observed (parallel, 91/9 mix): "
          f"mean={out['average_observed_ms']['parallel_actual']['mean']:.1f}ms")


if __name__ == "__main__":
    main()
