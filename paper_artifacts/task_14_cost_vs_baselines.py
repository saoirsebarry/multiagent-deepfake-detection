"""Task 14: inference cost of the multi-agent framework vs transformer baselines.

Answers the reviewer request for a quantitative inference-time and accelerator-memory
comparison against baseline models. Every row is measured on the SAME device with the
SAME protocol (batch size 1, fp32, warm-up then timed runs, device synchronised).

Each model is profiled in a FRESH SUBPROCESS so that accelerator-allocator caching from
one model cannot inflate the peak-memory reading of the next.

Provenance is recorded per row:
  "checkpointed" - released agent architecture, as used in the paper
  "architecture" - baseline rebuilt from its published architecture spec; weights are
                   randomly initialised because batch-1 inference cost is a function of
                   architecture and input shape, not of weight values. This is the same
                   assumption task_10_latency.py already makes for the agents.
Rows whose rebuild is smaller than the parameter count reported for the original model
are flagged "lower_bound": their true cost is at least the figure given.
"""
from __future__ import annotations

import json, os, platform, subprocess, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SEED, N_SAMPLES, N_RUNS, N_WARMUP = 42, 30, 10, 5
MB = 1024.0 ** 2

# name -> (builder key, reported params in the literature or None, provenance, note)
TARGETS = [
    ("Visual (XceptionNet)",        "xception",   None,      "checkpointed", "299x299 face crop"),
    ("Audio (FreqNet)",             "freqnet",    None,      "checkpointed", "224x224 Mel spectrogram"),
    ("Cross-Modal (Lip-Sync)",      "crossmodal", None,      "checkpointed", "20 frames + Mel spectrogram"),
    ("Biometric-Quality",           "biometric",  None,      "checkpointed", "5-channel 299x299"),
    ("Audio Forensics (ECAPA-TDNN)","ecapa",      None,      "checkpointed", "5 s waveform at 16 kHz"),
    ("Hybrid CNN-Transformer",      "hybrid",     6_420_000, "architecture", "EfficientNet-B0 + 4-layer transformer encoder, 224x224"),
    ("GenConViT AE",                "genconvit",  130_000_000,"architecture","autoencoder + ConvNeXt-T/Swin-T on original and reconstruction, 224x224"),
    ("Late-fusion ViT",             "vit",        181_000_000,"architecture","ViT-B/16 + 6-layer transformer audio branch, 224x224"),
]


# ===================================================================== worker
def worker(key: str) -> None:
    import torch, torch.nn as nn

    def pick():
        if torch.cuda.is_available(): return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    dev = pick()

    def sync():
        if dev.type == "cuda": torch.cuda.synchronize()
        elif dev.type == "mps": torch.mps.synchronize()

    def empty():
        if dev.type == "cuda": torch.cuda.empty_cache()
        elif dev.type == "mps": torch.mps.empty_cache()

    def live_mem():
        """Bytes currently held by live tensors on the accelerator."""
        if dev.type == "cuda": return torch.cuda.memory_allocated()
        if dev.type == "mps":  return torch.mps.current_allocated_memory()
        return 0

    class PeakSampler:
        """Poll live tensor allocation on a side thread to capture the forward-pass peak.

        The Metal driver heap (driver_allocated_memory) reports the allocator's reserved
        pool, which is dominated by a one-off ~1 GB reservation and does not track a
        model's working set. Sampling live allocation is a genuine, if approximate,
        measurement of peak tensor residency."""
        def __init__(s, hz=2000):
            s.interval, s.peak, s._run = 1.0 / hz, 0, False
        def __enter__(s):
            import threading
            s.peak, s._run = live_mem(), True
            s.t = threading.Thread(target=s._loop, daemon=True); s.t.start(); return s
        def _loop(s):
            while s._run:
                s.peak = max(s.peak, live_mem()); time.sleep(s.interval)
        def __exit__(s, *a):
            s._run = False; s.t.join(timeout=1.0); s.peak = max(s.peak, live_mem())

    sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "src" / "agents"))
    g = torch.Generator(device="cpu").manual_seed(SEED)
    rnd = lambda *s: torch.randn(*s, generator=g).to(dev)

    empty(); sync()
    base_mem = live_mem()

    model, call, extra = None, None, {}

    if key == "xception":
        from agents.visual_xception import XceptionDeepfakeDetector
        model = XceptionDeepfakeDetector(pretrained=False).to(dev).eval()
        x = rnd(1, 3, 299, 299); call = lambda: model(x)
    elif key == "freqnet":
        from agents.audio_freqnet import FreqNet
        model = FreqNet(num_classes=1).to(dev).eval()
        x = rnd(1, 3, 224, 224); call = lambda: model(x)
    elif key == "crossmodal":
        from agents.cross_modal_lipsync import CrossModal_CNN_LSTM
        model = CrossModal_CNN_LSTM().to(dev).eval()
        v, a = rnd(1, 20, 3, 224, 224), rnd(1, 1, 128, 313); call = lambda: model(v, a)
    elif key == "biometric":
        from agents.biometric_quality import FaceQualityNet
        model = FaceQualityNet().to(dev).eval()
        x = rnd(1, 5, 299, 299); call = lambda: model(x)
    elif key == "ecapa":
        from agents.audio_forensics_ecapa import OptimizedLightweightForensics
        from speechbrain.inference import EncoderClassifier
        ec = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(HERE / ".speechbrain_cache"))
        head = OptimizedLightweightForensics(192, 11).to(dev).eval()
        wav = torch.randn(1, 80000, generator=g)
        fv = rnd(1, 203)

        class Whole(nn.Module):
            def __init__(s):
                super().__init__(); s.enc = ec.mods.embedding_model; s.head = head
        model = Whole()

        def call():
            emb = ec.encode_batch(wav).reshape(1, -1)[:, :192].to(dev)
            return head(torch.cat([emb, fv[:, 192:]], 1))
        extra["note_runtime"] = ("ECAPA backbone runs on CPU: speechbrain's STFT falls "
                                 "back off MPS. Latency is an upper bound on this host.")
    elif key == "hybrid":
        import timm

        class Hybrid(nn.Module):
            def __init__(s, d=256, layers=4, heads=8):
                super().__init__()
                s.cnn = timm.create_model("efficientnet_b0", pretrained=False,
                                          num_classes=0, global_pool="")
                s.proj = nn.Conv2d(s.cnn.num_features, d, 1)
                s.enc = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d, heads, d * 4, batch_first=True), layers)
                s.head = nn.Linear(d, 1)
            def forward(s, x):
                f = s.proj(s.cnn.forward_features(x)).flatten(2).transpose(1, 2)
                return s.head(s.enc(f).mean(1))
        model = Hybrid().to(dev).eval()
        x = rnd(1, 3, 224, 224); call = lambda: model(x)
    elif key == "genconvit":
        import timm

        class AE(nn.Module):
            """Convolutional autoencoder matching GenConViT's AE branch role."""
            def __init__(s, ch=(64, 128, 256)):
                super().__init__()
                enc, dec, c = [], [], 3
                for o in ch:
                    enc += [nn.Conv2d(c, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True)]; c = o
                for o in reversed(ch[:-1]):
                    dec += [nn.ConvTranspose2d(c, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True)]; c = o
                dec += [nn.ConvTranspose2d(c, 3, 4, 2, 1), nn.Sigmoid()]
                s.enc, s.dec = nn.Sequential(*enc), nn.Sequential(*dec)
            def forward(s, x): return s.dec(s.enc(x))

        class GenConViT(nn.Module):
            """ConvNeXt-T + Swin-T applied to BOTH the input and its reconstruction,
            which is the structure that gives the published ~130M AE parameter count."""
            def __init__(s):
                super().__init__()
                s.ae = AE()
                s.cn1 = timm.create_model("convnext_tiny", pretrained=False, num_classes=0)
                s.sw1 = timm.create_model("swin_tiny_patch4_window7_224",
                                          pretrained=False, num_classes=0)
                s.cn2 = timm.create_model("convnext_tiny", pretrained=False, num_classes=0)
                s.sw2 = timm.create_model("swin_tiny_patch4_window7_224",
                                          pretrained=False, num_classes=0)
                d = s.cn1.num_features + s.sw1.num_features
                s.head = nn.Sequential(nn.Linear(2 * d, 512), nn.ReLU(True), nn.Linear(512, 1))
            def forward(s, x):
                xr = s.ae(x)
                a = torch.cat([s.cn1(x), s.sw1(x)], 1)
                b = torch.cat([s.cn2(xr), s.sw2(xr)], 1)
                return s.head(torch.cat([a, b], 1))
        model = GenConViT().to(dev).eval()
        x = rnd(1, 3, 224, 224); call = lambda: model(x)
    elif key == "vit":
        import timm

        class LateFusionViT(nn.Module):
            def __init__(s, d=768, alayers=6, heads=12):
                super().__init__()
                s.vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
                s.aproj = nn.Linear(128, d)
                s.audio = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d, heads, d * 4, batch_first=True), alayers)
                s.head = nn.Linear(s.vit.num_features + d, 1)
            def forward(s, v, a):
                return s.head(torch.cat([s.vit(v), s.audio(s.aproj(a)).mean(1)], 1))
        model = LateFusionViT().to(dev).eval()
        v, a = rnd(1, 3, 224, 224), rnd(1, 128, 128); call = lambda: model(v, a)
    else:
        raise SystemExit(f"unknown key {key}")

    with torch.no_grad():
        for _ in range(N_WARMUP): call()
        sync()
        resident = live_mem()          # model weights + fixed inputs, after warm-up
        lat = []
        with PeakSampler() as ps:
            for _ in range(N_SAMPLES):
                runs = []
                for _ in range(N_RUNS):
                    sync(); t0 = time.perf_counter(); call(); sync()
                    runs.append((time.perf_counter() - t0) * 1000)
                lat.append(float(np.mean(runs)))
            sync()
        peak = ps.peak

    lat = np.array(lat)
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("@@JSON@@" + json.dumps({
        "params_total": int(tot), "params_trainable": int(tr),
        "param_mem_mb": round(tot * 4 / MB, 2),
        "latency_ms": {"mean": float(lat.mean()),
                       "std": float(lat.std(ddof=1)) if len(lat) > 1 else 0.0,
                       "median": float(np.median(lat)), "p95": float(np.percentile(lat, 95))},
        "peak_tensor_mem_mb": round(max(peak - base_mem, 0) / MB, 2),
        "resident_tensor_mem_mb": round(max(resident - base_mem, 0) / MB, 2),
        "peak_activation_mem_mb": round(max(peak - resident, 0) / MB, 2),
        "device": dev.type, **extra}))


# ===================================================================== driver
def main() -> None:
    import torch
    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")
    dname = (torch.cuda.get_device_name(0) if dev == "cuda" else
             "Apple Metal (MPS, unified memory)" if dev == "mps" else
             f"CPU ({platform.processor() or platform.machine()})")
    print(f"device: {dname}  torch {torch.__version__}\n")
    print(f"{'model':32s} {'lat (ms)':>10s} {'params':>10s} {'p-mem MB':>10s} {'peak MB':>10s}")
    print("-" * 78)

    rows = []
    for name, key, reported, prov, note in TARGETS:
        r = subprocess.run([sys.executable, __file__, "--worker", key],
                           capture_output=True, text=True,
                           env={**os.environ, "TOKENIZERS_PARALLELISM": "false"})
        line = next((l for l in r.stdout.splitlines() if l.startswith("@@JSON@@")), None)
        if line is None:
            print(f"  [warn] {name}: worker failed\n{r.stderr[-600:]}")
            continue
        d = json.loads(line[len("@@JSON@@"):])
        row = {"name": name, "provenance": prov, "note": note,
               "reported_params_in_literature": reported, **d}
        if reported and d["params_total"] < 0.9 * reported:
            row["lower_bound"] = True
            row["fidelity_note"] = (f"rebuild has {d['params_total']/1e6:.1f}M parameters "
                                    f"against {reported/1e6:.0f}M reported for the original; "
                                    f"treat this row as a lower bound on the original's cost")
        rows.append(row)
        flag = "  (lower bound)" if row.get("lower_bound") else ""
        print(f"{name:32s} {d['latency_ms']['mean']:10.2f} {d['params_total']/1e6:9.2f}M "
              f"{d['param_mem_mb']:10.2f} {d['peak_tensor_mem_mb']:10.2f}{flag}")

    by = {r["name"]: r for r in rows}
    p1 = ["Biometric-Quality", "Audio Forensics (ECAPA-TDNN)", "Cross-Modal (Lip-Sync)"]
    p2 = ["Visual (XceptionNet)", "Audio (FreqNet)"]
    have = lambda ks: [k for k in ks if k in by]
    lat = lambda ks: [by[k]["latency_ms"]["mean"] for k in have(ks)]
    prm = lambda ks: sum(by[k]["params_total"] for k in have(ks))
    pkm = lambda ks: sum(by[k]["peak_tensor_mem_mb"] for k in have(ks))

    esc = 0.0879  # measured escalation rate at tau_disagree = 0.30 (Table 13)
    ph = {
        "phase1_only": dict(agents=have(p1), params_total=prm(p1),
                            param_mem_mb=round(prm(p1) * 4 / MB, 2),
                            latency_sequential_ms=round(sum(lat(p1)), 2),
                            latency_parallel_ms=round(max(lat(p1)), 2),
                            summed_peak_tensor_mem_mb=round(pkm(p1), 2)),
        "phase1_plus_2": dict(agents=have(p1 + p2), params_total=prm(p1 + p2),
                              param_mem_mb=round(prm(p1 + p2) * 4 / MB, 2),
                              latency_sequential_ms=round(sum(lat(p1 + p2)), 2),
                              latency_parallel_ms=round(max(lat(p1)) + max(lat(p2)), 2),
                              summed_peak_tensor_mem_mb=round(pkm(p1 + p2), 2)),
    }
    ph["expected_adaptive"] = dict(
        escalation_rate=esc,
        description="phase1 * (1 - escalation) + phase1+2 * escalation",
        latency_sequential_ms=round(ph["phase1_only"]["latency_sequential_ms"] * (1 - esc)
                                    + ph["phase1_plus_2"]["latency_sequential_ms"] * esc, 2),
        latency_parallel_ms=round(ph["phase1_only"]["latency_parallel_ms"] * (1 - esc)
                                  + ph["phase1_plus_2"]["latency_parallel_ms"] * esc, 2),
        param_mem_mb=round((ph["phase1_only"]["param_mem_mb"] * (1 - esc)
                            + ph["phase1_plus_2"]["param_mem_mb"] * esc), 2))

    out = {
        "device": dname, "device_type": dev, "torch": torch.__version__,
        "platform": platform.platform(),
        "protocol": dict(batch_size=1, dtype="float32", n_samples=N_SAMPLES,
                         n_runs_per_sample=N_RUNS, n_warmup=N_WARMUP, seed=SEED,
                         isolation="each model profiled in a fresh subprocess"),
        "caveats": [
            "Latencies exclude preprocessing (MTCNN face detection, librosa feature "
            "extraction, Mel-spectrogram computation) for every row alike.",
            "Rows with provenance=architecture are rebuilt from published architecture "
            "specifications with randomly initialised weights: batch-1 inference cost is "
            "a function of architecture and input shape, not of weight values.",
            "Rows flagged lower_bound have fewer parameters than the original model as "
            "reported in the literature, so the original's true cost is at least the "
            "figure given here.",
            "On Apple Metal, accelerator memory is unified with system memory. "
            "peak_tensor_mem_mb is the maximum live tensor allocation observed by a "
            "2 kHz sampling thread during the timed forwards, not the Metal driver heap "
            "(which is dominated by a one-off allocator reservation). Parameter memory "
            "(fp32) is exact and hardware independent, and is the portable column.",
            "Phase parallel latency is the best case if agents were dispatched "
            "concurrently; the released orchestrator runs them sequentially.",
        ],
        "models": rows, "phases": ph,
    }
    (HERE / "cost_vs_baselines.json").write_text(json.dumps(out, indent=2))
    print("-" * 78)
    for k in ("phase1_only", "phase1_plus_2", "expected_adaptive"):
        v = ph[k]
        print(f"{k:18s} {v['latency_sequential_ms']:7.1f} ms seq / "
              f"{v['latency_parallel_ms']:6.1f} ms par   params-mem {v['param_mem_mb']:7.1f} MB")
    print("\nwrote cost_vs_baselines.json")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        main()
