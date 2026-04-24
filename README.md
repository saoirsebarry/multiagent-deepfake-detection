# A Multi-Agent Framework for Explainable Multi-Modal Deepfake Detection

Code and data release accompanying:

> Saoirse Barry, Pancham Shukla, Susmitha Vekkot.
> *A Multi-Agent Framework with Adaptive Orchestration for Explainable Multi-Modal Deepfake Detection*.
> AI Open, 2026.

**Headline result.** On the PolyGlotFake test set (2,162 samples) the five-agent ensemble achieves AUC-ROC = 1.000, average precision = 1.000, and 100.00 % accuracy at the operating threshold τ = 0.37, with zero false positives and zero false negatives. Real- and fake-class aggregate-score distributions are fully separable (max real = 0.357 < min fake = 0.388).

---

## 1. What is in this release

| Path | Contents |
|---|---|
| [`src/`](src/) | Production source: multi-agent pipeline (`detect.py`), YouTube variant, orchestration scripts, per-agent model definitions, preprocessing, XAI utilities. |
| [`checkpoints/`](checkpoints/) | The five trained agent checkpoints used to produce every number in the paper, plus ECAPA forensic feature-normalisation statistics. Total ~235 MB. |
| [`shape_predictor_68_face_landmarks.dat`](shape_predictor_68_face_landmarks.dat) | dlib 68-point face-landmark predictor, required by the biometric-quality XAI overlay. |
| [`paper_artifacts/`](paper_artifacts/) | All numerical artifacts cited in the paper (JSON/CSV/TeX/PDF/PNG), the scripts that produced them, source CSVs, and a reproducibility harness (`run.sh`, `self_check.py`). |
| [`paper_artifacts/source_csvs/`](paper_artifacts/source_csvs/) | The three saved orchestrator-run CSVs every paper number is derived from. |
| [`requirements.txt`](requirements.txt) | Pinned Python dependencies (Python 3.12). |
| [`LICENSE`](LICENSE) | MIT for source code (see note about third-party assets). |
| [`CITATION.cff`](CITATION.cff) | Machine-readable citation. |

---

## 2. Quick-start

### 2.1 Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Optional (needed only for report-generation / VLM / transcription agents):
#   export GROQ_API_KEY=...        # Llama-3-70B report generator
#   export GEMINI_API_KEY=...      # Gemini-1.5-Flash visual context
#   python -m transformers.utils.download  openai/whisper-tiny ./whisper-tiny-local
```

### 2.2 Reproduce every number in the paper

```bash
cd paper_artifacts
./run.sh          # ~5 min on CPU, produces all JSONs / CSVs / figures / SUMMARY.md
python self_check.py
```

The self-check verifies headline accuracy == bootstrap-point accuracy == ROC/PR operating-point == ablation-baseline accuracy, and fails loudly on any mismatch.

### 2.3 Run the multi-agent pipeline on new video

```bash
# Preprocess a video directory into the .npz format expected by the agents
python src/data_preprocessing/preprocess.py --input_dir /path/to/videos --output_dir data/preprocessed --image_size 299

# Run the adaptive 5-agent pipeline with XAI on the preprocessed data
python src/detect.py --data_dir data/preprocessed
```

Output CSVs are written to `results/` and XAI artifacts to `xai_results/`.

---

## 3. Architecture at a glance

### 3.1 Five detection agents

| Agent | File | Backbone | Total / Trainable | Forensic signal |
|---|---|---|---:|---|
| Visual (Xception) | [`src/agents/visual_xception.py`](src/agents/visual_xception.py) | timm Xception, fine-tuned from block 11 | 24.0 M / 3.2 M | Spatial facial artifacts |
| Audio (FreqNet) | [`src/agents/audio_freqnet.py`](src/agents/audio_freqnet.py) | ResNet-style with `torch.fft.fft2`/`ifft2` | 1.9 M / 1.9 M | Frequency-domain anomalies |
| Cross-modal lip-sync | [`src/agents/cross_modal_lipsync.py`](src/agents/cross_modal_lipsync.py) | MobileNetV2 + BiLSTM + cross-attention | 6.1 M / 3.9 M | Audio-visual sync |
| Biometric-quality | [`src/agents/biometric_quality.py`](src/agents/biometric_quality.py) | EfficientNet-B0 on 5-channel input (RGB + blur-variance + exposure) | 4.4 M / 4.4 M | Image quality + geometry |
| ECAPA forensic | [`src/agents/audio_forensics_ecapa.py`](src/agents/audio_forensics_ecapa.py) | SpeechBrain ECAPA-TDNN (frozen) + 11 hand-crafted features + dual-stream MLP | 20.8 M / 0.04 M | Speaker & prosody forensics |
| **System total** | | | **57.1 M / 13.3 M** | |

The 11 hand-crafted ECAPA features group into 5 prosodic (pitch mean/std/range, RMS energy mean/std), 3 spectral (HF/LF energy ratio, spectral flux, spectral centroid), 2 temporal (inter-window embedding distance mean and std across three sliding windows), and 1 embedding-variance feature.

### 3.2 Adaptive orchestration

```
Phase 1 (always)                    Phase 2 (on disagreement)
┌─────────────────────┐              ┌─────────────────────┐
│ Biometric-Quality   │              │ Visual (XceptionNet)│
│ ECAPA forensic      │              │ Audio (FreqNet)     │
│ Cross-Modal         │──d > 0.30──▶ │                     │
└─────────────────────┘              └─────────────────────┘
         │                                       │
         └────────────── weighted mean ──────────┘
                              │
                     ŝ ≥ 0.37  ⇒  Deepfake
                     ŝ < 0.37  ⇒  Real
```

- **Weights** (selected on validation split):
  `(w_visual, w_freqnet, w_ecapa, w_crossmodal, w_biometric) = (0.20, 0.15, 0.20, 0.25, 0.20)`.
  In Phase 1, the remaining three weights are renormalised to sum to 1.
- **Disagreement metric.** `d = std(phase-1 scores)`. If any two Phase-1 agents disagree on verdict at τ = 0.37, `d ← max(d, 0.30)` — forcing Phase 2.
- **Decision threshold τ = 0.37.** Selected on validation. Accuracy is a plateau over τ ∈ [0.30, 0.40] (all ≥ 99.86 %); τ = 0.5 is not privileged because the aggregate score was never a direct BCE target.

### 3.3 Integrated explainability

| Agent | XAI method | Output |
|---|---|---|
| Visual | GradCAM on last Conv2d | Saliency heatmap over face frames |
| FreqNet | GradCAM on log-Mel spectrogram | Saliency + waveform overlay |
| Cross-Modal | Cross-attention weights | Frame-level attention heatmap |
| Biometric | dlib landmarks + quality overlay | 2×2 figure (face / landmarks / edges / blur-exposure bar) |
| ECAPA | SHAP `KernelExplainer` | Waterfall over 11 forensic features |
| Transcription | Whisper-Tiny | Plain-text transcript |
| Visual context | Gemini 1.5 Flash | Natural-language scene description |
| Report | Llama-3-70B via Groq | Structured forensic summary |

All XAI images produced per inference are placed under `xai_results/{agent}_{basename}.png`.

---

## 4. Reproducing every number and figure in the paper

Everything the paper cites can be regenerated from the three CSVs in `paper_artifacts/source_csvs/`. Run `paper_artifacts/run.sh` and the following are produced:

| Paper claim | Task script | Output file |
|---|---|---|
| Abstract / §4.1 headline (100.00 % / 0 errors at τ = 0.37) | `task_01_headline.py`, `task_05b_ablation_tau037.py` | `headline_metrics.json`, `ablation_table_tau037.csv` |
| AUC = 1.000, AP = 1.000, operating-point marker | `task_02_roc_pr.py` | `roc_curve_system.{pdf,png}`, `pr_curve_system.{pdf,png}` |
| Threshold robustness (Table: accuracy ≥ 99.86 % over [0.30, 0.40]) | `task_03_threshold.py` | `threshold_robustness.csv`, `threshold_robustness_table.tex` |
| 95 % CI on every metric | `task_04_bootstrap.py` | `bootstrap_cis.json` |
| Ablation at τ = 0.37 (Table 10 of paper) | `task_05b_ablation_tau037.py` | `ablation_table_tau037.{csv,tex}` |
| Ablation at τ = 0.5 (threshold-sensitivity study) | `task_05_ablation.py` | `ablation_table.{csv,tex}` |
| 3-agent baseline row of Table 8 at τ = 0.37 | `task_06_three_agent.py` | `three_agent_metrics.json` (also re-check at τ=0.37 in `self_check.py`) |
| Disagreement-threshold sweep at operating τ | `task_07b_disagreement_tau037.py` | `disagreement_sweep_tau037.{csv,tex}` |
| YouTube confusion matrix and metrics at τ = 0.37 | `task_08b_youtube_tau037.py` | `youtube_metrics_tau037.json`, `youtube_confusion_matrix_tau037.{csv,tex}` |
| Per-agent parameter counts (Table 6) | `task_09_params.py` | `parameter_counts.json` |
| Inference latency | `task_10_latency.py` | `latency_benchmark.json` |
| McNemar vs. transformer baselines | `task_11_mcnemar.py` | `mcnemar_tests.json` **— prediction CSVs not saved; see §7 below** |
| Reliability diagram / ECE | `task_12_calibration.py` | `reliability_diagram.{pdf,png}`, `calibration_metrics.json` |
| Paper-ready summary | `task_13_summary.py` | `SUMMARY.md` |

A more detailed mapping between paper tables/figures and artifact files is in [`paper_artifacts/SUMMARY.md`](paper_artifacts/SUMMARY.md).

---

## 5. Data availability

### 5.1 PolyGlotFake

Training / validation / test videos are from the public PolyGlotFake benchmark:

> Y. Wang, L. Ren, H. Lu, Z. Yang, et al. *PolyGlotFake: A Novel Multilingual and Multimodal DeepFake Dataset.* 2024.

Download instructions are in the dataset authors' repository. Once the raw videos are in place, preprocess with:

```bash
python src/data_preprocessing/preprocessed_all_unbalanced.py \
    --real_dir <path/to/real> \
    --fake_dir <path/to/fake> \
    --output_dir data/polyglot_processed_all_unbalanced
```

The resulting `.npz` files (one per clip, containing `faces`, `waveform`, `label`) are consumed directly by the agent scripts.

Our identity-based test split has 118 real + 2,044 fake = **2,162 samples**. The exact filepath list is in `paper_artifacts/source_csvs/analysis_results_with_5_agents.csv` (column `filepath`).

### 5.2 YouTube distribution-shift benchmark

49 clips drawn from BBC News livestreams (real) and HeyGen sample content (fake). Per-clip URLs and timestamps are listed in `paper_artifacts/source_csvs/analysis_results_with_5_agents_orchestration.csv` (column `filepath`); raw videos are not redistributed (copyright).

---

## 6. Hyperparameters (as coded)

### 6.1 Decision pipeline

| Parameter | Value | Source |
|---|---|---|
| Decision threshold τ | 0.37 | [`src/detect.py:81`](src/detect.py#L81) |
| Disagreement threshold τ_d | 0.30 | [`src/detect.py:85`](src/detect.py#L85) |
| Agent weights | (0.20, 0.15, 0.20, 0.25, 0.20) | [`src/orchestrator.py:82-88`](src/orchestrator.py#L82) |
| Aggregation | Weighted mean, renormalised | [`src/orchestrator.py:635-641`](src/orchestrator.py#L635) |

### 6.2 Per-agent training

| Agent | Optimizer | LR | Epochs | Batch | Notes |
|---|---|---|---|---|---|
| Xception stage 1 (head) | AdamW (wd 1e-4) | 1e-3 | 5 | 32 | ImageNet backbone frozen, classifier dropout 0.5 / 0.35 |
| Xception stage 2 (fine-tune) | AdamW, OneCycleLR | 1e-5 | 20 | 32 | Unfreeze from block 11, Mixup, label smoothing 0.1 |
| FreqNet | AdamW, OneCycleLR | 1e-3 | 50 | 16 | 224 Mel bands, SpecAugment, early-stop patience 10 |
| Cross-modal | Adam, ReduceLROnPlateau | 1e-4 (base) / 1e-5 (fine-tune) | 50 | 8 | patience 10, 20 max faces, 128×313 audio Mel |
| Biometric stage 1 | Adam, CosineAnnealingLR | 1e-3 | 50 | 32 | EfficientNet-B0 backbone frozen, 5-channel adapter |
| Biometric stage 2 | Adam | 1e-5 | 20 | 32 | Unfreeze at epoch 5, patience 7 on val AUC |
| ECAPA head | AdamW, OneCycleLR (max 1e-3) | 1e-4 | 50 | 16 | ECAPA-TDNN backbone frozen, patience 7 on val AUC |

### 6.3 Preprocessing

- Face detection: MTCNN with `confidence > 0.95` ([`src/data_preprocessing/preprocess.py:66`](src/data_preprocessing/preprocess.py#L66))
- Frame stride: 10 ([`src/data_preprocessing/preprocess.py:201`](src/data_preprocessing/preprocess.py#L201))
- Audio: `librosa.load(..., sr=16000, mono=True)`, padded/truncated to 5 s (80,000 samples) at inference time
- Image size: 299 × 299 for Xception / Biometric, 224 × 224 for Cross-Modal / FreqNet (log-Mel)

---

## 7. Known limitations and unresolved items

The evaluation is complete except where explicitly noted below. These notes match the "Unresolved items" section in `paper_artifacts/SUMMARY.md`:

1. **McNemar tests against transformer baselines could not be completed in this release** because per-sample prediction CSVs were not saved for GenConViT AE/VAE, LIPINC-V2, the Custom ViT late-fusion model, and the Hybrid CNN-Transformer. The comparison code in `paper_artifacts/task_11_mcnemar.py` is ready; it will run the 2 × 2 exact-binomial test as soon as prediction CSVs aligned to the test `filepath` column are dropped into the expected location. See [`paper_artifacts/task_11_mcnemar.py`](paper_artifacts/task_11_mcnemar.py) for the expected schema.
2. **Latency numbers in `paper_artifacts/latency_benchmark.json` are MPS-timed** (Apple Silicon) because no CUDA GPU was available at the time of artifact generation. Rerun `task_10_latency.py` on the deployment GPU to refresh the numbers. Per-agent forward-pass latency is reproducible up to hardware noise; the parallel/sequential ratio is architectural.
3. **YouTube evaluation uses 49 parseable rows**, not the 50 sometimes mentioned in earlier draft material. All 49 appear with their final scores in `paper_artifacts/source_csvs/analysis_results_with_5_agents_orchestration.csv`.
4. **Calibration is deliberately imperfect**: ECE ≈ 0.10 at 10 bins. The aggregate score is a weighted mean of five independently trained sigmoids, not a jointly calibrated probability. Platt or isotonic scaling on a held-out split would restore calibrated probabilities if a downstream use case demanded it; the paper claims discrimination only.
5. **The orchestrator currently executes Phase-1 agents sequentially** in `_run_first_wave`. The agents share no state and are trivially parallelisable; the "parallel" latency numbers in `latency_benchmark.json` assume a user-provided parallel dispatch.

---

## 8. Provenance of the headline numbers

Every number cited in the paper traces back to one of three CSVs:

| CSV | Produces | Used for |
|---|---|---|
| `paper_artifacts/source_csvs/analysis_results_with_5_agents.csv` | 2,162-row PolyGlotFake test set with 5-agent per-sample scores and orchestrator `final_score` (weighted mean) | Headline, AUC, AP, CIs, ablation, threshold sweep, disagreement sweep, calibration |
| `paper_artifacts/source_csvs/analysis_results_with_3_agents.csv` | 2,162-row PolyGlotFake test set with 3-agent Phase-1 scores | 3-agent comparison row of Table 8 |
| `paper_artifacts/source_csvs/analysis_results_with_5_agents_orchestration.csv` | 49-row YouTube benchmark with 5-agent scores, recorded phase, consensus rate, and analysis time | YouTube evaluation (§4.10), distribution-shift analysis |

Each CSV can be regenerated by running the matching orchestrator script in `src/` against the preprocessed dataset. The published CSVs are the exact files used to produce the paper; checksums are recorded in `paper_artifacts/source_csvs/.sha256`.

---

## 9. Citing this work

If you use this code or the artifact package in academic work, please cite:

```bibtex
@article{barry2026multiagent,
  title   = {A Multi-Agent Framework with Adaptive Orchestration for Explainable Multi-Modal Deepfake Detection},
  author  = {Barry, Saoirse and Shukla, Pancham and Vekkot, Susmitha},
  journal = {AI Open},
  year    = {2026},
  note    = {Code and data release: \url{https://github.com/...}}
}
```

A machine-readable version is in [`CITATION.cff`](CITATION.cff).

---

## 10. License

Source code: MIT ([`LICENSE`](LICENSE)).

Third-party assets distributed with this release retain their original licenses:
- `shape_predictor_68_face_landmarks.dat` — Boost Software License 1.0 (dlib).
- PolyGlotFake videos are NOT redistributed here; users must obtain them from the dataset authors.
- Pretrained backbones (ECAPA-TDNN from SpeechBrain, ImageNet weights for Xception / MobileNetV2 / EfficientNet-B0) are downloaded at runtime from their respective upstream repositories and retain their original licenses.

---

## 11. Contact

Corresponding author: Saoirse Barry (saoirsebarry02@gmail.com). Department of Computing, Imperial College London.

Bug reports and questions: open an issue on the companion GitHub repository (URL in CITATION).
