# Reproducibility guide

Every number, table, and figure cited in the paper can be regenerated deterministically from the three CSVs in `paper_artifacts/source_csvs/`. This guide walks through the three reproduction paths.

## Path A — Just reproduce paper numbers (fastest, ~5 min CPU)

No GPU required. No raw video required. Just the three saved CSVs and the artifact scripts.

```bash
cd publication
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd paper_artifacts
bash run.sh
python self_check.py      # cross-task consistency check; must print "self-check OK"
```

This produces (or overwrites) every file listed in README §4 inside `paper_artifacts/`.

The self-check is the primary correctness oracle: it loads all JSON outputs and verifies that headline accuracy equals bootstrap-point accuracy equals ROC/PR operating-point accuracy equals ablation-baseline accuracy, to within 1e-4.

## Path B — Reproduce checkpoints from scratch

Requires the PolyGlotFake dataset (see README §5.1) and a GPU.

```bash
# 1. Preprocess
python src/data_preprocessing/preprocessed_all_unbalanced.py \
    --real_dir   <path/to/real> \
    --fake_dir   <path/to/fake> \
    --output_dir data/polyglot_processed_all_unbalanced

# 2. Train each agent
#
# The five scripts do NOT share a common data flag. Two take one, one takes none
# at all, and two hard-code a path relative to the working directory. Satisfy both
# hard-coded conventions with symlinks from the repo root, then every script runs
# unmodified. Each expects train/ and val/ subdirectories.

ln -sfn "$PWD/data/polyglot_processed_all_unbalanced" polyglot_processed_all_unbalanced

# no CLI flag - reads ./polyglot_processed_all_unbalanced
python src/agents/visual_xception.py

# no CLI flag - reads ./polyglot_processed_all_unbalanced
python src/agents/cross_modal_lipsync.py

# no data flag at all - reads data/polyglot_processed_all_unbalanced from CONFIG
python src/agents/audio_forensics_ecapa.py

# takes --dataroot
python src/agents/audio_freqnet.py \
    --dataroot data/polyglot_processed_all_unbalanced

# takes --data_dir
python src/agents/biometric_quality.py \
    --data_dir data/polyglot_processed_all_unbalanced
```

Each training script writes its best checkpoint to the corresponding directory under `checkpoints/`. Re-running with the same seeds (seed 42 throughout) reproduces within float noise.

## Path C — Re-run the orchestrator against the preprocessed test set

This regenerates the three source CSVs and in turn every downstream artifact.

```bash
# 5-agent PolyGlotFake run (produces analysis_results_with_5_agents.csv)
#
# orchestrator.py takes NO arguments: it reads CONFIG["data_dir"]/test and writes
# analysis_results_with_5_agents.csv into the working directory. Flags passed on the
# command line are silently ignored, so a stale tree at the hard-coded path will be
# evaluated instead - always check the row count of the CSV it produces.
python src/orchestrator.py
cp analysis_results_with_5_agents.csv paper_artifacts/source_csvs/

# YouTube run (produces analysis_results_with_5_agents_orchestration.csv)
# orchestrator_adaptive.py likewise takes no arguments; point CONFIG["data_dir"] at
# the YouTube tree (or symlink it) before running.
python src/orchestrator_adaptive.py
cp analysis_results_with_5_agents_orchestration.csv paper_artifacts/source_csvs/

# Full 5-agent pipeline with XAI artifacts
python src/detect.py \
    --data_dir data/polyglot_processed_all_unbalanced \
    --xai_output_dir xai_results/
```

After re-running C, run Path A to regenerate all downstream artifacts from the fresh CSVs.

## Known-good environment

- Python 3.12
- macOS 14+ with Apple Metal, or Linux with CUDA 11.8+
- Dependencies: pinned in `requirements.txt`

The paper's accuracy / AUC / AP / F1 numbers are invariant across CPU/GPU/MPS; only `latency_benchmark.json` is device-dependent.

## Randomness

- All Python `random`, `numpy`, and `torch` seeds are fixed to 42 in evaluation scripts.
- Bootstrap uses `numpy.random.default_rng(42)` — 10,000 iterations, identical across machines.
- Training scripts also fix seed 42 but neural-network training involves small device-dependent nondeterminism; the saved checkpoints in `checkpoints/` are the authoritative artifacts used in the paper.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'speechbrain'`**. Install `speechbrain>=1.0`; ECAPA-TDNN backbone is loaded from HuggingFace Hub at first run.
- **`checkpoints/speechbrain_cache/...` download errors**. Set `HF_TOKEN` env var to avoid rate limits; or pre-populate the cache by running `python -c "from speechbrain.inference import EncoderClassifier; EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir='checkpoints/speechbrain_cache')"`.
- **MPS `Symbol not found` during torchvision import**. Use Python 3.12; older Python versions ship torchvision wheels that are out of sync with torch 2.x.
- **Whisper-Tiny path not found**. Run `python -c "from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor; AutoModelForSpeechSeq2Seq.from_pretrained('openai/whisper-tiny').save_pretrained('./whisper-tiny-local'); AutoProcessor.from_pretrained('openai/whisper-tiny').save_pretrained('./whisper-tiny-local')"` from the publication root.
- **`GROQ_API_KEY` / `GEMINI_API_KEY` not set**. The headline detection agents do NOT require these; only the natural-language report and VLM context generation do. The pipeline degrades gracefully and emits a warning when they are unset.
