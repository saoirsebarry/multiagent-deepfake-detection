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
python src/agents/visual_xception.py \
    --data_dir data/polyglot_processed_all_unbalanced

python src/agents/audio_freqnet.py \
    --data_dir data/polyglot_processed_all_unbalanced

python src/agents/cross_modal_lipsync.py \
    --data_dir data/polyglot_processed_all_unbalanced

python src/agents/biometric_quality.py \
    --data_dir data/polyglot_processed_all_unbalanced

python src/agents/audio_forensics_ecapa.py \
    --data_dir data/polyglot_processed_all_unbalanced
```

Each training script writes its best checkpoint to the corresponding directory under `checkpoints/`. Re-running with the same seeds (seed 42 throughout) reproduces within float noise.

## Path C — Re-run the orchestrator against the preprocessed test set

This regenerates the three source CSVs and in turn every downstream artifact.

```bash
# 5-agent PolyGlotFake run (produces analysis_results_with_5_agents.csv)
python src/orchestrator.py \
    --data_dir data/polyglot_processed_all_unbalanced \
    --output_file paper_artifacts/source_csvs/analysis_results_with_5_agents.csv

# YouTube run (produces analysis_results_with_5_agents_orchestration.csv)
python src/orchestrator_adaptive.py \
    --data_dir data/youtube_processed \
    --output_file paper_artifacts/source_csvs/analysis_results_with_5_agents_orchestration.csv

# Full 5-agent pipeline with XAI artifacts
python src/detect.py \
    --data_dir data/polyglot_processed_all_unbalanced \
    --xai_output_dir xai_results/
```

After re-running C, run Path A to regenerate all downstream artifacts from the fresh CSVs.

## Path D — Reproduce the operating-point provenance (weights + τ "selected on validation")

The paper selects the decision threshold τ = 0.37 (and validates the agent weights) on the
**validation** split, then freezes them and reports test accuracy. This path makes that
claim reproducible end-to-end: it derives τ from the validation scores only, and never
uses the test labels to choose anything.

```bash
# 1. Score the validation split with the SAME weighted orchestrator that wrote the test CSV.
#    (Requires the preprocessed val/ split from Path B and the checkpoints.)
python src/orchestrator.py --split val \
    --output_file paper_artifacts/source_csvs/analysis_results_with_5_agents_VAL.csv

# 2. Derive τ on validation, freeze (weights, τ), and evaluate once on test.
#    Stdlib only — no GPU, no heavy deps.
python paper_artifacts/task_00_select_operating_point.py \
    --val  paper_artifacts/source_csvs/analysis_results_with_5_agents_VAL.csv \
    --test paper_artifacts/source_csvs/analysis_results_with_5_agents.csv
```

This writes `paper_artifacts/operating_point_provenance.json` containing: per-agent
validation AUCs, the validation-derived threshold `tau_star` and the rule used to pick it
(argmax validation balanced accuracy; midpoint of the tied plateau), the frozen test
metrics at `tau_star`, and a `reproduces_paper_tau` flag. Ship both the
`analysis_results_with_5_agents_VAL.csv` and `operating_point_provenance.json` so a third
party can verify the operating point was selected on validation rather than tuned on test.

`--weights {paper,auc,search}` reports how the paper's hand-picked weights compare on the
validation split against an AUC-proportional vector and a simplex margin search. The script
is honest by construction: if the validation-derived `tau_star` is not ≈ 0.37, it says so.

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
