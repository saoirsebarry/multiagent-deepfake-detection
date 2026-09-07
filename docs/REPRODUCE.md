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
# There is no --real_dir / --fake_dir. The script reads one dataset root that must
# contain json_file/, real/ and fake/ (the layout of the PolyGlotFake release).
#
# --splits limits which splits are WRITTEN; the train/val/test partition is always
# computed over the full file list, so restricting it does not change which clip lands
# in which split. No training script reads test/, so 'train,val' is enough to retrain
# and saves roughly 8-10 GiB.
#
# --limit N stops after N new clips; re-running resumes, because clips already written
# are skipped. --min_free_gib aborts before filling the output volume.
# --workers shards whole clips across processes, each building its own MTCNN.
# 1 (the default when the flag is absent) is the original serial loop and is the
# reference path. Writes are staged and renamed, so an interrupted run leaves a
# .partial file the resume scan ignores rather than a truncated .npz it would
# treat as done. --verify_existing cleans up truncated files left by older runs.
python src/data_preprocessing/preprocessed_all_unbalanced.py \
    --data_dir   <path/to/PolyGlotFake> \
    --output_dir data/polyglot_processed_all_unbalanced \
    --splits     train,val \
    --workers    4

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

# ECAPA-TDNN. Precompute the feature vectors first: the script otherwise re-runs four
# SpeechBrain encodes and a librosa.pyin per sample on every epoch, which dominates the
# stage. The cached vectors are identical to the ones the live dataset produces.
python tools/precompute_ecapa_features.py \
    --data_dir data/polyglot_processed_all_unbalanced \
    --out_dir data/ecapa_features --splits train val --workers 6

python src/agents/audio_forensics_ecapa.py \
    --data_dir data/polyglot_processed_all_unbalanced \
    --feature_cache data/ecapa_features \
    --output_dir audio_forensic_trained_models

# takes --dataroot
python src/agents/audio_freqnet.py \
    --dataroot data/polyglot_processed_all_unbalanced

# takes --data_dir
python tools/train_biometric.py \
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

# YouTube evaluation set (produces analysis_results_youtube.csv)
# 1. download the 25 source videos listed in youtube_eval/manifest.json (yt-dlp,
#    bestvideo+bestaudio merged to mp4, named "<title>_<youtube_id>.mp4");
# 2. extract the gated clips; 3. score them with the released system; 4. attach the
#    frozen per-clip labels from youtube_eval/clip_provenance.json.
python youtube_eval/extract_clips.py --videos_dir videos/ --labels_json labels.json --out_dir yt_npz/
ls yt_npz/*.npz > clip_list.txt
python youtube_eval/score_clips.py clip_list.txt yt_scores.csv

# Full 5-agent pipeline with XAI artifacts
python src/detect.py \
    --data_dir data/polyglot_processed_all_unbalanced \
    --xai_output_dir xai_results/
```

After re-running C, run Path A to regenerate all downstream artifacts from the fresh CSVs.

## Path D — Reproduce the operating-point provenance (validation-selected weights)

The paper selects the agent weight vector on the **validation** partition by exhaustive
grid search (0.05-step simplex, all five agents active; minimise validation errors of the
deployed tiered verdict at the conventional τ = 0.5, tie-break on the two-sided clearance
around the threshold), then freezes it and reports the
single test read-out. This path reruns that selection from the released validation scores
and asserts it lands on the released vector.

```bash
python paper_artifacts/task_00_select_operating_point.py
```

This reruns the released grid search on `analysis_results_VAL.csv`, fails loudly if the
argmax is not the released vector, and writes
`paper_artifacts/operating_point_provenance.json`: the grid size, the number of vectors
that classify validation perfectly (102 of 3,876), the selected vector's validation margin
and band (the conventional τ = 0.5 sits inside it), and the frozen test read-out.

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
- **Segmentation fault in ECAPA-TDNN before the first training step.** The librosa wheel ships a precompiled numba kernel cache that faults when loaded against a mismatched NumPy binary interface, inside `pyin`'s interpolation gufunc. Point the cache somewhere writable and empty: `export NUMBA_CACHE_DIR=/tmp/numba_cache`. The training script sets this itself; set it manually if you call the feature code directly.
- **MPS `Symbol not found` during torchvision import**. Use Python 3.12; older Python versions ship torchvision wheels that are out of sync with torch 2.x.
- **Whisper-Tiny path not found**. Run `python -c "from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor; AutoModelForSpeechSeq2Seq.from_pretrained('openai/whisper-tiny').save_pretrained('./whisper-tiny-local'); AutoProcessor.from_pretrained('openai/whisper-tiny').save_pretrained('./whisper-tiny-local')"` from the publication root.
- **`GROQ_API_KEY` / `GEMINI_API_KEY` not set**. The headline detection agents do NOT require these; only the natural-language report and VLM context generation do. The pipeline degrades gracefully and emits a warning when they are unset.

## Training curves

Per-epoch train/validation loss for all five agents is regenerated from the recovered logs by:

```bash
python paper_artifacts/task_20_training_curves_figure.py \
    --recovered paper_artifacts/recovered_curves.json \
    --biometric paper_artifacts/biometric_training_history.json \
    --ecapa_csv paper_artifacts/ecapa_training_log.csv \
    --out paper_artifacts/training_curves_all_agents
```

`recovered_curves.json` holds the XceptionNet, FreqNet and Cross-Modal histories parsed from their
re-run logs; `biometric_training_history.json` is written by `tools/train_biometric.py`;
`ecapa_training_log.csv` is written by the ECAPA trainer above.

## Checkpoints

`checkpoints/` ships the released checkpoint set — the one that produced every released
score file: XceptionNet, FreqNet, Cross-Modal and the ECAPA head trained with seed 42 in
the pinned environment, and the Biometric-Quality agent trained by
`tools/train_biometric.py` (per-pixel forensic-map channels; two-stage recipe with a
validation-selected fine-tune). Run the system with them via:

```bash
python src/orchestrator.py --split test --output_file /tmp/rescored.csv
```

This configuration is verified reproducible: re-scoring the released test partition
reproduces `paper_artifacts/source_csvs/analysis_results_with_5_agents.csv`
(AUC-ROC 0.99999, 99.91% accuracy at τ = 0.5). In an 8-clip cross-machine spot-check, 39 of
40 per-agent scores matched within 0.02. `SHA256SUMS` lists the checkpoint digests.

The released per-sample score CSVs in `paper_artifacts/source_csvs/` are the authoritative
record behind every number in the paper, and every reported figure re-derives from them
deterministically via `paper_artifacts/`.
