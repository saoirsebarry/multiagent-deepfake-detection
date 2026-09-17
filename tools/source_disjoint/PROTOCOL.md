# Source-disjoint re-evaluation protocol (pre-registered 2026-09-15, before any run)

**Why.** PolyGlotFake derives about nineteen forgeries from each authentic source video. The
released clip-level partition (seed 42, 70/15/15 within language) places a source's clips in
different partitions: of the 697 test sources that carry a forgery, 480 have their authentic
clip in the training partition and 111 in validation (476 and 111 by the released train list, `make_split.py --audit`; `paper_artifacts/task_23_by_source.py` infers 480 without it).
The reported test accuracy therefore measures within-source generalisation.

**Partition.** `make_split.py` assigns whole source videos (or whole identity groups when
`cluster_identities.py` finds the same face in several sources) to one of train, validation or
test: 70/15/15 by unit within each language, seed 42, fake class downsampled to the real count
in train and validation, test left unbalanced. The manifest hash is recorded before training.

**Training.** Every agent is trained from its pretrained backbone with the released recipe
(`src/agents/visual_xception.py`, `src/agents/audio_freqnet.py`, `tools/train_crossmodal.py`,
`tools/train_biometric.py` with validation-loss selection, `src/agents/audio_forensics_ecapa.py`),
then the released robustness fine-tunes with their validation-only adoption rules, warm-started
from the run's own checkpoints. Seeds 42, 43 and 44; nothing else varies between runs.

**Read-out.** Equal weights. The threshold is set on each run's validation scores by the
released rule (lowest-error band with no missed fake; midpoint). The test partition is read
once per run. Intervals are source-clustered bootstraps beside clip-level ones. Across runs:
mean and standard deviation of accuracy, AUC-ROC and AP; per-agent AUC; McNemar between the
five-agent system and the Phase-1 trio and each leave-one-out configuration, per run.

**Reporting.** Both partitions are reported side by side; the clip-level numbers are kept and
labelled as within-source. No run is discarded, whatever its outcome.
