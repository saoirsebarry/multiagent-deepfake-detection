# Revision notebooks

Two Colab notebooks supporting the reviewer response for the *Informatics* submission.

| Notebook | Reviewer comment | Produces |
|---|---|---|
| `colab_A_training_curves.ipynb` | 4 — training/validation loss curves for all agents | `training_curves.json`, `convergence_summary.csv`, `figure_training_curves.png` |
| `colab_B_deepfake_eval_2024.ipynb` | 5 — in-the-wild generalisation | `deepfake_eval_2024_results.json`, `dfe2024_comparison.csv`, `dfe2024_coverage.json` |

Open either directly in Colab:

- [notebook A](https://colab.research.google.com/github/saoirsebarry/multiagent-deepfake-detection/blob/main/notebooks/colab_A_training_curves.ipynb)
- [notebook B](https://colab.research.google.com/github/saoirsebarry/multiagent-deepfake-detection/blob/main/notebooks/colab_B_deepfake_eval_2024.ipynb)

## Two things that will bite you otherwise

**The training scripts do not share a data flag.** `visual_xception.py` and
`cross_modal_lipsync.py` hard-code a path relative to the working directory,
`audio_forensics_ecapa.py` takes no data flag at all and reads its own `CONFIG`, while only
`audio_freqnet.py` (`--dataroot`) and `biometric_quality.py` (`--data_dir`) accept one. The
notebooks satisfy every convention with symlinks rather than editing five scripts.

**`orchestrator.py` takes no arguments.** It reads `CONFIG["data_dir"]/test` and writes
`analysis_results_with_5_agents.csv` into the working directory. Flags passed on the command
line are silently ignored, so pointing it at a new dataset with `--data_dir` will quietly
evaluate whatever sits at the hard-coded path instead. Notebook B redirects that path with a
symlink and asserts the output row count matches the benchmark it just preprocessed.
