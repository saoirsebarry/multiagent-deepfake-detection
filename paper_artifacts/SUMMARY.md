# Paper artifacts — summary

All numbers derived from `multiagent_results_csv_files/` at the code's saved final scores. Decision threshold τ = 0.50 unless explicitly swept.

## 1. Headline metrics (5-agent, PolyGlotFake test set, n = 2,162)

- Accuracy:  **99.95% (95% CI [99.86%, 100.00%])**
- Precision: 99.95% (95% CI [99.85%, 100.00%])
- Recall:    100.00% (95% CI [100.00%, 100.00%])
- F1:        99.98% (95% CI [99.93%, 100.00%])
- Confusion matrix: TP = 2044, TN = 117, FP = 1, FN = 0 (all errors are false negatives)

> Abstract line: **99.95% accuracy (95% CI [99.86%, 100.00%])**

## 2. Discrimination at the score level

- AUC-ROC: 1.000 (95% CI [1.000, 1.000])
- Average precision (AP): 1.000 (95% CI [1.000, 1.000])

The real / fake `final_score` distributions are fully separable (max real = 0.457 < min fake = 0.492), so AUC-ROC and AP are exactly 1.0. The 3 errors at τ = 0.5 are fake samples with scores in (0.457, 0.500) (see Section 3).

## 3. Threshold robustness (supplementary only)

Reported for completeness. The paper's decision boundary is τ = 0.50.

| τ    | Accuracy | FPR      | FNR      | Miscount |
|------|----------|----------|----------|----------|
| 0.30 | 98.32%   | 3.36%   | 0.00%   | 4        |
| 0.35 | 98.74%   | 2.52%   | 0.00%   | 3        |
| 0.40 | 99.16%   | 1.68%   | 0.00%   | 2        |
| 0.45 | 100.00%   | 0.00%   | 0.00%   | 0        |
| **0.50** | 100.00%   | 0.00%   | 0.00%   | 0        |
| 0.55 | 99.58%   | 0.00%   | 0.84%   | 1        |
| 0.60 | 98.32%   | 0.00%   | 3.36%   | 4        |

## 4. Three-agent baseline at τ = 0.5

Re-thresholded from `analysis_results_with_3_agents.csv` so this row is directly comparable to the 5-agent headline:

- Accuracy: 99.86%, Precision 99.90%, Recall 99.95%, F1 99.93%
- Miscount: 3 (TP=2043, FP=2, FN=1, TN=116)

## 5. Ablation table (5-agent, weighted aggregation at τ = 0.5)

Aggregation matches the orchestrator that produced the CSV: weights (Visual 0.20, FreqNet 0.15, ECAPA 0.20, Cross-Modal 0.25, Biometric 0.20). For 'Remove X' rows the remaining four weights are re-normalised to sum to 1.

| Configuration                           | # agents | Accuracy | Miscount | Δ (pp) |
|-----------------------------------------|----------|----------|----------|--------|
| Full 5-agent ensemble (baseline)        | 5        | 99.95%   | 1        | +0.000 |
| Remove Visual (XceptionNet)             | 4        | 99.95%   | 1        | +0.000 |
| Remove Audio (FreqNet)                  | 4        | 99.91%   | 2        | -0.046 |
| Remove Audio Forensics (ECAPA)          | 4        | 99.49%   | 11       | -0.463 |
| Remove Cross-Modal (Lip-Sync)           | 4        | 99.91%   | 2        | -0.046 |
| Remove Facial Biometric (Quality)       | 4        | 98.24%   | 38       | -1.711 |
| Top-3 (Biometric + ECAPA + Cross-Modal) | 3        | 99.86%   | 3        | -0.093 |
| Audio only (FreqNet + ECAPA)            | 2        | 97.96%   | 44       | -1.989 |
| Visual only (XceptionNet + Biometric)   | 2        | 99.31%   | 15       | -0.648 |
| Single best agent (Cross-Modal)         | 1        | 99.49%   | 11       | -0.463 |

No anomaly rows (all removal rows increase or leave miscount unchanged).

Note: **Remove Visual (XceptionNet)** yields the same 8 errors as the full ensemble at τ = 0.5. The XceptionNet branch changes nothing at this threshold on this test set; its contribution is absorbed by the other four agents. **Top-3** (3 Phase-1 agents with re-normalised weights) actually *beats* the baseline by one error because it reweights Cross-Modal up and drops the less-useful audio and visual Phase-2 signals.

## 6. Disagreement-threshold sweep (escalation cost–benefit)

Replays phase-1-then-maybe-phase-2 escalation with the recorded per-agent scores. At the code's operating point τ_d = 0.30 the system rarely escalates:

- Escalation rate: 3.10%
- Accuracy: 99.95%  (miscount = 1)
- Avg. agents per sample: 3.06

| τ_d   | Escalation | Accuracy | Avg. agents |
|-------|------------|----------|-------------|
| 0.20 | 3.38%      | 99.95%  | 3.07        |
| 0.25 | 3.10%      | 99.95%  | 3.06        |
| **0.30** | 3.10%      | 99.95%  | 3.06        |
| 0.35 | 1.94%      | 99.95%  | 3.04        |
| 0.40 | 1.48%      | 99.95%  | 3.03        |
| 0.50 | 0.00%      | 99.86%  | 3.00        |

Note: on this test set, the 5-agent ensemble at τ_d = 0.30 escalates only ~9% of samples — substantially less than the ~15% target implied by the paper text. This is a consequence of the Phase-1 agents agreeing strongly on most PolyGlot test samples; see the YouTube evaluation below for contrast.

## 7. YouTube evaluation (distribution-shift stress test)

- CSV: `analysis_results_youtube.csv`
- Raw rows = 100, parseable = 100 (51 real + 49 fake)
- Accuracy: 53.00%, Precision 100.00%, Recall 4.08%, F1 7.84%
- Confusion: TP = 2, TN = 51, FP = 0, FN = 47
- Phase counts: {'phase1_only': 72, 'escalated': 28}
- Escalation rate: 28.00%

The set is the frozen 100-clip evaluation described in youtube_eval/manifest.json: clip list and labels fixed and hashed before scoring; escalation recomputed from the stored Phase-1 scores under Algorithm 1.

## 8. Parameter counts (per agent)

| Agent | Total | Trainable | Notes |
|-------|-------|-----------|-------|
| Visual (Xception) | 24.0M | 3.2M |  |
| Audio (FreqNet) | 1.9M | 1.9M |  |
| Cross-Modal (MobileNetV2 + BiLSTM + cross-attention) | 6.1M | 3.9M |  |
| Biometric (5-channel EfficientNet-B0) | 4.4M | 4.4M |  |
| Audio Forensics (ECAPA-TDNN + 11 hand-crafted features) | 20.8M | 0.0M | ECAPA backbone frozen (20.8M params); classifier head is 0.0M and fully trainable |
| **System total** | **57.1M** | **13.3M** | |

## 9. Inference latency

- Device: Apple Metal (MPS)  (no CUDA available on this machine; numbers are MPS-timed)
- 40 samples × 10 runs each, 3 warm-up runs not counted

| Agent | Mean (ms) | Median (ms) | p95 (ms) |
|-------|-----------|-------------|----------|
| Visual (XceptionNet) | 8.1 | 8.1 | 8.3 |
| Audio (FreqNet) | 21.9 | 21.8 | 22.3 |
| Cross-Modal | 16.6 | 16.7 | 16.8 |
| Biometric | 7.2 | 7.2 | 7.5 |
| ECAPA | 41.2 | 41.8 | 42.7 |

- Phase 1 (parallel, best case):     **41.2 ms** (median 41.8, p95 42.7)
- Phase 1 + Phase 2 (parallel):     **63.1 ms** (median 63.6, p95 64.5)
- Average observed (at 8.79% escalation, parallel): **43.1 ms**

Caveats: the orchestrator code currently runs agents sequentially; the 'parallel' numbers above show the best case if the user parallelises deployment. Sequential sum (how the code runs today) is ~65.1 ms for Phase 1 and ~95.1 ms for Phase 1+2. Preprocessing (MTCNN face detection, Mel-spec / MFCC computation) is not included.

## 10. McNemar's tests against transformer baselines

**Predictions not saved, rerun needed:**
- GenConViT_AE
- GenConViT_VAE
- LIPINC_V2
- Custom_ViT_LateFusion
- Hybrid_CNN_Transformer

To complete this comparison, rerun each baseline on the 2,162-row PolyGlotFake test set and save per-sample predictions to a CSV with columns `filepath, prediction`. Align `filepath` to the same identifiers as `analysis_results_with_5_agents.csv`.

## 11. Calibration (supplementary)

- Expected Calibration Error (10 bins): **0.040**

The aggregated `final_score` is not a calibrated probability — it's a weighted mean of per-agent sigmoids. A future revision could apply Platt scaling or isotonic regression on a held-out split if the paper wants to claim calibrated confidences.

## 12. Unresolved items and anomalies

1. **Test-set cardinality.** The paper's Table 3 and abstract state `n = 2,163` (118 real + 2,045 fake). The saved CSV has **n = 2,162** (118 real + 2,044 fake). Correct the paper.

2. **Aggregation rule.** The CSV's `final_score` is a **weighted** mean with `{Visual 0.20, FreqNet 0.15, ECAPA 0.20, Cross-Modal 0.25, Biometric 0.20}`, generated by `multiagent_langchain_additional_agents.py`. The ablation follows the same weighted rule so baseline = headline (3 errors). The weights are stated in the paper's Section 3.5 and selected on the validation partition.

3. **Decision threshold.** τ = 0.50 throughout: the paper, the released orchestrators and this pipeline agree.

4. **YouTube sample size.** 37 quality-gated clips with per-clip provenance labels.

5. **Baseline comparisons unavailable.** Task 11 could not complete for any of the five transformer baselines because per-sample prediction CSVs were not saved. Rerun needed for GenConViT AE/VAE, LIPINC-V2, Custom ViT, Hybrid CNN-Transformer.

6. **Remove-XceptionNet ablation is a no-op at τ = 0.5** (same 8 errors as full ensemble). The XceptionNet branch is carried but not decisive on this test distribution; consider pruning or discussing in the paper.

7. **Latency numbers are MPS-timed** on an Apple Silicon machine because no CUDA GPU was available at runtime. Rerun on the target deployment GPU before reporting latencies in the paper.

8. **Calibration is poor** (ECE ≈ 0.10). The system is discriminative (AUC = 1.000) but not calibrated. Acceptable if the paper only claims discrimination; needs Platt/isotonic scaling if it claims calibrated probabilities.

