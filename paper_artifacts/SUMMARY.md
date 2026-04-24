# Paper artifacts — summary

All numbers derived from `multiagent_results_csv_files/` at the code's saved final scores. Decision threshold τ = 0.50 unless explicitly swept.

## 1. Headline metrics (5-agent, PolyGlotFake test set, n = 2,162)

- Accuracy:  **99.63% (95% CI [99.35%, 99.86%])**
- Precision: 100.00% (95% CI [100.00%, 100.00%])
- Recall:    99.61% (95% CI [99.31%, 99.85%])
- F1:        99.80% (95% CI [99.66%, 99.93%])
- Confusion matrix: TP = 2036, TN = 118, FP = 0, FN = 8 (all errors are false negatives)

> Abstract line: **99.63% accuracy (95% CI [99.35%, 99.86%])**

## 2. Discrimination at the score level

- AUC-ROC: 1.000 (95% CI [1.000, 1.000])
- Average precision (AP): 1.000 (95% CI [1.000, 1.000])

The real / fake `final_score` distributions are fully separable (max real = 0.357 < min fake = 0.388). The 8 errors at τ = 0.5 are fake samples with scores in (0.388, 0.500) and disappear at τ = 0.37 (see §3).

## 3. Threshold robustness (supplementary only)

Reported for completeness. The paper's decision boundary is τ = 0.50.

| τ    | Accuracy | FPR      | FNR      | Miscount |
|------|----------|----------|----------|----------|
| 0.30 | 99.86%   | 2.54%   | 0.00%   | 3        |
| 0.35 | 99.91%   | 1.69%   | 0.00%   | 2        |
| 0.40 | 99.86%   | 0.00%   | 0.15%   | 3        |
| 0.45 | 99.72%   | 0.00%   | 0.29%   | 6        |
| **0.50** | 99.63%   | 0.00%   | 0.39%   | 8        |
| 0.55 | 99.26%   | 0.00%   | 0.78%   | 16        |
| 0.60 | 98.66%   | 0.00%   | 1.42%   | 29        |

## 4. Three-agent baseline at τ = 0.5

Re-thresholded from `analysis_results_with_3_agents.csv` so this row is directly comparable to the 5-agent headline:

- Accuracy: 99.77%, Precision 100.00%, Recall 99.76%, F1 99.88%
- Miscount: 5 (TP=2039, FP=0, FN=5, TN=118)

## 5. Ablation table (5-agent, weighted aggregation at τ = 0.5)

Aggregation matches the orchestrator that produced the CSV: weights (Visual 0.20, FreqNet 0.15, ECAPA 0.20, Cross-Modal 0.25, Biometric 0.20). For 'Remove X' rows the remaining four weights are re-normalised to sum to 1.

| Configuration                           | # agents | Accuracy | Miscount | Δ (pp) |
|-----------------------------------------|----------|----------|----------|--------|
| Full 5-agent ensemble (baseline)        | 5        | 99.63%   | 8        | +0.000 |
| Remove Visual (XceptionNet)             | 4        | 99.63%   | 8        | +0.000 |
| Remove Audio (FreqNet)                  | 4        | 99.58%   | 9        | -0.046 |
| Remove Audio Forensics (ECAPA)          | 4        | 99.26%   | 16       | -0.370 |
| Remove Cross-Modal (Lip-Sync)           | 4        | 99.03%   | 21       | -0.601 |
| Remove Facial Biometric (Quality)       | 4        | 99.12%   | 19       | -0.509 |
| Top-3 (Biometric + ECAPA + Cross-Modal) | 3        | 99.72%   | 6        | +0.093 |
| Audio only (FreqNet + ECAPA)            | 2        | 95.84%   | 90       | -3.793 |
| Visual only (XceptionNet + Biometric)   | 2        | 95.93%   | 88       | -3.700 |
| Single best agent (Cross-Modal)         | 1        | 98.47%   | 33       | -1.156 |

No anomaly rows (all removal rows increase or leave miscount unchanged).

Note: **Remove Visual (XceptionNet)** yields the same 8 errors as the full ensemble at τ = 0.5. The XceptionNet branch changes nothing at this threshold on this test set; its contribution is absorbed by the other four agents. **Top-3** (3 Phase-1 agents with re-normalised weights) actually *beats* the baseline by one error because it reweights Cross-Modal up and drops the less-useful audio and visual Phase-2 signals.

## 6. Disagreement-threshold sweep (escalation cost–benefit)

Replays phase-1-then-maybe-phase-2 escalation with the recorded per-agent scores. At the code's operating point τ_d = 0.30 the system rarely escalates:

- Escalation rate: 8.79%
- Accuracy: 99.63%  (miscount = 8)
- Avg. agents per sample: 3.18

| τ_d   | Escalation | Accuracy | Avg. agents |
|-------|------------|----------|-------------|
| 0.20 | 10.04%      | 99.63%  | 3.20        |
| 0.25 | 9.44%      | 99.63%  | 3.19        |
| **0.30** | 8.79%      | 99.63%  | 3.18        |
| 0.35 | 7.77%      | 99.72%  | 3.16        |
| 0.40 | 6.89%      | 99.72%  | 3.14        |
| 0.50 | 0.00%      | 99.72%  | 3.00        |

Note: on this test set, the 5-agent ensemble at τ_d = 0.30 escalates only ~9% of samples — substantially less than the ~15% target implied by the paper text. This is a consequence of the Phase-1 agents agreeing strongly on most PolyGlot test samples; see the YouTube evaluation below for contrast.

## 7. YouTube evaluation (distribution-shift stress test)

- CSV: `analysis_results_with_5_agents_orchestration.csv`
- Raw rows = 49, parseable = 49 (24 real + 25 fake)
- Accuracy: 77.55%, Precision 93.75%, Recall 60.00%, F1 73.17%
- Confusion: TP = 15, TN = 23, FP = 1, FN = 10
- Phases recorded in CSV: {'iterative': 31, 'quick': 15, 'strong': 3}
- Escalation rate: 69.39% (quick = phase-1 only; iterative/strong = escalated)

**Reconciliation with the thesis's 50-sample / 78% claim.** The saved orchestration CSV contains 49 parseable rows (not 50). At τ = 0.5 the accuracy is 77.55%, which matches the thesis figure within 1 sample. The paper should either state `n = 49` honestly or rerun the evaluation to produce a 50-sample CSV. Escalation is ~69.4% (34 / 49), consistent with the paper's phrasing about Phase-2 activation rising on out-of-distribution content.

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

- Expected Calibration Error (10 bins): **0.103**

The aggregated `final_score` is not a calibrated probability — it's a weighted mean of per-agent sigmoids. A future revision could apply Platt scaling or isotonic regression on a held-out split if the paper wants to claim calibrated confidences.

## 12. Unresolved items and anomalies

1. **Test-set cardinality.** The paper's Table 3 and abstract state `n = 2,163` (118 real + 2,045 fake). The saved CSV has **n = 2,162** (118 real + 2,044 fake). Correct the paper.

2. **Aggregation rule.** The CSV's `final_score` is a **weighted** mean with `{Visual 0.20, FreqNet 0.15, ECAPA 0.20, Cross-Modal 0.25, Biometric 0.20}`, generated by `multiagent_langchain_additional_agents.py`. The ablation follows the same weighted rule so baseline = headline (8 errors). The paper's Methods should state these weights (it currently implies equal weighting; the other orchestrator script uses equal weighting).

3. **Decision threshold.** The paper states τ = 0.50 but the orchestrator code uses τ = 0.37 (`multiagent_xai_five_agents.py`) or τ = 0.38 (`multiagent_langchain_additional_agents.py`). Per this instruction τ = 0.50 is fixed for the paper; the code should be updated to match, or the paper should note τ = 0.37/0.38 as the code value.

4. **YouTube sample size.** 49 parseable rows rather than 50.

5. **Baseline comparisons unavailable.** Task 11 could not complete for any of the five transformer baselines because per-sample prediction CSVs were not saved. Rerun needed for GenConViT AE/VAE, LIPINC-V2, Custom ViT, Hybrid CNN-Transformer.

6. **Remove-XceptionNet ablation is a no-op at τ = 0.5** (same 8 errors as full ensemble). The XceptionNet branch is carried but not decisive on this test distribution; consider pruning or discussing in the paper.

7. **Latency numbers are MPS-timed** on an Apple Silicon machine because no CUDA GPU was available at runtime. Rerun on the target deployment GPU before reporting latencies in the paper.

8. **Calibration is poor** (ECE ≈ 0.10). The system is discriminative (AUC = 1.000) but not calibrated. Acceptable if the paper only claims discrimination; needs Platt/isotonic scaling if it claims calibrated probabilities.

