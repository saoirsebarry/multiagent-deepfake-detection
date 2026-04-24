"""Task 13: Assemble SUMMARY.md from all upstream JSON outputs.

Number precision (per spec):
  - Accuracies, precisions, recalls, F1 : 2 dp (e.g. 99.63%)
  - AUC, AP, ECE                        : 3 dp
  - Latencies (ms)                      : 1 dp
  - p-values                            : 3 sig figs or "< 0.001"
  - Parameter counts                    : 0.1 M
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def load(name: str):
    path = OUT / name
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def fmt_pct(x, nd=2):
    if x is None:
        return "—"
    return f"{x * 100:.{nd}f}%"


def fmt_ci_pct(entry, nd=2):
    return f"{fmt_pct(entry['point'], nd)} (95% CI [{fmt_pct(entry['ci_low'], nd)}, {fmt_pct(entry['ci_high'], nd)}])"


def fmt_num(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def fmt_ci_num(entry, nd=3):
    return (f"{fmt_num(entry['point'], nd)} "
            f"(95% CI [{fmt_num(entry['ci_low'], nd)}, {fmt_num(entry['ci_high'], nd)}])")


def fmt_pvalue(p):
    if p is None:
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3g}"


def fmt_params(n):
    if n is None:
        return "—"
    return f"{n / 1e6:.1f}M"


def main() -> None:
    headline = load("headline_metrics.json")
    boot = load("bootstrap_cis.json")
    roc_pr = load("roc_pr_summary.json")
    three = load("three_agent_metrics.json")
    disagree = load("escalation_at_tau_disagree_0.3.json")
    disagree_sweep = list(csv.DictReader(open(OUT / "disagreement_sweep.csv")))
    youtube = load("youtube_metrics.json")
    params = load("parameter_counts.json")
    latency = load("latency_benchmark.json")
    mcnemar = load("mcnemar_tests.json")
    calib = load("calibration_metrics.json")
    ablation_rows = list(csv.DictReader(open(OUT / "ablation_table.csv")))
    threshold_rows = list(csv.DictReader(open(OUT / "threshold_robustness.csv")))

    lines: list[str] = []
    push = lines.append

    push("# Paper artifacts — summary")
    push("")
    push("All numbers derived from `multiagent_results_csv_files/` at the code's "
         "saved final scores. Decision threshold τ = 0.50 unless explicitly swept.")
    push("")

    # 1. Headline
    push("## 1. Headline metrics (5-agent, PolyGlotFake test set, n = 2,162)")
    push("")
    push(f"- Accuracy:  **{fmt_ci_pct(boot['accuracy'])}**")
    push(f"- Precision: {fmt_ci_pct(boot['precision'])}")
    push(f"- Recall:    {fmt_ci_pct(boot['recall'])}")
    push(f"- F1:        {fmt_ci_pct(boot['f1'])}")
    push(f"- Confusion matrix: TP = {headline['confusion_matrix']['TP']}, "
         f"TN = {headline['confusion_matrix']['TN']}, "
         f"FP = {headline['confusion_matrix']['FP']}, "
         f"FN = {headline['confusion_matrix']['FN']} (all errors are false negatives)")
    push("")
    push(f"> Abstract line: **{fmt_pct(boot['accuracy']['point'])} accuracy "
         f"(95% CI [{fmt_pct(boot['accuracy']['ci_low'])}, "
         f"{fmt_pct(boot['accuracy']['ci_high'])}])**")
    push("")

    # 2. ROC / PR
    push("## 2. Discrimination at the score level")
    push("")
    push(f"- AUC-ROC: {fmt_ci_num(boot['auc_roc'])}")
    push(f"- Average precision (AP): {fmt_ci_num(boot['average_precision'])}")
    push("")
    push(f"The real / fake `final_score` distributions are fully separable "
         f"(max real = 0.357 < min fake = 0.388). The 8 errors at τ = 0.5 are "
         f"fake samples with scores in (0.388, 0.500) and disappear at τ = 0.37 "
         f"(see §3).")
    push("")

    # 3. Threshold robustness
    push("## 3. Threshold robustness (supplementary only)")
    push("")
    push("Reported for completeness. The paper's decision boundary is τ = 0.50.")
    push("")
    push("| τ    | Accuracy | FPR      | FNR      | Miscount |")
    push("|------|----------|----------|----------|----------|")
    for r in threshold_rows:
        tau = float(r['tau'])
        acc = float(r['accuracy']) * 100
        fpr = float(r['fpr']) * 100
        fnr = float(r['fnr']) * 100
        mis = int(r['miscount'])
        bold = abs(tau - 0.50) < 1e-9
        row = f"| {'**' if bold else ''}{tau:.2f}{'**' if bold else ''} | {acc:.2f}%   | {fpr:.2f}%   | {fnr:.2f}%   | {mis}        |"
        push(row)
    push("")

    # 4. Three-agent
    push("## 4. Three-agent baseline at τ = 0.5")
    push("")
    push("Re-thresholded from `analysis_results_with_3_agents.csv` so this row "
         "is directly comparable to the 5-agent headline:")
    push("")
    push(f"- Accuracy: {fmt_pct(three['accuracy'])}, Precision {fmt_pct(three['precision'])}, "
         f"Recall {fmt_pct(three['recall'])}, F1 {fmt_pct(three['f1'])}")
    push(f"- Miscount: {three['miscount']} (TP={three['confusion_matrix']['TP']}, "
         f"FP={three['confusion_matrix']['FP']}, FN={three['confusion_matrix']['FN']}, "
         f"TN={three['confusion_matrix']['TN']})")
    push("")

    # 5. Ablation
    push("## 5. Ablation table (5-agent, weighted aggregation at τ = 0.5)")
    push("")
    push("Aggregation matches the orchestrator that produced the CSV: weights "
         "(Visual 0.20, FreqNet 0.15, ECAPA 0.20, Cross-Modal 0.25, Biometric 0.20). "
         "For 'Remove X' rows the remaining four weights are re-normalised to sum to 1.")
    push("")
    push("| Configuration                           | # agents | Accuracy | Miscount | Δ (pp) |")
    push("|-----------------------------------------|----------|----------|----------|--------|")
    for r in ablation_rows:
        acc = float(r['accuracy']) * 100
        delta = float(r['delta_pp'])
        delta_s = f"{delta:+.3f}"
        config = r['configuration']
        if r.get('anomaly'):
            config += " †"
        push(f"| {config:<39s} | {r['n_agents']}        | {acc:.2f}%   | {r['miscount']:<8s} | {delta_s} |")
    # anomalies
    anomalies = [r for r in ablation_rows if r.get('anomaly')]
    push("")
    if anomalies:
        push("Anomaly flags († rows):")
        for a in anomalies:
            push(f"- {a['configuration']}: {a['anomaly']}")
    else:
        push("No anomaly rows (all removal rows increase or leave miscount unchanged).")
    push("")
    push("Note: **Remove Visual (XceptionNet)** yields the same 8 errors as the "
         "full ensemble at τ = 0.5. The XceptionNet branch changes nothing at "
         "this threshold on this test set; its contribution is absorbed by the "
         "other four agents. **Top-3** (3 Phase-1 agents with re-normalised weights) "
         "actually *beats* the baseline by one error because it reweights Cross-Modal "
         "up and drops the less-useful audio and visual Phase-2 signals.")
    push("")

    # 6. Disagreement sweep
    push("## 6. Disagreement-threshold sweep (escalation cost–benefit)")
    push("")
    push("Replays phase-1-then-maybe-phase-2 escalation with the recorded per-agent "
         "scores. At the code's operating point τ_d = 0.30 the system rarely "
         "escalates:")
    push("")
    op = next(r for r in disagree_sweep if abs(float(r['tau_disagree']) - 0.30) < 1e-9)
    push(f"- Escalation rate: {float(op['escalation_rate_pct']):.2f}%")
    push(f"- Accuracy: {float(op['accuracy_pct']):.2f}%  (miscount = {op['miscount']})")
    push(f"- Avg. agents per sample: {float(op['avg_agents_per_sample']):.2f}")
    push("")
    push("| τ_d   | Escalation | Accuracy | Avg. agents |")
    push("|-------|------------|----------|-------------|")
    for r in disagree_sweep:
        td = float(r['tau_disagree'])
        esc = float(r['escalation_rate_pct'])
        acc = float(r['accuracy_pct'])
        ag = float(r['avg_agents_per_sample'])
        bold = abs(td - 0.30) < 1e-9
        b = "**" if bold else ""
        push(f"| {b}{td:.2f}{b} | {esc:.2f}%      | {acc:.2f}%  | {ag:.2f}        |")
    push("")
    push(f"Note: on this test set, the 5-agent ensemble at τ_d = 0.30 escalates "
         f"only ~9% of samples — substantially less than the ~15% target implied "
         f"by the paper text. This is a consequence of the Phase-1 agents agreeing "
         f"strongly on most PolyGlot test samples; see the YouTube evaluation below "
         f"for contrast.")
    push("")

    # 7. YouTube
    push("## 7. YouTube evaluation (distribution-shift stress test)")
    push("")
    push(f"- CSV: `{youtube['csv_file']}`")
    push(f"- Raw rows = {youtube['n_rows_raw']}, parseable = {youtube['n_rows_parseable']} "
         f"({youtube['n_real']} real + {youtube['n_fake']} fake)")
    push(f"- Accuracy: {fmt_pct(youtube['accuracy'])}, Precision {fmt_pct(youtube['precision'])}, "
         f"Recall {fmt_pct(youtube['recall'])}, F1 {fmt_pct(youtube['f1'])}")
    push(f"- Confusion: TP = {youtube['confusion_matrix']['TP']}, "
         f"TN = {youtube['confusion_matrix']['TN']}, "
         f"FP = {youtube['confusion_matrix']['FP']}, "
         f"FN = {youtube['confusion_matrix']['FN']}")
    push(f"- Phases recorded in CSV: {youtube['phase_counts']}")
    push(f"- Escalation rate: {fmt_pct(youtube['escalation_rate'])} "
         f"(quick = phase-1 only; iterative/strong = escalated)")
    push("")
    push("**Reconciliation with the thesis's 50-sample / 78% claim.** The saved "
         "orchestration CSV contains 49 parseable rows (not 50). At τ = 0.5 the "
         "accuracy is 77.55%, which matches the thesis figure within 1 sample. "
         "The paper should either state `n = 49` honestly or rerun the evaluation "
         "to produce a 50-sample CSV. Escalation is ~69.4% (34 / 49), consistent "
         "with the paper's phrasing about Phase-2 activation rising on out-of-"
         "distribution content.")
    push("")

    # 8. Parameter counts
    push("## 8. Parameter counts (per agent)")
    push("")
    push("| Agent | Total | Trainable | Notes |")
    push("|-------|-------|-----------|-------|")
    order = [
        ("XceptionNet", "Visual (Xception)"),
        ("FreqNet", "Audio (FreqNet)"),
        ("CrossModal_CNN_BiLSTM_CrossAttn", "Cross-Modal (MobileNetV2 + BiLSTM + cross-attention)"),
        ("Biometric_5ch_EfficientNet_B0", "Biometric (5-channel EfficientNet-B0)"),
        ("ECAPA_TDNN_plus_11features", "Audio Forensics (ECAPA-TDNN + 11 hand-crafted features)"),
    ]
    total_all = 0
    train_all = 0
    for k, label in order:
        p = params.get(k, {})
        t = p.get("total")
        tr = p.get("trainable")
        note = ""
        if k == "ECAPA_TDNN_plus_11features":
            note = (f"ECAPA backbone frozen "
                    f"({fmt_params(p.get('breakdown', {}).get('ecapa_backbone', {}).get('encoder_params'))} params); "
                    f"classifier head is {fmt_params(p.get('breakdown', {}).get('classifier_head', {}).get('total'))} "
                    f"and fully trainable")
        push(f"| {label} | {fmt_params(t)} | {fmt_params(tr)} | {note} |")
        if t: total_all += t
        if tr: train_all += tr
    push(f"| **System total** | **{fmt_params(total_all)}** | **{fmt_params(train_all)}** | |")
    push("")

    # 9. Latency
    push("## 9. Inference latency")
    push("")
    push(f"- Device: {latency['gpu']}  (no CUDA available on this machine; "
         f"numbers are MPS-timed)")
    push(f"- {latency['n_samples']} samples × {latency['n_runs_per_sample']} runs each, "
         f"{latency['n_warmup']} warm-up runs not counted")
    push("")
    push("| Agent | Mean (ms) | Median (ms) | p95 (ms) |")
    push("|-------|-----------|-------------|----------|")
    for agent, stats in latency['per_agent_ms'].items():
        push(f"| {agent} | {stats['mean']:.1f} | {stats['median']:.1f} | {stats['p95']:.1f} |")
    push("")
    p1 = latency['phase1_only_ms']['parallel']
    p12 = latency['phase1_plus_2_ms']['parallel']
    avg = latency['average_observed_ms']['parallel_actual']
    push(f"- Phase 1 (parallel, best case):     **{p1['mean']:.1f} ms** (median {p1['median']:.1f}, p95 {p1['p95']:.1f})")
    push(f"- Phase 1 + Phase 2 (parallel):     **{p12['mean']:.1f} ms** (median {p12['median']:.1f}, p95 {p12['p95']:.1f})")
    esc = latency['average_observed_ms']['escalation_rate_actual_tau_d_0.3']
    push(f"- Average observed (at {esc * 100:.2f}% escalation, parallel): **{avg['mean']:.1f} ms**")
    push("")
    push("Caveats: the orchestrator code currently runs agents sequentially; "
         "the 'parallel' numbers above show the best case if the user "
         "parallelises deployment. Sequential sum (how the code runs today) is "
         f"~{latency['phase1_only_ms']['sequential']['mean']:.1f} ms for Phase 1 "
         f"and ~{latency['phase1_plus_2_ms']['sequential']['mean']:.1f} ms for Phase 1+2. "
         "Preprocessing (MTCNN face detection, Mel-spec / MFCC computation) is not "
         "included.")
    push("")

    # 10. McNemar
    push("## 10. McNemar's tests against transformer baselines")
    push("")
    ok = [k for k, v in mcnemar['baselines'].items() if v.get('status') == 'ok']
    missing = [k for k, v in mcnemar['baselines'].items() if v.get('status') != 'ok']
    if ok:
        push("| Baseline | a | b | c | d | p-value |")
        push("|----------|---|---|---|---|---------|")
        for k in ok:
            b = mcnemar['baselines'][k]
            cm = b['contingency']
            push(f"| {k} | {cm['a']} | {cm['b']} | {cm['c']} | {cm['d']} | {fmt_pvalue(b['p_value'])} |")
        push("")
    if missing:
        push("**Predictions not saved, rerun needed:**")
        for k in missing:
            push(f"- {k}")
        push("")
        push("To complete this comparison, rerun each baseline on the 2,162-row "
             "PolyGlotFake test set and save per-sample predictions to a CSV with "
             "columns `filepath, prediction`. Align `filepath` to the same "
             "identifiers as `analysis_results_with_5_agents.csv`.")
    push("")

    # 11. Calibration
    push("## 11. Calibration (supplementary)")
    push("")
    ece = calib['ece_10bins']
    push(f"- Expected Calibration Error (10 bins): **{ece:.3f}**")
    push("")
    push("The aggregated `final_score` is not a calibrated probability — it's a "
         "weighted mean of per-agent sigmoids. A future revision could apply "
         "Platt scaling or isotonic regression on a held-out split if the paper "
         "wants to claim calibrated confidences.")
    push("")

    # 12. Unresolved / anomalies
    push("## 12. Unresolved items and anomalies")
    push("")
    push("1. **Test-set cardinality.** The paper's Table 3 and abstract state "
         "`n = 2,163` (118 real + 2,045 fake). The saved CSV has **n = 2,162** "
         "(118 real + 2,044 fake). Correct the paper.")
    push("")
    push("2. **Aggregation rule.** The CSV's `final_score` is a **weighted** mean with "
         "`{Visual 0.20, FreqNet 0.15, ECAPA 0.20, Cross-Modal 0.25, Biometric 0.20}`, "
         "generated by `multiagent_langchain_additional_agents.py`. The ablation "
         "follows the same weighted rule so baseline = headline (8 errors). The "
         "paper's Methods should state these weights (it currently implies equal "
         "weighting; the other orchestrator script uses equal weighting).")
    push("")
    push("3. **Decision threshold.** The paper states τ = 0.50 but the orchestrator "
         "code uses τ = 0.37 (`multiagent_xai_five_agents.py`) or τ = 0.38 "
         "(`multiagent_langchain_additional_agents.py`). Per this instruction τ = 0.50 "
         "is fixed for the paper; the code should be updated to match, or the "
         "paper should note τ = 0.37/0.38 as the code value.")
    push("")
    push("4. **YouTube sample size.** 49 parseable rows rather than 50.")
    push("")
    push("5. **Baseline comparisons unavailable.** Task 11 could not complete for "
         "any of the five transformer baselines because per-sample prediction "
         "CSVs were not saved. Rerun needed for GenConViT AE/VAE, LIPINC-V2, "
         "Custom ViT, Hybrid CNN-Transformer.")
    push("")
    push("6. **Remove-XceptionNet ablation is a no-op at τ = 0.5** (same 8 errors "
         "as full ensemble). The XceptionNet branch is carried but not decisive "
         "on this test distribution; consider pruning or discussing in the "
         "paper.")
    push("")
    push("7. **Latency numbers are MPS-timed** on an Apple Silicon machine because "
         "no CUDA GPU was available at runtime. Rerun on the target deployment "
         "GPU before reporting latencies in the paper.")
    push("")
    push("8. **Calibration is poor** (ECE ≈ 0.10). The system is discriminative "
         "(AUC = 1.000) but not calibrated. Acceptable if the paper only claims "
         "discrimination; needs Platt/isotonic scaling if it claims calibrated "
         "probabilities.")
    push("")

    text = "\n".join(lines) + "\n"
    (OUT / "SUMMARY.md").write_text(text)
    print(f"[task13] SUMMARY.md written ({len(text)} chars, {len(lines)} lines)")


if __name__ == "__main__":
    main()
