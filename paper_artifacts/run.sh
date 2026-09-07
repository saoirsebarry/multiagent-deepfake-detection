#!/usr/bin/env bash
# Re-invoke every task in order. Idempotent; fixed seeds.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=.venv/bin/python
if [[ ! -x "$PY" ]]; then
    echo "venv missing; create with:"
    echo "  /opt/homebrew/bin/python3.12 -m venv .venv"
    echo "  .venv/bin/pip install -r paper_artifacts/requirements.txt"
    exit 1
fi

echo "=== Task 0: weight-selection provenance ==="
$PY paper_artifacts/task_00_select_operating_point.py

echo "=== Task 1: headline metrics ==="
$PY paper_artifacts/task_01_headline.py

echo "=== Task 2: ROC/PR curves ==="
$PY paper_artifacts/task_02_roc_pr.py

echo "=== Task 3: threshold robustness ==="
$PY paper_artifacts/task_03_threshold.py

echo "=== Task 4: bootstrap CIs ==="
$PY paper_artifacts/task_04_bootstrap.py

echo "=== Task 5: ablation (tau=0.5, threshold-sensitivity study) ==="
$PY paper_artifacts/task_05_ablation.py

echo "=== Task 6: three-agent baseline ==="
$PY paper_artifacts/task_06_three_agent.py

echo "=== Task 7: disagreement sweep (tau=0.5) ==="
$PY paper_artifacts/task_07_disagreement.py

echo "=== Task 8: YouTube reconciliation (tau=0.5) ==="
$PY paper_artifacts/task_08_youtube.py

echo "=== Task 9: parameter counts ==="
$PY paper_artifacts/task_09_params.py

echo "=== Task 10: inference latency ==="
$PY paper_artifacts/task_10_latency.py

echo "=== Task 11: McNemar tests ==="
$PY paper_artifacts/task_11_mcnemar.py

echo "=== Task 12: calibration ==="
$PY paper_artifacts/task_12_calibration.py

echo "=== Task 13: SUMMARY.md ==="
$PY paper_artifacts/task_13_summary.py

echo "=== Self-check ==="
$PY paper_artifacts/self_check.py
