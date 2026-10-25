#!/usr/bin/env bash
set -e

EVAL_DIR="/home/jovyan/work/dermlip_slic_v8/evaluations/dermlip_isic2018_eval_2594_010_clinical_robust_20261023_009_rescued"
RUNNER_DIR="$EVAL_DIR/runner"
OUT_DIR="$EVAL_DIR/output"
LOG_FILE="$EVAL_DIR/eval.log"

mkdir -p "$RUNNER_DIR"
mkdir -p "$OUT_DIR"

cp /home/jovyan/work/proyectonp/repo_v8_canonical/scripts/evaluate_dermlip_semi.py "$RUNNER_DIR/evaluate_dermlip_semi.py"

CHECKPOINT="/home/jovyan/work/dermlip_slic_v8/runs/dermlip-semi-supervised-v8-isic2018-2594-hair-010/selection/best_teacher_model.pt"
MANIFEST="/home/jovyan/work/dermlip_slic_v8/evaluations/dermlip_isic2018_eval_2594_009_20261023_001/inputs/images/manifest.eval-descriptive.json"
RECEIPT="/home/jovyan/work/dermlip_slic_v8/evaluations/dermlip_isic2018_eval_2594_009_20261023_001/inputs/images/manifest.eval-descriptive.receipt.json"
REF="/home/jovyan/work/dermlip_slic_v8/evaluations/dermlip_isic2018_eval_2594_009_20261023_001/inputs/masks/manifest.eval-descriptive.json"
REF_CASES="/home/jovyan/work/dermlip_slic_v8/evaluations/dermlip_isic2018_eval_2594_010_clinical_robust_20261023_007/output/cases.jsonl"

export PYTHONPATH=/home/jovyan/work/proyectonp/repo_v8_canonical/src:/home/jovyan/work/dermlip_slic_v8/src
PYTHON=/home/jovyan/work/proyectonp/.venv/bin/python

echo "=== Launching Run 009: V2 Evaluator with Rescued Mechanisms (Adaptive Hair + Micro Fallback) ===" | tee -a "$LOG_FILE"
echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOG_FILE"

nohup $PYTHON "$RUNNER_DIR/evaluate_dermlip_semi.py" \
    --checkpoint "$CHECKPOINT" \
    --checkpoint-component teacher \
    --eval-manifest "$MANIFEST" \
    --references-manifest "$REF" \
    --full-pair-receipt "$RECEIPT" \
    --out-dir "$OUT_DIR" \
    --device cuda:1 \
    --threshold 0.5 \
    --hausdorff \
    --hair-removal \
    --vignette-mask \
    --tisage-slic \
    --largest-component \
    --center-prior \
    --fill-holes \
    --adaptive-hair-rescue \
    --micro-lesion-rescue \
    --reference-cases "$REF_CASES" \
    >> "$LOG_FILE" 2>&1 &

echo "EVAL_PID: $!" | tee -a "$LOG_FILE"
