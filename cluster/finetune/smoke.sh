#!/usr/bin/env bash
# Three-minute wiring check on ONE GPU before queueing the 16-job array.
# Runs 1 epoch of resnet50 on a stratified subset of HAM10000 and exercises
# every stage: data, split, offline weights, training, metrics, W&B, MLflow.
#
#   bash cluster/finetune/smoke.sh                 # resnet50 / ham10000
#   bash cluster/finetune/smoke.sh dinov2_vitb14 isic2019
set -euo pipefail

MODEL="${1:-resnet50}"
DATASET="${2:-ham10000}"
SCRATCH_BASE="${SCRATCH:-$HOME/scratch}/dermssl"
REPO_DIR="${REPO_DIR:-$HOME/proyecto-grado-ssl-lesiones}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH_BASE/data}"

export HF_HOME="$SCRATCH_BASE/hf-cache"
export HF_HUB_OFFLINE=1
export TORCH_HOME="$SCRATCH_BASE/torch-cache"
export WANDB_MODE=offline
export WANDB_DIR="$SCRATCH_BASE/wandb"
mkdir -p "$WANDB_DIR"

source "$HOME/venvs/dermssl-v6/bin/activate"
python "$REPO_DIR/scripts/finetune_baseline.py" \
    --model "$MODEL" --dataset "$DATASET" --smoke \
    --data-root "$DATA_ROOT" \
    --output-dir "$SCRATCH_BASE/finetune/smoke-$MODEL-$DATASET" \
    --num-workers 4 \
    --mlflow-tracking-uri "file:$SCRATCH_BASE/mlruns"
