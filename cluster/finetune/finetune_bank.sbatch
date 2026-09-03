#!/bin/bash
#SBATCH --job-name=dermssl-ft
#SBATCH --array=0-15%4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Fine-tuning bank: 16 jobs (8 backbones x 2 datasets), one GPU each, four at
# a time on the 4-GPU priority queue. Job index -> (model, dataset) comes from
# src/dermssl/finetune/protocol.py (dataset-major: HAM10000 first).
# Adjust partition/account to the cluster's names:
##SBATCH --partition=gpu
##SBATCH --account=<PROJECT_ACCOUNT>

set -euo pipefail

SCRATCH_BASE="${SCRATCH:-$HOME/scratch}/dermssl"
REPO_DIR="${REPO_DIR:-$HOME/proyecto-grado-ssl-lesiones}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH_BASE/data}"          # ham10000/ isic2020/
OUTPUT_DIR="$SCRATCH_BASE/finetune/runs/job${SLURM_ARRAY_TASK_ID}-${SLURM_ARRAY_JOB_ID}"

# Offline weights (see cluster/prepare_offline_assets.sh): no internet needed.
export HF_HOME="$SCRATCH_BASE/hf-cache"
export HF_HUB_OFFLINE=1
export TORCH_HOME="$SCRATCH_BASE/torch-cache"

# Tracking: W&B offline (sync later, section 6 of the README) + MLflow file store.
export WANDB_MODE=offline
export WANDB_DIR="$SCRATCH_BASE/wandb"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" logs

source "$HOME/venvs/dermssl-v6/bin/activate"

python "$REPO_DIR/scripts/finetune_baseline.py" \
    --job-index "$SLURM_ARRAY_TASK_ID" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --results-dir "$REPO_DIR/results/finetune" \
    --num-workers 8 \
    --precision "${PRECISION:-bf16}" \
    --mlflow-tracking-uri "file:$SCRATCH_BASE/mlruns"
