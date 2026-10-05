#!/bin/bash
#SBATCH --job-name=dermssl-vits14
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Small arm: proposed method (SimSiam over DINOv2 ViT-S/14) on the 4-GPU
# priority queue. Adjust partition/account to the cluster's names:
##SBATCH --partition=gpu
##SBATCH --account=<PROJECT_ACCOUNT>

set -euo pipefail

# --- paths (adjust SCRATCH to the cluster's scratch filesystem) ---
SCRATCH_BASE="${SCRATCH:-$HOME/scratch}/dermssl"
REPO_DIR="${REPO_DIR:-$HOME/proyecto-grado-ssl-lesiones}"
IMAGE_ROOT="$SCRATCH_BASE/pool"            # unlabeled image pool (no CSVs)
OUTPUT_DIR="$SCRATCH_BASE/runs/vits14-$SLURM_JOB_ID"
export TORCH_HOME="$SCRATCH_BASE/torch-cache"   # pre-seeded, see prepare_offline_assets.sh

# --- environment ---
source "$HOME/venvs/dermssl-v6/bin/activate"
export WANDB_MODE=offline                  # no internet: sync later from laptop
export WANDB_DIR="$SCRATCH_BASE/wandb"
mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" logs

# --- run: proposed method, ViT-S/14, 4 GPUs ---
torchrun --standalone --nproc_per_node=4 "$REPO_DIR/scripts/pretrain.py" \
    --method simsiam_dinov2 \
    --backbone dinov2_vits14 \
    --image-root "$IMAGE_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size-per-gpu 32 \
    --precision bf16 \
    --num-workers 8 \
    --budget-gpu-hours 40 \
    --seed 42 \
    --tracking-uri "file:$SCRATCH_BASE/mlruns"
