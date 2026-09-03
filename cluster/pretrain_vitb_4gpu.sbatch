#!/bin/bash
#SBATCH --job-name=dermssl-vitb14
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
# Big arm: proposed method over DINOv2 ViT-B/14, also on the 4-GPU priority
# queue (decision 2026-08-28: the 8-GPU queue wait is not worth it). Adjust partition/account to the cluster's names:
##SBATCH --partition=gpu
##SBATCH --account=<PROJECT_ACCOUNT>

set -euo pipefail

SCRATCH_BASE="${SCRATCH:-$HOME/scratch}/dermssl"
REPO_DIR="${REPO_DIR:-$HOME/proyecto-grado-ssl-lesiones}"
IMAGE_ROOT="$SCRATCH_BASE/pool"
OUTPUT_DIR="$SCRATCH_BASE/runs/vitb14-$SLURM_JOB_ID"
export TORCH_HOME="$SCRATCH_BASE/torch-cache"

source "$HOME/venvs/dermssl-v6/bin/activate"
export WANDB_MODE=offline
export WANDB_DIR="$SCRATCH_BASE/wandb"
mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" logs

# Both arms use 32/GPU (effective 128) so the S-vs-B comparison keeps a
# single variable: the backbone size. The LR rule scales with the batch.
torchrun --standalone --nproc_per_node=4 "$REPO_DIR/scripts/pretrain.py" \
    --method simsiam_dinov2 \
    --backbone dinov2_vitb14 \
    --image-root "$IMAGE_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size-per-gpu 32 \
    --precision bf16 \
    --num-workers 8 \
    --budget-gpu-hours 80 \
    --seed 42 \
    --tracking-uri "file:$SCRATCH_BASE/mlruns"
