#!/usr/bin/env bash
# One-shot launcher for Dr. Karen: prepares the environment, checks every
# prerequisite, runs a smoke test and queues the 24-job fine-tuning bank.
# Re-runnable: every step is idempotent (skips what already exists).
#
#   bash cluster/finetune/run_all.sh                 # full flow
#   bash cluster/finetune/run_all.sh --smoke-only    # stop after the smoke
#   bash cluster/finetune/run_all.sh --no-smoke      # queue directly
#   bash cluster/finetune/run_all.sh --dry-run       # print what would run
#
# Inputs it expects (override with environment variables):
#   SCRATCH      scratch filesystem              (default: $HOME/scratch)
#   ASSETS_TAR   dermssl-weights-cache.tar.gz    (default: $SCRATCH/dermssl/dermssl-weights-cache.tar.gz)
#   DATA_TAR     dermssl-data.tar.gz             (default: $SCRATCH/dermssl/dermssl-data.tar.gz)
#   WANDB_ENTITY W&B team/user that owns the project (optional)
#   PRECISION    bf16 (A100/H100) | fp16 (V100/T4)  (default: bf16)
set -euo pipefail

# ---------------------------------------------------------------- settings
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRATCH_BASE="${SCRATCH:-$HOME/scratch}/dermssl"
ENV_DIR="${ENV_DIR:-$HOME/venvs/dermssl-v6}"
ASSETS_TAR="${ASSETS_TAR:-$SCRATCH_BASE/dermssl-weights-cache.tar.gz}"
DATA_TAR="${DATA_TAR:-$SCRATCH_BASE/dermssl-data.tar.gz}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH_BASE/data}"
export PRECISION="${PRECISION:-bf16}"
export REPO_DIR SCRATCH_BASE DATA_ROOT

DRY_RUN=0; DO_SMOKE=1; SMOKE_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --no-smoke) DO_SMOKE=0 ;;
        --smoke-only) SMOKE_ONLY=1 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

log()  { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
run()  { if [ "$DRY_RUN" = 1 ]; then echo "  (dry-run) $*"; else "$@"; fi; }

# ---------------------------------------------------------------- 1. checks
log "1/6 prerequisites"
command -v python3.11 >/dev/null 2>&1 || die "python3.11 not found (module load python/3.11?)"
command -v git >/dev/null 2>&1 || die "git not found"
if command -v sbatch >/dev/null 2>&1; then SCHEDULER=slurm; else SCHEDULER=local; fi
echo "  repo:      $REPO_DIR"
echo "  scratch:   $SCRATCH_BASE"
echo "  scheduler: $SCHEDULER"
echo "  precision: $PRECISION"
[ -f "$REPO_DIR/scripts/finetune_baseline.py" ] || die "not a proyecto-grado-ssl-lesiones checkout: $REPO_DIR"
run mkdir -p "$SCRATCH_BASE" "$REPO_DIR/logs"

# ---------------------------------------------------------------- 2. venv
log "2/6 virtualenv ($ENV_DIR)"
if [ -x "$ENV_DIR/bin/python" ] && "$ENV_DIR/bin/python" -c "import torch, timm, sklearn, pandas" 2>/dev/null; then
    echo "  already installed, skipping"
else
    run bash "$REPO_DIR/cluster/setup_env.sh" "$ENV_DIR"
fi

# ---------------------------------------------------------------- 3. weights
log "3/6 offline weight caches"
if [ -d "$SCRATCH_BASE/hf-cache" ] && [ -d "$SCRATCH_BASE/torch-cache" ]; then
    echo "  caches present, skipping"
elif [ -f "$ASSETS_TAR" ]; then
    run tar -xzf "$ASSETS_TAR" -C "$SCRATCH_BASE"
else
    die "weight caches missing. Copy dermssl-weights-cache.tar.gz to $ASSETS_TAR (GitHub Release cluster-assets-v2) or set ASSETS_TAR"
fi
export HF_HOME="$SCRATCH_BASE/hf-cache" TORCH_HOME="$SCRATCH_BASE/torch-cache" HF_HUB_OFFLINE=1

# ---------------------------------------------------------------- 4. data
log "4/6 datasets ($DATA_ROOT)"
if [ -f "$DATA_ROOT/ham10000/HAM10000_metadata.csv" ] || find "$DATA_ROOT/ham10000" -name HAM10000_metadata.csv -print -quit 2>/dev/null | grep -q .; then
    echo "  datasets present, skipping"
elif [ -f "$DATA_TAR" ]; then
    run mkdir -p "$DATA_ROOT"
    run tar -xzf "$DATA_TAR" -C "$DATA_ROOT"
else
    die "datasets missing. Copy dermssl-data.tar.gz to $DATA_TAR (see cluster/prepare_datasets.sh) or set DATA_ROOT"
fi
for d in ham10000 isic2019 isic2020; do
    [ "$DRY_RUN" = 1 ] || [ -d "$DATA_ROOT/$d" ] || die "missing $DATA_ROOT/$d"
done

# ---------------------------------------------------------------- 5. smoke
if [ "$DO_SMOKE" = 1 ]; then
    log "5/6 smoke test (resnet50 / ham10000, 1 epoch, few minutes)"
    if [ "$SCHEDULER" = slurm ]; then
        run srun --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:20:00 \
            bash "$REPO_DIR/cluster/finetune/smoke.sh" resnet50 ham10000
    else
        run bash "$REPO_DIR/cluster/finetune/smoke.sh" resnet50 ham10000
    fi
    echo "  smoke OK"
    [ "$SMOKE_ONLY" = 1 ] && { log "smoke-only requested, stopping here"; exit 0; }
else
    log "5/6 smoke skipped (--no-smoke)"
fi

# ---------------------------------------------------------------- 6. bank
log "6/6 fine-tuning bank: 24 jobs (8 models x 3 datasets)"
if [ "$SCHEDULER" = slurm ]; then
    run sbatch --export=ALL "$REPO_DIR/cluster/finetune/finetune_bank.sbatch"
    cat <<MSG

  Queued. Monitor with:   squeue -u \$USER
  Logs:                   $REPO_DIR/logs/dermssl-ft-*.out
  Results (JSON):         $REPO_DIR/results/finetune/<dataset>/
  When all 24 finished:   bash cluster/finetune/publish_results.sh
MSG
else
    echo "  no SLURM: running the 24 jobs sequentially on the local GPU"
    source "$ENV_DIR/bin/activate"
    export WANDB_MODE=offline WANDB_DIR="$SCRATCH_BASE/wandb"
    mkdir -p "$WANDB_DIR"
    for i in $(seq 0 23); do
        run python "$REPO_DIR/scripts/finetune_baseline.py" --job-index "$i" \
            --data-root "$DATA_ROOT" \
            --output-dir "$SCRATCH_BASE/finetune/runs/job$i" \
            --results-dir "$REPO_DIR/results/finetune" \
            --precision "$PRECISION" \
            --mlflow-tracking-uri "file:$SCRATCH_BASE/mlruns"
    done
    run bash "$REPO_DIR/cluster/finetune/publish_results.sh"
fi
