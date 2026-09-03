#!/usr/bin/env bash
# Create the virtualenv for the v6 pretraining runs (run once on the cluster).
# Requires: python3.11 available (module load python/3.11 or system install).
set -euo pipefail

ENV_DIR="${1:-$HOME/venvs/dermssl-v6}"

python3.11 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --upgrade pip

pip install -r "$(dirname "$0")/requirements.txt" \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple

python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("devices:", torch.cuda.device_count())
PY

echo "OK. Activate with: source $ENV_DIR/bin/activate"
