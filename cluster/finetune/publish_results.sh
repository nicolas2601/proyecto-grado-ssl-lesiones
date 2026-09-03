#!/usr/bin/env bash
# Push the results JSONs to GitHub (the only outbound network the cluster
# has) and print the W&B sync command for a machine with internet.
#
#   bash cluster/finetune/publish_results.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/proyecto-grado-ssl-lesiones}"
SCRATCH_BASE="${SCRATCH:-$HOME/scratch}/dermssl"
cd "$REPO_DIR"

n=$(find results/finetune -name 'results_*_ft*.json' 2>/dev/null | wc -l)
if [ "$n" -eq 0 ]; then
    echo "no results under results/finetune yet"; exit 1
fi
git add results/finetune
if git diff --cached --quiet; then
    echo "nothing new to publish ($n JSON files already committed)"
else
    git commit -m "results(finetune): $n runs from the cluster ($(hostname), $(date +%F))"
    git push origin HEAD
    echo "pushed $n result files"
fi

echo
echo "W&B offline runs are under: $SCRATCH_BASE/wandb/wandb"
echo "From a machine with internet (after copying that folder, or from the login"
echo "node if it has outbound access):"
echo "  wandb login"
echo "  wandb sync $SCRATCH_BASE/wandb/wandb/offline-run-*"
