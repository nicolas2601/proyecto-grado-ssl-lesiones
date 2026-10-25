#!/usr/bin/env bash
set -e
export PYTHONPATH=/home/jovyan/work/proyectonp/repo_v8_canonical/src:/home/jovyan/work/dermlip_slic_v8/src:/home/jovyan/work/usl_net_v8/runtime-direct-20261020/repo/src:/home/jovyan/work/vendor/Derm1M/src:$PYTHONPATH
PYTHON=/home/jovyan/work/proyectonp/.venv/bin/python

mkdir -p /home/jovyan/work/dermlip_slic_v8/runs/dermlip-semi-supervised-v8-isic2018-2594-v3-tripleprior
rm -rf /home/jovyan/work/dermlip_slic_v8/runs/dermlip-semi-supervised-v8-isic2018-2594-v3-tripleprior/selection

nohup $PYTHON /home/jovyan/work/dermlip_slic_v8/train_dermlip_semi.py \
    --execute \
    --runtime-config /home/jovyan/work/dermlip_slic_v8/runtime_poolv2_20261022_001/plan-segmentacion/metodos-v8/launch-preparation/configs/dermlip.selection.runtime.20261024.011.json \
    --execution-gate /home/jovyan/work/dermlip_slic_v8/runtime_poolv2_20261022_001/preflight/dermlip-selection-execution-gate-v2-20261024-011.json \
    --execution-claim-path /home/jovyan/work/dermlip_slic_v8/runs/dermlip-semi-supervised-v8-isic2018-2594-v3-tripleprior/selection/claims/dermlip-selection-20261024-011.selection.claim.json \
    > /home/jovyan/work/dermlip_slic_v8/runs/dermlip-semi-supervised-v8-isic2018-2594-v3-tripleprior/training.log 2>&1 &

echo "LAUNCHED_PID: $!"
