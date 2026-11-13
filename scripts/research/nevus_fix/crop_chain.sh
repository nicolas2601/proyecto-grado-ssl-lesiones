#!/usr/bin/env bash
# usage: crop_chain.sh <gpu> <crop> <outdir> <sets...>
set -u
gpu=$1; crop=$2; out=$3; shift 3
for s in "$@"; do
  /opt/conda/bin/python3 scripts/extract_d4_features.py --code /home/jovyan/work/derm-v10.2-evidence/models/classifier/code \
    --checkpoint /home/jovyan/work/release_v10_1/build-d4-candidate/classifier.pt \
    --checkpoint-sha256 6c3a8dfb2d5c8cabeb86554925e26613c45edaee57ada07b3b33839008489309 \
    --rows rows/$s.jsonl --out $out/$s.npz --device cuda:$gpu --crop $crop > logs/$out-$s.log 2>&1
  echo "$s $?" >> logs/$out.status
done
