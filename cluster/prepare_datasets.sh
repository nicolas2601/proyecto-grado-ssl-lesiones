#!/usr/bin/env bash
# Run on a machine WITH internet and a Kaggle token (~/.kaggle/kaggle.json).
# Downloads the two datasets the bank uses and lays them out exactly as
# scripts/finetune_baseline.py expects under --data-root:
#   <out>/ham10000/   HAM10000_metadata.csv              + *.jpg  (kmader, ~2.9 GB)
#   <out>/isic2019/   ISIC_2019_Training_GroundTruth.csv + *.jpg  (andrewmvd, ~9 GB)
# Then packs a tarball to ship through the channel the cluster admins allow.
#
# isic2019 must come from andrewmvd/isic-2019, NOT from the RESIZED release:
# only this one ships the 8-class one-hot ground truth the bank trains on. The
# RESIZED variant carries a binary target and a synthetic patient_id.
set -euo pipefail

OUT="${1:-/tmp/dermssl-data}"
mkdir -p "$OUT"
declare -A SLUGS=(
    [ham10000]="kmader/skin-cancer-mnist-ham10000"
    [isic2019]="andrewmvd/isic-2019"
)
for name in ham10000 isic2019; do
    if [ -d "$OUT/$name" ] && [ -n "$(ls -A "$OUT/$name")" ]; then
        echo "skip $name (already present)"; continue
    fi
    kaggle datasets download -d "${SLUGS[$name]}" -p "$OUT/$name" --unzip
    echo "downloaded $name"
done
tar -C "$OUT" -czf "$OUT/dermssl-data.tar.gz" ham10000 isic2019
du -sh "$OUT"/*
echo "Ship $OUT/dermssl-data.tar.gz to the cluster and extract into \$SCRATCH/dermssl/data/"
