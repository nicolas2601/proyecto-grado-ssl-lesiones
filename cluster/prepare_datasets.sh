#!/usr/bin/env bash
# Run on a machine WITH internet and a Kaggle token (~/.kaggle/kaggle.json).
# Downloads the three datasets the bank uses and lays them out exactly as
# scripts/finetune_baseline.py expects under --data-root:
#   <out>/ham10000/   HAM10000_metadata.csv + *.jpg      (kmader, ~2.9 GB)
#   <out>/isic2019/   train-metadata.csv   + *.jpg      (nischaydnk RESIZED, 372 MB)
#   <out>/isic2020/   train-metadata.csv   + *.jpg      (nischaydnk RESIZED, 624 MB)
# Then packs a tarball to ship through the channel the cluster admins allow.
set -euo pipefail

OUT="${1:-/tmp/dermssl-data}"
mkdir -p "$OUT"
declare -A SLUGS=(
    [ham10000]="kmader/skin-cancer-mnist-ham10000"
    [isic2019]="nischaydnk/isic-2019-jpg-224x224-resized"
    [isic2020]="nischaydnk/isic-2020-jpg-256x256-resized"
)
for name in ham10000 isic2019 isic2020; do
    if [ -d "$OUT/$name" ] && [ -n "$(ls -A "$OUT/$name")" ]; then
        echo "skip $name (already present)"; continue
    fi
    kaggle datasets download -d "${SLUGS[$name]}" -p "$OUT/$name" --unzip
    echo "downloaded $name"
done
tar -C "$OUT" -czf "$OUT/dermssl-data.tar.gz" ham10000 isic2019 isic2020
du -sh "$OUT"/*
echo "Ship $OUT/dermssl-data.tar.gz to the cluster and extract into \$SCRATCH/dermssl/data/"
