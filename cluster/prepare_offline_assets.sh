#!/usr/bin/env bash
# Run this on a machine WITH internet (Nicolas' laptop), NOT on the cluster.
#
# The cluster only reaches github.com (git clone). torch.hub.load(
# "facebookresearch/dinov2", ...) clones the hub repo from GitHub (fine) but
# downloads the pretrained weights from dl.fbaipublicfiles.com (BLOCKED).
# torch.hub checks its local cache first, so we pre-build the cache here and
# ship it to the cluster via a GitHub Release asset of our own repo.
set -euo pipefail

OUT_DIR="${1:-/tmp/dermssl-offline-assets}"
export TORCH_HOME="$OUT_DIR/torch-cache"
mkdir -p "$TORCH_HOME"

python - <<'PY'
import torch
# Clones facebookresearch/dinov2 into $TORCH_HOME/hub and downloads the
# pretrained weights into $TORCH_HOME/hub/checkpoints/.
for name in ("dinov2_vits14", "dinov2_vitb14"):
    model = torch.hub.load("facebookresearch/dinov2", name, pretrained=True)
    print(name, "cached OK")
PY

tar -C "$OUT_DIR" -czf "$OUT_DIR/torch-cache-dinov2.tar.gz" torch-cache
echo
echo "Asset ready: $OUT_DIR/torch-cache-dinov2.tar.gz"
echo "Upload it to a GitHub Release of this repo, e.g.:"
echo "  gh release create cluster-assets-v1 $OUT_DIR/torch-cache-dinov2.tar.gz \\"
echo "      --title 'Offline assets for cluster runs' --notes 'DINOv2 ViT-S/14 + ViT-B/14 torch.hub cache'"
echo
echo "On the cluster:"
echo "  gh release download cluster-assets-v1 -p torch-cache-dinov2.tar.gz  # or curl the asset URL"
echo "  tar -xzf torch-cache-dinov2.tar.gz -C \$SCRATCH/dermssl/"
echo "  export TORCH_HOME=\$SCRATCH/dermssl/torch-cache"
