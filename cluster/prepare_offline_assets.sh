#!/usr/bin/env bash
# Run this on a machine WITH internet (Nicolas' laptop), NOT on the cluster.
#
# The cluster only reaches github.com. Every pretrained weight the runs need
# is fetched here into two caches and shipped as a tarball:
#   torch-cache/  torch.hub (DINOv2 repo + weights, SimSiam FAIR checkpoint,
#                 torchvision AlexNet)
#   hf-cache/     Hugging Face Hub (timm tags of the 6 HF-hosted baselines)
# On the cluster: export TORCH_HOME=.../torch-cache HF_HOME=.../hf-cache
# HF_HUB_OFFLINE=1 and both loaders resolve from cache without any network.
set -euo pipefail

OUT_DIR="${1:-/tmp/dermssl-offline-assets}"
export TORCH_HOME="$OUT_DIR/torch-cache"
export HF_HOME="$OUT_DIR/hf-cache"
mkdir -p "$TORCH_HOME" "$HF_HOME"

python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src") if "__file__" in globals() else "src")
import torch
import timm
from dermssl.finetune import protocol

# v6 pretraining arms (torch.hub) ----------------------------------------
for name in ("dinov2_vits14", "dinov2_vitb14"):
    torch.hub.load("facebookresearch/dinov2", name, pretrained=True)
    print("torch.hub cached:", name)

# fine-tuning bank: the 6 HF-hosted baselines via timm --------------------
for model, tag in protocol.TIMM_HF_TAGS.items():
    kwargs = {"img_size": protocol.RESOLUTION} if "patch14" in tag else {}
    timm.create_model(tag, pretrained=True, num_classes=0, **kwargs)
    print("hf cached:", model, "->", tag)

# the two documented exceptions -----------------------------------------
from torchvision.models import AlexNet_Weights, alexnet
alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
print("torch.hub cached: alexnet (torchvision)")
torch.hub.load_state_dict_from_url(protocol.SIMSIAM_CHECKPOINT_URL, map_location="cpu")
print("torch.hub cached: simsiam_r50 (FAIR checkpoint)")
PY

tar -C "$OUT_DIR" -czf "$OUT_DIR/dermssl-weights-cache.tar.gz" torch-cache hf-cache
echo
echo "Asset ready: $OUT_DIR/dermssl-weights-cache.tar.gz"
echo "Upload it to a GitHub Release of this repo, e.g.:"
echo "  gh release create cluster-assets-v2 $OUT_DIR/dermssl-weights-cache.tar.gz \\"
echo "      --title 'Offline weight caches for cluster runs' \\"
echo "      --notes 'torch.hub (DINOv2, SimSiam, AlexNet) + HF Hub (timm baselines)'"
echo
echo "On the cluster:"
echo "  gh release download cluster-assets-v2 -p dermssl-weights-cache.tar.gz  # or curl the asset URL"
echo "  tar -xzf dermssl-weights-cache.tar.gz -C \$SCRATCH/dermssl/"
echo "  export TORCH_HOME=\$SCRATCH/dermssl/torch-cache HF_HOME=\$SCRATCH/dermssl/hf-cache HF_HUB_OFFLINE=1"
