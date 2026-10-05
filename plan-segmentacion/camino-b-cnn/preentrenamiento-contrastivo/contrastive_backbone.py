"""MoCo v2 on a ResNet50 trunk, split BatchNorm and backbone feature providers.

Written from the MoCo v1/v2 papers (He et al. 2020, Chen et al. 2020); no code
was copied. The pilot uses ONE contrastive method (MoCo v2 style). The
``FeatureProvider`` interface lets the SimCLR + MoCo v1 + MoCo v2 fusion of
USL-Net be added later with ``FusedFeatureProvider`` without touching CCAM.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ckpt_io import load_checkpoint, save_checkpoint, strip_state_dict_prefixes, unwrap_state_dict

TRUNK_CHANNELS = 2048
LAYER3_CHANNELS = 1024
EXTERNAL_PREFIXES = ("module.", "encoder_q.", "encoder.", "backbone.", "trunk.")


class SplitBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm that computes training statistics over ``num_splits`` sub-batches.

    Emulates MoCo's shuffling BN on a single device: the key encoder sees a
    shuffled batch, so its sub-batch statistics differ from the query's and the
    model cannot cheat through intra-batch statistics. The state-dict layout is
    identical to ``nn.BatchNorm2d`` so weights load into a plain trunk.
    """

    def __init__(self, num_features: int, num_splits: int, **kwargs) -> None:
        super().__init__(num_features, **kwargs)
        self.num_splits = num_splits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.training or not self.track_running_stats):
            return super().forward(x)
        batch, channels, height, width = x.shape
        splits = self.num_splits
        if batch % splits:
            raise ValueError(f"batch {batch} is not divisible by num_splits {splits}")
        running_mean = self.running_mean.repeat(splits)
        running_var = self.running_var.repeat(splits)
        out = F.batch_norm(
            x.reshape(-1, channels * splits, height, width),
            running_mean,
            running_var,
            self.weight.repeat(splits),
            self.bias.repeat(splits),
            True,
            self.momentum,
            self.eps,
        )
        self.running_mean.data.copy_(running_mean.view(splits, channels).mean(dim=0))
        self.running_var.data.copy_(running_var.view(splits, channels).mean(dim=0))
        return out.reshape(batch, channels, height, width)


def convert_batchnorm_to_split(module: nn.Module, num_splits: int) -> nn.Module:
    """Replace every ``nn.BatchNorm2d`` in ``module`` by ``SplitBatchNorm2d`` (in place)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d) and not isinstance(child, SplitBatchNorm2d):
            replacement = SplitBatchNorm2d(
                child.num_features,
                num_splits,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats,
            )
            replacement.load_state_dict(child.state_dict())
            setattr(module, name, replacement)
        else:
            convert_batchnorm_to_split(child, num_splits)
    return module


class ResNet50Trunk(nn.Module):
    """ResNet50 from random init; optionally stride 1 in the last stage (USL-Net)."""

    def __init__(self, layer4_stride: int = 2, bn_splits: int = 1) -> None:
        super().__init__()
        base = torchvision.models.resnet50(weights=None)
        if layer4_stride == 1:
            first_block = base.layer4[0]
            first_block.conv2.stride = (1, 1)
            first_block.downsample[0].stride = (1, 1)
        elif layer4_stride != 2:
            raise ValueError("layer4_stride must be 1 or 2")
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1, self.layer2 = base.layer1, base.layer2
        self.layer3, self.layer4 = base.layer3, base.layer4
        if bn_splits > 1:
            convert_batchnorm_to_split(self, bn_splits)

    def forward_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 3, H, W) -> layer3 (B, 1024, H/16, W/16), layer4 (B, 2048, h4, w4)
        x = self.layer2(self.layer1(self.stem(x)))
        layer3 = self.layer3(x)
        return layer3, self.layer4(layer3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> pooled (B, 2048)
        return self.forward_maps(x)[1].mean(dim=(2, 3))


class ProjectionEncoder(nn.Module):
    """Trunk plus MoCo v2 MLP head; returns L2-normalized float32 embeddings."""

    def __init__(self, trunk: ResNet50Trunk, embed_dim: int, hidden_dim: int = 2048) -> None:
        super().__init__()
        self.trunk = trunk
        self.head = nn.Sequential(
            nn.Linear(TRUNK_CHANNELS, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(self.trunk(x)).float(), dim=1)


class MoCoV2(nn.Module):
    """Momentum encoder, queue of negatives and InfoNCE loss (MoCo v2)."""

    def __init__(
        self,
        embed_dim: int = 128,
        queue_size: int = 16384,
        momentum: float = 0.99,
        temperature: float = 0.2,
        layer4_stride: int = 2,
        bn_splits: int = 4,
    ) -> None:
        super().__init__()
        self.momentum = momentum
        self.temperature = temperature
        trunk = ResNet50Trunk(layer4_stride=layer4_stride, bn_splits=bn_splits)
        self.encoder_q = ProjectionEncoder(trunk, embed_dim)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for parameter in self.encoder_k.parameters():
            parameter.requires_grad = False
        self.register_buffer("queue", F.normalize(torch.randn(queue_size, embed_dim), dim=1))
        self.register_buffer("queue_ptr", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self) -> None:
        keys = list(self.encoder_k.parameters())
        queries = [p.detach() for p in self.encoder_q.parameters()]
        torch._foreach_mul_(keys, self.momentum)
        torch._foreach_add_(keys, queries, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def encode_key(self, images: torch.Tensor) -> torch.Tensor:
        """Key embeddings computed on a shuffled batch, returned in the input order."""
        permutation = torch.randperm(images.shape[0], device=images.device)
        keys = self.encoder_k(images[permutation])
        return keys[torch.argsort(permutation)]

    def loss(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """InfoNCE: the positive key against every queued negative."""
        positive = (queries * keys).sum(dim=1, keepdim=True)
        negatives = queries @ self.queue.t().detach()
        logits = torch.cat([positive, negatives], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor) -> None:
        size = self.queue.shape[0]
        if keys.shape[0] > size:
            raise ValueError("queue_size must be at least the global batch size")
        start = int(self.queue_ptr)
        positions = (start + torch.arange(keys.shape[0], device=keys.device)) % size
        self.queue[positions] = keys
        self.queue_ptr.fill_(int(positions[-1] + 1) % size)


# --------------------------------------------------------- persistence
def save_backbone(path: Path, trunk: ResNet50Trunk, config: dict) -> None:
    """Persist only the trunk, the artifact later stages consume."""
    save_checkpoint(path, {"trunk": trunk.state_dict(), "config": config})


def load_external_weights(trunk: ResNet50Trunk, path: Path) -> tuple[list[str], list[str]]:
    """Opt-in: load external self-supervised weights (a deviation from the pure setting).

    Classifier and projection-head keys are dropped. Returns
    ``(missing_keys, unexpected_keys)`` of the trunk after loading.
    """
    state = strip_state_dict_prefixes(unwrap_state_dict(load_checkpoint(path)), EXTERNAL_PREFIXES)
    state = {k: v for k, v in state.items() if not k.startswith(("fc.", "head."))}
    result = trunk.load_state_dict(_to_trunk_keys(state, trunk), strict=False)
    return list(result.missing_keys), list(result.unexpected_keys)


def _to_trunk_keys(torchvision_state: dict, trunk: ResNet50Trunk) -> dict:
    """Map torchvision ResNet keys (``conv1.``, ``bn1.``) onto the trunk layout."""
    if any(key.startswith("stem.") for key in torchvision_state):
        return torchvision_state
    mapped = {}
    for key, value in torchvision_state.items():
        if key.startswith("conv1."):
            key = "stem.0." + key[len("conv1.") :]
        elif key.startswith("bn1."):
            key = "stem.1." + key[len("bn1.") :]
        mapped[key] = value
    return mapped


# ------------------------------------------------------ feature providers
class FeatureProvider(Protocol):
    """Anything that turns a normalized image batch into a spatial feature map."""

    out_channels: int

    def extract(self, images: torch.Tensor) -> torch.Tensor:  # (B, 3, H, W) -> (B, C, h, w)
        ...


class ResNet50MoCoFeatures(nn.Module):
    """Frozen trunk with last-stage stride 1; concatenates layer3 and layer4 maps."""

    out_channels = LAYER3_CHANNELS + TRUNK_CHANNELS

    def __init__(self, trunk: ResNet50Trunk) -> None:
        super().__init__()
        self.trunk = trunk.eval()
        for parameter in self.trunk.parameters():
            parameter.requires_grad = False

    @classmethod
    def from_checkpoint(cls, path: Path) -> ResNet50MoCoFeatures:
        state = load_checkpoint(path)
        trunk = ResNet50Trunk(layer4_stride=1, bn_splits=1)
        trunk.load_state_dict(state["trunk"])
        return cls(trunk)

    def train(self, mode: bool = True) -> ResNet50MoCoFeatures:
        super().train(False)  # frozen: BatchNorm always uses running statistics
        return self

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        layer3, layer4 = self.trunk.forward_maps(images)
        if layer3.shape[-2:] != layer4.shape[-2:]:
            layer3 = F.interpolate(layer3, size=layer4.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([layer3, layer4], dim=1)


class FusedFeatureProvider(nn.Module):
    """Channel-wise concatenation of several providers (USL-Net style fusion)."""

    def __init__(self, providers: Sequence[nn.Module]) -> None:
        super().__init__()
        self.providers = nn.ModuleList(providers)
        self.out_channels = sum(provider.out_channels for provider in providers)

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        maps = [provider.extract(images) for provider in self.providers]
        size = maps[0].shape[-2:]
        maps = [
            m if m.shape[-2:] == size else F.interpolate(m, size=size, mode="bilinear", align_corners=False)
            for m in maps
        ]
        return torch.cat(maps, dim=1)
