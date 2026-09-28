"""S4-UNet: a frozen DINOv2 ViT with a light DPT-style decoder (docs/27).

Registered on 2026-09-28, before this code existed: the public DINOv2
ViT-B/14 stays frozen; the patch tokens of blocks 3, 6, 9 and 12 (one-based)
are reassembled into four maps at 4x, 2x, 1x and 0.5x the patch grid and
fused coarse to fine, as in DPT (Ranftl et al., 2021) / UNETR. Only the
decoder is trained, on pseudo-masks, so no reference mask is ever used.

The backbone contract is DINOv2's ``get_intermediate_layers(x, n, reshape,
norm)`` plus ``patch_size`` and ``embed_dim``; the tests use a tiny fake.
The checkpoint stores the decoder and the backbone identity, never the
backbone weights: ``model_from_checkpoint`` rebuilds it through a factory.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

ARCH = "s4_dinov2_dpt"
REGISTERED_BLOCKS = (3, 6, 9, 12)
DEFAULT_BLOCK_COUNT = 12
# Resampling factors are DPT's, relative to the patch grid. The widths are half
# of DPT-base, so the decoder (5.0 M parameters at ViT-B) stays lighter than
# the student U-Net it is compared with (7.77 M).
REASSEMBLE_CHANNELS = (48, 96, 192, 384)
REASSEMBLE_FACTORS = (4, 2, 1, 0.5)
DEFAULT_FEATURES = 128
HEAD_HIDDEN = 32
DINOV2_HUB = "facebookresearch/dinov2"


def zero_based(blocks: Sequence[int]) -> list[int]:
    """The registration counts blocks from 1; DINOv2 indexes them from 0."""
    return [int(block) - 1 for block in blocks]


def _validate_blocks(blocks: Sequence[int], depth: int) -> tuple[int, ...]:
    blocks = tuple(int(block) for block in blocks)
    ascending = all(a < b for a, b in zip(blocks, blocks[1:]))
    if len(blocks) != len(REASSEMBLE_FACTORS) or not ascending:
        raise ValueError(f"expected 4 distinct ascending blocks, got {blocks}")
    if blocks[0] < 1 or blocks[-1] > depth:
        raise ValueError(f"blocks {blocks} outside 1..{depth}")
    return blocks


def _resampler(channels: int, factor: float) -> nn.Module:
    if factor > 1:
        size = int(factor)
        return nn.ConvTranspose2d(channels, channels, size, stride=size)
    if factor < 1:
        return nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    return nn.Identity()


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.relu(self.conv1(F.relu(x))))


class FusionBlock(nn.Module):
    """Add the skip, refine, and upsample to the next (finer) map's size.

    The coarsest block has no skip (as in DPT), so it has no skip unit either:
    an unused unit would never receive a gradient.
    """

    def __init__(self, features: int, has_skip: bool = True):
        super().__init__()
        self.skip_unit = ResidualConvUnit(features) if has_skip else None
        self.unit = ResidualConvUnit(features)
        self.project = nn.Conv2d(features, features, 1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None, size) -> torch.Tensor:
        if self.skip_unit is not None:
            x = x + self.skip_unit(skip)
        x = F.interpolate(self.unit(x), size=size, mode="bilinear", align_corners=False)
        return self.project(x)


class DPTDecoder(nn.Module):
    """Four token maps (B, D, g, g) -> one logit map at the input resolution."""

    def __init__(self, embed_dim: int, features: int = DEFAULT_FEATURES):
        super().__init__()
        self.reassemble = nn.ModuleList(
            nn.Sequential(nn.Conv2d(embed_dim, width, 1), _resampler(width, factor))
            for width, factor in zip(REASSEMBLE_CHANNELS, REASSEMBLE_FACTORS)
        )
        self.to_features = nn.ModuleList(
            nn.Conv2d(width, features, 3, padding=1, bias=False) for width in REASSEMBLE_CHANNELS
        )
        last = len(REASSEMBLE_CHANNELS) - 1
        self.fusion = nn.ModuleList(
            FusionBlock(features, has_skip=index < last) for index in range(last + 1)
        )
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, 3, padding=1),
            nn.Conv2d(features // 2, HEAD_HIDDEN, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(HEAD_HIDDEN, 1, 1),
        )

    def forward(self, maps: Sequence[torch.Tensor], size) -> torch.Tensor:
        levels = [
            project(reassemble(tokens))
            for tokens, reassemble, project in zip(maps, self.reassemble, self.to_features)
        ]
        # Coarse to fine: level 3 (0.5x) ... level 0 (4x), then twice level 0.
        path = None
        for index in reversed(range(len(levels))):
            finer = levels[index - 1].shape[-2:] if index else [2 * s for s in levels[0].shape[-2:]]
            source = levels[index] if path is None else path
            path = self.fusion[index](source, None if path is None else levels[index], finer)
        logits = self.head(path)
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


class S4UNet(nn.Module):
    """Frozen DINOv2 encoder + trainable DPT decoder; forward returns logits."""

    def __init__(self, backbone: nn.Module, blocks: Sequence[int] = REGISTERED_BLOCKS,
                 features: int = DEFAULT_FEATURES):
        super().__init__()
        depth = len(getattr(backbone, "blocks", ())) or DEFAULT_BLOCK_COUNT
        self.blocks = _validate_blocks(blocks, depth)
        self.features = int(features)
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.decoder = DPTDecoder(int(backbone.embed_dim), self.features)
        self.backbone.eval()

    def train(self, mode: bool = True) -> S4UNet:
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        patch = int(self.backbone.patch_size)
        if x.shape[-1] % patch or x.shape[-2] % patch:
            raise ValueError(f"input {tuple(x.shape[-2:])} is not a multiple of patch {patch}")
        maps = self.backbone.get_intermediate_layers(
            x, n=zero_based(self.blocks), reshape=True, norm=True
        )
        return [m.detach() for m in maps]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x), x.shape[-2:])


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def decoder_state(model: S4UNet) -> dict:
    return {f"decoder.{k}": v for k, v in model.decoder.state_dict().items()}


def checkpoint_config(model: S4UNet, *, backbone_arch: str, backbone_source: str,
                      backbone_sha256: str, resolution: int) -> dict:
    return {
        "arch": ARCH, "backbone_arch": backbone_arch, "backbone_source": backbone_source,
        "backbone_sha256": backbone_sha256, "blocks": list(model.blocks),
        "features": model.features, "resolution": int(resolution),
    }


BackboneFactory = Callable[[str, str], nn.Module]


def model_from_checkpoint(checkpoint: Mapping, factory: BackboneFactory) -> S4UNet:
    """Rebuild the backbone from its identity, then load the decoder (eval mode)."""
    config = checkpoint["config"]
    if config.get("arch") != ARCH:
        raise ValueError(f"not an S4-UNet checkpoint: arch={config.get('arch')!r}")
    backbone = factory(config["backbone_arch"], config["backbone_source"])
    model = S4UNet(backbone, blocks=config["blocks"], features=config["features"])
    prefix = "decoder."
    state = {k[len(prefix):]: v for k, v in checkpoint["model"].items() if k.startswith(prefix)}
    model.decoder.load_state_dict(state)
    return model.eval()


def hub_backbone(arch: str, source: str) -> nn.Module:
    """DINOv2 from torch.hub: public weights, or a state_dict file (A* / B*)."""
    if source == "public":
        return torch.hub.load(DINOV2_HUB, arch, pretrained=True)
    model = torch.hub.load(DINOV2_HUB, arch, pretrained=False)
    model.load_state_dict(torch.load(source, map_location="cpu", weights_only=True))
    return model
