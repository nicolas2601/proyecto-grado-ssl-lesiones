"""Own USL-Net components following the prose; no checkpoint downloads or trainer."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def masked_bce_logits(logits: torch.Tensor, pseudo: torch.Tensor) -> torch.Tensor:
    """Mean BCE over confident pixels; text interpretation, not printed eq.3.

    Fully uncertain input is an error requiring abstention. Inputs are 0/.5/1;
    ignore pixels contribute no gradient. No arbitrary .25 loss factor is added.
    """
    if logits.shape != pseudo.shape or not torch.isfinite(logits).all() or not torch.isfinite(pseudo).all():
        raise ValueError('Matching finite logits and pseudo-labels required')
    if not ((pseudo == 0) | (pseudo == .5) | (pseudo == 1)).all():
        raise ValueError('USL pseudo-label values must be 0, .5, 1')
    valid = pseudo != .5
    if not valid.any():
        raise ValueError('All-uncertain batch must abstain')
    return F.binary_cross_entropy_with_logits(logits[valid], pseudo[valid])


class USLTextCCAM(nn.Module):
    """1x1 background CCAM; foreground is its complement (paper sec.3.3).

    Caller supplies concatenated contrastive features. This head alone does not
    recreate contrastive pretraining or the original CCAM objective.
    """
    def __init__(self, in_channels: int, *, normalize_logits: bool = False, spatial_epsilon: float = 1e-6):
        super().__init__()
        if spatial_epsilon <= 0:
            raise ValueError('Spatial logit epsilon must be positive')
        self.normalize_logits = normalize_logits
        self.spatial_epsilon = float(spatial_epsilon)
        self.background = nn.Conv2d(in_channels, 1, kernel_size=1)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.background(features)
        if self.normalize_logits:
            logits = logits.float()
            mean = logits.mean(dim=(2, 3), keepdim=True)
            variance = logits.var(dim=(2, 3), unbiased=False, keepdim=True)
            logits = (logits - mean) / (variance + self.spatial_epsilon).sqrt()
        return logits

    def forward(self, features: torch.Tensor, output_size: tuple[int, int] = (224, 224)) -> torch.Tensor:
        background = torch.sigmoid(self.logits(features))
        return 1 - F.interpolate(background, size=output_size, mode='bilinear', align_corners=False)


class USLTextFCNDecoder(nn.Module):
    """Three ResNet stages -> 256 channels -> sum -> 1x1 -> bilinear logits.

    For torchvision ResNet50 these are layer1/2/3 (paper conv2/3/4), with
    256/512/1024 channels. 1x1 lateral projections are an explicit implementation
    choice; no unreported BN/ReLU is introduced. Backbone/weights remain separate.
    """
    def __init__(self, channels: tuple[int, int, int] = (256, 512, 1024)):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, 256, 1) for c in channels)
        self.output = nn.Conv2d(256, 1, 1)

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                output_size: tuple[int, int] = (224, 224)) -> torch.Tensor:
        if len(features) != 3:
            raise ValueError('Three ResNet stage features required')
        size = features[0].shape[-2:]
        projected = [F.interpolate(layer(x), size=size, mode='bilinear', align_corners=False)
                     for layer, x in zip(self.lateral, features, strict=True)]
        return F.interpolate(self.output(sum(projected)), size=output_size,
                             mode='bilinear', align_corners=False)
