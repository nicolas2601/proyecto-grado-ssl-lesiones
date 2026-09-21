"""Class-agnostic activation map head and contrastive loss (CCAM / C2AM).

Re-implemented from Xie et al., "C2AM: Contrastive learning of Class-agnostic
Activation Map ...", CVPR 2022 (the official repository is MIT licensed; no
code was copied). Per image, the activation map P splits the feature map Z
into a foreground vector ``v_f = sum(P * Z)`` and a background vector
``v_b = sum((1 - P) * Z)``. Cosine similarities across the batch give:

* negative pairs: foreground of image i against background of image j;
* positive pairs: foreground-foreground and background-background of
  different images, weighted by ``exp(-alpha * rank)`` so only the most
  similar pairs pull together (ranking-based weighting).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-6


class CCAMHead(nn.Module):
    """Convolution + BatchNorm + sigmoid producing a one-channel activation map."""

    def __init__(self, in_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size, padding=kernel_size // 2)
        self.norm = nn.BatchNorm2d(1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, C, h, w) -> activation in (0, 1): (B, 1, h, w)
        return torch.sigmoid(self.norm(self.conv(features)))


@torch.no_grad()
def rank_weights(similarity: torch.Tensor, alpha: float) -> torch.Tensor:
    """``exp(-alpha * rank)`` per row, ranking the off-diagonal similarities (0 = most similar)."""
    masked = similarity.detach().clone()
    masked.fill_diagonal_(float("-inf"))
    ranks = masked.argsort(dim=1, descending=True).argsort(dim=1).float()
    return torch.exp(-alpha * ranks)


def _positive_loss(similarity: torch.Tensor, alpha: float) -> torch.Tensor:
    count = similarity.shape[0]
    weights = rank_weights(similarity, alpha)
    off_diagonal = 1.0 - torch.eye(count, device=similarity.device)
    log_similarity = torch.log(similarity.clamp(min=EPSILON))
    return -(weights * log_similarity * off_diagonal).sum() / (count * (count - 1))


def ccam_loss(features: torch.Tensor, activation: torch.Tensor, alpha: float) -> torch.Tensor:
    """Contrastive CCAM loss ``L_POS + L_NEG`` over a batch (float32).

    ``features`` are non-negative (post-ReLU), so cosine similarities lie in [0, 1].
    """
    count = features.shape[0]
    if count < 2:
        raise ValueError("the CCAM loss needs at least two images per batch")
    flat_features = features.float().flatten(2)  # (B, C, hw)
    flat_activation = activation.float().flatten(2)  # (B, 1, hw)
    foreground = F.normalize((flat_features * flat_activation).sum(-1), dim=1, eps=EPSILON)
    background = F.normalize((flat_features * (1 - flat_activation)).sum(-1), dim=1, eps=EPSILON)

    negative = -torch.log((1.0 - (foreground @ background.t()).clamp(max=1.0 - EPSILON))).mean()
    positive_fg = _positive_loss(foreground @ foreground.t(), alpha)
    positive_bg = _positive_loss(background @ background.t(), alpha)
    return positive_fg + positive_bg + negative
