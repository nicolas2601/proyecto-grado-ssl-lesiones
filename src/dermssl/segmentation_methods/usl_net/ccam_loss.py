"""CCAM objective matching the published pairwise cosine/rank formulation.

This is a documented mathematical reimplementation of the reference loss at
https://github.com/CVI-SZU/CCAM/blob/master/CUSTOM/core/loss.py. USL-Net keeps
its own 1x1 head and pooled foreground/background descriptors, so this is an
objective adaptation rather than a reproduction of the reference architecture.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def _cosine_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.normalize(left, dim=1) @ F.normalize(right, dim=1).T


def _rank_weighted_positive(descriptors: torch.Tensor, alpha: float) -> torch.Tensor:
    similarity = _cosine_matrix(descriptors, descriptors).clamp(0.0005, 0.9995)
    # The reference ranks every entry in each row, including the self-pair.
    order = similarity.detach().sort(descending=True, dim=1).indices
    rank = order.sort(dim=1).indices.to(dtype=torch.float32) - 1.0
    weights = torch.exp(-rank * alpha)
    loss = -torch.log(similarity)
    return (loss * weights).mean()


def ccam_loss_components(features: torch.Tensor, foreground: torch.Tensor,
                         alpha: float = 0.25) -> dict[str, torch.Tensor]:
    """Return the reference-form positive FG/BG and negative pair losses.

    The reduction is the mean over B squared similarities, with diagonal
    entries included, as in the published source. Float32 computation avoids
    half-precision loss/gradient underflow while preserving the model graph.
    """
    if features.ndim != 4 or foreground.ndim != 4 or foreground.shape[1] != 1:
        raise ValueError('Expected feature maps [B,C,H,W] and foreground [B,1,H,W]')
    if features.shape[0] < 2 or features.shape[0] != foreground.shape[0]:
        raise ValueError('CCAM pairwise objective requires matching batch size >= 2')
    if features.shape[2:] != foreground.shape[2:]:
        foreground = F.interpolate(foreground, size=features.shape[2:], mode='bilinear', align_corners=False)
    value = features.float().flatten(2)
    activation = foreground.float().flatten(2)
    fg = (value * activation).sum(-1)
    bg = (value * (1.0 - activation)).sum(-1)
    fg = F.normalize(fg, dim=1)
    bg = F.normalize(bg, dim=1)

    cross = _cosine_matrix(bg, fg).clamp(0.0005, 0.9995)
    negative = -torch.log(1.0 - cross).mean()
    positive_fg = _rank_weighted_positive(fg, alpha)
    positive_bg = _rank_weighted_positive(bg, alpha)
    return {'positive_foreground': positive_fg, 'positive_background': positive_bg,
            'negative': negative, 'total': positive_fg + positive_bg + negative}


def ccam_loss(features: torch.Tensor, foreground: torch.Tensor,
              alpha: float = 0.25) -> torch.Tensor:
    return ccam_loss_components(features, foreground, alpha)['total']
