"""SimSiam from RANDOM initialization - the purist SSL arm.

This arm answers one question for the thesis: what does SSL achieve using our
skin pool ALONE, with zero external weights of any kind? It is expected to
lose to continual DINOv2 (48-70k images vs 142M), and reporting that honestly
is part of the contribution (docs/15, option A).

Architecture follows Chen & He 2021: ResNet-50 encoder (weights=None - that
single argument is what makes this arm purist), a 3-layer projection MLP and
a 2-layer prediction MLP with a bottleneck. The loss is symmetrized negative
cosine similarity with stop-gradient on the target branch; the stop-gradient
is what prevents collapse without negative pairs or momentum encoders.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

PROJ_DIM = 2048
PRED_HIDDEN = 512


class SimSiam(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = resnet50(weights=None)  # random init: the purist guarantee
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, PROJ_DIM, bias=False),
            nn.BatchNorm1d(PROJ_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(PROJ_DIM, PROJ_DIM, bias=False),
            nn.BatchNorm1d(PROJ_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(PROJ_DIM, PROJ_DIM, bias=False),
            nn.BatchNorm1d(PROJ_DIM, affine=False),
        )
        self.predictor = nn.Sequential(
            nn.Linear(PROJ_DIM, PRED_HIDDEN, bias=False),
            nn.BatchNorm1d(PRED_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(PRED_HIDDEN, PROJ_DIM),
        )

    def forward(
        self, view1: torch.Tensor, view2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z1 = self.projector(self.backbone(view1))
        z2 = self.projector(self.backbone(view2))
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


def simsiam_loss(
    p1: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor
) -> torch.Tensor:
    """Symmetrized negative cosine similarity; targets are already detached."""

    def neg_cosine(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return -F.cosine_similarity(p, z, dim=-1).mean()

    return 0.5 * (neg_cosine(p1, z2) + neg_cosine(p2, z1))


def collapse_indicator(z: torch.Tensor) -> float:
    """Std of the L2-normalized embedding, averaged over dimensions.

    Healthy training hovers near 1/sqrt(dim); a slide toward 0 means the
    representations are collapsing to a constant. Logged to MLflow every
    step window so a collapsing run is visible long before the budget burns.
    """
    z_norm = F.normalize(z, dim=-1)
    return float(z_norm.std(dim=0).mean().item())
