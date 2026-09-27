"""SimSiam - the ablation arm (ResNet from scratch) and the PROPOSED method.

Two configurations share this module:

* ``simsiam`` (ablation): ResNet-50 with weights=None - zero external weights,
  trained on the skin pool alone. Expected to lose to the ViT arms (48-70k
  images vs 142M); reporting that honestly bounds what own-data SSL achieves.
* ``simsiam_dinov2`` (proposed): the SimSiam objective over the DINOv2
  ViT-S/14 backbone, starting from the public label-free weights. The thesis'
  proposed method: a 2-view objective (no multi-crop, no teacher head) makes
  the continual adaptation cheaper than full DINO at the same label-free
  guarantee.

Heads follow Chen & He 2021: a 3-layer projection MLP and a 2-layer
prediction MLP with a bottleneck. The loss is symmetrized negative cosine
similarity with stop-gradient on the target branch; the stop-gradient is what
prevents collapse without negative pairs or momentum encoders.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

PROJ_DIM = 2048
PRED_HIDDEN = 512


def _build_backbone(backbone_name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """(backbone, feature_dim) for either backbone family."""
    if backbone_name == "resnet50":
        backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feat_dim
    backbone = torch.hub.load(
        "facebookresearch/dinov2", backbone_name, pretrained=pretrained
    )
    return backbone, backbone.embed_dim


class SimSiam(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = False,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if backbone is None:
            backbone, feat_dim = _build_backbone(backbone_name, pretrained)
        else:
            feat_dim = backbone.embed_dim
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
        self, view1: torch.Tensor, view2: torch.Tensor, return_features: bool = False
    ) -> tuple[torch.Tensor, ...]:
        """``(p1, p2, z1, z2)``, plus the backbone features when asked.

        The features come out of the SAME forward so DDP syncs their
        gradients; a separate method call on the wrapped module would not.
        """
        h1, h2 = self.backbone(view1), self.backbone(view2)
        z1, z2 = self.projector(h1), self.projector(h2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        if return_features:
            return p1, p2, z1.detach(), z2.detach(), h1, h2
        return p1, p2, z1.detach(), z2.detach()


def simsiam_loss(
    p1: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor
) -> torch.Tensor:
    """Symmetrized negative cosine similarity; targets are already detached."""

    def neg_cosine(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return -F.cosine_similarity(p, z, dim=-1).mean()

    return 0.5 * (neg_cosine(p1, z2) + neg_cosine(p2, z1))


def original_transformed_consistency_loss(
    p_original: torch.Tensor,
    p_transformed: torch.Tensor,
    z_original: torch.Tensor,
    z_transformed: torch.Tensor,
) -> torch.Tensor:
    """Representation consistency between one original and its transformation.

    Identical to ``simsiam_loss`` by construction: when the two views ARE the
    original and its tone-shifted copy, the symmetrised negative cosine over
    that pair IS the consistency. Kept as a named entry point for callers
    that build the pair themselves; the training loop uses
    ``paired_objective`` instead, which is what stops the term from being
    added to itself.
    """
    return simsiam_loss(p_original, p_transformed, z_original, z_transformed)


def paired_objective(
    ssl_loss: torch.Tensor, weight: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(loss, consistency)`` for the original/transformed arm.

    The proposed arm feeds the original as view 1 and its tone-shifted copy
    as view 2, so ``ssl_loss`` already measures the consistency between them.
    Summing a separately computed "consistency" term over the same four
    tensors - which is what this loop did until 2026-09-22 - doubles the
    gradient and makes the arm incomparable with the plain one (AdamW
    absorbs most of a constant factor, but not the weight decay term, and
    the reported losses become indistinguishable). Weighting is the knob;
    the default of 1.0 keeps both arms on the same footing.
    """
    return weight * ssl_loss, ssl_loss


def collapse_indicator(z: torch.Tensor) -> float:
    """Std of the L2-normalized embedding, averaged over dimensions.

    Healthy training hovers near 1/sqrt(dim); a slide toward 0 means the
    representations are collapsing to a constant. Logged to MLflow every
    step window so a collapsing run is visible long before the budget burns.
    """
    z_norm = F.normalize(z, dim=-1)
    return float(z_norm.std(dim=0).mean().item())
