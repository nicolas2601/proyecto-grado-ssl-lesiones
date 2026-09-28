"""A tiny stand-in for DINOv2, shared by the S4-UNet tests.

It honours the two contracts the real ViT offers the S4 code:
``get_intermediate_layers`` (S4-UNet encoder) and ``forward_features``
(S-diag tokens, used by the A1 saliency builder). Weights are seeded in the
constructor, so two instances are identical, as two loads of DINOv2 are.
"""

from __future__ import annotations

import torch

EMBED = 12
PATCH = 14


class FakeVit(torch.nn.Module):
    patch_size = PATCH
    embed_dim = EMBED

    def __init__(self, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.patch = torch.nn.Conv2d(3, EMBED, PATCH, stride=PATCH)
        with torch.no_grad():
            for parameter in self.patch.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
        self.blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in range(12))
        self.requested = []

    def get_intermediate_layers(self, x, n, reshape=False, norm=True):
        self.requested.append((tuple(n), reshape, norm))
        base = self.patch(x)
        return tuple(base * (index + 1) for index in n)

    def forward_features(self, x):
        """Mean colour of each patch plus a constant, like the S-diag test fake."""
        pooled = torch.nn.functional.avg_pool2d(x, PATCH).flatten(2).transpose(1, 2)
        ones = torch.ones(*pooled.shape[:2], 1)
        return {"x_norm_patchtokens": torch.cat([pooled, ones], dim=2)}
