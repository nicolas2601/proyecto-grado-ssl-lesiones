"""Tiny random models for the release tests: no weights, no network.

The DINOv2 architecture comes from the facebookresearch/dinov2 source (the
same code torch.hub loads on the pod), looked up in ``$DINOV2_REPO`` and then
in the torch hub cache. Absent source skips the test on a laptop and fails it
under ``CI_REQUIRE_FULL_DEPS=1``, where CI clones the repository.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.check_release_guides import dinov2_source, tiny_hub


def require_dinov2_source() -> Path:
    source = dinov2_source()
    if source is None:
        message = "DINOv2 source not found: set DINOV2_REPO to a facebookresearch/dinov2 checkout"
        if os.environ.get("CI_REQUIRE_FULL_DEPS") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return source


def tiny_hub_model(seed: int = 0, **overrides):
    """A random DinoVisionTransformer built like the hub's dinov2_vitb14, only smaller.

    img_size 518 keeps the 37x37 position grid of the real model, so the
    interpolation to 224 is exercised. LayerScale gammas are randomised so a
    mapping that drops them cannot pass. Same builder as the guide check.
    """
    return tiny_hub(require_dinov2_source(), seed=seed, **overrides)
