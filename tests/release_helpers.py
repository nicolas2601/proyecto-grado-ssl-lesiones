"""Tiny random models for the release tests: no weights, no network.

The DINOv2 architecture comes from the facebookresearch/dinov2 source (the
same code torch.hub loads on the pod). It is looked up in ``$DINOV2_REPO``,
then in the torch hub cache. Absent source skips the test on a laptop and
fails it under ``CI_REQUIRE_FULL_DEPS=1``, where CI clones the repository.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

TINY = dict(embed_dim=32, depth=2, num_heads=2)


def dinov2_source() -> Path | None:
    candidates = [os.environ.get("DINOV2_REPO")]
    try:
        import torch  # noqa: PLC0415

        candidates.append(str(Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"))
    except ImportError:
        pass
    for candidate in candidates:
        if candidate and (Path(candidate) / "dinov2" / "models").is_dir():
            return Path(candidate)
    return None


def require_dinov2_source() -> Path:
    source = dinov2_source()
    if source is None:
        message = "DINOv2 source not found: set DINOV2_REPO to a facebookresearch/dinov2 checkout"
        if os.environ.get("CI_REQUIRE_FULL_DEPS") == "1":
            pytest.fail(message)
        pytest.skip(message)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return source


def tiny_hub_model(seed: int = 0, **overrides):
    """A random DinoVisionTransformer built like the hub's dinov2_vitb14, only smaller.

    img_size 518 keeps the 37x37 position grid of the real model, so the
    interpolation to 224 is exercised. LayerScale gammas are randomised so a
    mapping that drops them cannot pass.
    """
    import torch  # noqa: PLC0415

    require_dinov2_source()
    from dinov2.models.vision_transformer import DinoVisionTransformer  # noqa: PLC0415

    torch.manual_seed(seed)
    kwargs = dict(img_size=518, patch_size=14, mlp_ratio=4, init_values=1.0, block_chunks=0,
                  num_register_tokens=0, interpolate_antialias=False, interpolate_offset=0.1,
                  **TINY)
    kwargs.update(overrides)
    model = DinoVisionTransformer(**kwargs)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(torch.randn_like(param) * (0.5 if "gamma" in name else 0.2))
    return model.eval()
