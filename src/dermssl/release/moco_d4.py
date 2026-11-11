"""MoCo classifier at 448 px with the D4 test-time augmentation averaged inside the exported graph.

Same two-head contract as ``moco_bundle`` (CLS4 -> row L2 -> HAM7/PAD6 softmax), but each image is
encoded under the 8 dihedral views and the CLS4 vectors are averaged before normalization, which is
the exact ``cls4_d4`` recipe of the development evidence. Checkpoints use schema v2 and cannot be
loaded as v1 (or the reverse).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from dermssl.release.moco_bundle import TASK_CLASSES, MoCoClassifier, _validate_primitive_metadata
from dermssl.release.plain_dinov2 import PlainDinov2

SCHEMA = "dermssl.moco-classifier.v2"
RESOLUTION = 448
RECIPE = {"features": "cls4", "layers": [8, 9, 10, 11], "layer_norm": True, "normalization": "row_l2",
          "resolution": RESOLUTION, "views": 8, "tta": "d4-mean", "prior": False,
          "standard_scaler": False, "tau": 0.0, "softmax_temperature": 1.0}


def _rot90(pixels: torch.Tensor, k: int) -> torch.Tensor:
    """torch.rot90 over (H, W) built from flip+transpose, which the ONNX exporter supports."""
    if k == 0:
        return pixels
    if k == 1:
        return torch.flip(pixels, dims=(3,)).transpose(2, 3)
    if k == 2:
        return torch.flip(pixels, dims=(2, 3))
    return torch.flip(pixels, dims=(2,)).transpose(2, 3)


def d4_views(pixels: torch.Tensor) -> list[torch.Tensor]:
    """The 8 dihedral views, in the same construction as the development extractor."""
    views = []
    for k in range(4):
        rotated = _rot90(pixels, k)
        views += [rotated, torch.flip(rotated, dims=(3,))]
    return views


class D4MoCoClassifier(MoCoClassifier):
    """Heads read the row-normalized mean of CLS4 over the 8 D4 views."""

    def features(self, pixels: torch.Tensor) -> torch.Tensor:
        cls4 = torch.stack([self.backbone.forward_cls4(view) for view in d4_views(pixels)])
        return F.normalize(cls4.mean(0), p=2, dim=-1, eps=1e-12)


def _model_for_config(config: dict) -> D4MoCoClassifier:
    backbone = PlainDinov2.from_config(config)
    width = 4 * backbone.layernorm.normalized_shape[0]
    heads = {task: {"weight": torch.zeros(len(names), width), "bias": torch.zeros(len(names))}
             for task, names in TASK_CLASSES.items()}
    return D4MoCoClassifier(backbone, heads)


def save_checkpoint(path, model: D4MoCoClassifier, config: dict, provenance: dict) -> Path:
    path = Path(path)
    _validate_primitive_metadata(config, provenance)
    expected = _model_for_config(config)
    expected.load_state_dict(model.state_dict(), strict=True)
    if any(not torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
        raise ValueError("checkpoint tensors must be finite")
    metadata = {"schema": SCHEMA, "classes": {t: list(n) for t, n in TASK_CLASSES.items()},
                "recipe": RECIPE, "config": config, "provenance": provenance}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        torch.save(metadata | {"state_dict": state}, stream)
    return path


def load_checkpoint(path) -> tuple[D4MoCoClassifier, dict]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != SCHEMA:
        raise ValueError("unsupported D4 checkpoint schema")
    if checkpoint.get("recipe") != RECIPE or checkpoint.get("classes") != {
            t: list(n) for t, n in TASK_CLASSES.items()}:
        raise ValueError("checkpoint recipe/classes do not match the D4 contract")
    _validate_primitive_metadata(checkpoint.get("config"), checkpoint.get("provenance"))
    model = _model_for_config(checkpoint["config"])
    state = checkpoint["state_dict"]
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("checkpoint state tensors must be finite")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, {k: v for k, v in checkpoint.items() if k != "state_dict"}
