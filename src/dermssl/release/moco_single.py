"""MoCo classifier at 448 px with a single view (no test-time augmentation).

Same encoder, CLS4 features and two-head contract as ``moco_d4``, but each image is encoded once:
one ViT pass instead of eight. Checkpoints use schema v3 and cannot be loaded as v1 or v2.
"""
from __future__ import annotations

from pathlib import Path

import torch

from dermssl.release import moco_d4
from dermssl.release.moco_bundle import TASK_CLASSES, MoCoClassifier, _validate_primitive_metadata
from dermssl.release.plain_dinov2 import PlainDinov2

SCHEMA = "dermssl.moco-classifier.v3"
RECIPE = {**moco_d4.RECIPE, "views": 1, "tta": "none"}
CLASSES = {task: list(names) for task, names in TASK_CLASSES.items()}


class SingleViewMoCoClassifier(MoCoClassifier):
    """Heads read the row-normalized CLS4 of the image itself (the base MoCo feature)."""


def _model_for_config(config: dict) -> SingleViewMoCoClassifier:
    backbone = PlainDinov2.from_config(config)
    width = 4 * backbone.layernorm.normalized_shape[0]
    heads = {task: {"weight": torch.zeros(len(names), width), "bias": torch.zeros(len(names))}
             for task, names in TASK_CLASSES.items()}
    return SingleViewMoCoClassifier(backbone, heads)


def from_d4(model: moco_d4.D4MoCoClassifier, ham_weight: torch.Tensor | None = None,
            ham_bias: torch.Tensor | None = None) -> SingleViewMoCoClassifier:
    """Same encoder and heads as a D4 model, optionally with a HAM head refit for the single view."""
    heads = {"HAM": {"weight": model.ham_weight if ham_weight is None else ham_weight,
                     "bias": model.ham_bias if ham_bias is None else ham_bias},
             "PAD": {"weight": model.pad_weight, "bias": model.pad_bias}}
    return SingleViewMoCoClassifier(model.backbone, heads).eval()


def _check_finite(state: dict) -> None:
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("checkpoint tensors must be finite")


def save_checkpoint(path, model: SingleViewMoCoClassifier, config: dict, provenance: dict) -> Path:
    path = Path(path)
    _validate_primitive_metadata(config, provenance)
    _model_for_config(config).load_state_dict(model.state_dict(), strict=True)
    _check_finite(model.state_dict())
    metadata = {"schema": SCHEMA, "classes": CLASSES, "recipe": RECIPE, "config": config, "provenance": provenance}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        torch.save(metadata | {"state_dict": state}, stream)
    return path


def load_checkpoint(path) -> tuple[SingleViewMoCoClassifier, dict]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != SCHEMA:
        raise ValueError("unsupported single-view checkpoint schema")
    if checkpoint.get("recipe") != RECIPE or checkpoint.get("classes") != CLASSES:
        raise ValueError("checkpoint recipe/classes do not match the single-view contract")
    _validate_primitive_metadata(checkpoint.get("config"), checkpoint.get("provenance"))
    _check_finite(checkpoint["state_dict"])
    model = _model_for_config(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, {k: v for k, v in checkpoint.items() if k != "state_dict"}
