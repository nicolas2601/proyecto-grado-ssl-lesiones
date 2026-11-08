"""Safe, self-contained runtime bundle for the two-head MoCo classifier."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from dermssl.release.plain_dinov2 import PlainDinov2

TASK_CLASSES = {
    "HAM": ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"),
    "PAD": ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK"),
}
OUTPUT_NAMES = ("ham7_probabilities", "pad6_probabilities")
SCHEMA = "dermssl.moco-classifier.v1"
RECIPE = {
    "features": "cls4",
    "layers": [8, 9, 10, 11],
    "layer_norm": True,
    "normalization": "row_l2",
    "views": 1,
    "tta": False,
    "prior": False,
    "standard_scaler": False,
    "tau": 0.0,
    "softmax_temperature": 1.0,
}


class MoCoClassifier(nn.Module):
    """CLS4 row-normalized DINOv2 features with HAM and PAD linear heads."""

    def __init__(self, backbone: PlainDinov2, heads: dict):
        super().__init__()
        if len(backbone.encoder.layer) < 4:
            raise ValueError("MoCo CLS4 requires at least four backbone layers")
        if set(heads) != set(TASK_CLASSES):
            raise ValueError("heads must contain exactly HAM and PAD")
        self.backbone = backbone
        dim = 4 * backbone.layernorm.normalized_shape[0]
        for task in TASK_CLASSES:
            head = heads[task]
            weight, bias = head.get("weight"), head.get("bias")
            count = len(TASK_CLASSES[task])
            if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
                raise ValueError(f"{task} head weight and bias must be tensors")
            if tuple(weight.shape) != (count, dim) or tuple(bias.shape) != (count,):
                raise ValueError(f"{task} head must have shapes ({count}, {dim}) and ({count},)")
            if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
                raise ValueError(f"{task} head tensors must be finite")
            self.register_buffer(f"{task.lower()}_weight", weight.detach().float().clone())
            self.register_buffer(f"{task.lower()}_bias", bias.detach().float().clone())

    def features(self, pixels: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone.forward_cls4(pixels), p=2, dim=-1, eps=1e-12)

    def forward(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(pixels)
        ham = features @ self.ham_weight.T + self.ham_bias
        pad = features @ self.pad_weight.T + self.pad_bias
        return torch.softmax(ham, dim=-1), torch.softmax(pad, dim=-1)


def _validate_metadata(metadata: dict) -> None:
    if metadata.get("schema") != SCHEMA:
        raise ValueError("unsupported MoCo checkpoint schema")
    classes = {task: list(names) for task, names in TASK_CLASSES.items()}
    if metadata.get("classes") != classes:
        raise ValueError("checkpoint classes do not match the MoCo contract")
    if metadata.get("recipe") != RECIPE:
        raise ValueError("checkpoint recipe does not match the MoCo contract")


def _model_for_config(config: dict) -> MoCoClassifier:
    try:
        backbone = PlainDinov2.from_config(config)
        width = 4 * backbone.layernorm.normalized_shape[0]
        heads = {
            task: {"weight": torch.zeros(len(names), width), "bias": torch.zeros(len(names))}
            for task, names in TASK_CLASSES.items()
        }
        return MoCoClassifier(backbone, heads)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid checkpoint config: {exc}") from exc


def _validate_primitive_metadata(config: dict, provenance: dict) -> None:
    if not isinstance(config, dict) or not isinstance(provenance, dict):
        raise ValueError("config and provenance metadata must be dictionaries of primitives")
    try:
        json.dumps({"config": config, "provenance": provenance}, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("config and provenance must contain only finite JSON primitives") from exc


def save_checkpoint(path, model: MoCoClassifier, config: dict, provenance: dict) -> Path:
    """Write a full weights-only-loadable checkpoint without replacing files."""
    path = Path(path)
    _validate_primitive_metadata(config, provenance)
    expected = _model_for_config(config)
    try:
        expected.load_state_dict(model.state_dict(), strict=True)
    except RuntimeError as exc:
        raise ValueError(f"checkpoint config does not match model state: {exc}") from exc
    if any(not torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
        raise ValueError("checkpoint tensors must be finite")
    metadata = {
        "schema": SCHEMA,
        "classes": {task: list(names) for task, names in TASK_CLASSES.items()},
        "recipe": RECIPE,
        "config": config,
        "provenance": provenance,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        torch.save(metadata | {"state_dict": state}, stream)
    return path


def load_checkpoint(path) -> tuple[MoCoClassifier, dict]:
    """Load and validate tensors and metadata without executable pickle objects."""
    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"could not safely load MoCo checkpoint: {exc}") from exc
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("checkpoint must contain a state_dict")
    _validate_metadata(checkpoint)
    config = checkpoint.get("config")
    _validate_primitive_metadata(config, checkpoint.get("provenance"))
    model = _model_for_config(config)
    state = checkpoint["state_dict"]
    if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
           for value in state.values()):
        raise ValueError("checkpoint state tensors must be finite tensors")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"checkpoint state shape mismatch: {exc}") from exc
    model.eval()
    return model, {key: value for key, value in checkpoint.items() if key != "state_dict"}
