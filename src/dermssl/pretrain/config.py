"""Configuration and hard gates for the v6 SSL pretraining runs.

Why this module exists
----------------------
v5 proved that the acceptance logic must never live inside a notebook or a
launch script: code without tests ships bugs (the gate-averaging bug of the
first Kaggle run). v6 trains on a multi-GPU cluster where a wasted run costs
GPU-hours, so every decision that can be validated cheaply is validated here,
before torch is even imported.

The central invariant of the whole thesis rewrite: the pretraining stage is
COMPLETELY label-free. This module makes that invariant executable:

* the config has no field where a label file could even be expressed,
* ``assert_label_free`` scans every field for label-looking artifacts anyway
  (defense in depth against a future edit), and
* ``collect_images`` only ever returns image files, never metadata tables.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from dermssl import patch_guard

METHODS = ("simsiam", "dino_continual")
DEFAULT_BACKBONES = {"simsiam": "resnet50", "dino_continual": "dinov2_vits14"}
PRECISIONS = ("bf16", "fp16", "fp32")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Base learning rates follow the linear scaling rule of each method's paper:
# SimSiam uses 0.05 x batch/256 (SGD, from scratch). For continual pretraining
# we start from converged DINOv2 weights, so the base LR is deliberately small:
# 5e-4 x batch/1024 (AdamW). Both are provisional until the lab shares the
# config of a real DINOv2 run (see docs/16, pending item 1).
LR_RULES = {
    "simsiam": (0.05, 256),
    "dino_continual": (5e-4, 1024),
}

# Any config field whose name or value smells like supervision trips the gate.
FORBIDDEN_FIELD_TOKENS = ("label", "target", "class", "dx")
FORBIDDEN_VALUE_SUFFIXES = (".csv", ".tsv", ".parquet")


class LabelLeakError(RuntimeError):
    """Raised when anything label-shaped reaches the pretraining config."""


@dataclass(frozen=True)
class PretrainConfig:
    method: str
    image_root: str
    output_dir: str
    backbone: str = ""  # empty -> resolved from method by validate()
    resolution: int = 224
    local_resolution: int = 98  # 7 x 14; the DINO default 96 breaks patch 14
    n_local_crops: int = 6
    batch_size_per_gpu: int = 64
    budget_gpu_hours: float = 8.0
    checkpoint_fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)
    warmup_fraction: float = 0.05
    weight_decay: float = 1e-4
    momentum_teacher: float = 0.996  # dino_continual only (EMA start value)
    seed: int = 42
    num_workers: int = 8
    precision: str = "bf16"
    color_constancy_probs: tuple[tuple[str, float], ...] = (
        ("identity", 0.4),
        ("gray_world", 0.3),
        ("shades_of_gray", 0.3),
    )
    lesion_guided_crops: bool = True
    mlflow_experiment: str = "obj2-ssl-pretraining"
    tracking_uri: str = ""  # empty -> mlflow default (file:./mlruns or env)


def validate(cfg: PretrainConfig) -> PretrainConfig:
    """Return a normalized copy of ``cfg`` or raise with an actionable message."""
    if cfg.method not in METHODS:
        raise ValueError(f"unknown method {cfg.method!r}; expected one of {METHODS}")
    if cfg.precision not in PRECISIONS:
        raise ValueError(f"unknown precision {cfg.precision!r}; expected one of {PRECISIONS}")
    if cfg.budget_gpu_hours <= 0:
        raise ValueError(f"budget_gpu_hours must be positive, got {cfg.budget_gpu_hours}")
    _validate_fractions(cfg.checkpoint_fractions)

    backbone = cfg.backbone or DEFAULT_BACKBONES[cfg.method]
    patch_guard.assert_compatible(cfg.resolution, backbone)
    if cfg.method == "dino_continual":
        patch_guard.assert_compatible(cfg.local_resolution, backbone)

    normalized = dataclasses.replace(cfg, backbone=backbone)
    assert_label_free(normalized)
    return normalized


def _validate_fractions(fractions: tuple[float, ...]) -> None:
    if not fractions or list(fractions) != sorted(fractions):
        raise ValueError(f"checkpoint fractions must be ascending, got {fractions}")
    if any(f <= 0 or f > 1 for f in fractions):
        raise ValueError(f"checkpoint fractions must be in (0, 1], got {fractions}")
    if fractions[-1] != 1.0:
        raise ValueError(f"last checkpoint fraction must be 1.0, got {fractions[-1]}")


def assert_label_free(cfg: PretrainConfig) -> None:
    """Defense in depth: no field may carry anything label-shaped.

    The dataclass already has no label field; this guards against a future
    edit adding one, or a path pointing at a metadata table instead of an
    image directory.
    """
    for field in dataclasses.fields(cfg):
        if any(token in field.name.lower() for token in FORBIDDEN_FIELD_TOKENS):
            raise LabelLeakError(
                f"config field {field.name!r} looks like supervision; "
                "pretraining must stay label-free"
            )
        value = getattr(cfg, field.name)
        if isinstance(value, str) and value.lower().endswith(FORBIDDEN_VALUE_SUFFIXES):
            raise LabelLeakError(
                f"config field {field.name!r} points at a table ({value!r}); "
                "pretraining consumes image directories only"
            )


def collect_images(root: str | Path) -> list[Path]:
    """Every image under ``root``, and nothing that is not an image.

    Metadata tables (csv/tsv) are ignored by construction, which is the
    structural half of the label-free guarantee: the data pipeline cannot read
    labels because it never sees the files that contain them.
    """
    root = Path(root)
    found = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not found:
        raise FileNotFoundError(f"no images ({sorted(IMAGE_SUFFIXES)}) under {root}")
    return found


def effective_lr(cfg: PretrainConfig, world_size: int) -> float:
    """Linear-scaling learning rate for the total batch across all GPUs."""
    base, denominator = LR_RULES[cfg.method]
    total_batch = cfg.batch_size_per_gpu * world_size
    return base * total_batch / denominator


def mlflow_params(cfg: PretrainConfig) -> dict[str, str | int | float | bool]:
    """Flatten the config into MLflow-safe primitives.

    Tuples become comma-joined strings so every run in the UI shows the exact
    values without truncation or repr noise.
    """
    params: dict[str, str | int | float | bool] = {}
    for field in dataclasses.fields(cfg):
        value = getattr(cfg, field.name)
        if isinstance(value, tuple):
            if value and isinstance(value[0], tuple):
                value = ",".join(f"{name}:{prob}" for name, prob in value)
            else:
                value = ",".join(str(v) for v in value)
        params[field.name] = value
    return params
