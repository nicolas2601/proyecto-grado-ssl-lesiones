"""Run configuration for one fine-tuning job, validated before torch loads.

A wrong flag must fail in milliseconds on the login node, not after waiting
in the SLURM queue. Everything here is plain Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dermssl.finetune import protocol


class ConfigError(ValueError):
    pass


@dataclass
class FinetuneConfig:
    model: str
    dataset: str
    data_root: str
    output_dir: str
    epochs: int = protocol.EPOCHS
    batch_size: int = protocol.BATCH
    lr_backbone: float = protocol.LR_BACKBONE
    lr_head: float = protocol.LR_HEAD
    weight_decay: float = protocol.WEIGHT_DECAY
    seed: int = protocol.SEED
    num_workers: int = 8
    smoke: bool = False
    precision: str = "fp16"   # "fp16" | "bf16" | "fp32"
    wandb_project: str = protocol.WANDB_PROJECT
    wandb_entity: str = ""
    wandb_mode: str = "offline"  # "offline" | "online" | "disabled"
    mlflow_tracking_uri: str = ""
    tensorboard_dir: str = ""
    extra_tags: list[str] = field(default_factory=list)

    @property
    def job_name(self) -> str:
        return f"ft-{self.model}-{self.dataset}-s{self.seed}"

    @property
    def dataset_spec(self) -> protocol.DatasetSpec:
        return protocol.DATASETS[self.dataset]

    @property
    def dataset_dir(self) -> Path:
        return Path(self.data_root) / self.dataset_spec.subdir

    @property
    def effective_epochs(self) -> int:
        return protocol.SMOKE_EPOCHS if self.smoke else self.epochs

    @property
    def effective_batch(self) -> int:
        return protocol.batch_size_for(self.model, self.batch_size)

    def as_params(self) -> dict:
        """Flat dict for the trackers (every value JSON-serialisable)."""
        return {
            "modelo": self.model, "dataset": self.dataset, "smoke": self.smoke,
            "epocas": self.effective_epochs, "batch": self.effective_batch,
            "lr_backbone": self.lr_backbone, "lr_head": self.lr_head,
            "weight_decay": self.weight_decay, "semilla": self.seed,
            "resolucion": protocol.RESOLUTION, "precision": self.precision,
            "fuente_pesos": weight_source(self.model),
        }


def weight_source(model: str) -> str:
    if model in protocol.TIMM_HF_TAGS:
        return f"hf:timm/{protocol.TIMM_HF_TAGS[model]}"
    if model in protocol.TORCHVISION_MODELS:
        return "torchvision:IMAGENET1K_V1"
    if model == "simsiam_r50":
        return protocol.SIMSIAM_CHECKPOINT_URL
    raise ConfigError(f"unknown model {model!r}")


def validate(cfg: FinetuneConfig, check_paths: bool = True) -> FinetuneConfig:
    if cfg.model not in protocol.MODELS:
        raise ConfigError(f"model must be one of {protocol.MODELS}, got {cfg.model!r}")
    if cfg.dataset not in protocol.DATASETS:
        raise ConfigError(
            f"dataset must be one of {tuple(protocol.DATASETS)}, got {cfg.dataset!r}")
    if cfg.epochs < 1:
        raise ConfigError("epochs must be >= 1")
    if cfg.batch_size < 1:
        raise ConfigError("batch_size must be >= 1")
    if cfg.lr_backbone <= 0 or cfg.lr_head <= 0:
        raise ConfigError("learning rates must be positive")
    if cfg.precision not in ("fp16", "bf16", "fp32"):
        raise ConfigError("precision must be fp16, bf16 or fp32")
    if cfg.wandb_mode not in ("offline", "online", "disabled"):
        raise ConfigError("wandb_mode must be offline, online or disabled")
    if check_paths:
        spec = cfg.dataset_spec
        if not cfg.dataset_dir.is_dir():
            raise ConfigError(
                f"dataset folder not found: {cfg.dataset_dir} "
                f"(expected <data-root>/{spec.subdir}/ with {spec.metadata_csv})")
        if not any(cfg.dataset_dir.rglob(spec.metadata_csv)):
            raise ConfigError(
                f"{spec.metadata_csv} not found under {cfg.dataset_dir}")
    return cfg
