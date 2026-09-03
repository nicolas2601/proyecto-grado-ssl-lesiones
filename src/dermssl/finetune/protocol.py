"""Single source of truth for the fine-tuning protocol.

Everything a reviewer would ask about is a named constant here: which
backbones, where their weights come from, which datasets, and the exact
optimization recipe. ``scripts/build_finetune_tests.py`` (Kaggle notebooks)
and ``scripts/finetune_baseline.py`` (cluster) both follow these values;
``tests/test_finetune_protocol.py`` fails if either one diverges.

Only the standard library lives here on purpose: the job manifest, run names
and output paths must be computable on a login node without torch.
"""

from __future__ import annotations

from dataclasses import dataclass

MODELS: tuple[str, ...] = (
    "alexnet",
    "vgg19",
    "efficientnet_b0",
    "resnet50",
    "vit_b16",
    "simsiam_r50",
    "dinov2_vits14",
    "dinov2_vitb14",
)

# Hugging Face Hub tags served by timm (team decision with Dr. Karen,
# 2026-08-28). .tv_in1k / .tv2_in1k are the exact torchvision tensors.
TIMM_HF_TAGS: dict[str, str] = {
    "vgg19": "vgg19.tv_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "resnet50": "resnet50.tv2_in1k",
    "vit_b16": "vit_base_patch16_224.augreg_in1k",
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
}

# Documented exceptions without official HF weights.
TORCHVISION_MODELS: tuple[str, ...] = ("alexnet",)
SIMSIAM_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/simsiam/models/100ep-256bs/"
    "pretrain/checkpoint_0099.pth.tar"
)

# Models whose 224px activations do not fit batch 32 on a 16 GB card.
LOW_MEMORY_MODELS: tuple[str, ...] = ("vgg19", "vit_b16", "dinov2_vitb14")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    subdir: str            # folder under --data-root on the cluster
    metadata_csv: str      # file name searched recursively under subdir
    task: str              # "multiclass" | "binary"
    group_column: str      # column used for the anti-leak split ("" = none)
    json_suffix: str       # results_<model>_ft<json_suffix>.json


DATASETS: dict[str, DatasetSpec] = {
    "ham10000": DatasetSpec(
        name="ham10000", subdir="ham10000", metadata_csv="HAM10000_metadata.csv",
        task="multiclass", group_column="lesion_id", json_suffix="",
    ),
    "isic2020": DatasetSpec(
        name="isic2020", subdir="isic2020", metadata_csv="train-metadata.csv",
        task="binary", group_column="patient_id", json_suffix="_isic2020",
    ),
}

# Optimization recipe (identical for the 8 models and the 3 datasets).
EPOCHS = 8
BATCH = 32
LR_BACKBONE = 5e-5
LR_HEAD = 5e-4
WEIGHT_DECAY = 1e-4
SEED = 42
RESOLUTION = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SMOKE_EPOCHS = 1
SMOKE_FRACTION = {"ham10000": (0.08, 12, 6), "isic2020": (0.04, 40, 20)}  # (frac, min train/cls, min val/cls)

WANDB_PROJECT = "dermssl-baselines-finetune"
MLFLOW_EXPERIMENT_PREFIX = "finetune-"


@dataclass(frozen=True)
class Job:
    index: int
    model: str
    dataset: str

    @property
    def run_name(self) -> str:
        return f"ft-{self.model}-{self.dataset}-s{SEED}"

    @property
    def results_file(self) -> str:
        return f"results_{self.model}_ft{DATASETS[self.dataset].json_suffix}.json"

    @property
    def checkpoint_file(self) -> str:
        return f"ft_{self.model}{DATASETS[self.dataset].json_suffix}.pth"


def manifest() -> list[Job]:
    """The 16 (model, dataset) jobs in a fixed order: dataset-major.

    Dataset-major means the SLURM array finishes one whole dataset before
    starting the next, so a partial run still yields a complete table.
    """
    jobs: list[Job] = []
    for dataset in DATASETS:
        for model in MODELS:
            jobs.append(Job(index=len(jobs), model=model, dataset=dataset))
    return jobs


def job_from_index(index: int) -> Job:
    jobs = manifest()
    if not 0 <= index < len(jobs):
        raise ValueError(f"job index {index} out of range 0..{len(jobs) - 1}")
    return jobs[index]


def batch_size_for(model: str, requested: int) -> int:
    """Halve the batch for the three memory-hungry models (notebook README)."""
    if model in LOW_MEMORY_MODELS and requested > 16:
        return 16
    return requested
