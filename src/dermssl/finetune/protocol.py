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

# Expected feature width per backbone: GAP of the last conv block (CNN) or the
# CLS token (ViT). Asserted at load time by both runners.
#
# vgg19 is the reason this exists. timm's `num_classes=0` returns VGG's 4096-d
# pre-logits (fc6/fc7 of the classifier head), not the 512-d GAP of conv5, so
# the HF migration silently moved the extraction point two fully-connected
# layers deeper and made that row incomparable with the rest of the bank.
FEATURE_DIMS: dict[str, int] = {
    "alexnet": 256,
    "vgg19": 512,
    "efficientnet_b0": 1280,
    "resnet50": 2048,
    "vit_b16": 768,
    "simsiam_r50": 2048,
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
}

# Backbones whose timm head must be bypassed to reach the conv features.
CONV_GAP_FROM_TIMM: tuple[str, ...] = ("vgg19",)

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
    # The only bank whose task matches the Gessert anchor (8 classes, official
    # ISIC 2019). Also the only one with no real grouping column: andrewmvd's
    # release ships none and the RESIZED variant's patient_id is synthetic
    # ("randomly generated"), so the split is stratified only. An empty
    # group_column is that caveat in machine-readable form - not an oversight.
    "isic2019": DatasetSpec(
        name="isic2019", subdir="isic2019",
        metadata_csv="ISIC_2019_Training_GroundTruth.csv",
        task="multiclass", group_column="", json_suffix="_isic2019",
    ),
}

# One-hot columns of ISIC_2019_Training_GroundTruth.csv. UNK is released as all
# zeros: keeping it would invent a ninth class that never occurs and drag every
# macro metric down by a factor of 9/8.
ISIC2019_CLASSES: tuple[str, ...] = ("MEL", "NV", "BCC", "AK", "BKL", "DF",
                                     "VASC", "SCC")
ISIC2019_DROPPED_COLUMNS: tuple[str, ...] = ("UNK",)

# Datasets that go into the cluster fine-tuning manifest. Deliberately NOT all
# of DATASETS: the array is 8 models wide, so every dataset added costs a whole
# extra round on Dr. Karen's four GPUs.
#
# isic2019 replaced isic2020 here (Nico, 2026-09-04). It is the only bank whose
# task matches the Gessert anchor (8 classes, official challenge), which is what
# makes the fine-tuning table comparable against the literature. Two caveats
# travel with that choice and are documented in cluster/finetune/README.md:
#   1. No real grouping column, so the split is stratified only and the metrics
#      carry an optimistic bias (same caveat as the frozen isic2019 bank).
#   2. ISIC 2019's training set contains HAM10000, so the two datasets in this
#      manifest are NOT independent.
# isic2020 stays in DATASETS: its frozen bank is already published and its JSONs
# under results/baselines_isic2020/ must keep resolving.
FINETUNE_DATASETS: tuple[str, ...] = ("ham10000", "isic2019")

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
# (frac, min train/cls, min val/cls). isic2019 mirrors the frozen bank's own
# smoke subsample so both halves of the project exercise the same subset shape.
SMOKE_FRACTION = {"ham10000": (0.08, 12, 6), "isic2020": (0.04, 40, 20),
                  "isic2019": (0.08, 12, 6)}

WANDB_PROJECT = "dermssl-baselines-finetune"
MLFLOW_EXPERIMENT_PREFIX = "finetune-"


OUTSIDE_MANIFEST = -1


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

    Built from FINETUNE_DATASETS, not from DATASETS: a dataset can be worth a
    frozen bank and still not be worth cluster hours. isic2020 is exactly that
    case now - its frozen bank is published, but the fine-tuning table is the
    one that has to line up with the Gessert anchor, so the cluster spends its
    rounds on isic2019 instead.
    """
    jobs: list[Job] = []
    for dataset in FINETUNE_DATASETS:
        for model in MODELS:
            jobs.append(Job(index=len(jobs), model=model, dataset=dataset))
    return jobs


class UnknownJobError(ValueError):
    """The (model, dataset) pair is not one the protocol defines."""


def job_for(model: str, dataset: str) -> Job:
    """The job for any (model, dataset) in MODELS x DATASETS, manifest or not.

    ``manifest()`` is the SLURM array and stays at 16 jobs; a pair outside it
    (isic2020, docs/33 G1.3) gets ``index == OUTSIDE_MANIFEST`` but the same
    run name, result file and checkpoint conventions.
    """
    if model not in MODELS or dataset not in DATASETS:
        raise UnknownJobError(
            f"no job for model={model!r} dataset={dataset!r}; "
            f"models {MODELS}, datasets {tuple(DATASETS)}")
    for job in manifest():
        if (job.model, job.dataset) == (model, dataset):
            return job
    return Job(index=OUTSIDE_MANIFEST, model=model, dataset=dataset)


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
