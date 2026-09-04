"""The fine-tuning protocol has ONE definition; notebooks, CLI and SLURM
array must all follow it. These tests fail the moment any of them drifts."""

import re
import sys
from pathlib import Path

import pytest

from dermssl.finetune import protocol
from dermssl.finetune.config import ConfigError, FinetuneConfig, validate, weight_source

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_baseline_tests as frozen_builder  # noqa: E402
import build_finetune_tests as ft_builder  # noqa: E402


# ---------------------------------------------------------------- manifest
def test_manifest_has_16_jobs_dataset_major_and_stable_order():
    jobs = protocol.manifest()
    assert len(jobs) == 16
    assert [j.index for j in jobs] == list(range(16))
    assert [j.dataset for j in jobs[:8]] == ["ham10000"] * 8
    assert [j.model for j in jobs[:8]] == list(protocol.MODELS)
    assert jobs[15].model == "dinov2_vitb14" and jobs[15].dataset == "isic2019"


def test_job_from_index_bounds():
    assert protocol.job_from_index(0).model == "alexnet"
    with pytest.raises(ValueError):
        protocol.job_from_index(16)
    with pytest.raises(ValueError):
        protocol.job_from_index(-1)


def test_results_file_names_match_notebook_convention():
    assert protocol.job_from_index(3).results_file == "results_resnet50_ft.json"
    assert protocol.job_from_index(11).results_file == "results_resnet50_ft_isic2019.json"
    assert protocol.job_from_index(11).checkpoint_file == "ft_resnet50_isic2019.pth"


def test_low_memory_models_get_batch_16():
    assert protocol.batch_size_for("vgg19", 32) == 16
    assert protocol.batch_size_for("resnet50", 32) == 32
    assert protocol.batch_size_for("vgg19", 8) == 8


# ---------------------------------------------------------------- parity with notebooks
def _consts(code: str) -> dict:
    return {k: v for k, v in re.findall(r"^([A-Z_]+) = ([^#\n]+)", code, re.M)}


def test_models_and_hf_tags_match_the_notebook_builder():
    assert tuple(frozen_builder.MODELS) == protocol.MODELS
    tags = dict(re.findall(r'"(\w+)": "([\w./-]+)",', frozen_builder.BACKBONE_CODE))
    assert tags == protocol.TIMM_HF_TAGS
    assert protocol.SIMSIAM_CHECKPOINT_URL.split("/")[-1] in frozen_builder.BACKBONE_CODE


def test_hyperparameters_match_the_finetune_notebook_builder():
    c = _consts(ft_builder.CONFIG_FT_CODE)
    assert int(c["EPOCHS"]) == protocol.EPOCHS
    assert int(c["BATCH"]) == protocol.BATCH
    assert float(c["LR_BACKBONE"]) == protocol.LR_BACKBONE
    assert float(c["LR_HEAD"]) == protocol.LR_HEAD
    assert int(c["SEED"]) == protocol.SEED
    assert "weight_decay=1e-4" in ft_builder.TRAIN_FT_CODE
    assert protocol.WEIGHT_DECAY == 1e-4
    assert set(frozen_builder.DATASETS) == set(protocol.DATASETS)
    for name, spec in frozen_builder.DATASETS.items():
        assert spec["json_suffix"] == protocol.DATASETS[name].json_suffix


# ---------------------------------------------------------------- config validation
def test_config_rejects_bad_values_before_touching_paths():
    base = dict(data_root="/x", output_dir="/y")
    with pytest.raises(ConfigError):
        validate(FinetuneConfig(model="nope", dataset="ham10000", **base), check_paths=False)
    with pytest.raises(ConfigError):
        validate(FinetuneConfig(model="resnet50", dataset="nope", **base), check_paths=False)
    with pytest.raises(ConfigError):
        validate(FinetuneConfig(model="resnet50", dataset="ham10000", epochs=0, **base),
                 check_paths=False)
    with pytest.raises(ConfigError):
        validate(FinetuneConfig(model="resnet50", dataset="ham10000", precision="int8", **base),
                 check_paths=False)


def test_config_checks_dataset_layout(tmp_path):
    cfg = FinetuneConfig(model="resnet50", dataset="ham10000",
                         data_root=str(tmp_path), output_dir=str(tmp_path / "out"))
    with pytest.raises(ConfigError, match="ham10000"):
        validate(cfg)
    (tmp_path / "ham10000" / "sub").mkdir(parents=True)
    with pytest.raises(ConfigError, match="HAM10000_metadata.csv"):
        validate(cfg)
    (tmp_path / "ham10000" / "sub" / "HAM10000_metadata.csv").write_text("image_id\n")
    assert validate(cfg) is cfg


def test_smoke_and_low_memory_are_reflected_in_effective_values():
    cfg = FinetuneConfig(model="vgg19", dataset="isic2020", data_root="/x",
                         output_dir="/y", smoke=True)
    assert cfg.effective_epochs == protocol.SMOKE_EPOCHS
    assert cfg.effective_batch == 16
    assert cfg.as_params()["fuente_pesos"] == "hf:timm/vgg19.tv_in1k"
    assert weight_source("alexnet").startswith("torchvision")
    assert weight_source("simsiam_r50") == protocol.SIMSIAM_CHECKPOINT_URL


def test_the_cluster_isic2019_loader_matches_the_notebook_cell():
    """The cluster runner cannot import torch here, so this checks the parity
    that matters by source: both halves must read the 8-class one-hot ground
    truth the same way, or Dr. Karen's numbers stop being comparable with the
    Kaggle bank.

    The failure this guards against is silent: ``_load_isic`` asserts a
    ``target`` column that ISIC 2019 does not have, and ``_load_ham`` reads
    ``dx``/``lesion_id`` that it does not have either. Routing isic2019 to
    either one dies, or worse, mislabels."""
    runner = (REPO / "src" / "dermssl" / "finetune" / "torch_run.py").read_text()
    cell = frozen_builder.DATA_CODE_ISIC2019MC

    assert "def _load_isic2019" in runner, "the cluster has no ISIC 2019 loader"
    # isic2019 must be routed by name, before the task-based branch: it is
    # multiclass like HAM10000 but shares none of its columns.
    assert runner.index('spec.name == "isic2019"') < runner.index('spec.task == "multiclass"')

    for op, why in ((".idxmax(axis=1)", "one-hot -> class name"),
                    ("ISIC2019_DROPPED_COLUMNS", "UNK must be dropped")):
        assert op in runner, f"cluster loader missing: {why}"
    assert "idxmax(axis=1)" in cell and "UNK" in cell

    # The 8 classes are pinned in the protocol and asserted by the runner, so a
    # changed ground truth fails loudly instead of producing a shifted mapping.
    assert "protocol.ISIC2019_CLASSES" in runner
    assert protocol.ISIC2019_CLASSES == ("MEL", "NV", "BCC", "AK", "BKL", "DF",
                                         "VASC", "SCC")
    assert protocol.ISIC2019_DROPPED_COLUMNS == ("UNK",)


def test_isic2019_carries_its_split_caveat_into_the_results_json():
    """``split_desc`` is written to every results JSON, so the limitation
    travels with the numbers instead of living only in the README."""
    runner = (REPO / "src" / "dermssl" / "finetune" / "torch_run.py").read_text()
    body = runner.split("def _load_isic2019")[1].split("\ndef ")[0]
    assert "sin grupo" in body.lower() or "sin agrupacion" in body.lower()
    assert protocol.DATASETS["isic2019"].group_column == "", \
        "isic2019 must declare it has no grouping column"


# ---------------------------------------------------------------- cluster wiring
def test_sbatch_array_covers_exactly_the_manifest():
    text = (REPO / "cluster" / "finetune" / "finetune_bank.sbatch").read_text()
    m = re.search(r"--array=0-(\d+)%(\d+)", text)
    assert m, "array directive missing"
    assert int(m.group(1)) == len(protocol.manifest()) - 1
    assert int(m.group(2)) == 4, "4-GPU priority queue: four concurrent jobs"
    assert "--gres=gpu:1" in text
    assert "--job-index" in text and "SLURM_ARRAY_TASK_ID" in text
    assert "HF_HUB_OFFLINE=1" in text and "WANDB_MODE=offline" in text


def test_cli_scripts_compile_and_list_jobs(tmp_path):
    import py_compile
    import subprocess

    for name in ("finetune_baseline.py",):
        py_compile.compile(str(SCRIPTS / name), doraise=True)
    out = subprocess.run([sys.executable, str(SCRIPTS / "finetune_baseline.py"),
                          "--list-jobs"], capture_output=True, text=True, check=True)
    assert out.stdout.count("\n") == 16
    bad = subprocess.run([sys.executable, str(SCRIPTS / "finetune_baseline.py"),
                          "--job-index", "3", "--data-root", str(tmp_path),
                          "--output-dir", str(tmp_path)], capture_output=True, text=True)
    assert bad.returncode != 0 and "config error" in bad.stderr
