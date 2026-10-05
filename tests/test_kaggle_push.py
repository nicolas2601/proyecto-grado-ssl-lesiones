"""kaggle_push must stage a correct kernel-metadata.json and flip SMOKE
without touching anything else in the notebook."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import kaggle_push as kp  # noqa: E402


def test_every_bank_maps_to_a_known_dataset_source():
    for bank, (dataset, _) in kp.BANKS.items():
        assert dataset in kp.DATASET_SOURCES, bank


def test_metadata_is_private_gpu_internet_with_dataset_source():
    m = kp.metadata("nicorghost", "baselines_isic2020", "resnet50", "x.ipynb")
    assert m["id"] == "nicorghost/baselines-isic2020-resnet50"
    assert m["enable_gpu"] and m["enable_internet"] and m["is_private"]
    assert m["dataset_sources"] == ["nischaydnk/isic-2020-jpg-256x256-resized"]


def test_metadata_pins_the_t4_shape_for_the_kaggle_cli():
    """The CLI reads machine_shape from kernel-metadata.json. Without it Kaggle
    may hand out a P100 (sm_60), which its PyTorch build has no kernels for."""
    m = kp.metadata("nicorghost", "baselines", "resnet50", "x.ipynb")
    assert m["machine_shape"] == "NvidiaTeslaT4"


@pytest.mark.parametrize("bank", list(kp.BANKS))
def test_stage_flips_smoke_and_keeps_cell_count(tmp_path, bank):
    dataset, pattern = kp.BANKS[bank]
    # Not every bank covers all eight models: the LR probe is a two-model
    # diagnostic. Pick whichever model this bank actually generated.
    bank_dir = REPO / "notebooks" / bank
    model = "resnet50" if (bank_dir / "resnet50").is_dir() else sorted(
        d.name for d in bank_dir.iterdir() if d.is_dir())[0]
    src = bank_dir / model / pattern.format(m=model)
    original = json.loads(src.read_text())
    folder = kp.stage(bank, model, "u", real=True, staging=tmp_path,
                      run_token="run-expected")
    staged = json.loads((folder / src.name).read_text())
    assert len(staged["cells"]) == len(original["cells"])
    code = "".join("".join(c["source"]) for c in staged["cells"] if c["cell_type"] == "code")
    assert "SMOKE = False" in code and "SMOKE = True" not in code
    assert code.count("KAGGLE_RUN_TOKEN = ") == 1
    assert 'KAGGLE_RUN_TOKEN = "run-expected"' in code
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert meta["code_file"] == src.name
    assert meta["dataset_sources"] == kp.DATASET_SOURCES[dataset]
    assert meta["machine_shape"] == "NvidiaTeslaT4"


def test_push_payload_matches_kaggle_kernels_push_schema(tmp_path):
    folder = kp.stage("baselines_isic2020", "resnet50", "nicorghost", real=False,
                      staging=tmp_path)
    body = kp.push_payload(folder)
    assert body["slug"] == "nicorghost/baselines-isic2020-resnet50"
    assert body["kernelType"] == "notebook" and body["language"] == "python"
    assert body["enableGpu"] and body["enableInternet"] and body["isPrivate"]
    assert body["datasetDataSources"] == ["nischaydnk/isic-2020-jpg-256x256-resized"]
    assert json.loads(body["text"])["nbformat"] == 4


def test_every_generated_notebook_bank_is_pushable():
    """A bank of notebooks nobody can push is a bank that never runs.

    finetune_isic2019 was generated on 2026-09-04 when the project moved off
    ISIC 2020, but BANKS was never updated, so `--bank finetune_isic2019` was
    rejected by argparse and the eight notebooks sat unreachable on disk.
    """
    on_disk = {d.name for d in (REPO / "notebooks").iterdir()
               if d.is_dir() and any(d.glob("*/*.ipynb"))}
    missing = on_disk - set(kp.BANKS)
    assert not missing, f"notebook banks with no entry in BANKS: {sorted(missing)}"


def test_staged_notebook_compiles_after_run_token_injection(tmp_path):
    folder = kp.stage("baselines", "resnet50", "u", real=True, staging=tmp_path,
                      run_token="run-expected")
    notebook = json.loads(next(folder.glob("*.ipynb")).read_text())
    code = "".join(
        line for cell in notebook["cells"] if cell["cell_type"] == "code" for line in cell["source"]
    )

    compile(code, "<staged notebook>", "exec")
