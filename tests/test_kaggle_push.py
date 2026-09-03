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


@pytest.mark.parametrize("bank", list(kp.BANKS))
def test_stage_flips_smoke_and_keeps_cell_count(tmp_path, bank):
    dataset, pattern = kp.BANKS[bank]
    src = REPO / "notebooks" / bank / "resnet50" / pattern.format(m="resnet50")
    original = json.loads(src.read_text())
    folder = kp.stage(bank, "resnet50", "u", real=True, staging=tmp_path)
    staged = json.loads((folder / src.name).read_text())
    assert len(staged["cells"]) == len(original["cells"])
    code = "".join("".join(c["source"]) for c in staged["cells"] if c["cell_type"] == "code")
    assert "SMOKE = False" in code and "SMOKE = True" not in code
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert meta["code_file"] == src.name
    assert meta["dataset_sources"] == kp.DATASET_SOURCES[dataset]


def test_push_payload_matches_kaggle_kernels_push_schema(tmp_path):
    folder = kp.stage("baselines_isic2020", "resnet50", "nicorghost", real=False,
                      staging=tmp_path)
    body = kp.push_payload(folder)
    assert body["slug"] == "nicorghost/baselines-isic2020-resnet50"
    assert body["kernelType"] == "notebook" and body["language"] == "python"
    assert body["enableGpu"] and body["enableInternet"] and body["isPrivate"]
    assert body["datasetDataSources"] == ["nischaydnk/isic-2020-jpg-256x256-resized"]
    assert json.loads(body["text"])["nbformat"] == 4
