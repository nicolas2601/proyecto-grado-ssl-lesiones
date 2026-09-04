"""ISIC 2019 frozen bank: 8 real classes, and no real patient group.

This is the only bank whose task matches the Gessert anchor (0.636 balanced
accuracy on the official ISIC 2019 test set), so it is the only one that can be
compared against it head on. It is also the only bank without a real grouping
column: the RESIZED variant's patient_id is synthetic and andrewmvd/isic-2019
ships none at all, so the split is stratified only. That limit has to travel
with the numbers instead of living in someone's memory.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from dermssl.finetune import protocol  # noqa: E402

import build_baseline_tests as bbt  # noqa: E402
import kaggle_push as kp  # noqa: E402

BANK_DIR = REPO / "notebooks" / "baselines_isic2019"


def test_protocol_declares_isic2019_as_a_multiclass_dataset():
    spec = protocol.DATASETS["isic2019"]
    assert spec.task == "multiclass"
    assert spec.json_suffix == "_isic2019"


def test_isic2019_declares_that_it_has_no_real_grouping_column():
    """An empty group_column is the machine-readable form of the caveat."""
    assert protocol.DATASETS["isic2019"].group_column == ""
    # The other two banks do have one; this is what makes them comparable.
    assert protocol.DATASETS["ham10000"].group_column == "lesion_id"
    assert protocol.DATASETS["isic2020"].group_column == "patient_id"


def test_the_bank_points_at_the_multiclass_dataset_not_the_binary_resized():
    """nischaydnk/...-resized only ships a binary target; the eight classes
    that Gessert competed on live in andrewmvd/isic-2019."""
    assert kp.DATASET_SOURCES["isic2019"] == ["andrewmvd/isic-2019"]
    assert kp.BANKS["baselines_isic2019"] == ("isic2019", "test_{m}_isic2019.ipynb")


def test_builder_knows_the_bank():
    assert "isic2019" in bbt.DATASETS
    assert bbt.DATASETS["isic2019"]["out_dir"] == "baselines_isic2019"


def _code(model: str) -> str:
    path = BANK_DIR / model / f"test_{model}_isic2019.ipynb"
    assert path.is_file(), f"missing notebook {path}"
    return "".join("".join(c["source"])
                   for c in json.loads(path.read_text())["cells"]
                   if c["cell_type"] == "code")


def test_every_model_has_a_generated_notebook():
    for model in protocol.MODELS:
        assert _code(model)


def test_the_data_cell_reads_the_one_hot_ground_truth():
    code = _code("resnet50")
    assert "ISIC_2019_Training_GroundTruth.csv" in code
    assert "idxmax" in code, "one-hot columns must collapse to a single label"


def test_the_data_cell_drops_the_empty_unknown_column():
    """UNK is all zeros in the released ground truth; keeping it would invent a
    ninth class that never occurs and drag macro metrics down."""
    assert '"UNK"' in _code("resnet50")


def test_the_split_description_states_the_missing_group_out_loud():
    code = _code("resnet50")
    assert "sin grupo" in code.lower() or "sin patient_id" in code.lower()


def test_the_notebook_still_carries_the_feature_width_guard():
    assert "FEATURE_DIMS" in _code("vgg19")


def test_the_bank_writes_its_own_results_file():
    """The name is composed from JSON_SUFFIX, so the suffix is what to pin: a
    wrong one would overwrite another bank's results."""
    code = _code("resnet50")
    assert 'JSON_SUFFIX = "_isic2019"' in code
    assert 'f"results_{MODEL}{JSON_SUFFIX}.json"' in code


def test_isic2019_is_frozen_only_and_never_enters_the_cluster_manifest():
    """Adding it to DATASETS grew the SLURM array from 16 to 24 jobs, which
    would have handed Dr. Karen eight fine-tuning runs on the one dataset with
    no anti-leak split. The frozen bank can carry that caveat; cluster hours
    should not inherit it silently."""
    assert "isic2019" not in protocol.FINETUNE_DATASETS
    assert {j.dataset for j in protocol.manifest()} == set(protocol.FINETUNE_DATASETS)
    assert len(protocol.manifest()) == len(protocol.MODELS) * len(protocol.FINETUNE_DATASETS)


def test_the_result_name_the_runner_expects_matches_the_notebook():
    """kaggle_run_bank downloads by this name; a mismatch reports a stale run."""
    import kaggle_run_bank as krb
    assert krb.result_name("baselines_isic2019", "resnet50") == \
        "results_resnet50_isic2019.json"
