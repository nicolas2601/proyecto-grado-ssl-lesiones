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

import pytest

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


def _split_desc(model: str) -> str:
    """The SPLIT_DESC assignment only - not the whole notebook.

    Searching the full code blob passed even with the caveat deleted from
    SPLIT_DESC and left in a comment, which is the one place it does NOT
    travel: SPLIT_DESC is what lands in the results JSON and reaches whoever
    reads the numbers. Verified by falsifying it.
    """
    code = _code(model)
    start = code.index("SPLIT_DESC = ")
    tail = code[start:]
    end = tail.index("\nprint(") if "\nprint(" in tail else len(tail)
    return tail[:end]


def test_the_split_description_states_the_missing_group_out_loud():
    desc = _split_desc("resnet50").lower()
    assert "sin grupo" in desc or "sin patient_id" in desc, (
        "el caveat de fuga no esta en SPLIT_DESC, que es el campo que viaja "
        "dentro del results JSON")


def test_the_caveat_is_not_satisfied_by_a_comment_elsewhere():
    """Guards the guard: the assertion must read SPLIT_DESC, not the file."""
    desc = _split_desc("resnet50")
    assert "SPLIT_DESC" in desc and len(desc) < len(_code("resnet50"))


def test_the_notebook_still_carries_the_feature_width_guard():
    assert "FEATURE_DIMS" in _code("vgg19")


def test_the_bank_writes_its_own_results_file():
    """The name is composed from JSON_SUFFIX, so the suffix is what to pin: a
    wrong one would overwrite another bank's results."""
    code = _code("resnet50")
    assert 'JSON_SUFFIX = "_isic2019"' in code
    assert 'f"results_{MODEL}{JSON_SUFFIX}.json"' in code


def test_the_manifest_is_exactly_the_finetune_datasets():
    """The array size is derived, never hardcoded: FINETUNE_DATASETS is the one
    place that decides what Dr. Karen's cluster spends its rounds on."""
    assert {j.dataset for j in protocol.manifest()} == set(protocol.FINETUNE_DATASETS)
    assert len(protocol.manifest()) == len(protocol.MODELS) * len(protocol.FINETUNE_DATASETS)


def test_a_dataset_without_grouping_may_enter_the_cluster_only_if_documented():
    """Replaces the older rule that kept isic2019 out of the manifest entirely.

    Nico moved the fine-tuning bank from isic2020 to isic2019 (2026-09-04) to
    match the Gessert anchor's 8-class task. That trades a real ``patient_id``
    for a stratified-only split, so the numbers carry an optimistic bias that
    fine-tuning amplifies - the model can memorise a patient across the split.

    The decision is the user's; losing the caveat is not. Any dataset that
    enters FINETUNE_DATASETS without a grouping column has to say so in the
    document Dr. Karen actually reads, not only in a code comment."""
    ungrouped = [d for d in protocol.FINETUNE_DATASETS
                 if not protocol.DATASETS[d].group_column]
    if not ungrouped:
        pytest.skip("every fine-tuning dataset has a real grouping column")
    readme = (REPO / "cluster" / "finetune" / "README.md").read_text()
    for dataset in ungrouped:
        assert dataset in readme, f"{dataset} corre en el cluster y el README no lo nombra"
    assert "sin agrupacion" in readme.lower() or "sin agrupación" in readme.lower(), \
        "el README del cluster no documenta la limitacion del split sin grupo"


def test_the_result_name_the_runner_expects_matches_the_notebook():
    """kaggle_run_bank downloads by this name; a mismatch reports a stale run."""
    import kaggle_run_bank as krb
    assert krb.result_name("baselines_isic2019", "resnet50") == \
        "results_resnet50_isic2019.json"
