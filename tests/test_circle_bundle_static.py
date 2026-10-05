"""Static contract for the GPU-only CIRCLe Stage 2 feature-bundle builder."""

import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "kaggle_circle_bundle.py"


def test_bundle_builder_exists_and_compiles():
    assert BUILDER.is_file()
    py_compile.compile(str(BUILDER), doraise=True)


def test_bundle_builder_imports_torch_only_after_argument_validation():
    source = BUILDER.read_text(encoding="utf-8")
    before_run = source.split("def run(")[0]

    assert "import torch" not in before_run
    assert "import timm" not in before_run


def test_bundle_builder_writes_all_required_stage2_bundle_keys():
    source = BUILDER.read_text(encoding="utf-8")

    for required in (
        "x_train_original",
        "x_train_shifted",
        "y_train",
        "x_val",
        "y_val",
        "fst_buckets_val",
        "backbone_frozen",
    ):
        assert required in source


def test_bundle_builder_excludes_frozen_test_split_and_ineligible_masks():
    source = BUILDER.read_text(encoding="utf-8")

    assert "is_bundle_split" in source
    assert "partition_by_eligibility" in source


def test_bundle_builder_samples_target_bucket_instead_of_a_fixed_shift():
    source = BUILDER.read_text(encoding="utf-8")

    assert "sample_target_bucket" in source
    assert "shift_skin_tone_from_mask_record" in source
