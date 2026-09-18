"""Pure-logic contracts for the CIRCLe Stage 2 feature-bundle builder.

GPU-only parts (U-Net mask re-inference, frozen backbone forward pass,
NPZ assembly) are covered separately by a py_compile/ast static test,
mirroring tests/test_stage2_runner_static.py - never imported here.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import kaggle_circle_bundle as kcb  # noqa: E402


def _fst_record(image_id, *, whole_bucket, eligible, masked_bucket=None):
    return {
        "dataset": "target_pool",
        "image_id": image_id,
        "whole_image": {"bucket": whole_bucket},
        "masked": {
            "bucket": masked_bucket if masked_bucket is not None else "unavailable",
            "mask": {"eligible": eligible},
        },
    }


def test_is_bundle_split_accepts_train_and_validation():
    assert kcb.is_bundle_split("train") is True
    assert kcb.is_bundle_split("validation") is True


def test_is_bundle_split_rejects_frozen_test_split():
    """Test queda congelado (design.md): nunca entra a esta fase experimental."""
    assert kcb.is_bundle_split("test") is False


def test_partition_by_eligibility_keeps_only_eligible_records():
    records = [
        _fst_record("a", whole_bucket=1, eligible=True, masked_bucket=1),
        _fst_record("b", whole_bucket=5, eligible=False),
    ]

    kept, _ = kcb.partition_by_eligibility(records)

    assert [r["image_id"] for r in kept] == ["a"]


def test_partition_by_eligibility_counts_quarantine_by_whole_image_bucket():
    """El hallazgo de equidad: si la cuarentena concentra en un bucket FST,
    tiene que quedar contado explicitamente, no perdido en el descarte."""
    records = [
        _fst_record("a", whole_bucket=1, eligible=True, masked_bucket=1),
        _fst_record("b", whole_bucket=6, eligible=False),
        _fst_record("c", whole_bucket=6, eligible=False),
        _fst_record("d", whole_bucket=2, eligible=False),
    ]

    _, quarantine_counts = kcb.partition_by_eligibility(records)

    assert quarantine_counts == {6: 2, 2: 1}


def test_partition_by_eligibility_counts_unavailable_whole_bucket_explicitly():
    records = [
        _fst_record("a", whole_bucket="unavailable", eligible=False),
    ]

    _, quarantine_counts = kcb.partition_by_eligibility(records)

    assert quarantine_counts == {"unavailable": 1}


def test_sample_target_bucket_never_returns_the_source_bucket():
    rng = np.random.RandomState(42)

    for source in range(1, 7):
        for _ in range(20):
            target = kcb.sample_target_bucket(source, rng, max_shift=2)
            assert target != source


def test_sample_target_bucket_stays_within_max_shift_of_source():
    rng = np.random.RandomState(7)

    for source in range(1, 7):
        for _ in range(20):
            target = kcb.sample_target_bucket(source, rng, max_shift=2)
            assert abs(target - source) <= 2


def test_sample_target_bucket_is_deterministic_for_a_seeded_rng():
    first = kcb.sample_target_bucket(3, np.random.RandomState(123), max_shift=2)
    second = kcb.sample_target_bucket(3, np.random.RandomState(123), max_shift=2)

    assert first == second


def test_build_bundle_report_surfaces_quarantine_and_split_provenance():
    report = kcb.build_bundle_report(
        backbone="dino",
        split_hash="abc123",
        n_train=100,
        n_val=20,
        quarantine_counts={6: 2, 2: 1},
        seed=42,
    )

    assert report["backbone"] == "dino"
    assert report["split_hash"] == "abc123"
    assert report["n_train"] == 100
    assert report["n_val"] == 20
    assert report["quarantine_by_whole_image_bucket"] == {"6": 2, "2": 1}
    assert report["quarantine_total"] == 3
    assert report["seed"] == 42
