"""Contracts for source-preserving FST pool manifests and frozen splits."""

from dermssl import pool_manifest as pm


def test_freeze_grouped_splits_keeps_patients_and_exact_duplicates_together():
    records = [
        {
            "dataset": "ham10000",
            "image_id": "ham-a",
            "source_revision": "v1",
            "sha256": "a" * 64,
            "patient_or_lesion_group": "lesion-1",
        },
        {
            "dataset": "isic2019",
            "image_id": "isic-copy",
            "source_revision": "resized-v1",
            "sha256": "a" * 64,
            "patient_or_lesion_group": None,
        },
        {
            "dataset": "ham10000",
            "image_id": "ham-b",
            "source_revision": "v1",
            "sha256": "b" * 64,
            "patient_or_lesion_group": "lesion-2",
        },
        {
            "dataset": "isic2020",
            "image_id": "isic-2020-a",
            "source_revision": "resized-v1",
            "sha256": "c" * 64,
            "patient_or_lesion_group": "patient-2020-a",
        },
    ]

    result = pm.freeze_grouped_splits(records, seed=7)
    assigned = {record["image_id"]: record for record in result["images"]}

    assert assigned["ham-a"]["duplicate_group"] == assigned["isic-copy"]["duplicate_group"]
    assert assigned["ham-a"]["split"] == assigned["isic-copy"]["split"]
    assert result["split_hash"] == pm.split_hash(result["images"])
    assert any("ISIC2019" in limitation for limitation in result["limitations"])


def test_freeze_grouped_splits_keeps_near_perceptual_duplicates_together():
    records = [
        {
            "dataset": "ham10000",
            "image_id": "near-a",
            "source_revision": "v1",
            "sha256": "a" * 64,
            "perceptual_hash": "0000000000000000",
            "patient_or_lesion_group": None,
        },
        {
            "dataset": "isic2020",
            "image_id": "near-b",
            "source_revision": "v1",
            "sha256": "b" * 64,
            "perceptual_hash": "0000000000000001",
            "patient_or_lesion_group": None,
        },
    ]

    result = pm.freeze_grouped_splits(records, seed=11, perceptual_distance=1)
    assigned = {record["image_id"]: record for record in result["images"]}

    assert assigned["near-a"]["duplicate_group"].startswith("perceptual:")
    assert assigned["near-a"]["split"] == assigned["near-b"]["split"]


def test_validate_split_isolation_rejects_a_group_that_crosses_splits():
    images = [
        {
            "dataset": "ham10000",
            "image_id": "a",
            "source_revision": "v1",
            "duplicate_group": "exact:shared",
            "patient_or_lesion_group": "lesion-a",
            "split": "train",
        },
        {
            "dataset": "isic2019",
            "image_id": "b",
            "source_revision": "v1",
            "duplicate_group": "exact:shared",
            "patient_or_lesion_group": None,
            "split": "test",
        },
    ]

    assert pm.validate_split_isolation(images) == ["duplicate_group:exact:shared"]
