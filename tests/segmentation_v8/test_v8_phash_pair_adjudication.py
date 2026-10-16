"""Conservative pHash eligibility decisions; execute these fixtures only in UNABIA."""

from copy import deepcopy

import pytest

from scripts.adjudicate_v8_phash_pairs import adjudicate_pairs


@pytest.fixture
def inputs():
    pool = [
        {"image_id": "ISIC_1000001_downsampled", "dataset": "isic2019",
         "image_path": "/fixture/train/a.jpg", "image_sha256": "a" * 64,
         "bytes": 42, "width": 224, "height": 224},
        {"image_id": "ISIC_1000002", "dataset": "isic2020",
         "image_path": "/fixture/train/b.jpg", "image_sha256": "b" * 64,
         "bytes": 43, "width": 224, "height": 224},
        {"image_id": "ISIC_1000003", "dataset": "ham10000_p1",
         "image_path": "/fixture/train/c.jpg", "image_sha256": "c" * 64,
         "bytes": 44, "width": 224, "height": 224},
    ]
    evaluation = [
        {"eval_id": "ISIC_0000001", "eval_member": "evaluation/ISIC_0000001.jpg"},
        {"eval_id": "ISIC_0000002", "eval_member": "evaluation/ISIC_0000002.jpg"},
    ]
    pairs = [
        {"eval_id": "ISIC_0000002", "eval_member": "evaluation/ISIC_0000002.jpg",
         "train_id": "ISIC_1000001", "train_path": "/fixture/train/a.jpg",
         "hamming_distance_63bit": 0, "exact_file_sha256_match": False,
         "status": "review_pending", "exclusion_applied": False},
        {"eval_id": "ISIC_0000001", "eval_member": "evaluation/ISIC_0000001.jpg",
         "train_id": "ISIC_1000002", "train_path": "/fixture/train/b.jpg",
         "hamming_distance_63bit": 8, "exact_file_sha256_match": False,
         "status": "review_pending", "exclusion_applied": False},
        {"eval_id": "ISIC_0000001", "eval_member": "evaluation/ISIC_0000001.jpg",
         "train_id": "ISIC_1000001", "train_path": "/fixture/train/a.jpg",
         "hamming_distance_63bit": 2, "exact_file_sha256_match": False,
         "status": "review_pending", "exclusion_applied": False},
    ]
    return pairs, pool, evaluation


def test_preserves_every_edge_but_deduplicates_training_path_exclusions(inputs):
    # Catches nearest-only adjudication and counting edges as distinct training images.
    pairs, pool, evaluation = inputs

    result = adjudicate_pairs(pairs, pool, evaluation)

    assert [(row["eval_id"], row["train_path"])
            for row in result["pair_decisions"]] == [
        ("ISIC_0000002", "/fixture/train/a.jpg"),
        ("ISIC_0000001", "/fixture/train/b.jpg"),
        ("ISIC_0000001", "/fixture/train/a.jpg"),
    ]
    assert result["excluded_training_paths"] == [
        "/fixture/train/a.jpg", "/fixture/train/b.jpg",
    ]
    assert all(row["disposition"] == "exclude_training_path_conservatively"
               for row in result["pair_decisions"])


def test_zero_distance_does_not_certify_identity_or_independence(inputs):
    # Catches treating pHash equality as duplicate identity or a final-manifest gate.
    result = adjudicate_pairs(*inputs)

    assert result["pair_decisions"][0]["hamming_distance_63bit"] == 0
    assert all(row["duplicate_identity_status"] == "unverified"
               for row in result["pair_decisions"])
    assert result["manifest_frozen"] is False
    assert result["training_consumable"] is False
    assert result["patient_independence_accredited"] is False
    assert result["execution_enabled"] is False


def test_decisions_do_not_mutate_source_evidence(inputs):
    # Catches overwriting the historical pending register in place.
    originals = deepcopy(inputs)

    result = adjudicate_pairs(*inputs)
    result["pair_decisions"][0]["train_path"] = "/changed/decision-only.jpg"

    assert inputs == originals


@pytest.mark.parametrize("field,value", [
    ("eval_id", "ISIC_9999999"),
    ("eval_member", "evaluation/wrong.jpg"),
    ("train_path", "/fixture/train/unknown.jpg"),
    ("train_id", "ISIC_1000002"),
    ("train_path", ""),
    ("eval_id", None),
    ("hamming_distance_63bit", -1),
    ("hamming_distance_63bit", 9),
    ("hamming_distance_63bit", True),
    ("hamming_distance_63bit", "0"),
    ("exact_file_sha256_match", True),
    ("exact_file_sha256_match", "false"),
])
def test_rejects_malformed_or_unmapped_pair_without_partial_decisions(inputs, field, value):
    # Catches silently dropping suspect rows or excluding a different mapped image.
    pairs, pool, evaluation = inputs
    pairs[-1][field] = value
    originals = deepcopy(inputs)

    with pytest.raises(ValueError):
        adjudicate_pairs(pairs, pool, evaluation)

    assert inputs == originals


@pytest.mark.parametrize("field", [
    "eval_id", "eval_member", "train_id", "train_path",
    "hamming_distance_63bit", "exact_file_sha256_match",
])
def test_rejects_missing_pair_evidence(inputs, field):
    # Catches defaults that turn incomplete screening evidence into a decision.
    pairs, pool, evaluation = inputs
    del pairs[-1][field]

    with pytest.raises(ValueError):
        adjudicate_pairs(pairs, pool, evaluation)


def test_rejects_duplicate_pair_edges(inputs):
    # Catches inflating the adjudicated count by repeating the same edge.
    pairs, pool, evaluation = inputs
    pairs.append(deepcopy(pairs[0]))

    with pytest.raises(ValueError):
        adjudicate_pairs(pairs, pool, evaluation)


def test_rejects_ambiguous_pool_path_mapping(inputs):
    # Catches a last-write-wins map selecting a different image for the same path.
    pairs, pool, evaluation = inputs
    pool.append({**pool[0], "image_id": "ISIC_9999999"})

    with pytest.raises(ValueError):
        adjudicate_pairs(pairs, pool, evaluation)


def test_rejects_ambiguous_evaluation_mapping(inputs):
    # Catches accepting two different archive members for the same evaluation ID.
    pairs, pool, evaluation = inputs
    evaluation.append({**evaluation[0], "eval_member": "evaluation/other.jpg"})

    with pytest.raises(ValueError):
        adjudicate_pairs(pairs, pool, evaluation)
