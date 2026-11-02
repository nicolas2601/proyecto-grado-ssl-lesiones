"""Contracts for post-hoc, paired segmentation uncertainty analysis."""
import json

import numpy as np
import pytest

from scripts import analyze_ima_ablation_026_031 as analysis


def row(image_id, dice=0.8, image_hash="a" * 64, reference_hash="b" * 64):
    return {
        "image_id": image_id, "image_sha256": image_hash,
        "reference_mask_sha256": reference_hash, "status": "ok", "dice": dice,
        "reference_foreground_fraction": 0.1, "sensitivity": dice,
        "specificity": 1 - (0.1 - dice * 0.1) / 0.9,
    }


def write(path, rows):
    path.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")


def test_alignment_rejects_duplicate_and_mismatched_reference(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write(a, [row("one"), row("one")])
    with pytest.raises(ValueError, match="Duplicate"):
        analysis.load_raw(a)
    write(a, [row("one"), row("two")])
    write(b, [row("two"), row("one", reference_hash="c" * 64)])
    with pytest.raises(ValueError, match="reference"):
        analysis.align_runs({"a": analysis.load_raw(a), "b": analysis.load_raw(b)})


def test_alignment_uses_ids_and_checks_gt_and_image_hash(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write(a, [row("one"), row("two")])
    write(b, [row("two", 0.9), row("one", 0.7)])
    aligned = analysis.align_runs({"a": analysis.load_raw(a), "b": analysis.load_raw(b)})
    assert [x["dice"] for x in aligned["b"]] == [0.7, 0.9]
    write(b, [row("one", image_hash="d" * 64), row("two")])
    with pytest.raises(ValueError, match="image"):
        analysis.align_runs({"a": analysis.load_raw(a), "b": analysis.load_raw(b)})


@pytest.mark.parametrize("field,value", [("dice", float("nan")), ("dice", 1.2), ("sensitivity", 0.1)])
def test_invalid_scores_or_dice_identity_abort(tmp_path, field, value):
    record = row("one")
    record[field] = value
    path = tmp_path / "bad.jsonl"
    write(path, [record])
    with pytest.raises(ValueError):
        analysis.load_raw(path)


def test_bootstrap_keeps_seed_covariance_and_constant_effect():
    # Opposite seed effects cancel exactly per image. Independent resampling
    # per seed would incorrectly give the joint mean a non-zero interval.
    differences = np.array([[0.1, -0.1, 0.03], [-0.1, 0.1, 0.03], [0.2, -0.2, 0.03]])
    out = analysis.bootstrap_contrast(differences, resamples=200, seed=42)
    assert out["mean_across_fixed_seeds"]["delta"] == pytest.approx(0.01)
    assert out["mean_across_fixed_seeds"]["ci95"] == pytest.approx([0.01, 0.01])
    assert out["per_seed"][2]["ci95"] == pytest.approx([0.03, 0.03])
    assert analysis.sign_flip_sensitivity([0.1, 0.2, 0.3])["two_sided_p"] == 0.25


def test_bootstrap_rejects_empty_nonfinite_and_too_few_resamples():
    for values in [np.empty((0, 3)), np.array([[np.nan, 1, 2]])]:
        with pytest.raises(ValueError):
            analysis.bootstrap_contrast(values, resamples=100, seed=42)
    with pytest.raises(ValueError):
        analysis.bootstrap_contrast(np.ones((2, 3)), resamples=0, seed=42)


def test_training_recipe_and_source_cannot_change_between_seeds():
    def identity(seed, code="a" * 64):
        return {"code": {"source_sha256": {"trainer.py": code}},
                "config": {"seed": seed, "device": "cuda:0", "batch_size": 16},
                "unlabeled_manifest_sha256": "b" * 64,
                "ph2_manifest_sha256": "c" * 64,
                "train_case_ids_sha256": "d" * 64, "val_case_ids_sha256": "e" * 64}
    analysis.verify_shared_training_identity([identity(42), identity(43)])
    with pytest.raises(ValueError, match="training identity"):
        analysis.verify_shared_training_identity([identity(42), identity(43, "f" * 64)])
    changed = identity(43)
    changed["config"]["batch_size"] = 32
    with pytest.raises(ValueError, match="training identity"):
        analysis.verify_shared_training_identity([identity(42), changed])
