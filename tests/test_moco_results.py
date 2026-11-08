from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_moco_results as moco  # noqa: E402
import moco_results_inputs as inputs  # noqa: E402


def _readout(*, image_ids=None, groups=None, cv_seeds=(10, 20), variant="v"):
    matrices = [[[2, 1], [1, 2]], [[3, 0], [0, 3]]]
    runs = []
    for seed, matrix in zip(cv_seeds, matrices):
        runs.append({
            "seed": seed,
            "folds": [
                {"fold": 0, "train_groups_sha256": "train-0", "test_groups_sha256": "test-0"},
                {"fold": 1, "train_groups_sha256": "train-1", "test_groups_sha256": "test-1"},
            ],
            "metrics": {
                "classes": ["A", "B"],
                "confusion_matrix": matrix,
                "balanced_accuracy": 1.0 if matrix[0][0] == 3 else 2 / 3,
                "ece": 0.0 if matrix[0][0] == 3 else 0.2,
                "auc_ovr": {"A": 0.9, "B": 0.7},
            },
        })
    return {
        "kind": "PAD",
        "classes": ["A", "B"],
        "image_ids": image_ids or ["i1", "i2", "i3", "i4", "i5", "i6"],
        "groups": groups or ["g1", "g1", "g2", "g3", "g4", "g4"],
        "settings": {
            "reader": "balancedLR",
            "outer": 5,
            "inner": 3,
            "CV_seeds": list(cv_seeds),
            "C_grid": [0.3, 1.0],
            "tau": 0,
        },
        "support": {"A": {"images": 3, "groups": 2}, "B": {"images": 3, "groups": 2}},
        "runs": runs,
        "variant": variant,
    }


def test_validation_rejects_different_cohort_identity():
    base = _readout()
    changed = _readout(image_ids=["x", "i2", "i3", "i4", "i5", "i6"])
    with pytest.raises(ValueError, match="image_ids"):
        moco.validate_alignment([base, changed])


def test_validation_rejects_different_class_order_or_cv_settings():
    base = _readout()
    wrong_classes = _readout()
    wrong_classes["classes"] = ["B", "A"]
    with pytest.raises(ValueError, match="class order"):
        moco.validate_alignment([base, wrong_classes])
    wrong_cv = _readout()
    wrong_cv["settings"] = {**wrong_cv["settings"], "outer": 4}
    with pytest.raises(ValueError, match="CV settings"):
        moco.validate_alignment([base, wrong_cv])
    wrong_fold = _readout()
    wrong_fold["runs"][0]["folds"][0]["test_groups_sha256"] = "changed"
    with pytest.raises(ValueError, match="fold hashes"):
        moco.validate_alignment([base, wrong_fold])


def test_aggregate_preserves_nested_repetitions_and_class_support():
    result = moco.aggregate_method("test", [(42, _readout()), (43, _readout())])
    assert result["n_trained_seeds"] == 2
    assert result["n_cv_repetitions_per_seed"] == 2
    assert result["classes"][0]["precision"] == pytest.approx(5 / 6)
    assert result["classes"][0]["recall"] == pytest.approx(5 / 6)
    assert result["classes"][0]["f1"] == pytest.approx(5 / 6)
    assert result["classes"][0]["support_images"] == 3
    assert result["accuracy"] == pytest.approx(5 / 6)
    assert result["macro_f1"] == pytest.approx(5 / 6)
    assert result["auc_ovr_macro"] == pytest.approx(0.8)
    assert result["ece"] == pytest.approx(0.1)


def test_aggregate_rejects_negative_confusion_counts():
    invalid = _readout()
    invalid["runs"][0]["metrics"]["confusion_matrix"][0][1] = -1
    with pytest.raises(ValueError, match="confusion matrix counts"):
        moco.aggregate_method("invalid", [(42, invalid)])


def test_unseeded_reference_is_not_reported_as_ssl_training_seed():
    result = moco.aggregate_method("public", [(None, _readout())])
    assert result["n_trained_seeds"] == 0
    assert result["n_encoder_checkpoints"] == 1
    assert result["trained_seeds"] == []


def test_rendered_release_tables_exclude_private_ids_and_note_nominal_winners():
    records = [moco.aggregate_method("test", [(42, _readout())])]
    csv_text, markdown_text = moco.render_tables(records, source_hash="abc", protocol_hash="def")
    assert "image_ids" not in markdown_text
    assert "g1" not in markdown_text
    assert "Best nominal F1" in markdown_text
    assert "Best nominal recall" in markdown_text
    assert "source_sha256=abc" in markdown_text
    assert "source_sha256" in csv_text
    assert "support_groups" in csv_text
    assert "n_encoder_checkpoints" in csv_text
    assert "Trained SSL seeds" in markdown_text


def test_v5_registry_covers_each_reported_variant_once():
    assert set(inputs.moco_variant_specs()) == {
        "public-cls", "m0-cls", "m0-cls4", "m0-cls_patch", "quota4-cls",
        "quota4-cls4", "quota4-cls_patch", "quota4-interpolate05", "m0-teacher", "quota4-teacher",
    }


def test_output_writer_is_deterministic_and_aggregate_only(tmp_path):
    pad = moco.aggregate_method("test", [(42, _readout())])
    ham = {**pad, "cohort": "HAM"}
    report = {
        "release_version": "10.0.0",
        "source_plan_sha256": "plan-hash",
        "source_sha256": "abc",
        "protocol_sha256": "def",
        "aggregation": "mean, not ensemble",
        "scope": "development only",
        "cohorts": {
            "PAD": {"classes": ["A", "B"], "records": [pad]},
            "HAM": {"classes": ["A", "B"], "records": [ham]},
        },
    }
    moco.write_release_outputs(report, tmp_path)
    first = {path.name: path.read_text() for path in tmp_path.iterdir()}
    moco.write_release_outputs(report, tmp_path)
    second = {path.name: path.read_text() for path in tmp_path.iterdir()}
    assert first == second
    assert set(first) == {"metrics.json", "api-report.json", "PAD.csv", "HAM.csv", "README.md"}
    api_report = json.loads(first["api-report.json"])
    assert api_report["release"] == "v10.0.0"
    assert api_report["method"]["family"] == "MoCo-v3-style"
    assert api_report["classification"] == report["cohorts"]
    assert api_report["source"]["table_receipt_sha256"] == "abc"
    assert api_report["protocol"]["sha256"] == "def"
    assert api_report["table"]["markdown"] == "README.md"
    assert api_report["caveats"]
    assert "image_ids" not in first["metrics.json"]
    assert "g1" not in first["README.md"]
    assert "source_sha256=abc" in first["README.md"]
