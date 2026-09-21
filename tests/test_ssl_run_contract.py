"""Contracts for reproducible, label-free SSL/domain-adaptation runs."""

import json

import pytest

from dermssl.pretrain import config as cfg_mod
from dermssl.pretrain import run_contract


def make_config(**overrides):
    base = dict(
        method="dino_continual",
        image_root="/data/train-images-only",
        output_dir="/out/dino-adapted",
        split_hash="a" * 64,
        seed=17,
    )
    base.update(overrides)
    return cfg_mod.PretrainConfig(**base)


def test_run_manifest_records_reproducibility_and_resume_contract():
    cfg = cfg_mod.validate(make_config(resume_checkpoint="/runs/resume.pt"))

    manifest = run_contract.build_run_manifest(
        cfg,
        gpu_count=2,
        data_revision="pool-v1",
        code_revision="abc123",
    )

    assert manifest["split_hash"] == "a" * 64
    assert manifest["initial_weights"] == "dinov2-public-lvd142m"
    assert manifest["resume_checkpoint"] == "/runs/resume.pt"
    assert manifest["seed"] == 17
    assert manifest["gpu_count"] == 2
    assert manifest["label_free"] is True


def test_run_manifest_records_zero_gpu_count_for_cpu_execution():
    manifest = run_contract.build_run_manifest(
        cfg_mod.validate(make_config(max_steps=1)),
        gpu_count=0,
    )

    assert manifest["gpu_count"] == 0


def test_run_manifest_requires_a_step_limit_without_gpus():
    with pytest.raises(ValueError, match="max_steps"):
        run_contract.build_run_manifest(
            cfg_mod.validate(make_config()),
            gpu_count=0,
        )


def test_run_manifest_rejects_a_negative_gpu_count():
    with pytest.raises(ValueError, match="no puede ser negativo"):
        run_contract.build_run_manifest(
            cfg_mod.validate(make_config()),
            gpu_count=-1,
        )


def test_write_run_manifest_creates_missing_parent_directories(tmp_path):
    destination = tmp_path / "nested" / "run.json"

    run_contract.write_run_manifest({"schema_version": "1.0"}, destination)

    assert json.loads(destination.read_text(encoding="utf-8")) == {"schema_version": "1.0"}


def test_paired_matrix_rejects_variants_with_a_different_split_or_probe():
    base = run_contract.RunVariant(
        name="dinov2-base",
        backbone="dinov2_vits14",
        stage="base",
        split_hash="a" * 64,
        seed=17,
        probe="linear-v1",
        weights="dinov2-public-lvd142m",
    )
    adapted = run_contract.RunVariant(
        name="dinov2-adapted",
        backbone="dinov2_vits14",
        stage="adapted",
        split_hash="b" * 64,
        seed=17,
        probe="linear-v2",
        weights="/runs/dino-adapted.pt",
    )

    with pytest.raises(ValueError, match="split_hash"):
        run_contract.validate_paired_variants((base, adapted))


def test_paired_matrix_accepts_dinov2_and_imagenet_resnet_controls():
    matrix = (
        run_contract.RunVariant(
            "dinov2-base", "dinov2_vits14", "base", "a" * 64, 17, "linear-v1", "dinov2-public-lvd142m"
        ),
        run_contract.RunVariant(
            "dinov2-adapted", "dinov2_vits14", "adapted", "a" * 64, 17, "linear-v1", "/runs/dino.pt"
        ),
        run_contract.RunVariant(
            "resnet50-base", "resnet50", "base", "a" * 64, 17, "linear-v1", "imagenet1k_v2"
        ),
        run_contract.RunVariant(
            "resnet50-adapted", "resnet50", "adapted", "a" * 64, 17, "linear-v1", "/runs/resnet.pt"
        ),
    )

    assert run_contract.validate_paired_variants(matrix) == matrix


def test_validation_checkpoint_selection_prefers_best_eligible_macro_f1():
    baseline = run_contract.CheckpointEvaluation(
        checkpoint="baseline.pt",
        split_hash="a" * 64,
        seed=17,
        probe="linear-v1",
        evaluation_split="validation",
        macro_f1=0.70,
        global_auc=0.82,
        tone_gap=0.18,
    )
    candidates = (
        run_contract.CheckpointEvaluation(
            checkpoint="adapted_25pct.pt",
            split_hash="a" * 64,
            seed=17,
            probe="linear-v1",
            evaluation_split="validation",
            macro_f1=0.73,
            global_auc=0.80,
            tone_gap=0.14,
        ),
        run_contract.CheckpointEvaluation(
            checkpoint="adapted_final.pt",
            split_hash="a" * 64,
            seed=17,
            probe="linear-v1",
            evaluation_split="validation",
            macro_f1=0.75,
            global_auc=0.79,
            tone_gap=0.11,
        ),
    )

    selected = run_contract.select_validation_checkpoint(baseline, candidates)

    assert selected.checkpoint == "adapted_25pct.pt"
    assert selected.evaluation_split == "validation"


def test_checkpoint_promotion_record_is_validation_only_and_reproducible():
    baseline = run_contract.CheckpointEvaluation(
        checkpoint="baseline.pt",
        split_hash="a" * 64,
        seed=17,
        probe="linear-v1",
        evaluation_split="validation",
        macro_f1=0.70,
        global_auc=0.82,
        tone_gap=0.18,
    )
    candidate = run_contract.CheckpointEvaluation(
        checkpoint="adapted_50pct.pt",
        split_hash="a" * 64,
        seed=17,
        probe="linear-v1",
        evaluation_split="validation",
        macro_f1=0.74,
        global_auc=0.81,
        tone_gap=0.12,
    )

    record = run_contract.build_checkpoint_promotion_record(baseline, (candidate,))

    assert record == {
        "selection_split": "validation",
        "selected_checkpoint": "adapted_50pct.pt",
        "max_auc_drop": 0.02,
        "selection_rule": "max_validation_macro_f1_with_auc_and_tone_gap_gates",
    }


def test_run_manifest_records_original_transformed_consistency_objective():
    cfg = cfg_mod.validate(
        make_config(
                method="simsiam_dinov2",
                transformed_image_root="/data/transformed-images",
                transformed_manifest_path="/data/transformed-images/tone_transform_manifest.json",
                data_revision="pool-v1",
            transformed_data_revision="circle-v1",
            adaptation_mode="original_transformed_consistency",
            consistency_weight=0.5,
            max_steps=1,
        )
    )

    transformed_evidence = cfg_mod.TransformedManifestEvidence(
        pairs=(),
        manifest_path="/data/transformed-images/tone_transform_manifest.json",
        manifest_revision="circle-v1",
        manifest_sha256="b" * 64,
    )
    manifest = run_contract.build_run_manifest(
        cfg, gpu_count=2, transformed_manifest=transformed_evidence
    )

    assert manifest["adaptation_mode"] == "original_transformed_consistency"
    assert manifest["transformed_image_root"] == "/data/transformed-images"
    assert manifest["transformed_data_revision"] == "circle-v1"
    assert manifest["consistency_weight"] == 0.5
    assert manifest["max_steps"] == 1
    assert manifest["transformed_manifest_sha256"] == "b" * 64
