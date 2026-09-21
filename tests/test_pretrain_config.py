"""The pretraining config must be impossible to configure with labels."""

import dataclasses
from pathlib import Path

import pytest

from dermssl.pretrain import config as cfg_mod


def make_config(**overrides):
    base = dict(
        method="dino_continual",
        image_root="/data/pool",
        output_dir="/out",
        split_hash="a" * 64,
    )
    base.update(overrides)
    return cfg_mod.PretrainConfig(**base)


def test_default_backbone_follows_the_method():
    dino = cfg_mod.validate(make_config(method="dino_continual"))
    siam = cfg_mod.validate(make_config(method="simsiam"))
    assert dino.backbone == "dinov2_vits14"
    assert siam.backbone == "resnet50"


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="simsiam"):
        cfg_mod.validate(make_config(method="supervised"))


def test_resolutions_are_checked_against_the_patch_grid():
    # 256 is the ISIC 2020 trap: not a multiple of 14.
    with pytest.raises(Exception, match="252"):
        cfg_mod.validate(make_config(resolution=256))
    # 96 is the DINO default local crop, also invalid for patch 14; 98 = 7 x 14.
    with pytest.raises(Exception, match="98"):
        cfg_mod.validate(make_config(local_resolution=96))


def test_resnet_arm_has_no_patch_constraint():
    ok = cfg_mod.validate(make_config(method="simsiam", resolution=256))
    assert ok.resolution == 256


def test_budget_must_be_positive_and_fractions_must_end_at_one():
    with pytest.raises(ValueError, match="budget"):
        cfg_mod.validate(make_config(budget_gpu_hours=0))
    with pytest.raises(ValueError, match="fracciones"):
        cfg_mod.validate(make_config(checkpoint_fractions=(0.5, 0.25, 1.0)))
    with pytest.raises(ValueError, match="fracción"):
        cfg_mod.validate(make_config(checkpoint_fractions=(0.25, 0.5)))


def test_label_free_gate_rejects_any_label_looking_field():
    """The whole point of v6: no code path can smuggle supervision in."""
    ok = cfg_mod.validate(make_config())
    assert cfg_mod.assert_label_free(ok) is None  # silent when clean

    smuggled = dataclasses.replace(ok, image_root="/data/ham10000_metadata.csv")
    with pytest.raises(cfg_mod.LabelLeakError):
        cfg_mod.assert_label_free(smuggled)


def test_collect_images_ignores_non_image_files(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.PNG").write_bytes(b"x")
    (tmp_path / "metadata.csv").write_text("image,dx\na,mel\n")
    (tmp_path / "notes.txt").write_text("hi")
    found = cfg_mod.collect_images(tmp_path)
    assert sorted(p.name for p in found) == ["a.jpg", "b.PNG"]


def test_collect_images_fails_loudly_on_an_empty_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        cfg_mod.collect_images(tmp_path)


def test_collect_images_follows_symlinked_dataset_directories(tmp_path):
    """Staging a pool from several read-only mounts (docs/16) means
    ``--image-root`` is a directory of SYMLINKS to the real datasets, not a
    copy. ``Path.rglob("*")`` does NOT descend into symlinked directories by
    default - only the root itself being a symlink is followed - so a naive
    rglob silently returns zero images from every linked dataset."""
    real_dataset = tmp_path / "real_dataset"
    real_dataset.mkdir()
    (real_dataset / "ISIC_0000000.jpg").write_bytes(b"x")

    pool_root = tmp_path / "pool"
    pool_root.mkdir()
    link = pool_root / "linked_dataset"
    try:
        link.symlink_to(real_dataset, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"el entorno no permite crear symlinks: {exc}")

    found = cfg_mod.collect_images(pool_root)

    assert [p.name for p in found] == ["ISIC_0000000.jpg"]


def test_effective_lr_scales_linearly_with_total_batch():
    siam = cfg_mod.validate(make_config(method="simsiam", batch_size_per_gpu=64))
    # SimSiam rule: base 0.05 x total_batch / 256.
    assert cfg_mod.effective_lr(siam, world_size=4) == pytest.approx(0.05 * 256 / 256)
    dino = cfg_mod.validate(make_config(method="dino_continual", batch_size_per_gpu=64))
    # Continual pretraining rule: base 5e-4 x total_batch / 1024, then it is a
    # deliberately small LR because we start from converged DINOv2 weights.
    assert cfg_mod.effective_lr(dino, world_size=2) == pytest.approx(5e-4 * 128 / 1024)


def test_dino_scratch_shares_architecture_with_continual():
    """Variable isolation: same ViT, same objective, only the starting
    weights differ - so the default backbone must match the continual arm."""
    scratch = cfg_mod.validate(make_config(method="dino_scratch"))
    continual = cfg_mod.validate(make_config(method="dino_continual"))
    assert scratch.backbone == continual.backbone == "dinov2_vits14"


def test_dino_scratch_uses_the_from_scratch_lr_rule():
    scratch = cfg_mod.validate(
        make_config(method="dino_scratch", batch_size_per_gpu=64)
    )
    # From-scratch DINO rule: 5e-4 x total_batch / 256 (the paper's), NOT the
    # deliberately small continual rule that assumes converged weights.
    assert cfg_mod.effective_lr(scratch, world_size=2) == pytest.approx(
        5e-4 * 128 / 256
    )


def test_dino_scratch_local_crops_hit_the_patch_guard():
    # Same ViT-S/14 grid as the continual arm: 96 is invalid, 98 = 7 x 14.
    with pytest.raises(Exception, match="98"):
        cfg_mod.validate(make_config(method="dino_scratch", local_resolution=96))


def test_simsiam_dinov2_is_the_proposed_method():
    """The thesis' proposed method: SimSiam objective over the DINOv2 ViT.
    Backbone must default to the same ViT as the dino arms."""
    proposed = cfg_mod.validate(make_config(method="simsiam_dinov2"))
    assert proposed.backbone == "dinov2_vits14"
    assert "simsiam_dinov2" in cfg_mod.SIMSIAM_METHODS
    assert "simsiam" in cfg_mod.SIMSIAM_METHODS


def test_simsiam_dinov2_resolution_hits_the_patch_guard():
    # ViT-S/14 grid applies: 256 is not a multiple of 14.
    with pytest.raises(Exception, match="252"):
        cfg_mod.validate(make_config(method="simsiam_dinov2", resolution=256))


def test_simsiam_dinov2_uses_the_converged_weights_lr_rule():
    proposed = cfg_mod.validate(
        make_config(method="simsiam_dinov2", batch_size_per_gpu=64)
    )
    # Starts from converged DINOv2 weights, so the small continual rule
    # applies (5e-4 x total_batch / 1024), not SimSiam's from-scratch SGD rule.
    assert cfg_mod.effective_lr(proposed, world_size=2) == pytest.approx(
        5e-4 * 128 / 1024
    )


def test_mlflow_params_are_flat_primitives():
    params = cfg_mod.mlflow_params(cfg_mod.validate(make_config()))
    assert params["method"] == "dino_continual"
    assert params["checkpoint_fractions"] == "0.1,0.25,0.5,1.0"
    assert all(isinstance(v, (str, int, float, bool)) for v in params.values())


def test_precision_is_validated():
    with pytest.raises(ValueError, match="precisión"):
        cfg_mod.validate(make_config(precision="int8"))


def test_ssl_run_requires_a_frozen_split_hash():
    with pytest.raises(ValueError, match="split_hash"):
        cfg_mod.validate(make_config(split_hash=""))


def test_imagenet_resnet_control_has_its_own_explicit_method():
    control = cfg_mod.validate(make_config(method="simsiam_imagenet"))
    assert control.backbone == "resnet50"
    assert control.initial_weights == "imagenet1k_v2"


def test_original_transformed_adaptation_requires_a_transformed_image_root():
    with pytest.raises(ValueError, match="transformed_image_root"):
        cfg_mod.validate(
            make_config(
                method="simsiam_dinov2",
                adaptation_mode="original_transformed_consistency",
            )
        )

    adapted = cfg_mod.validate(
        make_config(
            method="simsiam_dinov2",
            adaptation_mode="original_transformed_consistency",
            transformed_image_root="/data/train-images-transformed",
            transformed_manifest_path="/data/train-images-transformed/tone_transform_manifest.json",
            data_revision="pool-v1",
            transformed_data_revision="circle-v1",
            consistency_weight=0.5,
        )
    )

    assert adapted.transformed_image_root == "/data/train-images-transformed"
    assert adapted.consistency_weight == 0.5


def test_original_transformed_adaptation_rejects_an_unsupported_dino_objective():
    with pytest.raises(ValueError, match="SimSiam"):
        cfg_mod.validate(
            make_config(
                method="dino_continual",
                adaptation_mode="original_transformed_consistency",
                transformed_image_root="/data/train-images-transformed",
            )
        )


def test_collect_paired_images_requires_a_transformation_for_each_original(tmp_path):
    originals = tmp_path / "originals"
    transformed = tmp_path / "transformed"
    (originals / "nested").mkdir(parents=True)
    (transformed / "nested").mkdir(parents=True)
    (originals / "nested" / "a.jpg").write_bytes(b"original-a")
    (originals / "b.jpg").write_bytes(b"original-b")
    (transformed / "nested" / "a.jpg").write_bytes(b"transformed-a")
    (transformed / "b.jpg").write_bytes(b"transformed-b")

    def file_hash(path):
        import hashlib

        return hashlib.sha256(path.read_bytes()).hexdigest()

    rows = [
        {
            "relative_path": relative,
            "source_sha256": file_hash(originals / relative),
            "transformed_sha256": file_hash(transformed / relative),
        }
        for relative in ("b.jpg", "nested/a.jpg")
    ]
    manifest = {
        "source_revision": "pool-v1",
        "transforms": rows,
    }
    import hashlib
    import json

    manifest["revision"] = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_path = transformed / "tone_transform_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    pairs = cfg_mod.collect_paired_images(
        originals,
        transformed,
        manifest_path=manifest_path,
        source_revision="pool-v1",
        transformed_revision=manifest["revision"],
    )

    assert [(left.relative_to(originals), right.relative_to(transformed)) for left, right in pairs] == [
        (Path("b.jpg"), Path("b.jpg")),
        (Path("nested/a.jpg"), Path("nested/a.jpg")),
    ]

    (transformed / "b.jpg").unlink()
    with pytest.raises(ValueError, match="exactamente"):
        cfg_mod.collect_paired_images(
            originals,
            transformed,
            manifest_path=manifest_path,
            source_revision="pool-v1",
            transformed_revision=manifest["revision"],
        )
