"""Behavioral contracts for paired transformed-image SSL smoke runs."""

from __future__ import annotations

import importlib.util
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dermssl import mask_manifest
from dermssl.pretrain import config as config_mod
from dermssl.pretrain import run_contract
from dermssl.pretrain.tone_transforms import generate_tone_transformations


ROOT = Path(__file__).resolve().parents[1]


def _load_pretrain_script():
    spec = importlib.util.spec_from_file_location(
        "pretrain_script_for_test", ROOT / "scripts" / "pretrain.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((16, 16, 3), [190, 140, 110], dtype=np.uint8)
    Image.fromarray(image).save(path)


def _accepted_mask() -> np.ndarray:
    mask = np.zeros((16, 16), dtype=bool)
    mask[5:11, 5:11] = True
    return mask


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def _write_mask_manifest(path: Path, mask: np.ndarray, *, eligible: bool = True) -> None:
    record = mask_manifest.build_mask_record(
        dataset="ham10000",
        image_id="sample",
        source_revision="pool-v1",
        mask=mask,
        producer="otsu",
        version="1",
    )
    if not eligible:
        record = {**record, "eligible": False, "quality": "degenerate"}
    mask_manifest.write_mask_manifest(
        mask_manifest.build_mask_manifest((record,)), path
    )


def _write_mask_manifest_for_images(path: Path, mask: np.ndarray, image_ids: tuple[str, ...]) -> None:
    records = tuple(
        mask_manifest.build_mask_record(
            dataset="ham10000",
            image_id=image_id,
            source_revision="pool-v1",
            mask=mask,
            producer="otsu",
            version="1",
        )
        for image_id in image_ids
    )
    mask_manifest.write_mask_manifest(mask_manifest.build_mask_manifest(records), path)


def _paired_config(
    original_root: Path,
    transformed_root: Path,
    output_root: Path,
    transform_manifest: dict,
    **overrides,
) -> config_mod.PretrainConfig:
    base = dict(
        method="simsiam",
        image_root=str(original_root),
        output_dir=str(output_root),
        split_hash="a" * 64,
        data_revision="pool-v1",
        transformed_image_root=str(transformed_root),
        transformed_manifest_path=str(transformed_root / "tone_transform_manifest.json"),
        transformed_data_revision=transform_manifest["revision"],
        adaptation_mode="original_transformed_consistency",
        batch_size_per_gpu=2,
        num_workers=0,
        resolution=32,
        precision="fp32",
        max_steps=1,
    )
    base.update(overrides)
    return config_mod.PretrainConfig(**base)


def test_transformed_smoke_keeps_the_pair_root_and_records_auditable_execution(tmp_path):
    script = _load_pretrain_script()
    args = script.parse_args(
        [
            "--method",
            "simsiam",
            "--image-root",
            "/data/original",
            "--transformed-image-root",
            "/data/transformed",
            "--transformed-manifest-path",
            "/data/transformed/tone_transform_manifest.json",
            "--data-revision",
            "pool-v1",
            "--transformed-data-revision",
            "b" * 64,
            "--output-dir",
            "/tmp/ssl-smoke",
            "--split-hash",
            "a" * 64,
            "--smoke",
        ]
    )

    cfg = script.build_config(args)
    transformed_evidence = config_mod.TransformedManifestEvidence(
        pairs=(),
        manifest_path="/data/transformed/tone_transform_manifest.json",
        manifest_revision="b" * 64,
        manifest_sha256="c" * 64,
    )
    manifest = run_contract.build_run_manifest(
        cfg, gpu_count=1, transformed_manifest=transformed_evidence
    )
    observed = run_contract.record_execution_evidence(
        manifest,
        device="cpu",
        steps_executed=1,
        image_count=2,
        pair_count=2,
        ssl_loss=0.125,
        consistency_loss=0.125,
        dataset_image_count=2,
        dataset_pair_count=2,
        samples_processed=2,
        total_loss=0.25,
    )

    assert cfg.max_steps == script.SMOKE_MAX_STEPS
    assert manifest["transformed_image_root"] == "/data/transformed"
    assert observed["execution"] == {
        "device": "cpu",
        "steps_executed": 1,
        "image_count": 2,
        "dataset_image_count": 2,
        "dataset_pair_count": 2,
        "samples_processed": 2,
        "pair_count": 2,
        "ssl_loss": 0.125,
        "consistency_loss": 0.125,
        "total_loss": 0.25,
    }
    path = tmp_path / "run.json"
    run_contract.write_run_manifest(observed, path)
    persisted = json.loads(path.read_text())
    assert persisted["transformed_image_root"] == "/data/transformed"
    assert persisted["execution"]["device"] == "cpu"
    assert persisted["execution"]["pair_count"] == 2


def test_tone_transform_generator_preserves_originals_and_writes_revision_manifest(tmp_path):
    original_root = tmp_path / "original"
    mask_root = tmp_path / "masks"
    output_root = tmp_path / "transformed"
    source = original_root / "nested" / "sample.jpg"
    mask = _accepted_mask()
    _write_image(source)
    before = source.read_bytes()
    _write_mask(mask_root / "nested" / "sample.png", mask)
    manifest_path = tmp_path / "mask_manifest.json"
    _write_mask_manifest(manifest_path, mask)

    result = generate_tone_transformations(
        image_root=original_root,
        mask_root=mask_root,
        mask_manifest_path=manifest_path,
        output_root=output_root,
        shift_buckets=1,
        source_revision="pool-v1",
    )

    output = output_root / "nested" / "sample.jpg"
    saved = json.loads((output_root / "tone_transform_manifest.json").read_text())
    assert source.read_bytes() == before
    assert output.is_file()
    assert result["summary"] == {"total": 1, "transformed": 1, "rejected": 0}
    assert saved["revision"] == result["revision"]
    assert len(saved["revision"]) == 64
    assert saved["transforms"][0]["relative_path"] == "nested/sample.jpg"


def test_tone_transform_generator_rejects_an_ineligible_mask_without_writing_output(tmp_path):
    original_root = tmp_path / "original"
    mask_root = tmp_path / "masks"
    output_root = tmp_path / "transformed"
    mask = _accepted_mask()
    _write_image(original_root / "sample.jpg")
    _write_mask(mask_root / "sample.png", mask)
    manifest_path = tmp_path / "mask_manifest.json"
    _write_mask_manifest(manifest_path, mask, eligible=False)

    with pytest.raises(ValueError, match="elegible"):
        generate_tone_transformations(
            image_root=original_root,
            mask_root=mask_root,
            mask_manifest_path=manifest_path,
            output_root=output_root,
            shift_buckets=1,
            source_revision="pool-v1",
        )

    assert not (output_root / "sample.jpg").exists()


def test_tone_transform_generator_rejects_an_output_root_that_overlaps_originals(tmp_path):
    original_root = tmp_path / "original"
    mask_root = tmp_path / "masks"
    mask = _accepted_mask()
    _write_image(original_root / "sample.jpg")
    _write_mask(mask_root / "sample.png", mask)
    manifest_path = tmp_path / "mask_manifest.json"
    _write_mask_manifest(manifest_path, mask)

    with pytest.raises(ValueError, match="separada"):
        generate_tone_transformations(
            image_root=original_root,
            mask_root=mask_root,
            mask_manifest_path=manifest_path,
            output_root=original_root,
            shift_buckets=1,
            source_revision="pool-v1",
        )


def test_paired_pretraining_requires_a_manifest_with_matching_revisions_and_hashes(tmp_path):
    original_root = tmp_path / "original"
    mask_root = tmp_path / "masks"
    transformed_root = tmp_path / "transformed"
    mask = _accepted_mask()
    _write_image(original_root / "sample.jpg")
    _write_mask(mask_root / "sample.png", mask)
    mask_path = tmp_path / "mask_manifest.json"
    _write_mask_manifest(mask_path, mask)
    transform_manifest = generate_tone_transformations(
        image_root=original_root,
        mask_root=mask_root,
        mask_manifest_path=mask_path,
        output_root=transformed_root,
        shift_buckets=1,
        source_revision="pool-v1",
    )

    evidence = config_mod.validate_transformed_manifest(
        _paired_config(original_root, transformed_root, tmp_path / "run", transform_manifest)
    )

    assert evidence.pairs[0][0].relative_to(original_root).as_posix() == "sample.jpg"
    assert evidence.pairs[0][1].relative_to(transformed_root).as_posix() == "sample.jpg"
    assert evidence.manifest_revision == transform_manifest["revision"]
    assert evidence.manifest_sha256 == sha256(
        (transformed_root / "tone_transform_manifest.json").read_bytes()
    ).hexdigest()

    with pytest.raises(ValueError, match="revisión"):
        config_mod.validate_transformed_manifest(
            _paired_config(
                original_root,
                transformed_root,
                tmp_path / "wrong-revision",
                transform_manifest,
                transformed_data_revision="wrong-revision",
            )
        )

    (transformed_root / "sample.jpg").write_bytes(b"altered")
    with pytest.raises(ValueError, match="hash"):
        config_mod.validate_transformed_manifest(
            _paired_config(original_root, transformed_root, tmp_path / "altered", transform_manifest)
        )


def test_paired_pretraining_rejects_an_extra_transformed_file(tmp_path):
    original_root = tmp_path / "original"
    mask_root = tmp_path / "masks"
    transformed_root = tmp_path / "transformed"
    mask = _accepted_mask()
    _write_image(original_root / "sample.jpg")
    _write_mask(mask_root / "sample.png", mask)
    mask_path = tmp_path / "mask_manifest.json"
    _write_mask_manifest(mask_path, mask)
    transform_manifest = generate_tone_transformations(
        image_root=original_root,
        mask_root=mask_root,
        mask_manifest_path=mask_path,
        output_root=transformed_root,
        shift_buckets=1,
        source_revision="pool-v1",
    )
    _write_image(transformed_root / "unexpected.jpg")

    with pytest.raises(ValueError, match="exactamente"):
        config_mod.validate_transformed_manifest(
            _paired_config(original_root, transformed_root, tmp_path / "run", transform_manifest)
        )


def test_torch_loop_cpu_step_records_actual_pair_usage_and_separate_losses(tmp_path):
    pytest.importorskip("torch")
    try:
        pytest.importorskip("torchvision")
    except RuntimeError as exc:
        pytest.skip(f"requires compatible torch and torchvision: {exc}")
    from dermssl.pretrain import torch_loop

    original_root = tmp_path / "original"
    mask_root = tmp_path / "masks"
    transformed_root = tmp_path / "transformed"
    mask = _accepted_mask()
    for image_id in ("sample", "sample-two"):
        _write_image(original_root / f"{image_id}.jpg")
        _write_mask(mask_root / f"{image_id}.png", mask)
    mask_path = tmp_path / "mask_manifest.json"
    _write_mask_manifest_for_images(mask_path, mask, ("sample", "sample-two"))
    transform_manifest = generate_tone_transformations(
        image_root=original_root,
        mask_root=mask_root,
        mask_manifest_path=mask_path,
        output_root=transformed_root,
        shift_buckets=1,
        source_revision="pool-v1",
    )
    output_root = tmp_path / "run" / "nested"

    torch_loop.train(
        _paired_config(original_root, transformed_root, output_root, transform_manifest)
    )

    run = json.loads((output_root / "run.json").read_text(encoding="utf-8"))
    execution = run["execution"]
    assert run["transformed_manifest_path"] == str(
        (transformed_root / "tone_transform_manifest.json").resolve()
    )
    assert run["transformed_manifest_revision"] == transform_manifest["revision"]
    assert run["transformed_manifest_sha256"] == sha256(
        (transformed_root / "tone_transform_manifest.json").read_bytes()
    ).hexdigest()
    assert run["gpu_count"] == 0
    assert execution["device"] == "cpu"
    assert execution["steps_executed"] == 1
    assert execution["dataset_image_count"] == 2
    assert execution["dataset_pair_count"] == 2
    assert execution["samples_processed"] == 2
    assert execution["pair_count"] == 2
    assert execution["ssl_loss"] is not None
    # The two views ARE the original and its tone-shifted copy, so the SimSiam
    # loss over that pair already is the consistency: the manifest reports the
    # same value under both names. Until 2026-09-22 this assertion read
    # ``total == ssl + consistency`` and so froze the defect it should have
    # caught - the arm trained with twice the gradient of the plain one.
    assert execution["consistency_loss"] == pytest.approx(execution["ssl_loss"])
    assert execution["total_loss"] == pytest.approx(execution["ssl_loss"])
