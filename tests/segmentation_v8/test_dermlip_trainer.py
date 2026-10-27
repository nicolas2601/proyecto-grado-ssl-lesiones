"""Unit tests for the Teacher-Student Semi-Supervised Segmentation Trainer."""
from __future__ import annotations

import torch
import json
from pathlib import Path
from PIL import Image
import pytest

from scripts.train_dermlip_semi import (
    UnlabeledDermDataset,
    load_dermlip_model_manifest,
    parse_args,
    validate_unlabeled_manifest,
)
import scripts.train_dermlip_semi as semi_trainer

from dermssl.segmentation_methods.dermlip_slic.trainer import (
    DermLIPSegmentationUNet,
    TeacherStudentTrainer,
    compute_dice_loss,
)


def test_unet_forward_shape():
    model = DermLIPSegmentationUNet(in_channels=3, num_classes=1, base_channels=8)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)


def test_compute_dice_loss():
    # Perfect overlap -> Dice loss should be ~0.0
    logits = torch.full((2, 1, 32, 32), 10.0)  # sigmoid ~ 1.0
    targets = torch.ones((2, 1, 32, 32))
    loss = compute_dice_loss(logits, targets)
    assert loss.item() < 0.05

    # Zero overlap -> Dice loss should be ~1.0
    targets_zero = torch.zeros((2, 1, 32, 32))
    loss_disjoint = compute_dice_loss(logits, targets_zero)
    assert loss_disjoint.item() > 0.90


def test_teacher_student_trainer_step():
    student = DermLIPSegmentationUNet(in_channels=3, num_classes=1, base_channels=8)
    trainer = TeacherStudentTrainer(
        student=student,
        device="cpu",
        ema_decay=0.95,
        gamma_semantic=0.5,
        kappa_uncertainty=0.693147,
    )

    optimizer = torch.optim.Adam(trainer.student.parameters(), lr=1e-3)

    # Synthetic batches (B=2, H=32, W=32)
    labeled_batch = (
        torch.randn(2, 3, 32, 32),
        (torch.rand(2, 1, 32, 32) > 0.5).float(),
    )
    unlabeled_batch = (
        torch.randn(2, 3, 32, 32),  # weak
        torch.randn(2, 3, 32, 32),  # strong
        torch.rand(2, 1, 32, 32),   # semantic prior
    )

    step_res = trainer.training_step(labeled_batch, unlabeled_batch, lambda_u=0.5)

    assert "total_loss" in step_res
    assert "loss_sup" in step_res
    assert "loss_unsup" in step_res
    assert "mean_alpha" in step_res
    assert torch.isfinite(step_res["total_loss"])

    # Test backpropagation
    optimizer.zero_grad()
    step_res["total_loss"].backward()
    optimizer.step()

    # Test EMA update
    initial_teacher_weight = trainer.teacher.enc1.conv[0].weight.clone()
    trainer.update_teacher()
    updated_teacher_weight = trainer.teacher.enc1.conv[0].weight
    # Check that teacher moved towards student
    assert not torch.equal(initial_teacher_weight, updated_teacher_weight)


def test_teacher_batchnorm_buffers_update():
    student = DermLIPSegmentationUNet(in_channels=3, num_classes=1, base_channels=8)
    trainer = TeacherStudentTrainer(
        student=student,
        device="cpu",
        ema_decay=0.90,
    )
    # Forward pass through student in train mode updates BatchNorm running stats
    student.train()
    x = torch.randn(4, 3, 32, 32) * 5.0 + 3.0
    _ = student(x)

    s_mean = student.enc1.conv[1].running_mean.clone()
    t_mean_before = trainer.teacher.enc1.conv[1].running_mean.clone()

    trainer.update_teacher()
    t_mean_after = trainer.teacher.enc1.conv[1].running_mean.clone()

    # Teacher buffer must have updated towards student running stats via EMA
    assert not torch.equal(t_mean_before, t_mean_after)
    expected = t_mean_before * 0.90 + s_mean * 0.10
    assert torch.allclose(t_mean_after, expected, atol=1e-5)


def test_unlabeled_pool_rejects_missing_or_incorrect_digest(tmp_path: Path):
    image_path = tmp_path / "pool.png"
    Image.new("RGB", (16, 16), (200, 100, 50)).save(image_path)
    manifest_path = tmp_path / "pool.json"
    manifest_path.write_text(json.dumps({"records": [{"image_id": "ISIC_1", "image_path": "pool.png"}]}))
    with pytest.raises(ValueError, match="image_sha256"):
        validate_unlabeled_manifest(manifest_path, [], tmp_path)


def test_unlabeled_pool_rejects_labeled_identity_overlap(tmp_path: Path):
    image_path = tmp_path / "pool.png"
    Image.new("RGB", (16, 16), (200, 100, 50)).save(image_path)
    import hashlib

    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "pool.json"
    manifest_path.write_text(json.dumps({"records": [{
        "image_id": "IMD001", "lesion_id": "IMD001", "image_path": "pool.png", "image_sha256": digest
    }]}))
    labeled = [{"image_id": "IMD001", "lesion_id": "IMD001", "image_sha256": digest}]
    with pytest.raises(ValueError, match="overlap"):
        validate_unlabeled_manifest(manifest_path, labeled, tmp_path)


@pytest.mark.parametrize("legacy_receipt", [None, {"schema": "legacy.receipt.v1"}])
def test_legacy_unlabeled_manifest_does_not_require_v2_receipt(tmp_path: Path, legacy_receipt):
    import hashlib

    image_path = tmp_path / "legacy.png"
    Image.new("RGB", (24, 18), (140, 100, 80)).save(image_path)
    record = {
        "image_id": "legacy_case_1",
        "dataset": "ISIC2019",
        "image_path": str(image_path),
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
    }
    manifest_path = tmp_path / "legacy.manifest.json"
    manifest_path.write_text(json.dumps({"records": [record]}), encoding="utf-8")
    if legacy_receipt is not None:
        (tmp_path / "receipt.json").write_text(json.dumps(legacy_receipt), encoding="utf-8")

    paths, _, audit = validate_unlabeled_manifest(manifest_path, [], tmp_path)

    assert paths == [image_path.resolve()]
    assert "pool_stage" not in audit


def test_v2_unlabeled_pool_requires_matching_adjacent_receipt(tmp_path: Path, monkeypatch):
    import hashlib
    import dermssl.segmentation_methods.common.protocol as protocol

    image_path = tmp_path / "source.png"
    Image.new("RGB", (37, 29), (120, 90, 70)).save(image_path)
    record = {
        "image_id": "ISIC2019_case_1",
        "dataset": "ISIC2019",
        "image_path": str(image_path),
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
    }
    contract_pins = protocol.V2_SHARED_POOL_CONTRACT
    monkeypatch.setitem(contract_pins, "selection_count", 1)
    monkeypatch.setitem(contract_pins, "selection_dataset_counts", {"ISIC2019": 1})
    monkeypatch.setitem(
        contract_pins,
        "selection_cohort_sha256",
        protocol.cohort_digest([record]),
    )
    manifest_path = tmp_path / "selection.manifest.json"
    contract = dict(contract_pins)
    contract.update({
        "stage": "selection",
        "stage_count": 1,
        "stage_dataset_counts": {"ISIC2019": 1},
        "stage_source_manifest_sha256": contract_pins["selection_source_manifest_sha256"],
        "stage_cohort_sha256": contract_pins["selection_cohort_sha256"],
    })
    manifest = {
        "schema_version": 1,
        "partition": "train",
        "count": 1,
        "pool_contract": contract,
        "eligibility": {
            "verified": True,
            "excludes_target_test": True,
            "patient_lesion_groups_preserved": True,
            "deduplication_preserved": True,
            "source_manifest_sha256": contract_pins["source_pool_manifest_sha256"],
            "source_receipt_sha256": contract_pins["source_receipt_sha256"],
            "exclusions_sha256": contract_pins["source_exclusions_sha256"],
            "verification_revision": "a" * 64,
        },
        "records": [record],
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    # No adjacent receipt: eligibility.verified and pool_contract alone are insufficient.
    with pytest.raises(ValueError, match="receipt"):
        validate_unlabeled_manifest(manifest_path, [], tmp_path)

    source_pins = {
        "split_v2_receipt": contract_pins["source_receipt_sha256"],
        "candidate_source_pool_manifest": contract_pins["source_pool_manifest_sha256"],
        "split_v2_exclusion_crosswalk": contract_pins["source_exclusions_sha256"],
        "selection_source_manifest": contract_pins["selection_source_manifest_sha256"],
        "refit_source_manifest": contract_pins["refit_source_manifest_sha256"],
        "isic2018_task1_evaluation_manifest": contract_pins["evaluation_manifest_sha256"],
        "isic2018_task1_exclusion_receipt": contract_pins[
            "evaluation_exclusion_receipt_sha256"
        ],
        "ph2_eligible_manifest": contract_pins["ph2_manifest_sha256"],
        "ph2_curator_exclusions": contract_pins["ph2_exclusions_sha256"],
    }
    receipt = {
        "schema": "v2_shared_segmentation_pool_bridge.v1",
        "status": "preflight_bridge_built_execution_disabled",
        "arms": ["DermLIP-SLED", "SLED v8"],
        "same_underlying_pool_for_both_arms": True,
        "source_artifacts": {
            name: {"path": f"/source/{name}", "sha256": digest}
            for name, digest in source_pins.items()
        },
        "pools": {
            "selection": {
                "path": str(manifest_path.resolve()),
                "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "count": 1,
                "cohort_sha256": contract_pins["selection_cohort_sha256"],
                "dataset_counts": {"ISIC2019": 1},
            }
        },
        "gates": {
            "execution_enabled": False,
            "execution_ready": False,
            "launch_ready": False,
            "training_started": False,
            "training_consumable": False,
            "pool_manifest_frozen": False,
        },
    }
    (tmp_path / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    paths, digest, audit = validate_unlabeled_manifest(manifest_path, [], tmp_path)
    assert paths == [image_path.resolve()]
    assert digest == hashlib.sha256(manifest_bytes).hexdigest()
    assert audit["pool_stage"] == "selection"
    assert audit["target_size"] == [224, 224]
    weak, strong, prior = UnlabeledDermDataset(paths)[0]
    assert weak.shape == strong.shape == (3, 224, 224)
    assert prior.shape == (1, 224, 224)

    receipt["pools"]["selection"]["sha256"] = "0" * 64
    (tmp_path / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind this manifest"):
        validate_unlabeled_manifest(manifest_path, [], tmp_path)

    manifest["records"][0]["diagnosis"] = "melanoma"
    contaminated_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest_path.write_bytes(contaminated_bytes)
    receipt["pools"]["selection"]["sha256"] = hashlib.sha256(
        contaminated_bytes
    ).hexdigest()
    (tmp_path / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="exact image-only schema"):
        validate_unlabeled_manifest(manifest_path, [], tmp_path)

    incomplete = json.loads(manifest_bytes)
    incomplete.pop("pool_contract")
    incomplete_bytes = json.dumps(incomplete, sort_keys=True).encode("utf-8")
    manifest_path.write_bytes(incomplete_bytes)
    receipt["pools"]["selection"]["sha256"] = hashlib.sha256(
        incomplete_bytes
    ).hexdigest()
    (tmp_path / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="missing its eligibility or contract"):
        validate_unlabeled_manifest(manifest_path, [], tmp_path)

    bypass_attempt = json.loads(manifest_bytes)
    bypass_attempt["pool_contract"]["contract_id"] = "unrecognized_contract"
    bypass_attempt["pool_contract"].pop("stage")
    bypass_attempt["eligibility"]["source_receipt_sha256"] = "0" * 64
    bypass_attempt["eligibility"]["source_manifest_sha256"] = "1" * 64
    bypass_bytes = json.dumps(bypass_attempt, sort_keys=True).encode("utf-8")
    manifest_path.write_bytes(bypass_bytes)
    with pytest.raises(ValueError, match="V2 shared-pool contract rejected"):
        validate_unlabeled_manifest(manifest_path, [], tmp_path)

    markerless_manifest = json.loads(manifest_bytes)
    markerless_manifest.pop("pool_contract")
    markerless_manifest.pop("eligibility")
    markerless_bytes = json.dumps(markerless_manifest, sort_keys=True).encode("utf-8")
    manifest_path.write_bytes(markerless_bytes)
    receipt["pools"]["selection"]["sha256"] = hashlib.sha256(markerless_bytes).hexdigest()
    (tmp_path / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="missing its eligibility or contract"):
        validate_unlabeled_manifest(manifest_path, [], tmp_path)


def test_dermlip_guidance_is_enabled_by_default():
    assert parse_args([]).enable_dermlip is True


def _pool_use_fixture(tmp_path: Path, *, stage: str = "selection", method: str = "dermlip_semi_supervised"):
    import hashlib

    from dermssl.segmentation_methods.common.pool_use import validate_pool_use_attestation
    from dermssl.segmentation_methods.common.protocol import cohort_digest

    pool_root = tmp_path / "shared-segmentation-pools-v2-test"
    pool_root.mkdir()
    receipt = pool_root / "receipt.json"
    receipt.write_text("immutable preflight receipt", encoding="utf-8")
    manifest = pool_root / f"{stage}.manifest.json"
    rows = [{
        "image_id": "ISIC2019_case_1",
        "dataset": "ISIC2019",
        "image_path": str(pool_root / "images" / "ISIC2019_case_1.jpg"),
        "image_sha256": "a" * 64,
    }]
    manifest_doc = {
        "schema_version": 1,
        "partition": "train",
        "count": 1,
        "pool_contract": {"contract_id": "classification_split_master_v2", "stage": stage},
        "records": rows,
    }
    manifest.write_text(json.dumps(manifest_doc, sort_keys=True), encoding="utf-8")
    stage_binding = {
        "path": str(manifest.resolve()),
        "sha256": semi_trainer._sha256(manifest),
        "record_count": 1,
        "dataset_counts": {"ISIC2019": 1},
        "cohort_sha256": cohort_digest(rows),
    }
    receipt_binding = {"path": str(receipt.resolve()), "sha256": semi_trainer._sha256(receipt)}
    bindings = {"pool_root": str(pool_root.resolve()), "stage": stage, "preflight_receipt": receipt_binding,
                "source_artifacts": {"selection_source_manifest_sha256": "b" * 64},
                "stages": {stage: stage_binding}}
    attestation = {
        "schema": "v2_method_pool_use_attestation.v1",
        "status": "pool_consumption_attested_execution_disabled",
        "method_scopes": ["dermlip_semi_supervised", "sled"],
        "stage": stage,
        "bindings": bindings,
        "stage_manifests": {
            stage: stage_binding
        },
        "preflight_receipt": receipt_binding,
        "pool_manifest_frozen": True,
        "freeze_semantics": "sha_identity_pinned_filesystem_immutability_not_claimed",
        "training_consumable": True,
        "patient_independence_accredited": False,
        "holdout_novelty": False,
        "schema_fields": ["image_id", "dataset", "image_path", "image_sha256"],
    }
    attestation_path = tmp_path / "pool-use-attestation.json"
    attestation_path.write_text(json.dumps(attestation, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(attestation_path.read_bytes()).hexdigest()
    validated = validate_pool_use_attestation(
        attestation_path,
        expected_sha256=digest,
        expected_bindings=bindings,
        manifest_path=manifest,
        receipt_path=receipt,
        method=method,
        stage=stage,
    )
    return attestation_path, manifest, receipt, attestation, digest, validated, bindings


@pytest.mark.parametrize("defect", ["legacy_schema", "duplicate_id", "stage_mismatch", "wrong_cohort"])
def test_v2_pool_attestation_rejects_unbound_or_malformed_manifest_inventory(tmp_path: Path, defect: str):
    import hashlib

    from dermssl.segmentation_methods.common.pool_use import validate_pool_use_attestation

    path, manifest_path, receipt_path, attestation, _, _, bindings = _pool_use_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if defect == "legacy_schema":
        manifest.pop("pool_contract")
    elif defect == "duplicate_id":
        manifest["records"].append(dict(manifest["records"][0], image_sha256="b" * 64))
        manifest["count"] = 2
    elif defect == "stage_mismatch":
        manifest["pool_contract"]["stage"] = "refit"
    else:
        manifest["records"][0]["image_id"] = "ISIC2019_other_case"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    stage_binding = attestation["stage_manifests"]["selection"]
    stage_binding["sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    attestation["bindings"]["stages"]["selection"]["sha256"] = stage_binding["sha256"]
    attestation_path = tmp_path / "pool-use-attestation.json"
    attestation_path.write_text(json.dumps(attestation, sort_keys=True), encoding="utf-8")
    attestation_sha = hashlib.sha256(attestation_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="manifest|cohort|identity|stage"):
        validate_pool_use_attestation(
            attestation_path,
            expected_sha256=attestation_sha,
            expected_bindings=bindings,
            manifest_path=manifest_path,
            receipt_path=receipt_path,
            method="dermlip_semi_supervised",
            stage="selection",
        )


@pytest.mark.parametrize("legacy_artifact", ["last_checkpoint.pt", "pseudo_masks/old.png", "cache/embeddings.npy"])
def test_dermlip_v2_preflight_rejects_existing_legacy_output_namespace(
    tmp_path: Path, monkeypatch, legacy_artifact: str
):
    from dermssl.segmentation_methods.common import pool_use

    manifest = tmp_path / "selection.manifest.json"
    manifest.write_text(json.dumps({"pool_contract": {"stage": "selection"}}), encoding="utf-8")
    ph2_manifest = tmp_path / "ph2.json"
    ph2_manifest.write_text(json.dumps({"records": [{"image_id": "case", "case_group_id": "case"}]}), encoding="utf-8")
    attestation = tmp_path / "attestation.json"
    attestation.write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "run"
    legacy_path = out_dir / legacy_artifact
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_bytes(b"legacy")
    args = parse_args([
        "--ph2-manifest", str(ph2_manifest), "--unlabeled-manifest", str(manifest),
        "--pool-use-attestation", str(attestation),
        "--disable-dermlip", "--out-dir", str(out_dir), "--batch-size", "1",
    ])

    monkeypatch.setattr(pool_use, "is_v2_pool_manifest", lambda *_args: True)
    monkeypatch.setattr(pool_use, "validate_pool_use_attestation", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(pool_use, "build_v2_pool_provenance", lambda **_kwargs: {
        "pool_stage": "selection",
        "pool_manifest_sha256": "a" * 64,
        "pool_use_attestation_sha256": "b" * 64,
        "source_artifact_sha256": {"selection_source_manifest_sha256": "c" * 64},
        "fresh_model_initialization": True,
        "legacy_checkpoint_reused": False,
        "legacy_feature_cache_reused": False,
        "legacy_pseudomasks_reused": False,
    })
    monkeypatch.setattr(semi_trainer, "validate_unlabeled_manifest", lambda *_args, **_kwargs: (
        [tmp_path / "pool.png"], "a" * 64, {"records_validated": 1}
    ))

    class OneCaseDataset:
        def __init__(self, *_args, **_kwargs):
            self.records = [{"image_id": "case", "case_group_id": "case"}]
        def __len__(self):
            return 1

    monkeypatch.setattr(semi_trainer, "PH2Dataset", OneCaseDataset)

    with pytest.raises(FileExistsError, match="fresh|output|legacy"):
        semi_trainer.run_training(args)


def test_v2_pool_use_attestation_accepts_only_pinned_method_stage_and_inputs(tmp_path: Path):
    _, _, _, _, _, validated, _ = _pool_use_fixture(tmp_path)
    assert validated["training_consumable"] is True
    assert validated["pool_manifest_frozen"] is True
    assert validated["patient_independence_accredited"] is False
    assert validated["holdout_novelty"] is False


def test_v2_pool_provenance_binds_manifest_attestation_receipt_and_fresh_policy(tmp_path: Path):
    from dermssl.segmentation_methods.common.pool_use import build_v2_pool_provenance

    attestation_path, manifest_path, _, attestation, digest, _, _ = _pool_use_fixture(tmp_path)
    provenance = build_v2_pool_provenance(
        attestation_path=attestation_path,
        attestation=attestation,
        attestation_sha256=digest,
        manifest_path=manifest_path,
        stage="selection",
    )

    assert provenance["pool_manifest_sha256"] == semi_trainer._sha256(manifest_path)
    assert provenance["pool_use_attestation_sha256"] == digest
    assert provenance["pool_stage"] == "selection"
    assert provenance["fresh_model_initialization"] is True
    assert provenance["legacy_checkpoint_reused"] is False
    assert provenance["legacy_feature_cache_reused"] is False
    assert provenance["legacy_pseudomasks_reused"] is False
    assert provenance["source_artifact_sha256"] == attestation["bindings"]["source_artifacts"]


def test_training_and_checkpoint_receipts_copy_exact_pool_provenance(tmp_path: Path):
    from scripts.train_dermlip_semi import (
        _write_checkpoint_receipt,
        _write_training_started_receipt,
    )

    provenance = {
        "pool_stage": "selection",
        "pool_manifest_sha256": "a" * 64,
        "pool_use_attestation_sha256": "b" * 64,
    }
    started = _write_training_started_receipt(
        tmp_path / "training_started.json",
        run_id="run-1",
        identity_fingerprint="c" * 64,
        method="dermlip_semi_supervised",
        stage="selection",
        output_dir=tmp_path,
        pool_provenance=provenance,
    )
    checkpoint = tmp_path / "last_checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_receipt = _write_checkpoint_receipt(
        checkpoint,
        run_id="run-1",
        identity_fingerprint="c" * 64,
        method="dermlip_semi_supervised",
        stage="selection",
        output_dir=tmp_path,
        epoch=1,
        pool_provenance=provenance,
    )

    assert started["pool_provenance"] == provenance
    assert checkpoint_receipt["pool_provenance"] == provenance


@pytest.mark.parametrize("tamper", ["missing", "sha", "stage", "method", "manifest_path"])
def test_v2_pool_use_attestation_rejects_missing_or_tampered_binding(tmp_path: Path, tamper: str):
    from dermssl.segmentation_methods.common.pool_use import validate_pool_use_attestation

    path, manifest, receipt, doc, digest, _, bindings = _pool_use_fixture(tmp_path)
    if tamper == "missing":
        path.unlink()
        with pytest.raises(ValueError, match="attestation"):
            validate_pool_use_attestation(
                path, expected_sha256=digest, manifest_path=manifest, receipt_path=receipt,
                expected_bindings=bindings,
                method="dermlip_semi_supervised", stage="selection",
            )
        return
    if tamper == "sha":
        manifest.write_text("changed manifest", encoding="utf-8")
    elif tamper == "stage":
        doc["stage"] = "refit"
        path.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
    elif tamper == "method":
        with pytest.raises(ValueError, match="method"):
            validate_pool_use_attestation(
                path, expected_sha256=digest, manifest_path=manifest, receipt_path=receipt,
                expected_bindings=bindings,
                method="rpi_net", stage="selection",
            )
        return
    else:
        doc["stage_manifests"]["selection"]["path"] = str(tmp_path / "other.json")
        path.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA|stage|path|pin"):
        validate_pool_use_attestation(
            path, expected_sha256=digest, manifest_path=manifest, receipt_path=receipt,
            expected_bindings=bindings,
            method="dermlip_semi_supervised", stage="selection",
        )


def test_v2_execution_requires_enabled_exact_stage_gate(tmp_path: Path):
    import hashlib

    from dermssl.segmentation_methods.common.pool_use import validate_execution_gate

    _, manifest, _, _, digest, _, _ = _pool_use_fixture(tmp_path)
    profile = {"pilot_samples": None, "allow_self_consistency_pilot": False}
    profile_sha256 = hashlib.sha256(
        json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    gate = {
        "schema": "dermlip_stage_execution_gate.v1",
        "method": "dermlip_semi_supervised",
        "stage": "selection",
        "execution_enabled": False,
        "launch_ready": False,
        "training_started": False,
        "pool_manifest_sha256": semi_trainer._sha256(manifest),
        "pool_use_attestation_sha256": digest,
        "profile": profile,
        "profile_sha256": profile_sha256,
        "pool_manifest": {"path": str(manifest.resolve()), "sha256": semi_trainer._sha256(manifest)},
    }
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="execution_enabled"):
        validate_execution_gate(
            gate_path, method="dermlip_semi_supervised", stage="selection",
            profile_sha256=profile_sha256,
            profile=profile,
            manifest_path=manifest, attestation_sha256=digest,
        )


@pytest.mark.parametrize("training_started", [False, True])
def test_v2_new_run_gate_requires_not_started_and_accepts_ready_gate(
    tmp_path: Path, training_started: bool
):
    import hashlib

    from dermssl.segmentation_methods.common.pool_use import validate_execution_gate

    _, manifest, _, _, digest, _, _ = _pool_use_fixture(tmp_path)
    profile = {"epochs": 300, "pilot_samples": None, "allow_self_consistency_pilot": False}
    claim_path = tmp_path / "claims" / "dermlip-selection.claim.json"
    profile["execution_claim_path"] = str(claim_path.resolve())
    profile_sha256 = hashlib.sha256(
        json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    gate = {
        "schema": "dermlip_stage_execution_gate.v1",
        "method": "dermlip_semi_supervised",
        "stage": "selection",
        "execution_enabled": True,
        "launch_ready": True,
        "training_started": training_started,
        "execution_claim_path": str(claim_path.resolve()),
        "pool_manifest_frozen": True,
        "training_consumable": True,
        "profile": profile,
        "profile_sha256": profile_sha256,
        "pool_manifest": {"path": str(manifest.resolve()), "sha256": semi_trainer._sha256(manifest)},
        "pool_use_attestation_sha256": digest,
    }
    gate_path = tmp_path / f"gate-started-{training_started}.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    if training_started:
        with pytest.raises(ValueError, match="training_started=false"):
            validate_execution_gate(
                gate_path, method="dermlip_semi_supervised", stage="selection",
                profile_sha256=profile_sha256, profile=profile, manifest_path=manifest,
                attestation_sha256=digest,
            )
    else:
        assert validate_execution_gate(
            gate_path, method="dermlip_semi_supervised", stage="selection",
            profile_sha256=profile_sha256, profile=profile, manifest_path=manifest,
            attestation_sha256=digest,
        )["training_started"] is False


def _write_test_execution_gate(
    tmp_path: Path, *, enabled: bool = True
) -> tuple[Path, Path, str, str, str, dict]:
    import hashlib

    _, manifest, _, _, attestation_sha256, _, _ = _pool_use_fixture(tmp_path)
    claim_path = tmp_path / "claims" / "dermlip-selection.claim.json"
    profile = {
        "stage": "selection",
        "epochs": 300,
        "pilot_samples": None,
        "allow_self_consistency_pilot": False,
        "execution_claim_path": str(claim_path.resolve()),
    }
    profile_sha256 = hashlib.sha256(
        json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    gate = {
        "schema": "dermlip_stage_execution_gate.v1",
        "method": "dermlip_semi_supervised",
        "stage": "selection",
        "execution_enabled": enabled,
        "launch_ready": enabled,
        "training_started": False,
        "pool_manifest_frozen": True,
        "training_consumable": True,
        "profile": profile,
        "profile_sha256": profile_sha256,
        "execution_claim_path": str(claim_path.resolve()),
        "pool_manifest": {"path": str(manifest.resolve()), "sha256": semi_trainer._sha256(manifest)},
        "pool_use_attestation_sha256": attestation_sha256,
    }
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    return gate_path, manifest, profile_sha256, attestation_sha256, str(claim_path.resolve()), profile


def test_v2_execution_gate_claim_is_one_time(tmp_path: Path):
    from dermssl.segmentation_methods.common.pool_use import claim_execution_gate

    gate_path, manifest, profile_sha256, attestation_sha256, claim_path, profile = _write_test_execution_gate(tmp_path)
    claim_path_obj = Path(claim_path)
    first = claim_execution_gate(
        gate_path,
        method="dermlip_semi_supervised",
        stage="selection",
        profile_sha256=profile_sha256,
        profile=profile,
        manifest_path=manifest,
        attestation_sha256=attestation_sha256,
        run_id="run-first",
        identity_fingerprint="f" * 64,
        out_dir=tmp_path / "run-first-output",
    )
    assert first["run_id"] == "run-first"
    assert first["gate_sha256"] == semi_trainer._sha256(gate_path)
    assert json.loads(claim_path_obj.read_text(encoding="utf-8")) == first
    with pytest.raises(FileExistsError, match="claim already exists"):
        claim_execution_gate(
            gate_path,
            method="dermlip_semi_supervised",
            stage="selection",
            profile_sha256=profile_sha256,
            profile=profile,
            manifest_path=manifest,
            attestation_sha256=attestation_sha256,
            run_id="run-second",
            identity_fingerprint="e" * 64,
            out_dir=tmp_path / "run-second-output",
        )


def test_false_v2_execution_gate_does_not_create_claim(tmp_path: Path):
    from dermssl.segmentation_methods.common.pool_use import claim_execution_gate

    gate_path, manifest, profile_sha256, attestation_sha256, claim_path, profile = _write_test_execution_gate(
        tmp_path, enabled=False
    )
    with pytest.raises(ValueError, match="execution_enabled"):
        claim_execution_gate(
            gate_path,
            method="dermlip_semi_supervised",
            stage="selection",
            profile_sha256=profile_sha256,
            profile=profile,
            manifest_path=manifest,
            attestation_sha256=attestation_sha256,
            run_id="run-denied",
            identity_fingerprint="d" * 64,
            out_dir=tmp_path / "run-denied-output",
        )
    assert not Path(claim_path).exists()


def _write_v2_resume_fixture(tmp_path: Path) -> dict:
    import hashlib

    from dermssl.segmentation_methods.common.pool_use import claim_execution_gate

    gate_path, manifest, profile_sha256, attestation_sha256, claim_path, profile = (
        _write_test_execution_gate(tmp_path)
    )
    run_id = "resume-run-001"
    identity_fingerprint = "a" * 64
    out_dir = tmp_path / "run-output"
    claim_execution_gate(
        gate_path,
        method="dermlip_semi_supervised",
        stage="selection",
        profile_sha256=profile_sha256,
        profile=profile,
        manifest_path=manifest,
        attestation_sha256=attestation_sha256,
        run_id=run_id,
        identity_fingerprint=identity_fingerprint,
        out_dir=out_dir,
    )
    out_dir.mkdir()
    (out_dir / "run_identity.json").write_text(json.dumps({
        "run_id": run_id,
        "identity_fingerprint": identity_fingerprint,
        "output_dir": str(out_dir.resolve()),
    }), encoding="utf-8")
    (out_dir / "training_started.json").write_text(json.dumps({
        "schema": "dermlip_training_started_receipt.v1",
        "run_id": run_id,
        "identity_fingerprint": identity_fingerprint,
        "method": "dermlip_semi_supervised",
        "stage": "selection",
        "output_dir": str(out_dir.resolve()),
        "timestamp_utc": "2026-10-22T10:00:00Z",
    }), encoding="utf-8")
    (out_dir / "input_validation.json").write_text("{}", encoding="utf-8")
    checkpoint_path = out_dir / "last_checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint bytes")
    checkpoint_receipt = {
        "schema": "dermlip_checkpoint_receipt.v1",
        "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "run_id": run_id,
        "identity_fingerprint": identity_fingerprint,
        "method": "dermlip_semi_supervised",
        "stage": "selection",
        "output_dir": str(out_dir.resolve()),
        "epoch": 12,
    }
    (out_dir / "last_checkpoint.receipt.json").write_text(
        json.dumps(checkpoint_receipt), encoding="utf-8"
    )
    return {
        "gate_path": gate_path,
        "manifest_path": manifest,
        "profile_sha256": profile_sha256,
        "profile": profile,
        "attestation_sha256": attestation_sha256,
        "claim_path": Path(claim_path),
        "out_dir": out_dir,
        "resume_path": checkpoint_path,
        "run_id": run_id,
        "identity_fingerprint": identity_fingerprint,
    }


def test_v2_resume_contract_accepts_matching_run(tmp_path: Path):
    from dermssl.segmentation_methods.common.pool_use import validate_v2_resume_contract

    inputs = _write_v2_resume_fixture(tmp_path)
    receipt = validate_v2_resume_contract(**inputs)
    assert receipt["run_id"] == inputs["run_id"]
    assert receipt["checkpoint_epoch"] == 12


def test_v2_resume_contract_rejects_copied_output_directory(tmp_path: Path):
    import shutil

    from dermssl.segmentation_methods.common.pool_use import validate_v2_resume_contract

    inputs = _write_v2_resume_fixture(tmp_path)
    copied_out = tmp_path / "copied-output"
    shutil.copytree(inputs["out_dir"], copied_out)
    inputs["out_dir"] = copied_out
    inputs["resume_path"] = copied_out / "last_checkpoint.pt"
    with pytest.raises(ValueError, match="claim|run identity"):
        validate_v2_resume_contract(**inputs)


@pytest.mark.parametrize("mismatch", ["identity", "gate"])
def test_v2_resume_contract_rejects_identity_or_gate_mismatch(tmp_path: Path, mismatch: str):
    from dermssl.segmentation_methods.common.pool_use import validate_v2_resume_contract

    inputs = _write_v2_resume_fixture(tmp_path)
    if mismatch == "identity":
        inputs["identity_fingerprint"] = "b" * 64
    else:
        inputs["gate_path"].write_text(
            inputs["gate_path"].read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
    with pytest.raises(ValueError, match="identity|gate SHA"):
        validate_v2_resume_contract(**inputs)


@pytest.mark.parametrize("missing", ["checkpoint", "start_receipt", "checkpoint_receipt"])
def test_v2_resume_contract_rejects_missing_checkpoint_or_receipt(
    tmp_path: Path, missing: str
):
    from dermssl.segmentation_methods.common.pool_use import validate_v2_resume_contract

    inputs = _write_v2_resume_fixture(tmp_path)
    targets = {
        "checkpoint": inputs["resume_path"],
        "start_receipt": inputs["out_dir"] / "training_started.json",
        "checkpoint_receipt": inputs["out_dir"] / "last_checkpoint.receipt.json",
    }
    targets[missing].unlink()
    with pytest.raises((ValueError, FileNotFoundError), match="checkpoint|receipt|missing"):
        validate_v2_resume_contract(**inputs)


def test_v2_resume_checkpoint_structure_must_match_receipt(tmp_path: Path):
    from scripts.train_dermlip_semi import _validate_resume_checkpoint_structure

    with pytest.raises(ValueError, match="structure is incomplete"):
        _validate_resume_checkpoint_structure(
            {"run_id": "wrong"},
            run_id="run-001",
            identity_fingerprint="a" * 64,
            checkpoint_epoch=12,
        )


def test_resume_attempt_receipt_is_separate_and_preserves_start_receipt(tmp_path: Path):
    from dermssl.segmentation_methods.common.pool_use import write_resume_attempt_receipt

    started = tmp_path / "training_started.json"
    started.write_text('{"schema":"dermlip_training_started_receipt.v1"}', encoding="utf-8")
    before = started.read_bytes()
    attempt_path = write_resume_attempt_receipt(
        tmp_path,
        gate_sha256="b" * 64,
        run_id="run-001",
        identity_fingerprint="a" * 64,
        method="dermlip_semi_supervised",
        stage="selection",
        checkpoint_sha256="c" * 64,
    )
    assert attempt_path.is_file()
    assert json.loads(attempt_path.read_text(encoding="utf-8"))["checkpoint_sha256"] == "c" * 64
    assert started.read_bytes() == before


def test_training_started_receipt_writer_binds_run_and_stage(tmp_path: Path):
    from scripts.train_dermlip_semi import _write_training_started_receipt

    receipt_path = tmp_path / "training_started.json"
    receipt = _write_training_started_receipt(
        receipt_path,
        run_id="run-001",
        identity_fingerprint="f" * 64,
        method="dermlip_semi_supervised",
        stage="selection",
        output_dir=tmp_path,
    )
    assert receipt["run_id"] == "run-001"
    assert receipt["identity_fingerprint"] == "f" * 64
    assert receipt["method"] == "dermlip_semi_supervised"
    assert receipt["stage"] == "selection"
    assert receipt["timestamp_utc"].endswith("Z")
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    assert not receipt_path.with_name("training_started.json.tmp").exists()


@pytest.mark.parametrize(
    "override",
    [{"pilot_samples": 10}, {"allow_self_consistency_pilot": True}],
)
def test_v2_execution_gate_rejects_sampling_not_authorized_by_profile(tmp_path: Path, override: dict):
    import hashlib

    from dermssl.segmentation_methods.common.pool_use import validate_execution_gate

    _, manifest, _, _, digest, _, _ = _pool_use_fixture(tmp_path)
    gate_profile = {
        "epochs": 300,
        "pilot_samples": None,
        "allow_self_consistency_pilot": False,
    }
    profile = {**gate_profile, **override}
    profile_sha256 = hashlib.sha256(
        json.dumps(gate_profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    gate = {
        "schema": "dermlip_stage_execution_gate.v1",
        "method": "dermlip_semi_supervised",
        "stage": "selection",
        "execution_enabled": True,
        "launch_ready": True,
        "training_started": False,
        "pool_manifest_frozen": True,
        "training_consumable": True,
        "profile": gate_profile,
        "profile_sha256": profile_sha256,
        "pool_manifest": {"path": str(manifest.resolve()), "sha256": semi_trainer._sha256(manifest)},
        "pool_use_attestation_sha256": digest,
    }
    gate_path = tmp_path / "enabled-gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="pilot_samples|allow_self_consistency_pilot|profile"):
        validate_execution_gate(
            gate_path, method="dermlip_semi_supervised", stage="selection",
            profile_sha256=hashlib.sha256(
                json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            profile=profile, manifest_path=manifest, attestation_sha256=digest,
        )


def test_v2_manifest_without_markers_or_receipt_is_not_accepted_as_legacy(tmp_path: Path):
    pool_root = tmp_path / "shared-segmentation-pools-v2-20261022-001"
    pool_root.mkdir()
    manifest = pool_root / "selection.manifest.json"
    manifest.write_text(json.dumps({"records": []}), encoding="utf-8")
    args = parse_args([
        "--unlabeled-manifest", str(manifest), "--disable-dermlip",
        "--out-dir", str(tmp_path / "run"),
    ])
    with pytest.raises(ValueError, match="attestation|stage"):
        semi_trainer.run_training(args)


def test_legacy_training_pool_is_rejected_even_with_opt_in(tmp_path: Path):
    manifest = tmp_path / "legacy.manifest.json"
    manifest.write_text(json.dumps({"records": []}), encoding="utf-8")
    args = parse_args([
        "--unlabeled-manifest", str(manifest), "--disable-dermlip",
        "--out-dir", str(tmp_path / "run"), "--ph2-manifest", str(tmp_path / "missing-ph2.json"),
    ])
    with pytest.raises(ValueError, match="[Ff]resh training requires.*v2"):
        semi_trainer.run_training(args)

    opted_in = parse_args([
        "--unlabeled-manifest", str(manifest), "--allow-legacy-pool", "--disable-dermlip",
        "--out-dir", str(tmp_path / "run-opt-in"), "--ph2-manifest", str(tmp_path / "missing-ph2.json"),
    ])
    with pytest.raises(ValueError, match="[Ff]resh training requires.*v2"):
        semi_trainer.run_training(opted_in)


def test_new_dermlip_run_rejects_legacy_pool_even_with_legacy_opt_in(tmp_path: Path):
    legacy_manifest = tmp_path / "legacy.manifest.json"
    legacy_manifest.write_text(json.dumps({"records": []}), encoding="utf-8")
    args = parse_args([
        "--unlabeled-manifest", str(legacy_manifest), "--allow-legacy-pool",
        "--disable-dermlip", "--out-dir", str(tmp_path / "new-run"),
        "--ph2-manifest", str(tmp_path / "missing-ph2.json"),
    ])

    with pytest.raises(ValueError, match="[Ff]resh training requires.*v2"):
        semi_trainer.run_training(args)


def test_training_profile_binds_sampling_flags():
    from scripts.train_dermlip_semi import _execution_profile

    default_args = parse_args([])
    profile = _execution_profile(default_args, "selection")
    assert profile["pilot_samples"] is None
    assert profile["allow_self_consistency_pilot"] is False


def test_dermlip_v2_preflight_stops_before_trainer_initialization(tmp_path: Path, monkeypatch):
    from dermssl.segmentation_methods.common import pool_use

    manifest = tmp_path / "selection.manifest.json"
    manifest.write_text(json.dumps({"pool_contract": {"stage": "selection"}}), encoding="utf-8")
    ph2_manifest = tmp_path / "ph2.json"
    ph2_manifest.write_text(json.dumps({"records": [{"image_id": "case", "case_group_id": "case"}]}), encoding="utf-8")
    model_manifest = tmp_path / "model.json"
    model_manifest.write_text("{}", encoding="utf-8")
    attestation = tmp_path / "pool-use-attestation.json"
    attestation.write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "run"
    args = parse_args([
        "--ph2-manifest", str(ph2_manifest), "--unlabeled-manifest", str(manifest),
        "--dermlip-model-manifest", str(model_manifest), "--pool-use-attestation", str(attestation),
        "--execution-gate", str(tmp_path / "selection-gate.json"),
        "--execution-claim-path", str(tmp_path / "claims" / "dermlip-selection.claim.json"),
        "--out-dir", str(out_dir),
        "--batch-size", "1",
    ])
    class OneCaseDataset:
        def __init__(self, *_args, **_kwargs):
            self.records = [{"image_id": "case", "case_group_id": "case"}]
        def __len__(self):
            return 1

    monkeypatch.setattr(pool_use, "is_v2_pool_manifest", lambda *_args: True)
    monkeypatch.setattr(pool_use, "validate_pool_use_attestation", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(pool_use, "build_v2_pool_provenance", lambda **_kwargs: {
        "pool_stage": "selection",
        "pool_manifest_sha256": "a" * 64,
        "pool_use_attestation_sha256": "b" * 64,
        "source_artifact_sha256": {"selection_source_manifest_sha256": "c" * 64},
        "fresh_model_initialization": True,
        "legacy_checkpoint_reused": False,
        "legacy_feature_cache_reused": False,
        "legacy_pseudomasks_reused": False,
    })
    monkeypatch.setattr(semi_trainer, "PH2Dataset", OneCaseDataset)
    monkeypatch.setattr(
        semi_trainer, "validate_unlabeled_manifest",
        lambda *_args, **_kwargs: ([tmp_path / "unlabeled.png"], "a" * 64, {"records_validated": 1}),
    )
    monkeypatch.setattr(semi_trainer, "load_dermlip_model_manifest", lambda *_args: {
        "model_name": "fake", "manifest_sha256": "b" * 64,
    })
    monkeypatch.setattr(semi_trainer, "verify_dermlip_hf_cache", lambda _identity: {"verified": True})
    monkeypatch.setattr(semi_trainer, "DermLIPSuperpixelCalibrator", lambda **_kwargs: object())
    monkeypatch.setattr(semi_trainer, "UnlabeledDermDataset", OneCaseDataset)
    monkeypatch.setattr(semi_trainer, "TeacherStudentTrainer", lambda **_kwargs: pytest.fail("trainer initialized"))
    result = semi_trainer.run_training(args)
    assert result["status"] == "preflight_validated_execution_disabled"
    assert result["training_started"] is False
    assert result["pool_provenance"]["pool_stage"] == "selection"
    assert not out_dir.exists()
    assert not (out_dir / "training_started.json").exists()
    assert not (tmp_path / "claims" / "dermlip-selection.claim.json").exists()


def _build_real_pool_use_attestation(tmp_path: Path, monkeypatch) -> tuple[Path, Path, str]:
    import hashlib

    from dermssl.segmentation_methods.common import pool_use
    from dermssl.segmentation_methods.common.protocol import cohort_digest

    pool_root = tmp_path / "shared-segmentation-pools-v2-20261022-001"
    pool_root.mkdir()
    manifest = pool_root / "selection.manifest.json"
    rows = [{
        "image_id": "ISIC2019_case_1", "dataset": "ISIC2019",
        "image_path": str(pool_root / "images" / "ISIC2019_case_1.jpg"),
        "image_sha256": "a" * 64,
    }]
    manifest.write_text(json.dumps({
        "schema_version": 1, "partition": "train", "count": 1,
        "pool_contract": {"contract_id": "classification_split_master_v2", "stage": "selection"},
        "records": rows,
    }, sort_keys=True), encoding="utf-8")
    receipt = pool_root / "receipt.json"
    receipt.write_text('{"gates":{"execution_enabled":false}}', encoding="utf-8")
    stage_binding = {
        "path": str(manifest.resolve()),
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "record_count": 1,
        "dataset_counts": {"ISIC2019": 1},
        "cohort_sha256": cohort_digest(rows),
    }
    receipt_binding = {
        "path": str(receipt.resolve()),
        "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
    }
    bindings = {
        "pool_root": str(pool_root.resolve()),
        "preflight_receipt": receipt_binding,
        "source_artifacts": {"selection_source_manifest_sha256": "b" * 64},
        "stages": {"selection": stage_binding},
    }
    attestation = {
        "schema": "v2_method_pool_use_attestation.v1",
        "status": "pool_consumption_attested_execution_disabled",
        "method_scopes": ["dermlip_semi_supervised", "sled"],
        "stage": "selection",
        "bindings": bindings,
        "stage_manifests": {"selection": stage_binding},
        "preflight_receipt": receipt_binding,
        "pool_manifest_frozen": True,
        "freeze_semantics": "sha_identity_pinned_filesystem_immutability_not_claimed",
        "training_consumable": True,
        "patient_independence_accredited": False,
        "holdout_novelty": False,
        "schema_fields": ["image_id", "dataset", "image_path", "image_sha256"],
    }
    attestation_path = tmp_path / "pool-use-attestation.json"
    attestation_path.write_text(json.dumps(attestation, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(attestation_path.read_bytes()).hexdigest()
    original_sha256 = pool_use._sha256
    script_pin = pool_use.V2_POOL_USE_ATTESTATION_SHA256

    def fixture_sha256(path: Path) -> str:
        observed = original_sha256(path)
        if Path(path).resolve() == attestation_path.resolve() and observed == digest:
            return script_pin
        return observed

    monkeypatch.setattr(pool_use, "V2_POOL_USE_BINDINGS", bindings)
    monkeypatch.setattr(pool_use, "_sha256", fixture_sha256)
    return manifest, attestation_path, digest


def test_v2_preflight_uses_real_attestation_validator_without_trainer_or_outputs(
    tmp_path: Path, monkeypatch
):
    from dermssl.segmentation_methods.common import pool_use

    manifest, attestation, _ = _build_real_pool_use_attestation(tmp_path, monkeypatch)
    ph2_manifest = tmp_path / "ph2.json"
    ph2_manifest.write_text(
        json.dumps({"records": [{"image_id": "case", "case_group_id": "case"}]}),
        encoding="utf-8",
    )
    out_dir = tmp_path / "run"

    class OneCaseDataset:
        def __init__(self, *_args, **_kwargs):
            self.records = [{"image_id": "case", "case_group_id": "case"}]
        def __len__(self):
            return 1

    monkeypatch.setattr(semi_trainer, "PH2Dataset", OneCaseDataset)
    monkeypatch.setattr(
        semi_trainer, "validate_unlabeled_manifest",
        lambda *_args, **_kwargs: ([tmp_path / "unlabeled.png"], "a" * 64, {"records_validated": 1}),
    )
    monkeypatch.setattr(semi_trainer, "get_device", lambda _device: "cpu")
    monkeypatch.setattr(semi_trainer.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(semi_trainer, "UnlabeledDermDataset", OneCaseDataset)
    monkeypatch.setattr(semi_trainer, "TeacherStudentTrainer", lambda **_kwargs: pytest.fail("trainer initialized"))
    args = parse_args([
        "--ph2-manifest", str(ph2_manifest), "--unlabeled-manifest", str(manifest),
        "--pool-use-attestation", str(attestation), "--disable-dermlip",
        "--out-dir", str(out_dir), "--batch-size", "1",
    ])
    result = semi_trainer.run_training(args)
    assert result["status"] == "preflight_validated_execution_disabled"
    assert not out_dir.exists()
    assert pool_use.validate_pool_use_attestation.__name__ == "validate_pool_use_attestation"


@pytest.mark.parametrize("invalid_pin", ["attestation", "gate"])
def test_invalid_v2_pins_fail_before_any_cuda_query(tmp_path: Path, monkeypatch, invalid_pin: str):
    from dermssl.segmentation_methods.common import pool_use

    manifest, attestation, digest = _build_real_pool_use_attestation(tmp_path, monkeypatch)
    claim_path = tmp_path / "claims" / "dermlip-selection.claim.json"
    ph2_manifest = tmp_path / "ph2.json"
    ph2_manifest.write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "run"
    command = [
        "--ph2-manifest", str(ph2_manifest), "--unlabeled-manifest", str(manifest),
        "--pool-use-attestation", str(attestation), "--disable-dermlip",
        "--out-dir", str(out_dir), "--execution-claim-path", str(claim_path),
    ]
    expected_error = "SHA-256 pin mismatch"
    if invalid_pin == "attestation":
        attestation.write_text(attestation.read_text(encoding="utf-8") + " ", encoding="utf-8")
    else:
        command.append("--execute")
        command.extend(["--execution-gate", str(tmp_path / "gate.json")])
        args_for_profile = parse_args(command)
        profile = semi_trainer._execution_profile(args_for_profile, "selection")
        gate = {
            "schema": "dermlip_stage_execution_gate.v1",
            "method": "dermlip_semi_supervised",
            "stage": "selection",
            "execution_enabled": False,
            "launch_ready": False,
            "training_started": False,
            "pool_manifest_frozen": True,
            "training_consumable": True,
            "profile": profile,
            "profile_sha256": semi_trainer._json_sha256(profile),
            "execution_claim_path": str(claim_path.resolve()),
            "pool_manifest": {"path": str(manifest.resolve()), "sha256": semi_trainer._sha256(manifest)},
            "pool_use_attestation_sha256": pool_use.V2_POOL_USE_ATTESTATION_SHA256,
        }
        Path(command[command.index("--execution-gate") + 1]).write_text(json.dumps(gate), encoding="utf-8")
        expected_error = "execution_enabled"

    monkeypatch.setattr(semi_trainer.torch.cuda, "is_available", lambda: pytest.fail("CUDA queried before gates"))
    monkeypatch.setattr(semi_trainer, "get_device", lambda _device: pytest.fail("device resolved before gates"))
    with pytest.raises(ValueError, match=expected_error):
        semi_trainer.run_training(parse_args(command))
    assert not out_dir.exists()
    assert not claim_path.exists()


DERMLIP_SOURCE_MODEL_REVISION = "b455d4ca3b39a742ae9a8ea7d3246216c6d4bf4f"
DERMLIP_SOURCE_TOKENIZER_REVISION = "b79526d6ef3645e0df4530322e266f24c829f5ef"
DERMLIP_CONFIG_SOURCE_SHA256 = "0d9dcaeebcd6fe219392578f3da73ab025d471d4c7289698b56788172ddc2ba1"
DERMLIP_CHECKPOINT_SOURCE_SHA256 = "c01315cc8bda77decb7bfaaafaf4668979d065d0ded33c1cb42108cd8d56fedf"


def _mock_known_dermlip_source_hashes(monkeypatch):
    original_sha256 = semi_trainer._sha256

    def source_hash(path: Path) -> str:
        if path.name == "open_clip_config.json":
            return DERMLIP_CONFIG_SOURCE_SHA256
        if path.name == "open_clip_model.safetensors":
            return DERMLIP_CHECKPOINT_SOURCE_SHA256
        return original_sha256(path)

    monkeypatch.setattr(semi_trainer, "_sha256", source_hash)


def _pinned_test_file_hash(name: str, path: Path) -> str:
    if name == "open_clip_config.json":
        return DERMLIP_CONFIG_SOURCE_SHA256
    if name == "open_clip_model.safetensors":
        return DERMLIP_CHECKPOINT_SOURCE_SHA256
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_complete_test_dermlip_manifest(
    tmp_path: Path,
    *,
    model_revision: str = DERMLIP_SOURCE_MODEL_REVISION,
    config_hash: str | None = None,
    terms_acceptance: dict | None = None,
) -> Path:
    import hashlib

    bundle = tmp_path / "dermlip-bundle"
    bundle.mkdir()
    bundle_files = {
        "open_clip_config.json": b"{}",
        "open_clip_model.safetensors": b"checkpoint",
        "tokenizer.json": b"{}",
    }
    for name, content in bundle_files.items():
        (bundle / name).write_bytes(content)
    hashes = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in bundle_files.items()
    }
    hashes["open_clip_config.json"] = semi_trainer.DERMLIP_CONFIG_SOURCE_SHA256
    hashes["open_clip_model.safetensors"] = semi_trainer.DERMLIP_CHECKPOINT_SOURCE_SHA256
    if config_hash is not None:
        hashes["open_clip_config.json"] = config_hash
    manifest = {
        "schema_version": 1,
        "model": {
            "repository": "redlessone/DermLIP_PanDerm-base-w-PubMed-256",
            "revision": model_revision,
        },
        "tokenizer": {
            "repository": "NeuML/pubmedbert-base-embeddings",
            "revision": DERMLIP_SOURCE_TOKENIZER_REVISION,
        },
        "bundle_path": "dermlip-bundle",
        "license": "CC-BY-NC-ND-4.0",
        "terms_acceptance": terms_acceptance or {"status": "unverified"},
        "usage": {"commercial_use": "prohibited", "weight_modification": "prohibited"},
        "files": hashes,
    }
    manifest_path = tmp_path / "dermlip-model.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _write_terms_attestation_receipt(tmp_path: Path) -> dict:
    import hashlib

    receipt = tmp_path / "terms-acceptance.txt"
    attestation = {
        "repository": "redlessone/DermLIP_PanDerm-base-w-PubMed-256",
        "model": "DermLIP PanDerm-base-w-PubMed-256",
        "model_revision": DERMLIP_SOURCE_MODEL_REVISION,
        "terms_reference": "DermLIP model access terms",
        "terms_version": "CC-BY-NC-ND-4.0",
        "license": "CC-BY-NC-ND-4.0",
        "accepted_by": "test user",
        "accepted_on": "2026-10-20",
        "attested_at": "2026-10-21T12:00:00Z",
    }
    receipt.write_text(json.dumps(attestation), encoding="utf-8")
    return {
        "status": "user_attested",
        "verification_method": "manual_user_attestation",
        "human_attestation": {
            **attestation,
            "receipt_path": receipt.name,
            "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        },
    }


def test_dermlip_manifest_rejects_unapproved_model_revision(tmp_path: Path):
    manifest_path = _write_complete_test_dermlip_manifest(tmp_path, model_revision="a" * 40)

    with pytest.raises(ValueError, match="pinned source revision"):
        load_dermlip_model_manifest(manifest_path)


def test_dermlip_manifest_rejects_self_consistent_but_unverified_model_config_hash(
    tmp_path: Path, monkeypatch
):
    import hashlib

    altered_config_hash = hashlib.sha256(b"{}").hexdigest()
    monkeypatch.setattr(
        semi_trainer,
        "_sha256",
        lambda path: {
            "open_clip_config.json": altered_config_hash,
            "open_clip_model.safetensors": DERMLIP_CHECKPOINT_SOURCE_SHA256,
        }.get(path.name, __import__("hashlib").sha256(path.read_bytes()).hexdigest()),
    )
    manifest_path = _write_complete_test_dermlip_manifest(tmp_path, config_hash=altered_config_hash)

    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        load_dermlip_model_manifest(manifest_path)


def test_dermlip_model_manifest_pins_bundle_and_file_hashes(tmp_path: Path, monkeypatch):
    _mock_known_dermlip_source_hashes(monkeypatch)
    bundle = tmp_path / "dermlip-bundle"
    bundle.mkdir()
    (bundle / "open_clip_config.json").write_text("{}", encoding="utf-8")
    (bundle / "open_clip_model.safetensors").write_bytes(b"checkpoint")
    (bundle / "tokenizer.json").write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "model": {
            "repository": "redlessone/DermLIP_PanDerm-base-w-PubMed-256",
            "revision": DERMLIP_SOURCE_MODEL_REVISION,
        },
        "tokenizer": {
            "repository": "NeuML/pubmedbert-base-embeddings",
            "revision": DERMLIP_SOURCE_TOKENIZER_REVISION,
        },
        "bundle_path": "dermlip-bundle",
        "license": "CC-BY-NC-ND-4.0",
        "terms_acceptance": _write_terms_attestation_receipt(tmp_path),
        "usage": {"commercial_use": "prohibited", "weight_modification": "prohibited"},
        "files": {
            name: _pinned_test_file_hash(name, bundle / name)
            for name in ("open_clip_config.json", "open_clip_model.safetensors", "tokenizer.json")
        },
    }
    manifest_path = tmp_path / "dermlip-model.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    identity = load_dermlip_model_manifest(manifest_path)

    assert identity["model_repository"] == manifest["model"]["repository"]
    assert identity["model_revision"] == manifest["model"]["revision"]
    assert identity["tokenizer_repository"] == manifest["tokenizer"]["repository"]
    assert identity["tokenizer_revision"] == manifest["tokenizer"]["revision"]
    assert identity["model_name"] == f"hf-hub:{manifest['model']['repository']}"
    assert identity["license"] == "CC-BY-NC-ND-4.0"
    assert identity["terms_acceptance_status"] == "user_attested"
    assert identity["terms_acceptance_verification_method"] == "manual_user_attestation"
    assert identity["model_source_hashes_verified"] is True
    assert identity["tokenizer_source_hashes_verified"] is False
    assert identity["tokenizer_bundle_integrity_verified"] is True
    assert identity["source_provenance_status"] == "model_files_verified_tokenizer_hashes_unverified"
    assert len(identity["manifest_sha256"]) == 64


def test_dermlip_manifest_unverified_terms_block_model_use(tmp_path: Path, monkeypatch):
    _mock_known_dermlip_source_hashes(monkeypatch)
    manifest_path = _write_complete_test_dermlip_manifest(
        tmp_path, terms_acceptance={"status": "unverified"}
    )

    with pytest.raises(ValueError, match="terms.*user_attested|human_attestation"):
        load_dermlip_model_manifest(manifest_path)


def test_unverified_terms_stop_training_before_model_or_loop(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    _mock_known_dermlip_source_hashes(monkeypatch)
    manifest_path = _write_complete_test_dermlip_manifest(tmp_path)
    created_models = []
    monkeypatch.setattr(
        semi_trainer,
        "DermLIPSuperpixelCalibrator",
        lambda **_kwargs: created_models.append("model"),
    )
    args = SimpleNamespace(
        seed=42,
        device="cpu",
        out_dir=tmp_path / "output",
        resume=None,
        enable_dermlip=True,
        dermlip_model_manifest=manifest_path,
    )

    with pytest.raises(ValueError, match="terms.*user_attested|human_attestation"):
        semi_trainer.run_training(args)

    assert created_models == []


def test_dermlip_manifest_accepts_structured_user_attestation(tmp_path: Path, monkeypatch):
    _mock_known_dermlip_source_hashes(monkeypatch)
    manifest_path = _write_complete_test_dermlip_manifest(
        tmp_path,
        terms_acceptance=_write_terms_attestation_receipt(tmp_path),
    )

    identity = load_dermlip_model_manifest(manifest_path)

    attestation = json.loads(manifest_path.read_text(encoding="utf-8"))["terms_acceptance"]
    assert identity["terms_acceptance_status"] == "user_attested"
    assert identity["terms_acceptance_verification_method"] == "manual_user_attestation"
    assert identity["terms_acceptance_evidence_sha256"] == attestation["human_attestation"]["receipt_sha256"]


def test_dermlip_manifest_rejects_arbitrary_acceptance_file(tmp_path: Path, monkeypatch):
    _mock_known_dermlip_source_hashes(monkeypatch)
    receipt = tmp_path / "terms-acceptance.txt"
    receipt.write_text("Accepted", encoding="utf-8")
    import hashlib

    manifest_path = _write_complete_test_dermlip_manifest(
        tmp_path,
        terms_acceptance={
            "status": "accepted",
            "evidence": {"path": receipt.name, "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()},
        },
    )

    with pytest.raises(ValueError, match="user_attested|human_attestation"):
        load_dermlip_model_manifest(manifest_path)


@pytest.mark.parametrize(
    "field", ["accepted_by", "accepted_on", "attested_at", "terms_reference", "receipt_sha256"]
)
def test_dermlip_manifest_rejects_incomplete_user_attestation(tmp_path: Path, monkeypatch, field: str):
    _mock_known_dermlip_source_hashes(monkeypatch)
    attestation = _write_terms_attestation_receipt(tmp_path)
    attestation["human_attestation"].pop(field)
    manifest_path = _write_complete_test_dermlip_manifest(tmp_path, terms_acceptance=attestation)

    with pytest.raises(ValueError, match="attestation|receipt"):
        load_dermlip_model_manifest(manifest_path)


def test_dermlip_attestation_keeps_user_acceptance_date_without_invented_time(
    tmp_path: Path, monkeypatch
):
    _mock_known_dermlip_source_hashes(monkeypatch)
    manifest_path = _write_complete_test_dermlip_manifest(
        tmp_path,
        terms_acceptance=_write_terms_attestation_receipt(tmp_path),
    )

    identity = load_dermlip_model_manifest(manifest_path)

    attestation = identity["terms_human_attestation"]
    assert attestation["accepted_on"] == "2026-10-20"
    assert attestation["attested_at"].endswith("Z")
    assert "accepted_at" not in attestation


def test_dermlip_model_manifest_rejects_missing_bundle_file_or_bad_hash(tmp_path: Path):
    bundle = tmp_path / "dermlip-bundle"
    bundle.mkdir()
    (bundle / "model.bin").write_bytes(b"checkpoint")
    manifest = {
        "schema_version": 1,
        "model": {
            "repository": "redlessone/DermLIP_PanDerm-base-w-PubMed-256",
            "revision": DERMLIP_SOURCE_MODEL_REVISION,
        },
        "tokenizer": {
            "repository": "NeuML/pubmedbert-base-embeddings",
            "revision": DERMLIP_SOURCE_TOKENIZER_REVISION,
        },
        "bundle_path": "dermlip-bundle",
        "license": "CC-BY-NC-ND-4.0",
        "terms_acceptance": {"status": "unverified"},
        "usage": {"commercial_use": "prohibited", "weight_modification": "prohibited"},
        "files": {"model.bin": "0" * 64, "tokenizer.json": "1" * 64},
    }
    manifest_path = tmp_path / "dermlip-model.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch|missing"):
        load_dermlip_model_manifest(manifest_path)


def test_dermlip_model_manifest_rejects_bundle_file_omitted_from_hash_inventory(tmp_path: Path, monkeypatch):
    _mock_known_dermlip_source_hashes(monkeypatch)

    bundle = tmp_path / "dermlip-bundle"
    bundle.mkdir()
    bundle_files = {
        "open_clip_config.json": b"{}",
        "open_clip_model.safetensors": b"checkpoint",
        "tokenizer.json": b"{}",
        "README.md": b"untracked bundle content",
    }
    for name, content in bundle_files.items():
        (bundle / name).write_bytes(content)
    manifest = {
        "schema_version": 1,
        "model": {
            "repository": "redlessone/DermLIP_PanDerm-base-w-PubMed-256",
            "revision": DERMLIP_SOURCE_MODEL_REVISION,
        },
        "tokenizer": {
            "repository": "NeuML/pubmedbert-base-embeddings",
            "revision": DERMLIP_SOURCE_TOKENIZER_REVISION,
        },
        "bundle_path": "dermlip-bundle",
        "license": "CC-BY-NC-ND-4.0",
        "terms_acceptance": {"status": "unverified"},
        "usage": {"commercial_use": "prohibited", "weight_modification": "prohibited"},
        "files": {
            name: _pinned_test_file_hash(name, bundle / name)
            for name, content in bundle_files.items()
            if name != "README.md"
        },
    }
    manifest_path = tmp_path / "dermlip-model.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="inventory|unlisted|omitted"):
        load_dermlip_model_manifest(manifest_path)


def test_dermlip_model_manifest_rejects_hash_inventory_entry_missing_from_bundle(tmp_path: Path, monkeypatch):
    _mock_known_dermlip_source_hashes(monkeypatch)

    bundle = tmp_path / "dermlip-bundle"
    bundle.mkdir()
    bundle_files = {
        "open_clip_config.json": b"{}",
        "open_clip_model.safetensors": b"checkpoint",
        "tokenizer.json": b"{}",
    }
    for name, content in bundle_files.items():
        (bundle / name).write_bytes(content)
    manifest = {
        "schema_version": 1,
        "model": {
            "repository": "redlessone/DermLIP_PanDerm-base-w-PubMed-256",
            "revision": DERMLIP_SOURCE_MODEL_REVISION,
        },
        "tokenizer": {
            "repository": "NeuML/pubmedbert-base-embeddings",
            "revision": DERMLIP_SOURCE_TOKENIZER_REVISION,
        },
        "bundle_path": "dermlip-bundle",
        "license": "CC-BY-NC-ND-4.0",
        "terms_acceptance": {"status": "unverified"},
        "usage": {"commercial_use": "prohibited", "weight_modification": "prohibited"},
        "files": {
            **{
                name: _pinned_test_file_hash(name, bundle / name)
                for name, content in bundle_files.items()
            },
            "extra.json": "c" * 64,
        },
    }
    manifest_path = tmp_path / "dermlip-model.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing"):
        load_dermlip_model_manifest(manifest_path)


def test_dermlip_model_manifest_requires_tokenizer_assets_at_bundle_root(tmp_path: Path, monkeypatch):
    _mock_known_dermlip_source_hashes(monkeypatch)

    bundle = tmp_path / "dermlip-bundle"
    nested_tokenizer = bundle / "tokenizer"
    nested_tokenizer.mkdir(parents=True)
    bundle_files = {
        "open_clip_config.json": b"{}",
        "open_clip_model.safetensors": b"checkpoint",
        "tokenizer/tokenizer.json": b"{}",
    }
    for name, content in bundle_files.items():
        (bundle / name).write_bytes(content)
    manifest = {
        "schema_version": 1,
        "model": {
            "repository": "redlessone/DermLIP_PanDerm-base-w-PubMed-256",
            "revision": DERMLIP_SOURCE_MODEL_REVISION,
        },
        "tokenizer": {
            "repository": "NeuML/pubmedbert-base-embeddings",
            "revision": DERMLIP_SOURCE_TOKENIZER_REVISION,
        },
        "bundle_path": "dermlip-bundle",
        "license": "CC-BY-NC-ND-4.0",
        "terms_acceptance": {"status": "unverified"},
        "usage": {"commercial_use": "prohibited", "weight_modification": "prohibited"},
        "files": {
            name: _pinned_test_file_hash(name, bundle / name)
            for name, content in bundle_files.items()
        },
    }
    manifest_path = tmp_path / "dermlip-model.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="tokenizer.*bundle root"):
        load_dermlip_model_manifest(manifest_path)


def test_dermlip_manifest_argument_is_available():
    args = parse_args(["--dermlip-model-manifest", "model.json"])

    assert args.dermlip_model_manifest == Path("model.json")


def _populate_hf_cache(
    cache_root: Path,
    *,
    model_ref: str = DERMLIP_SOURCE_MODEL_REVISION,
    tokenizer_ref: str = DERMLIP_SOURCE_TOKENIZER_REVISION,
):
    import hashlib

    model_repo = cache_root / "models--redlessone--DermLIP_PanDerm-base-w-PubMed-256"
    tokenizer_repo = cache_root / "models--NeuML--pubmedbert-base-embeddings"
    model_snapshot = model_repo / "snapshots" / DERMLIP_SOURCE_MODEL_REVISION
    tokenizer_snapshot = tokenizer_repo / "snapshots" / DERMLIP_SOURCE_TOKENIZER_REVISION
    model_snapshot.mkdir(parents=True)
    tokenizer_snapshot.mkdir(parents=True)
    (model_repo / "refs").mkdir()
    (tokenizer_repo / "refs").mkdir()
    (model_repo / "refs" / "main").write_text(model_ref, encoding="utf-8")
    (tokenizer_repo / "refs" / "main").write_text(tokenizer_ref, encoding="utf-8")
    consumed = {
        model_snapshot / "open_clip_config.json": b"model config",
        model_snapshot / "open_clip_model.safetensors": b"model weights",
        tokenizer_snapshot / "tokenizer.json": b"{}",
    }
    for path, content in consumed.items():
        path.write_bytes(content)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in consumed}
    return cache_root, hashes


def _accepted_dermlip_identity(tmp_path: Path, monkeypatch) -> dict:
    import hashlib

    config_hash = hashlib.sha256(b"model config").hexdigest()
    weights_hash = hashlib.sha256(b"model weights").hexdigest()
    monkeypatch.setattr(semi_trainer, "DERMLIP_CONFIG_SOURCE_SHA256", config_hash)
    monkeypatch.setattr(semi_trainer, "DERMLIP_CHECKPOINT_SOURCE_SHA256", weights_hash)
    original_sha256 = semi_trainer._sha256

    def source_hash(path: Path) -> str:
        if path.parent.name == "dermlip-bundle":
            return {
                "open_clip_config.json": config_hash,
                "open_clip_model.safetensors": weights_hash,
            }.get(path.name, original_sha256(path))
        return original_sha256(path)

    monkeypatch.setattr(semi_trainer, "_sha256", source_hash)
    manifest_path = _write_complete_test_dermlip_manifest(
        tmp_path,
        terms_acceptance=_write_terms_attestation_receipt(tmp_path),
    )
    return load_dermlip_model_manifest(manifest_path)


def test_dermlip_cache_requires_approved_active_refs_and_consumed_file_hashes(tmp_path: Path, monkeypatch):
    identity = _accepted_dermlip_identity(tmp_path, monkeypatch)
    cache_root, hashes = _populate_hf_cache(tmp_path / "cache")

    receipt = semi_trainer.verify_dermlip_hf_cache(identity, cache_root=cache_root)

    assert receipt["model_ref"] == DERMLIP_SOURCE_MODEL_REVISION
    assert receipt["tokenizer_ref"] == DERMLIP_SOURCE_TOKENIZER_REVISION
    assert receipt["model_config_sha256"] == semi_trainer.DERMLIP_CONFIG_SOURCE_SHA256
    assert receipt["model_weights_sha256"] == semi_trainer.DERMLIP_CHECKPOINT_SOURCE_SHA256
    assert receipt["tokenizer_source_hashes_verified"] is False
    assert receipt["tokenizer_cache_integrity_verified"] is True


def test_dermlip_cache_ref_mismatch_fails_closed(tmp_path: Path, monkeypatch):
    identity = _accepted_dermlip_identity(tmp_path, monkeypatch)
    cache_root, _ = _populate_hf_cache(tmp_path / "cache", model_ref="a" * 40)

    with pytest.raises(ValueError, match="model.*ref|active.*revision"):
        semi_trainer.verify_dermlip_hf_cache(identity, cache_root=cache_root)


def test_dermlip_tokenizer_cache_ref_mismatch_fails_closed(tmp_path: Path, monkeypatch):
    identity = _accepted_dermlip_identity(tmp_path, monkeypatch)
    cache_root, _ = _populate_hf_cache(tmp_path / "cache", tokenizer_ref="a" * 40)

    with pytest.raises(ValueError, match="tokenizer.*ref|active.*revision"):
        semi_trainer.verify_dermlip_hf_cache(identity, cache_root=cache_root)


def test_dermlip_cache_file_hash_mismatch_fails_closed(tmp_path: Path, monkeypatch):
    identity = _accepted_dermlip_identity(tmp_path, monkeypatch)
    cache_root, hashes = _populate_hf_cache(tmp_path / "cache")
    (cache_root / "models--redlessone--DermLIP_PanDerm-base-w-PubMed-256" / "snapshots"
     / DERMLIP_SOURCE_MODEL_REVISION / "open_clip_config.json").write_bytes(b"altered")

    with pytest.raises(ValueError, match="cache.*SHA-256|cache.*hash"):
        semi_trainer.verify_dermlip_hf_cache(identity, cache_root=cache_root)


def test_dermlip_cache_verification_is_read_only(tmp_path: Path, monkeypatch):
    import hashlib

    identity = _accepted_dermlip_identity(tmp_path, monkeypatch)
    cache_root, hashes = _populate_hf_cache(tmp_path / "cache")
    before = {
        path.relative_to(cache_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in cache_root.rglob("*") if path.is_file()
    }

    semi_trainer.verify_dermlip_hf_cache(identity, cache_root=cache_root)

    after = {
        path.relative_to(cache_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in cache_root.rglob("*") if path.is_file()
    }
    assert after == before


def _fake_vendor_open_clip(tmp_path: Path):
    import hashlib
    from types import SimpleNamespace

    vendor_root = tmp_path / "vendor" / "Derm1M"
    package_dir = vendor_root / "src" / "open_clip"
    package_dir.mkdir(parents=True)
    module_path = package_dir / "__init__.py"
    module_path.write_text("", encoding="utf-8")
    factory_path = package_dir / "factory.py"
    tokenizer_path = package_dir / "tokenizer.py"
    factory_path.write_bytes(b"factory source")
    tokenizer_path.write_bytes(b"tokenizer source")
    (vendor_root / ".git").mkdir()
    (vendor_root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    module = SimpleNamespace(
        __file__=str(module_path),
        call_PanDerm_base_visual=lambda: None,
    )
    hashes = {
        "factory": hashlib.sha256(factory_path.read_bytes()).hexdigest(),
        "tokenizer": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
    }
    return module, vendor_root, hashes


def _mock_vendor_git(monkeypatch, calibrator_module, *, head=None, status=None):
    from types import SimpleNamespace

    allowed_status = " M src/open_clip/factory.py\n M src/open_clip/tokenizer.py\n"

    def run(command, *_args, **_kwargs):
        if "status" in command:
            return SimpleNamespace(stdout=allowed_status if status is None else status)
        return SimpleNamespace(
            stdout=(calibrator_module.DERMLIP_VENDOR_COMMIT if head is None else head) + "\n"
        )

    monkeypatch.setattr(calibrator_module.subprocess, "run", run)


def test_open_clip_fork_rejects_module_from_wrong_path(tmp_path: Path):
    from types import SimpleNamespace

    from dermssl.segmentation_methods.dermlip_slic.calibrator import _validate_open_clip_fork

    wrong_module = SimpleNamespace(
        __file__=str(tmp_path / "site-packages" / "open_clip" / "__init__.py"),
        call_PanDerm_base_visual=lambda: None,
    )

    with pytest.raises(RuntimeError, match="vendor.*path|expected.*fork"):
        _validate_open_clip_fork(wrong_module, vendor_root=tmp_path / "vendor" / "Derm1M")


def test_open_clip_fork_rejects_wrong_vendor_commit(tmp_path: Path, monkeypatch):

    import dermssl.segmentation_methods.dermlip_slic.calibrator as calibrator_module

    module, vendor_root, hashes = _fake_vendor_open_clip(tmp_path)
    monkeypatch.setattr(calibrator_module, "DERMLIP_VENDOR_FACTORY_SHA256", hashes["factory"], raising=False)
    monkeypatch.setattr(calibrator_module, "DERMLIP_VENDOR_TOKENIZER_SHA256", hashes["tokenizer"], raising=False)
    _mock_vendor_git(monkeypatch, calibrator_module, head="a" * 40)

    with pytest.raises(RuntimeError, match="vendor.*commit|pinned.*commit"):
        calibrator_module._validate_open_clip_fork(module, vendor_root=vendor_root)


def test_open_clip_fork_rejects_vendor_file_hash_mismatch(tmp_path: Path, monkeypatch):
    import dermssl.segmentation_methods.dermlip_slic.calibrator as calibrator_module

    module, vendor_root, hashes = _fake_vendor_open_clip(tmp_path)
    monkeypatch.setattr(calibrator_module, "DERMLIP_VENDOR_FACTORY_SHA256", hashes["factory"], raising=False)
    monkeypatch.setattr(calibrator_module, "DERMLIP_VENDOR_TOKENIZER_SHA256", hashes["tokenizer"], raising=False)
    _mock_vendor_git(monkeypatch, calibrator_module)
    (vendor_root / "src" / "open_clip" / "factory.py").write_bytes(b"altered factory")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch.*factory|factory.*SHA"):
        calibrator_module._validate_open_clip_fork(module, vendor_root=vendor_root)


def test_open_clip_fork_rejects_additional_vendor_worktree_changes(tmp_path: Path, monkeypatch):
    import dermssl.segmentation_methods.dermlip_slic.calibrator as calibrator_module

    module, vendor_root, hashes = _fake_vendor_open_clip(tmp_path)
    monkeypatch.setattr(calibrator_module, "DERMLIP_VENDOR_FACTORY_SHA256", hashes["factory"], raising=False)
    monkeypatch.setattr(calibrator_module, "DERMLIP_VENDOR_TOKENIZER_SHA256", hashes["tokenizer"], raising=False)
    _mock_vendor_git(monkeypatch, calibrator_module, status=" M src/open_clip/factory.py\n?? unexpected.py\n")

    with pytest.raises(RuntimeError, match="vendor.*status|worktree.*changes"):
        calibrator_module._validate_open_clip_fork(module, vendor_root=vendor_root)


def test_open_clip_fork_rejects_vendor_repo_without_git_checkout(tmp_path: Path):
    module, vendor_root, _ = _fake_vendor_open_clip(tmp_path)
    import shutil

    shutil.rmtree(vendor_root / ".git")

    import dermssl.segmentation_methods.dermlip_slic.calibrator as calibrator_module

    with pytest.raises(RuntimeError, match="vendor.*git|checkout.*HEAD"):
        calibrator_module._validate_open_clip_fork(module, vendor_root=vendor_root)


def test_open_clip_fork_rejects_unverifiable_vendor_head(tmp_path: Path, monkeypatch):
    import subprocess

    import dermssl.segmentation_methods.dermlip_slic.calibrator as calibrator_module

    module, vendor_root, _ = _fake_vendor_open_clip(tmp_path)

    def unavailable_head(*_args, **_kwargs):
        raise subprocess.CalledProcessError(128, "git rev-parse HEAD")

    monkeypatch.setattr(calibrator_module.subprocess, "run", unavailable_head)

    with pytest.raises(RuntimeError, match="vendor.*commit|checkout.*HEAD"):
        calibrator_module._validate_open_clip_fork(module, vendor_root=vendor_root)


def test_open_clip_fork_rejects_missing_panderm_api(tmp_path: Path, monkeypatch):

    import dermssl.segmentation_methods.dermlip_slic.calibrator as calibrator_module

    module, vendor_root, hashes = _fake_vendor_open_clip(tmp_path)
    module.call_PanDerm_base_visual = None
    monkeypatch.setattr(calibrator_module, "DERMLIP_VENDOR_FACTORY_SHA256", hashes["factory"], raising=False)
    monkeypatch.setattr(calibrator_module, "DERMLIP_VENDOR_TOKENIZER_SHA256", hashes["tokenizer"], raising=False)
    _mock_vendor_git(monkeypatch, calibrator_module)

    with pytest.raises(RuntimeError, match="PanDerm.*API"):
        calibrator_module._validate_open_clip_fork(module, vendor_root=vendor_root)


def test_calibrator_rejects_fabricated_source_verification_before_import(monkeypatch):
    import builtins

    from dermssl.segmentation_methods.dermlip_slic.calibrator import DermLIPSuperpixelCalibrator

    imported = []
    original_import = builtins.__import__

    def import_guard(name, *args, **kwargs):
        if name == "open_clip":
            imported.append(name)
            raise AssertionError("untrusted OpenCLIP import attempted")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_guard)
    calibrator = DermLIPSuperpixelCalibrator(
        model_name=f"hf-hub:{semi_trainer.DERMLIP_MODEL_REPOSITORY}",
        source_verification={"verified": True, "model_revision": "main"},
    )

    with pytest.raises(RuntimeError, match="source verification|approved"):
        calibrator.initialize_model()

    assert imported == []


def _complete_calibrator_source_claim(tmp_path: Path, monkeypatch) -> dict:
    import hashlib

    import dermssl.segmentation_methods.dermlip_slic.calibrator as calibrator_module

    config_path = tmp_path / "cache" / f"models--{calibrator_module.DERMLIP_MODEL_REPOSITORY.replace('/', '--')}" \
        / "snapshots" / calibrator_module.DERMLIP_MODEL_REVISION
    tokenizer_path = tmp_path / "cache" \
        / f"models--{calibrator_module.DERMLIP_TOKENIZER_REPOSITORY.replace('/', '--')}" \
        / "snapshots" / calibrator_module.DERMLIP_TOKENIZER_REVISION
    config_path.mkdir(parents=True)
    tokenizer_path.mkdir(parents=True)
    config_hash = hashlib.sha256(b"config source").hexdigest()
    weights_hash = hashlib.sha256(b"weights source").hexdigest()
    tokenizer_hash = hashlib.sha256(b"tokenizer source").hexdigest()
    monkeypatch.setattr(calibrator_module, "DERMLIP_CONFIG_SOURCE_SHA256", config_hash)
    monkeypatch.setattr(calibrator_module, "DERMLIP_CHECKPOINT_SOURCE_SHA256", weights_hash)
    (config_path / "open_clip_config.json").write_bytes(b"config source")
    (config_path / "open_clip_model.safetensors").write_bytes(b"weights source")
    (tokenizer_path / "tokenizer.json").write_bytes(b"tokenizer source")
    model_cache = config_path.parents[1]
    tokenizer_cache = tokenizer_path.parents[1]
    (model_cache / "refs").mkdir()
    (tokenizer_cache / "refs").mkdir()
    (model_cache / "refs" / "main").write_text(calibrator_module.DERMLIP_MODEL_REVISION, encoding="utf-8")
    (tokenizer_cache / "refs" / "main").write_text(calibrator_module.DERMLIP_TOKENIZER_REVISION, encoding="utf-8")

    attestation = {
        "repository": calibrator_module.DERMLIP_MODEL_REPOSITORY,
        "model": calibrator_module.DERMLIP_MODEL_NAME,
        "model_revision": calibrator_module.DERMLIP_MODEL_REVISION,
        "terms_reference": "DermLIP model access terms",
        "terms_version": calibrator_module.DERMLIP_LICENSE,
        "license": calibrator_module.DERMLIP_LICENSE,
        "accepted_by": "test user",
        "accepted_on": "2026-10-20",
        "attested_at": "2026-10-21T12:00:00Z",
    }
    receipt_path = tmp_path / "attestation.json"
    receipt_path.write_text(json.dumps(attestation), encoding="utf-8")
    receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    return {
        "model_name": f"hf-hub:{calibrator_module.DERMLIP_MODEL_REPOSITORY}",
        "model_repository": calibrator_module.DERMLIP_MODEL_REPOSITORY,
        "model_revision": calibrator_module.DERMLIP_MODEL_REVISION,
        "tokenizer_repository": calibrator_module.DERMLIP_TOKENIZER_REPOSITORY,
        "tokenizer_revision": calibrator_module.DERMLIP_TOKENIZER_REVISION,
        "license": calibrator_module.DERMLIP_LICENSE,
        "terms_acceptance_status": "user_attested",
        "terms_acceptance_verification_method": "manual_user_attestation",
        "terms_acceptance_evidence_path": str(receipt_path),
        "terms_acceptance_evidence_sha256": receipt_sha256,
        "terms_human_attestation": {**attestation, "receipt_sha256": receipt_sha256},
        "model_source_hashes_verified": True,
        "tokenizer_source_hashes_verified": False,
        "tokenizer_bundle_integrity_verified": True,
        "files": {
            "open_clip_config.json": calibrator_module.DERMLIP_CONFIG_SOURCE_SHA256,
            "open_clip_model.safetensors": calibrator_module.DERMLIP_CHECKPOINT_SOURCE_SHA256,
            "tokenizer.json": tokenizer_hash,
        },
        "cache_verification": {
            "verified": True,
            "cache_root": str(tmp_path / "cache"),
            "cache_mutated": False,
            "model_ref": calibrator_module.DERMLIP_MODEL_REVISION,
            "tokenizer_ref": calibrator_module.DERMLIP_TOKENIZER_REVISION,
            "model_config_sha256": calibrator_module.DERMLIP_CONFIG_SOURCE_SHA256,
            "model_weights_sha256": calibrator_module.DERMLIP_CHECKPOINT_SOURCE_SHA256,
            "tokenizer_source_hashes_verified": False,
            "tokenizer_cache_integrity_verified": True,
            "tokenizer_file_sha256": {"tokenizer.json": tokenizer_hash},
        },
    }


def test_calibrator_rejects_full_claim_mapping_without_real_attestation_receipt(
    tmp_path: Path, monkeypatch
):
    import dermssl.segmentation_methods.dermlip_slic.calibrator as calibrator_module

    source = _complete_calibrator_source_claim(tmp_path, monkeypatch)
    source["terms_acceptance_evidence_path"] = str(tmp_path / "missing-receipt.json")

    with pytest.raises(RuntimeError, match="attestation receipt|receipt SHA"):
        calibrator_module._validate_dermlip_source_verification(
            f"hf-hub:{calibrator_module.DERMLIP_MODEL_REPOSITORY}", source
        )


def test_calibrator_rejects_full_claim_with_arbitrary_hashed_receipt(tmp_path: Path, monkeypatch):
    import hashlib

    import dermssl.segmentation_methods.dermlip_slic.calibrator as calibrator_module

    source = _complete_calibrator_source_claim(tmp_path, monkeypatch)
    receipt_path = Path(source["terms_acceptance_evidence_path"])
    receipt_path.write_text("Accepted", encoding="utf-8")
    receipt_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    source["terms_acceptance_evidence_sha256"] = receipt_hash
    source["terms_human_attestation"]["receipt_sha256"] = receipt_hash

    with pytest.raises(RuntimeError, match="structured JSON|attestation receipt fields"):
        calibrator_module._validate_dermlip_source_verification(
            f"hf-hub:{calibrator_module.DERMLIP_MODEL_REPOSITORY}", source
        )


def test_calibrator_accepts_source_claim_only_when_receipt_is_real_and_matching(
    tmp_path: Path, monkeypatch
):
    import dermssl.segmentation_methods.dermlip_slic.calibrator as calibrator_module

    source = _complete_calibrator_source_claim(tmp_path, monkeypatch)

    calibrator_module._validate_dermlip_source_verification(
        f"hf-hub:{calibrator_module.DERMLIP_MODEL_REPOSITORY}", source
    )


def test_calibrator_rejects_claim_when_verified_hf_cache_changes(tmp_path: Path, monkeypatch):
    import dermssl.segmentation_methods.dermlip_slic.calibrator as calibrator_module

    source = _complete_calibrator_source_claim(tmp_path, monkeypatch)
    model_cache = Path(source["cache_verification"]["cache_root"])
    model_repo = model_cache / f"models--{calibrator_module.DERMLIP_MODEL_REPOSITORY.replace('/', '--')}"
    (model_repo / "refs" / "main").write_text("a" * 40, encoding="utf-8")

    with pytest.raises(RuntimeError, match="cache.*ref|active.*revision"):
        calibrator_module._validate_dermlip_source_verification(
            f"hf-hub:{calibrator_module.DERMLIP_MODEL_REPOSITORY}", source
        )


def test_calibrator_sets_offline_mode_before_open_clip_import(tmp_path: Path, monkeypatch):
    import builtins

    from dermssl.segmentation_methods.dermlip_slic.calibrator import DermLIPSuperpixelCalibrator

    _, vendor_root, _ = _fake_vendor_open_clip(tmp_path)
    original_import = builtins.__import__
    observed = {}

    def import_guard(name, *args, **kwargs):
        if name == "open_clip":
            observed.update({key: __import__("os").environ.get(key) for key in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"
            )})
            raise RuntimeError("stop before importing OpenCLIP")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_guard)
    import dermssl.segmentation_methods.dermlip_slic.calibrator as calibrator_module
    monkeypatch.setattr(calibrator_module, "DERMLIP_VENDOR_REPO_ROOT", str(vendor_root))
    calibrator = DermLIPSuperpixelCalibrator(
        model_name=f"hf-hub:{semi_trainer.DERMLIP_MODEL_REPOSITORY}",
        source_verification=_complete_calibrator_source_claim(tmp_path, monkeypatch),
    )

    with pytest.raises(RuntimeError, match="stop before importing"):
        calibrator.initialize_model()

    assert observed == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }


def test_dermlip_guidance_failure_is_logged_and_not_replaced_with_neutral_map(tmp_path: Path):
    class FailingCalibrator:
        def calibrate_image(self, *_args, **_kwargs):
            raise RuntimeError("model unavailable")

    image_path = tmp_path / "case_17.png"
    Image.new("RGB", (16, 16), (200, 100, 50)).save(image_path)
    ledger = tmp_path / "out" / "input_failures.jsonl"
    dataset = UnlabeledDermDataset([image_path], target_size=(16, 16), calibrator=FailingCalibrator(), failure_ledger_path=ledger)
    with pytest.raises(RuntimeError, match="DermLIP guidance failed for case_17.png"):
        dataset[0]
    assert '"image_id": "case_17"' in ledger.read_text(encoding="utf-8")


def test_ph2_preflight_failure_receipt_keeps_safe_diagnostic_and_manifest_hash(tmp_path: Path, monkeypatch):
    manifest_path = tmp_path / "ph2.json"
    manifest_path.write_text(json.dumps({"records": [{"image_id": "PH2_001"}]}), encoding="utf-8")
    out_dir = tmp_path / "run"
    monkeypatch.setattr(
        semi_trainer,
        "PH2Dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError(
                "bad mask at /home/jovyan/work/ph2/images/PH2_001.bmp and "
                r"C:\Users\jovyan\data\ph2\masks\PH2_001.png and "
                "C:/Users/jovyan/data/ph2/PH2_001.png and /secret /etc/passwd; "
                "api_key=key123 api-key=key456 Authorization: Bearer auth123 "
                "Bearer standalone123; see https://example.org/docs; "
                "token=hf_abcdefghijklmnopqrstuvwxyz1234567890" + "x" * 700
            )
        ),
    )

    args = parse_args([
        "--ph2-manifest", str(manifest_path),
        "--out-dir", str(out_dir),
        "--disable-dermlip",
    ])
    with pytest.raises(ValueError, match="bad mask"):
        semi_trainer.run_training(args)

    receipt = json.loads((out_dir / "preflight_failures.jsonl").read_text(encoding="utf-8"))
    assert receipt["error_type"] == "ValueError"
    assert receipt["error_message"].startswith("bad mask at [LOCAL_PATH] and [LOCAL_PATH]")
    assert "/home/jovyan" not in receipt["error_message"]
    assert "C:\\Users\\jovyan" not in receipt["error_message"]
    assert "C:/Users/jovyan" not in receipt["error_message"]
    assert "/secret" not in receipt["error_message"]
    assert "/etc/passwd" not in receipt["error_message"]
    assert "api_key=key123" not in receipt["error_message"]
    assert "api-key=key456" not in receipt["error_message"]
    assert "Authorization: Bearer auth123" not in receipt["error_message"]
    assert "Bearer standalone123" not in receipt["error_message"]
    assert "api_key=[REDACTED]" in receipt["error_message"]
    assert "api-key=[REDACTED]" in receipt["error_message"]
    assert "Authorization=[REDACTED]" in receipt["error_message"]
    assert "Bearer [REDACTED]" in receipt["error_message"]
    assert "https://example.org/docs" in receipt["error_message"]
    assert len(receipt["error_message"]) <= 500
    assert "abcdefghijklmnopqrstuvwxyz" not in receipt["error_message"]
    assert receipt["ph2_manifest_sha256"] == semi_trainer._sha256(manifest_path)


def test_evaluate_reports_both_student_and_teacher_metrics():
    student = DermLIPSegmentationUNet(in_channels=3, num_classes=1, base_channels=8)
    trainer = TeacherStudentTrainer(
        student=student,
        device="cpu",
        ema_decay=0.90,
    )
    dataset = torch.utils.data.TensorDataset(
        torch.randn(4, 3, 32, 32),
        (torch.rand(4, 1, 32, 32) > 0.5).float(),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    metrics = trainer.evaluate(loader)

    for key in ("student_dice", "student_iou", "teacher_dice", "teacher_iou"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0
    assert metrics["ph2_validation_cases"] == 4
    assert "mean_dice" not in metrics
    assert "mean_iou" not in metrics


def test_adaptive_ema_warmup_prevents_initial_noise_persistence():
    student = DermLIPSegmentationUNet(in_channels=3, num_classes=1, base_channels=8)
    trainer = TeacherStudentTrainer(
        student=student,
        device="cpu",
        ema_decay=0.99,
    )
    student.enc1.conv[0].weight.data.fill_(5.0)
    trainer.teacher.enc1.conv[0].weight.data.fill_(0.0)

    # Step 0: decay = min(1 - 1/1, 0.99) = 0.0 -> teacher must immediately adopt student weights
    trainer.update_teacher(step=0)
    assert torch.allclose(trainer.teacher.enc1.conv[0].weight, student.enc1.conv[0].weight)

    # Step 9: decay = min(1 - 1/10, 0.99) = 0.90
    student.enc1.conv[0].weight.data.fill_(10.0)
    trainer.update_teacher(step=9)
    # Expected: 5.0 * 0.90 + 10.0 * 0.10 = 5.5
    expected = torch.full_like(trainer.teacher.enc1.conv[0].weight, 5.5)
    assert torch.allclose(trainer.teacher.enc1.conv[0].weight, expected)


def test_confidence_gated_background_suppression_and_cutmix_slic():
    student = DermLIPSegmentationUNet(in_channels=3, num_classes=1, base_channels=8)
    trainer = TeacherStudentTrainer(
        student=student,
        device="cpu",
        ema_decay=0.95,
        gamma_semantic=0.5,
        superpixel_affinity_weight=0.1,
    )

    # Teacher tries to over-segment peripheral skin (logit +1.1 -> prob ~0.75),
    # while peripheral-ring calibrated semantic prior strongly indicates background (prob 0.15 < 0.35).
    student_logits = torch.zeros(2, 1, 32, 32, requires_grad=True)
    teacher_logits = torch.full((2, 1, 32, 32), 1.1)
    sem_probs = torch.full((2, 1, 32, 32), 0.15)

    loss_u, mean_alpha, pseudo_targets = trainer.unsupervised_loss(
        student_logits, teacher_logits, sem_probs
    )
    # Background prior must suppress false-positive teacher prediction below 0.5
    assert float(pseudo_targets.max().item()) == 0.0
    assert torch.isfinite(loss_u)
    assert 0.0 < float(mean_alpha.item()) <= 1.0

    # Verify CutMix + SLIC affinity step with B=2
    labeled_batch = (torch.randn(2, 3, 32, 32), (torch.rand(2, 1, 32, 32) > 0.5).float())
    unlabeled_batch = (
        torch.randn(2, 3, 32, 32),
        torch.randn(2, 3, 32, 32),
        torch.rand(2, 1, 32, 32),
    )
    slic_labels = torch.zeros((2, 32, 32), dtype=torch.long)
    slic_labels[:, :, 16:] = 1
    res = trainer.training_step(
        labeled_batch,
        unlabeled_batch,
        lambda_u=1.0,
        unlabeled_slic_labels=slic_labels,
    )
    assert torch.isfinite(res["total_loss"])
    assert torch.isfinite(res["loss_sp"])
    res["total_loss"].backward()


def test_uncertainty_band_pseudo_label_masking():
    """Verify that ambiguous probability transitions are zero-weighted while confident regions guide training."""
    student = DermLIPSegmentationUNet(in_channels=3, num_classes=1, base_channels=8)
    trainer = TeacherStudentTrainer(
        student=student,
        device="cpu",
        gamma_semantic=0.0,
        pseudo_label_threshold_high=0.65,
        pseudo_label_threshold_low=0.35,
    )

    teacher_logits = torch.tensor([[[[-1.3863, 0.0, 1.3863]]]])
    student_logits = torch.tensor([[[[0.0, 0.0, 0.0]]]], requires_grad=True)

    loss_u, mean_alpha, pseudo_targets = trainer.unsupervised_loss(
        student_logits, teacher_logits, None
    )

    assert pseudo_targets[0, 0, 0, 0].item() == 0.0
    assert pseudo_targets[0, 0, 0, 1].item() == 0.0
    assert pseudo_targets[0, 0, 0, 2].item() == 1.0

    loss_u.backward()
    grad = student_logits.grad

    assert torch.isclose(grad[0, 0, 0, 1], torch.tensor(0.0), atol=1e-5)
    assert not torch.isclose(grad[0, 0, 0, 0], torch.tensor(0.0), atol=1e-5)
    assert not torch.isclose(grad[0, 0, 0, 2], torch.tensor(0.0), atol=1e-5)


def test_resnet_unet_and_v9_copy_paste_soft_kl_step():
    """Verify DermLIPResNetUNet, Scale-Jittered Copy-Paste, Log-Linear Fusion, and Soft KL distillation."""
    from dermssl.segmentation_methods.dermlip_slic.trainer import (
        DermLIPResNetUNet,
        build_segmentation_model,
    )

    student = build_segmentation_model(backbone="resnet34", pretrained=False)
    assert isinstance(student, DermLIPResNetUNet)
    param_groups = student.get_parameter_groups(base_lr=1e-4)
    assert len(param_groups) == 2
    assert param_groups[0]["lr"] == pytest.approx(5e-5)
    assert param_groups[1]["lr"] == pytest.approx(2e-4)

    trainer = TeacherStudentTrainer(
        student=student,
        device="cpu",
        ema_decay=0.95,
        gamma_semantic=0.5,
        pseudo_label_threshold_high=0.60,
        pseudo_label_threshold_low=0.40,
        soft_kl_weight=0.5,
        use_log_linear_fusion=True,
        copy_paste_prob=1.0,
        use_fp_dropout=True,
    )

    # Create a labeled batch with a clear central lesion (32x32)
    imgs_l = torch.full((2, 3, 64, 64), 0.2)
    masks_l = torch.zeros((2, 1, 64, 64))
    masks_l[:, :, 16:48, 16:48] = 1.0  # 25% area lesion

    imgs_u_weak = torch.full((2, 3, 64, 64), 0.7)
    imgs_u_strong = torch.full((2, 3, 64, 64), 0.65)
    sem_probs_u = torch.full((2, 1, 64, 64), 0.2)

    step_res = trainer.training_step(
        (imgs_l, masks_l),
        (imgs_u_weak, imgs_u_strong, sem_probs_u),
        lambda_u=1.0,
    )
    assert torch.isfinite(step_res["total_loss"])
    step_res["total_loss"].backward()

    # Verify soft_kl_weight > 0 provides non-zero gradient in transition zone (p=0.5)
    teacher_logits = torch.tensor([[[[-1.3863, 0.05, 1.3863]]]])
    student_logits = torch.tensor([[[[0.0, 0.0, 0.0]]]], requires_grad=True)
    loss_u, _, _ = trainer.unsupervised_loss(student_logits, teacher_logits, None)
    loss_u.backward()
    assert not torch.isclose(student_logits.grad[0, 0, 0, 1], torch.tensor(0.0), atol=1e-6)




