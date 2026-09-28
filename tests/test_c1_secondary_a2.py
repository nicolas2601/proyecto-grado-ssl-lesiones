"""C1-v7 secondary: the same mask grid with S4-UNet A2's masks (docs/27).

A2 met its prediction, so C1 gets a SECONDARY run whose masks come from A2
instead of student_r3. The artifact keeps the exact format (``c1_predict_masks``
already runs any checkpoint ``eval_student.load_student`` accepts); what has to
change is the identity: the producer is derived from the checkpoint's sha, as
for student_r3, and the A2 sha fails closed until it is measured on the pod.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from dermssl import c1_mask_export as mx
from dermssl import c1_mask_provenance as mp

REPO = Path(__file__).resolve().parents[1]
_A2_SHA = "2" * 64
_A2_COMMIT = "a2c0mm1"
_FINAL_SHA = "f" * 64
_INFERENCE = {
    "path": "plan-segmentacion/comun/evaluacion/eval_student.predict_masks",
    "tta": False,
    "resolution": 224,
    "threshold": 0.5,
    "upsample": "bilinear_prob_then_threshold",
}


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setattr(mp, "FINAL_STUDENT_R3_SHA256", _FINAL_SHA)
    monkeypatch.setattr(mp, "S4_UNET_A2_SHA256", _A2_SHA)
    monkeypatch.setattr(mp, "S4_UNET_A2_COMMIT", _A2_COMMIT)


def _built(**over):
    payload = {
        "n_images": 1, "image_ids": ["ISIC_0000001"], "masks_sha256": "a" * 64,
        "checkpoint_sha256": _A2_SHA, "checkpoint_path": "s4_unet/a2/best.pt",
        "code_revision": "abc1234", "mask_shape": [450, 600],
        "segmentation_run": mp.S4_UNET_A2_RUN, "segmentation_commit": _A2_COMMIT,
        "inference": dict(_INFERENCE),
    }
    payload.update(over)
    return mx.build_masks_manifest(**payload)


def test_the_producer_is_derived_from_the_run_not_passed_in():
    assert _built()["producer"] == "s4_unet_a2"
    assert _built(segmentation_run="final", checkpoint_sha256=_FINAL_SHA)["producer"] == (
        "student_r3")


def test_the_a2_manifest_says_it_is_label_free_and_secondary():
    manifest = _built()
    assert manifest["producer_is_label_free"] is True
    assert "SECONDARY" in manifest["segmentation_run_note"]
    assert "DINOv2" in manifest["producer_note"]


def test_an_unknown_run_cannot_be_written():
    with pytest.raises(ValueError, match="segmentation_run"):
        _built(segmentation_run="v2")


def test_every_run_producer_is_an_allowed_producer():
    assert set(mx.RUN_PRODUCERS.values()) == set(mp.ALLOWED_MASK_PRODUCERS)


def test_a2_fails_closed_while_its_sha_is_unmeasured(monkeypatch):
    monkeypatch.setattr(mp, "S4_UNET_A2_SHA256", None)
    assert all(run != mp.S4_UNET_A2_RUN for run, _ in mp.checkpoint_registry().values())
    with pytest.raises(ValueError, match="registrada|registrado"):
        mp.resolve_run(_A2_SHA)


def test_a_sha_without_its_commit_is_not_registered(monkeypatch):
    monkeypatch.setattr(mp, "S4_UNET_A2_SHA256", _A2_SHA)
    monkeypatch.setattr(mp, "S4_UNET_A2_COMMIT", None)
    assert _A2_SHA not in mp.checkpoint_registry()


def test_the_student_registration_is_untouched_by_a2(registered):
    assert mp.resolve_run(_FINAL_SHA) == mp.PUBLISHED_SEGMENTATION
    assert mp.resolve_run(_A2_SHA) == (mp.S4_UNET_A2_RUN, _A2_COMMIT)


def test_a_registered_a2_artifact_is_accepted_by_the_consumer(registered):
    source = mp.MaskSource.from_manifest(_built())
    assert source.producer == "s4_unet_a2"
    assert source.segmentation_run == mp.S4_UNET_A2_RUN


def test_a2_masks_cannot_be_labelled_student_r3(registered):
    manifest = _built()
    manifest["producer"] = "student_r3"
    with pytest.raises(ValueError, match="producer"):
        mp.MaskSource.from_manifest(manifest)


def test_student_masks_cannot_be_labelled_a2(registered):
    manifest = _built(segmentation_run="final", segmentation_commit="d81a492",
                      checkpoint_sha256=_FINAL_SHA)
    manifest["producer"] = "s4_unet_a2"
    with pytest.raises(ValueError, match="producer"):
        mp.MaskSource.from_manifest(manifest)


def test_producer_for_run_names_both_segmenters():
    assert mp.producer_for_run("final") == "student_r3"
    assert mp.producer_for_run(mp.S4_UNET_A2_RUN) == "s4_unet_a2"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location(
        "c1_predict_masks", REPO / "scripts" / "c1_predict_masks.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_step0_derives_the_a2_identity_from_the_checkpoint(script, registered):
    assert script.step0_identity(_A2_SHA, _A2_SHA) == (
        "s4_unet_a2", mp.S4_UNET_A2_RUN, _A2_COMMIT)
    assert script.step0_identity(_FINAL_SHA, _FINAL_SHA) == (
        "student_r3", "final", "d81a492")


def test_step0_refuses_a_checkpoint_other_than_the_expected_one(script, registered):
    with pytest.raises(SystemExit, match="mismatch"):
        script.step0_identity(_A2_SHA, _FINAL_SHA)
