import hashlib
import json

from PIL import Image

from demo.backend.model_service import ModelService
from demo.backend.segmented_model_service import SegmentedModelService
from demo.backend.f4_runtime import B71_SHA256, SEGMENTER_SHA256


def _service(tmp_path):
    return SegmentedModelService(tmp_path / "B.pt", tmp_path / "B.json", tmp_path / "B.npz",
                                 f4_dir=tmp_path, scientific_root=tmp_path / "source")


def test_missing_or_partial_segmentation_cannot_be_advertised(tmp_path):
    service = _service(tmp_path)
    assert service.status()["segmentation_ready"] is False
    (tmp_path / "F4_classifier_manifest.json").write_text('{}')
    assert service.approved_f4_manifest() is None


def test_serving_requires_bound_twenty_case_golden_receipt(tmp_path):
    service = _service(tmp_path)
    probe = tmp_path / "F4_probe.npz"
    probe.write_bytes(b"synthetic probe")
    (tmp_path / "student_r3_final_best.pt").write_bytes(b"synthetic checkpoint; actual load checks hash")
    from demo.backend.f4_runtime import EXPECTED_RECIPE
    manifest = {**EXPECTED_RECIPE, "probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest()}
    path = tmp_path / "F4_classifier_manifest.json"
    path.write_text(json.dumps(manifest))
    receipt = {"passed": True, "n_cases": 20, "max_abs_probability_difference": 0.001,
               "probe_sha256": manifest["probe_sha256"], "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "backbone_sha256": B71_SHA256, "segmenter_sha256": SEGMENTER_SHA256}
    verification = tmp_path / "F4_golden_verification.json"
    verification.write_text(json.dumps(receipt))
    assert service.approved_f4_manifest() == manifest
    receipt["n_cases"] = 19
    verification.write_text(json.dumps(receipt))
    assert service.approved_f4_manifest() is None
    receipt["n_cases"] = 20
    receipt["probe_sha256"] = "0" * 64
    verification.write_text(json.dumps(receipt))
    assert service.approved_f4_manifest() is None


def test_full_analysis_and_preview_have_explicit_different_identity(tmp_path, monkeypatch):
    service = _service(tmp_path)
    monkeypatch.setattr(service, "approved_f4_manifest", lambda: {"probe_sha256": "a" * 64})
    monkeypatch.setattr(service, "_load_f4", lambda: type("F4", (), {"predict": lambda self, image: {"inference_mode": "F4clean"}})())
    monkeypatch.setattr(ModelService, "status", lambda self: {"classification_ready": True, "identity": {"model_id": "B71"}})
    monkeypatch.setattr(ModelService, "predict", lambda self, image, mode: {"inference_mode": mode})
    image = Image.new("RGB", (4, 4))
    assert service.predict(image)["inference_mode"] == "F4clean"
    preview = service.predict(image, "preview")
    assert preview["approximate"] is True and preview["inference_mode"] == "preview_CLS_without_segmentation"
    assert preview["identity"]["model_id"] == "B71"


def test_unload_releases_backbone_and_segmenter_references(tmp_path):
    service = _service(tmp_path)
    service._backbone = object()
    service._f4 = object()
    service.unload()
    assert service._backbone is None and service._f4 is None
