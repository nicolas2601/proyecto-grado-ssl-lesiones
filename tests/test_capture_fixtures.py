from __future__ import annotations

import base64
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "release_v10_2"))

import capture_fixtures as cf  # noqa: E402


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class FakeClient:
    """Stands in for the in-process TestClient; canned bodies are test doubles, not displayed data."""

    def __init__(self, *, ready=True, segmentation_ok=True):
        self.ready = ready
        self.segmentation_ok = segmentation_ok
        self.simulate_calls = []

    def get(self, path):
        return FakeResponse(200, {
            "classification_ready": self.ready,
            "reason": "ready" if self.ready else "not ready",
            "release": "v10.1.0",
            "identity": {"model_sha256": "c" * 64, "release": "v10.1.0"},
            "segmenter": {
                "ready": self.ready,
                "reason": "ready" if self.ready else "segmenter_error",
                "identity": {"model_id": "m", "run": "r", "release": "v10.2.0", "model_sha256": "d" * 64,
                             "input_size": 224},
            },
        })

    def post(self, path, files=None, data=None):
        if path.startswith("/api/analyze"):
            segmentation = (
                {"status": "ok", "reason": None, "mask_png_base64": base64.b64encode(b"mask").decode()}
                if self.segmentation_ok
                else {"status": "unavailable", "reason": "empty_mask", "mask_png_base64": None}
            )
            return FakeResponse(200, {"schema": "dermssl.analyze.v1", "segmentation": segmentation,
                                      "classification": {"pred": "nv"}})
        self.simulate_calls.append(data["delta_l"])
        return FakeResponse(200, {"schema": "dermssl.simulate-tone.v1", "delta_l": float(data["delta_l"])})


def test_delta_list_is_the_four_real_steps_without_identity():
    assert cf.DELTAS == (-9.73, -4.86, 4.86, 9.73)
    assert 0 not in cf.DELTAS


def test_output_names_follow_the_contract():
    assert cf.analyze_name("reference-tschandl-600") == "analyze-MOCO_HAM7-reference-tschandl-600.json"
    assert cf.simulate_name(-9.73, "reference-tschandl-600") == "simulate-tone--9.73-reference-tschandl-600.json"
    assert cf.simulate_name(4.86, "x") == "simulate-tone-4.86-x.json"


def test_check_artifacts_rejects_pin_mismatch(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "manifest.json").write_bytes(b"{}")
    (root / "model.onnx").write_bytes(b"model")

    with pytest.raises(cf.CaptureError, match="manifest"):
        cf.check_artifacts(root, "0" * 64, "seg", "0" * 64, segmenter_check=False)


def test_check_artifacts_fails_when_model_is_missing(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    manifest = b"{}"
    (root / "manifest.json").write_bytes(manifest)
    pin = hashlib.sha256(manifest).hexdigest()

    with pytest.raises(cf.CaptureError, match="model.onnx"):
        cf.check_artifacts(root, pin, None, None, segmenter_check=False)


def test_run_capture_writes_analyze_simulations_and_manifest(tmp_path):
    image = tmp_path / "reference.webp"
    image.write_bytes(b"not-really-webp-but-bytes")
    out = tmp_path / "out"
    clock = lambda: datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)  # noqa: E731
    client = FakeClient()

    cf.run_capture(client, [image], out, clock=clock, git={"commit": "abc123", "dirty": False},
                   pins={"classifier_manifest_sha256": "a" * 64, "segmenter_manifest_sha256": "b" * 64})

    names = sorted(path.name for path in out.iterdir())
    assert "analyze-MOCO_HAM7-reference.json" in names
    for delta in cf.DELTAS:
        assert cf.simulate_name(delta, "reference") in names
    assert client.simulate_calls == ["-9.73", "-4.86", "4.86", "9.73"]
    manifest = json.loads((out / "capture_manifest.json").read_text())
    assert manifest["captured_at"] == "2026-11-11T12:00:00+00:00"
    assert manifest["git_commit"] == "abc123"
    assert manifest["git_dirty"] is False
    assert manifest["images"][0]["sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert manifest["images"][0]["simulations"] == "captured"
    assert manifest["segmenter"]["identity"]["model_sha256"] == "d" * 64


def test_unavailable_segmentation_is_recorded_not_fabricated(tmp_path):
    image = tmp_path / "reference.webp"
    image.write_bytes(b"bytes")
    out = tmp_path / "out"

    cf.run_capture(FakeClient(segmentation_ok=False), [image], out,
                   clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
                   git={"commit": "abc", "dirty": True},
                   pins={"classifier_manifest_sha256": "a" * 64, "segmenter_manifest_sha256": "b" * 64})

    entry = json.loads((out / "capture_manifest.json").read_text())["images"][0]
    assert entry["simulations"] == "skipped"
    assert entry["simulation_skip_reason"] == "empty_mask"
    assert not any(path.name.startswith("simulate-tone") for path in out.iterdir())


def test_not_ready_service_fails_before_any_output(tmp_path):
    image = tmp_path / "reference.webp"
    image.write_bytes(b"bytes")
    out = tmp_path / "out"

    with pytest.raises(cf.CaptureError, match="not ready"):
        cf.run_capture(FakeClient(ready=False), [image], out, clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
                       git={"commit": "abc", "dirty": False},
                       pins={"classifier_manifest_sha256": "a" * 64, "segmenter_manifest_sha256": "b" * 64})

    assert not out.exists() or not any(out.iterdir())


def test_unsupported_image_type_fails_loudly(tmp_path):
    image = tmp_path / "reference.avif"
    image.write_bytes(b"bytes")

    with pytest.raises(cf.CaptureError, match="unsupported"):
        cf.run_capture(FakeClient(), [image], tmp_path / "out", clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
                       git={"commit": "abc", "dirty": False},
                       pins={"classifier_manifest_sha256": "a" * 64, "segmenter_manifest_sha256": "b" * 64})


def test_git_state_from_explicit_commit_records_archive_provenance_without_fabricating_dirty():
    assert cf.git_state_from_argument("0d13e8a") == {"commit": "0d13e8a", "dirty": None, "source": "archive"}


def test_git_state_rejects_malformed_commit():
    with pytest.raises(cf.CaptureError, match="commit"):
        cf.git_state_from_argument("not a sha")


def test_manifest_name_lets_independent_parallel_captures_share_an_output_dir(tmp_path):
    image = tmp_path / "one.webp"
    image.write_bytes(b"bytes")
    out = tmp_path / "out"

    path = cf.run_capture(FakeClient(), [image], out, clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
                          git={"commit": "abc", "dirty": False},
                          pins={"classifier_manifest_sha256": "a" * 64, "segmenter_manifest_sha256": "b" * 64},
                          manifest_name="capture_manifest-one.json")

    assert path.name == "capture_manifest-one.json"
    assert (out / "capture_manifest-one.json").is_file()
    assert not (out / "capture_manifest.json").exists()
