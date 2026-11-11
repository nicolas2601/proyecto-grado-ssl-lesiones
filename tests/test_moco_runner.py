from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/release_v10/runner.py"
spec = importlib.util.spec_from_file_location("moco_v10_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_manifest_verification_rejects_corrupt_files_and_traversal(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"test-model")
    digest = hashlib.sha256(b"test-model").hexdigest()
    document = {"schema": "dermssl.moco-bundle.v1", "release": "v10.0.0",
                "files": {"model.onnx": digest}, "tasks": runner.TASKS}
    (tmp_path / "manifest.json").write_text(json.dumps(document))
    assert runner.verify_bundle(tmp_path)["release"] == "v10.0.0"
    (tmp_path / "model.onnx").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        runner.verify_bundle(tmp_path)
    document["files"] = {"../outside": digest}
    (tmp_path / "manifest.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="path"):
        runner.verify_bundle(tmp_path)


def test_optional_alternate_format_is_not_required_for_the_selected_runtime(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"onnx")
    document = {"schema": "dermssl.moco-bundle.v1", "release": "v10.0.0", "tasks": runner.TASKS,
                "files": {"model.onnx": hashlib.sha256(b"onnx").hexdigest(),
                          "classifier.pt": hashlib.sha256(b"torch").hexdigest()}}
    (tmp_path / "manifest.json").write_text(json.dumps(document))
    assert runner.verify_bundle(tmp_path, runtime="onnx")["release"] == "v10.0.0"
    with pytest.raises(ValueError, match="digest"):
        runner.verify_bundle(tmp_path, runtime="torch")


def test_manifest_rejects_wrong_class_order(tmp_path):
    document = {"schema": "dermssl.moco-bundle.v1", "files": {}, "tasks": {"HAM": {"classes": []}}}
    (tmp_path / "manifest.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="task|class"):
        runner.verify_bundle(tmp_path)


def test_input_size_comes_from_the_manifest_and_defaults_to_224():
    assert runner.input_size({"input": {"shape": ["batch", 3, 448, 448]}}) == 448
    assert runner.input_size({}) == 224
    with pytest.raises(ValueError, match="input shape"):
        runner.input_size({"input": {"shape": ["batch", 3, 448, 224]}})
