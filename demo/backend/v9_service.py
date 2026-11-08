from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .model_service import ModelServiceError, ProbeArtifactError


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
RELEASE = "v9.0.0"
THREADS = 2
MODEL_HASHES = {
    "A71": "fd999e669ef2f1a9aa6a19ecd6293bd871df3e9c4a7e951adfd50d77a48f89bc",
    "B71": "7fd581afc14a5c61da37f8d3873ce78c228276b0186a98e39096e9130285bbd6",
}
RUNTIME_HASHES = {
    "runner.py": "2e57b7e0e3fa45da46aa9190240206603ff91c89d298615243ee069407e02387",
    "preprocess.py": "aa9debec464c82dd546ec765ab7b84e34594d804491de383b084651842d00da1",
}


class V9ServiceError(ModelServiceError):
    """The V9 bundle is missing, invalid, or failed during inference."""


def _load_onnxruntime():
    try:
        import onnxruntime
    except ImportError as exc:
        raise V9ServiceError("onnxruntime is required for V9 inference") from exc
    return onnxruntime


class V9ModelService:
    """Verified V9 ONNX inference with a resident CPU session."""

    def __init__(self, bundle_dir: Path, expected_arm: str | None = None):
        self.bundle_dir = Path(bundle_dir)
        self.expected_arm = expected_arm
        self._lock = threading.RLock()
        self._session: Any | None = None
        self._input_name: str | None = None
        self._preprocess_module: Any | None = None
        self._manifest_fingerprint: str | None = None
        self._identity: dict[str, Any] | None = None

    def _manifest(self) -> dict[str, Any]:
        try:
            manifest_bytes = (self.bundle_dir / "manifest.json").read_bytes()
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise V9ServiceError("V9 manifest is missing or invalid") from exc
        if not isinstance(manifest, dict):
            raise V9ServiceError("V9 manifest must be an object")
        recipe = manifest.get("recipe")
        if (manifest.get("schema") != "dermssl.clean-bundle.v1" or manifest.get("arm") not in ("A71", "B71")
                or manifest.get("verified") is not True or manifest.get("classes") != CLASSES
                or not isinstance(recipe, dict) or recipe.get("tta") != "D4"
                or recipe.get("aggregation") != "mean_probabilities" or recipe.get("n_views") != 8
                or recipe.get("feature") != ("cls_attn" if manifest.get("arm") == "A71" else "cls4")
                or recipe.get("tau") != (0.5 if manifest.get("arm") == "A71" else 0.0)):
            raise V9ServiceError("V9 manifest does not declare the full D4 recipe")
        if self.expected_arm is not None and manifest["arm"] != self.expected_arm:
            raise V9ServiceError("V9 manifest arm does not match the requested model")
        if not isinstance(manifest.get("files"), dict):
            raise V9ServiceError("V9 manifest is missing file hashes")
        manifest["_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        return manifest

    def _checked_model(self, manifest: dict[str, Any]) -> Path:
        path = self.bundle_dir / "model.onnx"
        expected = manifest.get("files", {}).get("model.onnx")
        if expected is None:
            raise V9ServiceError("V9 manifest has no model.onnx SHA256")
        if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise V9ServiceError("V9 manifest has invalid model.onnx SHA256")
        if expected != MODEL_HASHES.get(manifest.get("arm")):
            raise V9ServiceError("V9 model.onnx SHA256 mismatch with published release")
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise V9ServiceError("V9 ONNX model is missing") from exc
        if digest.hexdigest() != expected:
            raise V9ServiceError("V9 model.onnx SHA256 mismatch")
        return path

    def _verify_bundle_files(self, manifest: dict[str, Any]) -> None:
        self._checked_model(manifest)
        for name in ("runner.py", "preprocess.py"):
            expected = manifest.get("files", {}).get(name)
            if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
                raise V9ServiceError(f"V9 manifest has no valid SHA256 for {name}")
            if expected != RUNTIME_HASHES[name]:
                raise V9ServiceError(f"V9 published SHA256 mismatch for {name}")
            path = self.bundle_dir / name
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise V9ServiceError(f"V9 bundle file is missing: {name}") from exc
            if digest != expected:
                raise V9ServiceError(f"V9 SHA256 mismatch for {name}")

    def _load_session(self, manifest: dict[str, Any]) -> Any:
        fingerprint = manifest["_manifest_sha256"]
        if self._session is not None and self._manifest_fingerprint == fingerprint:
            return self._session
        if self._manifest_fingerprint != fingerprint:
            self.unload()
        path = self.bundle_dir / "model.onnx"
        ort = _load_onnxruntime()
        options = ort.SessionOptions()
        options.intra_op_num_threads = THREADS
        options.inter_op_num_threads = 1
        try:
            session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
            inputs = session.get_inputs()
            if len(inputs) != 1:
                raise V9ServiceError("V9 ONNX model must expose exactly one input")
            self._input_name = inputs[0].name
        except V9ServiceError:
            raise
        except Exception as exc:  # noqa: BLE001 - model runtime boundary
            raise V9ServiceError("V9 ONNX session could not be loaded") from exc
        self._session = session
        self._manifest_fingerprint = fingerprint
        return session

    def _preprocess(self, image: Image.Image, bundle_manifest: dict[str, Any]) -> np.ndarray:
        if self._preprocess_module is None or self._manifest_fingerprint != bundle_manifest["_manifest_sha256"]:
            preprocess_path = self.bundle_dir / "preprocess.py"
            spec = importlib.util.spec_from_file_location("dermssl_v9_preprocess", preprocess_path)
            if spec is None or spec.loader is None:
                raise V9ServiceError("V9 canonical preprocess implementation could not be loaded")
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as exc:  # noqa: BLE001 - verified preprocessing boundary
                raise V9ServiceError("V9 canonical preprocess implementation failed to load") from exc
            self._preprocess_module = module
        try:
            batch = np.ascontiguousarray(self._preprocess_module.preprocess_image(image)[None, ...], dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - image preprocessing boundary
            raise V9ServiceError("V9 canonical preprocessing failed") from exc
        if batch.shape != (1, 3, 224, 224) or not np.isfinite(batch).all():
            raise V9ServiceError("V9 canonical preprocessing returned invalid input")
        return batch

    def status(self) -> dict[str, Any]:
        with self._lock:
            try:
                manifest = self._manifest()
                self._verify_bundle_files(manifest)
                model_path = self.bundle_dir / "model.onnx"
            except V9ServiceError as exc:
                return {"backbone_ready": False, "classification_ready": False, "reason": str(exc),
                        "release": RELEASE, "release_arm": None}
            identity = {"model_id": f"dermssl-v9-{manifest['arm']}", "release": RELEASE,
                        "arm": manifest["arm"], "model_sha256": manifest["files"]["model.onnx"],
                        "sha256_verified": True, "recipe": manifest["recipe"]}
            self._identity = identity
            return {"backbone_ready": True, "classification_ready": True, "reason": "ready",
                    "release": RELEASE, "release_arm": manifest["arm"], "device": "cpu",
                    "threads": THREADS, "bundle_path": str(self.bundle_dir),
                    "model_path": str(model_path), "identity": identity,
                    "recipe": manifest["recipe"]}

    def predict(self, image: Image.Image, mode: str = "base") -> dict[str, Any]:
        if mode not in ("base", "tta"):
            raise ProbeArtifactError(f"unsupported inference mode: {mode}")
        with self._lock:
            manifest = self._manifest()
            self._verify_bundle_files(manifest)
            session = self._load_session(manifest)
            batch = self._preprocess(image, manifest)
            views = [np.ascontiguousarray(np.rot90(np.flip(batch, -1) if flip else batch, k, (-2, -1)))
                     for flip in (False, True) for k in range(4)]
            try:
                probabilities = np.stack([session.run(None, {self._input_name: view})[0] for view in views]).mean(0)
            except Exception as exc:  # noqa: BLE001 - model runtime boundary
                raise V9ServiceError("V9 ONNX inference failed") from exc
            if probabilities.shape != (1, len(CLASSES)) or not np.isfinite(probabilities).all() or (probabilities < 0).any():
                raise V9ServiceError("V9 classifier returned invalid probabilities")
            if not np.allclose(probabilities.sum(1), 1.0, rtol=0, atol=1e-5):
                raise V9ServiceError("V9 classifier probabilities do not sum to one")
            probs = probabilities[0]
            index = int(np.argmax(probs))
            return {"pred": CLASSES[index], "confidence": round(float(probs[index]), 6),
                    "classes": CLASSES, "probs": {name: round(float(prob), 6) for name, prob in zip(CLASSES, probs)},
                    "feature_dim": 1536 if manifest["arm"] == "A71" else 3072,
                    "mode": mode, "requested_mode": mode, "inference_mode": "D4", "views": 8}

    def extract_features(self, image: Image.Image) -> dict[str, Any]:
        del image
        raise V9ServiceError("V9 exposes classifier probabilities, not CLS feature vectors")

    def unload(self) -> None:
        with self._lock:
            self._session = None
            self._input_name = None
            self._preprocess_module = None
            self._manifest_fingerprint = None
