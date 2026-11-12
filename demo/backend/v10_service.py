from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .model_service import ModelServiceError
from .runtime_threads import intra_op_threads
from .v10_artifacts import (
    OUTPUT_NAMES,
    TASK_CLASSES,
    V10ArtifactError as MoCoArtifactError,
    load_preprocessor,
    read_manifest,
    spec_for_manifest,
    verify_bundle_file,
    verify_bundle_files,
)


class MoCoServiceError(ModelServiceError):
    """Base error for the verified V10 MoCo runtime."""


class MoCoUnsupportedModeError(MoCoServiceError):
    """A requested inference mode is not part of the MoCo V10 recipe."""


def _create_onnx_session(model_path: Path) -> Any:
    try:
        import onnxruntime
    except ImportError as exc:
        raise MoCoServiceError("onnxruntime is required for V10 inference") from exc
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = intra_op_threads()
    options.inter_op_num_threads = 1
    try:
        return onnxruntime.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:  # noqa: BLE001 - ONNX runtime boundary
        raise MoCoServiceError("V10 ONNX session could not be loaded") from exc


class MoCoModelService:
    def __init__(
        self,
        bundle_dir: str | Path,
        expected_manifest_sha256: str | None,
        session_factory: Callable[[Path], Any] | None = None,
        preprocess_fn: Callable[[Any], np.ndarray] | None = None,
    ):
        if expected_manifest_sha256 is None:
            raise MoCoArtifactError("manifest SHA256 pin is required")
        self.bundle_dir = Path(bundle_dir)
        self.expected_manifest_sha256 = expected_manifest_sha256
        self._session_factory = session_factory or _create_onnx_session
        self._preprocess_fn = preprocess_fn
        self._lock = threading.RLock()
        self._session: Any | None = None
        self._input_name: str | None = None
        self._manifest_fingerprint: str | None = None
        self._identity: dict[str, Any] | None = None

    def _manifest(self) -> dict[str, Any]:
        return read_manifest(self.bundle_dir, self.expected_manifest_sha256)

    def _verify_bundle_files(self, manifest: dict[str, Any]) -> dict[str, Path]:
        return verify_bundle_files(self.bundle_dir, manifest)

    def verify_artifact(self, relative_path: str) -> str:
        manifest = self._manifest()
        verify_bundle_file(self.bundle_dir, manifest, relative_path)
        return manifest["files"][relative_path]

    @staticmethod
    def _validate_probabilities(values: Any, classes: list[str]) -> np.ndarray:
        probabilities = np.asarray(values, dtype=np.float32)
        if probabilities.shape != (1, len(classes)):
            raise MoCoServiceError("V10 classifier returned an invalid probability shape")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise MoCoServiceError("V10 classifier returned invalid probabilities")
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-5):
            raise MoCoServiceError("V10 classifier probabilities do not sum to one")
        return probabilities

    def _ensure_ready(self) -> tuple[dict[str, Any], Any]:
        manifest = self._manifest()
        fingerprint = manifest["_manifest_sha256"]
        if self._session is not None and self._manifest_fingerprint == fingerprint:
            return manifest, self._session
        self.unload()
        spec = spec_for_manifest(manifest)
        paths = self._verify_bundle_files(manifest)
        if self._preprocess_fn is None:
            self._preprocess_fn = load_preprocessor(paths["preprocess.py"], spec.input_size)
        warmup_input = self._preprocess(Image.new("RGB", (spec.input_size, spec.input_size)), spec.input_size)
        session = self._session_factory(paths["model.onnx"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        expected_names = ["ham7_probabilities", "pad6_probabilities"]
        if len(inputs) != 1 or [item.name for item in outputs] != expected_names:
            raise MoCoArtifactError("V10 ONNX inputs or named outputs do not match manifest")
        try:
            warmup = session.run(expected_names, {inputs[0].name: warmup_input})
        except Exception as exc:  # noqa: BLE001 - ONNX warmup boundary
            raise MoCoServiceError("V10 ONNX warmup failed") from exc
        if not isinstance(warmup, (list, tuple)) or len(warmup) != 2:
            raise MoCoServiceError("V10 ONNX warmup did not return both classifier heads")
        self._validate_probabilities(warmup[0], TASK_CLASSES["MOCO_HAM7"])
        self._validate_probabilities(warmup[1], TASK_CLASSES["MOCO_PAD6"])
        self._session = session
        self._input_name = inputs[0].name
        self._manifest_fingerprint = fingerprint
        self._identity = {
            "model_id": "dermssl-v10-moco",
            "release": manifest["release"],
            "method": "moco_v3_style",
            "conversion": "V10 shared-backbone ONNX",
            "backbone": "DINOv2 ViT-B/14",
            "backbone_sha256": manifest["backbone_sha256"],
            "model_sha256": manifest["files"]["model.onnx"],
            "recipe": manifest["recipe"],
        }
        return manifest, session

    def _preprocess(self, image: Any, size: int) -> np.ndarray:
        preprocess = self._preprocess_fn
        if preprocess is None:
            raise MoCoServiceError("hash-verified V10 preprocessing is unavailable")
        try:
            batch = np.asarray(preprocess(image), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - image preprocessing boundary
            raise MoCoServiceError("canonical V10 preprocessing failed") from exc
        if batch.shape != (1, 3, size, size) or not np.isfinite(batch).all():
            raise MoCoServiceError("canonical V10 preprocessing returned invalid input")
        return np.ascontiguousarray(batch)

    def predict(self, image: Any, model_key: str = "MOCO_HAM7", mode: str = "base") -> dict[str, Any]:
        if model_key not in TASK_CLASSES:
            raise MoCoArtifactError("unsupported V10 model key")
        if mode == "tta":
            raise MoCoUnsupportedModeError("TTA is not supported for MoCo V10")
        if mode != "base":
            raise MoCoUnsupportedModeError("unsupported MoCo V10 inference mode")
        task_name = "HAM" if model_key == "MOCO_HAM7" else "PAD"
        classes = TASK_CLASSES[model_key]
        with self._lock:
            manifest, session = self._ensure_ready()
            spec = spec_for_manifest(manifest)
            batch = self._preprocess(image, spec.input_size)
            try:
                outputs = session.run(list(OUTPUT_NAMES), {self._input_name: batch})
            except Exception as exc:  # noqa: BLE001 - ONNX inference boundary
                raise MoCoServiceError("V10 ONNX inference failed") from exc
            if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
                raise MoCoServiceError("V10 ONNX inference did not return both classifier heads")
            ham = self._validate_probabilities(outputs[0], TASK_CLASSES["MOCO_HAM7"])
            pad = self._validate_probabilities(outputs[1], TASK_CLASSES["MOCO_PAD6"])
            probabilities = ham if task_name == "HAM" else pad
            scores = probabilities[0]
            index = int(np.argmax(scores))
            head_sha = manifest["files"][f"heads/{task_name}-reader.{spec.head_ext}"]
            task_sha = manifest["files"][f"heads/{task_name}-reader.json"]
            sha256 = {
                "model": manifest["files"]["model.onnx"],
                "head": head_sha,
                "task": task_sha,
                "backbone": manifest["backbone_sha256"],
            }
            identity = {
                **(self._identity or {}),
                "task": task_name,
                "model_key": model_key,
                "head_sha256": head_sha,
                "task_sha256": task_sha,
                "backbone_sha256": manifest["backbone_sha256"],
                "sha256": sha256,
            }
            return {
                "pred": classes[index],
                "confidence": round(float(scores[index]), 6),
                "confidence_calibrated": False,
                "confidence_note": "Uncalibrated model score; not a clinical probability.",
                "classes": list(classes),
                "probs": {label: round(float(score), 6) for label, score in zip(classes, scores)},
                "feature_dim": 3072,
                "mode": "base",
                "requested_mode": mode,
                "inference_mode": spec.inference_mode,
                "views": spec.views,
                "model_key": model_key,
                "identity": identity,
            }

    def status(self) -> dict[str, Any]:
        with self._lock:
            try:
                manifest, _ = self._ensure_ready()
            except Exception as exc:  # noqa: BLE001 - readiness boundary
                return {
                    "backbone_ready": False,
                    "classification_ready": False,
                    "reason": str(exc) or "V10 model is not ready",
                    "release": "v10.0.0",
                    "release_arm": None,
                    "model_key": None,
                }
            return {
                "backbone_ready": True,
                "classification_ready": True,
                "reason": "ready",
                "release": manifest["release"],
                "release_arm": None,
                "device": "cpu",
                "threads": intra_op_threads(),
                "model_key": "MOCO_HAM7",
                "identity": self._identity,
            }

    def unload(self) -> None:
        with self._lock:
            self._session = None
            self._input_name = None
            self._manifest_fingerprint = None
            self._identity = None
