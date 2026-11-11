"""Optional lesion segmenter (ONNX) for /api/analyze. It never alters the classifier input.

Artifacts: ``segmenter_manifest.json`` (schema dermssl.segmenter.v1, pinned by SHA256 from the
environment) and ``segmenter.onnx`` (hash listed in the manifest). Preprocessing is read from the
manifest ``input`` block; the output is ``logits`` (1, 1, S, S) thresholded at sigmoid > 0.5.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .segmenter_geometry import (
    encode_mask_png,
    largest_component,
    outer_contour,
    simplify_contour,
)
from .tone_sim import MAX_AREA_FRACTION, MIN_AREA_FRACTION

SCHEMA = "dermssl.segmenter.v1"
MANIFEST_NAME = "segmenter_manifest.json"
MODEL_NAME = "segmenter.onnx"
THRESHOLD = 0.5
THREADS = 2
MAX_MASK_SIDE = 512
SHA256_PATTERN = r"[a-f0-9]{64}"
LOGIT_CLIP = 60.0
PROBABILITY_DECIMALS = 4


class SegmenterError(RuntimeError):
    """A segmentation did not produce a usable mask; ``reason`` is the public reason code."""

    def __init__(self, reason: str, message: str | None = None):
        super().__init__(message or reason)
        self.reason = reason


@dataclass(frozen=True)
class InputConfig:
    size: int
    scale: float
    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class Loaded:
    session: Any
    input_name: str
    config: InputConfig
    identity: dict[str, Any]


@dataclass(frozen=True)
class Segmentation:
    payload: dict[str, Any]
    full_mask: np.ndarray


def _create_session(model_path: Path) -> Any:
    try:
        import onnxruntime
    except ImportError as exc:
        raise SegmenterError("segmenter_error", "onnxruntime is required for the segmenter") from exc
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1
    try:
        return onnxruntime.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001 - ONNX runtime boundary
        raise SegmenterError("segmenter_error", "segmenter ONNX session could not be loaded") from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest(directory: Path, expected_sha256: str | None) -> dict[str, Any]:
    if expected_sha256 is None or not re.fullmatch(SHA256_PATTERN, expected_sha256):
        raise SegmenterError("segmenter_error", "segmenter manifest SHA256 pin is missing or invalid")
    try:
        content = (directory / MANIFEST_NAME).read_bytes()
    except OSError as exc:
        raise SegmenterError("segmenter_error", "segmenter manifest is missing") from exc
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise SegmenterError("segmenter_error", "segmenter manifest SHA256 mismatch")
    try:
        manifest = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SegmenterError("segmenter_error", "segmenter manifest is invalid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise SegmenterError("segmenter_error", "unsupported segmenter manifest schema")
    return manifest


def _input_config(manifest: dict[str, Any]) -> InputConfig:
    block = manifest.get("input")
    shape = block.get("shape") if isinstance(block, dict) else None
    if (not isinstance(block, dict) or block.get("name") != "pixel_values" or block.get("dtype") != "float32"
            or block.get("color") != "RGB" or not isinstance(shape, list) or len(shape) != 4):
        raise SegmenterError("segmenter_error", "segmenter input block does not match the contract")
    size = shape[2]
    if shape[1] != 3 or shape[2] != shape[3] or not isinstance(size, int) or size <= 0:
        raise SegmenterError("segmenter_error", "segmenter input shape must be (batch, 3, S, S)")
    scale, mean, std = block.get("scale"), block.get("mean"), block.get("std")
    if not isinstance(scale, (int, float)) or scale <= 0:
        raise SegmenterError("segmenter_error", "segmenter input scale is invalid")
    if not (isinstance(mean, list) and isinstance(std, list) and len(mean) == 3 and len(std) == 3
            and all(value > 0 for value in std)):
        raise SegmenterError("segmenter_error", "segmenter mean/std are invalid")
    return InputConfig(size=size, scale=float(scale), mean=np.asarray(mean, np.float32),
                       std=np.asarray(std, np.float32))


def preprocess(rgb: np.ndarray, config: InputConfig) -> np.ndarray:
    """Single configurable preprocessing path: RGB, bilinear resize to S, scale, normalise, NCHW."""
    resized = Image.fromarray(rgb, mode="RGB").resize((config.size, config.size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / np.float32(config.scale)
    pixels = (pixels - config.mean) / config.std
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype=np.float32)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -LOGIT_CLIP, LOGIT_CLIP)))


def _raster_size(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, MAX_MASK_SIDE / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def _contour_to_image(points: list[tuple[int, int]], raster_wh, image_wh) -> list[list[int]]:
    (mw, mh), (w, h) = raster_wh, image_wh
    out = []
    for x, y in points:
        px = min(w - 1, int(round((x + 0.5) * w / mw - 0.5)))
        py = min(h - 1, int(round((y + 0.5) * h / mh - 0.5)))
        out.append([max(0, px), max(0, py)])
    return out


class SegmenterService:
    def __init__(
        self,
        segmenter_dir: str | Path,
        expected_manifest_sha256: str | None,
        session_factory: Callable[[Path], Any] | None = None,
    ):
        self._dir = Path(segmenter_dir)
        self._pin = expected_manifest_sha256
        self._factory = session_factory or _create_session
        self._lock = threading.RLock()
        self._loaded: Loaded | None = None
        self._failure: SegmenterError | None = None
        self._attempted = False

    def _ensure_loaded(self) -> Loaded:
        with self._lock:
            if not self._attempted:
                self._attempted = True
                try:
                    self._loaded = self._load()
                except SegmenterError as exc:
                    self._failure = exc
        if self._loaded is None:
            raise self._failure or SegmenterError("segmenter_error", "segmenter is not configured")
        return self._loaded

    def _load(self) -> Loaded:
        manifest = _read_manifest(self._dir, self._pin)
        config = _input_config(manifest)
        files = manifest.get("files")
        model_sha = files.get(MODEL_NAME) if isinstance(files, dict) else None
        if not isinstance(model_sha, str) or not re.fullmatch(SHA256_PATTERN, model_sha):
            raise SegmenterError("segmenter_error", "segmenter manifest has no model SHA256")
        model_path = self._dir / MODEL_NAME
        if not model_path.is_file() or _sha256(model_path) != model_sha:
            raise SegmenterError("segmenter_error", "segmenter ONNX is missing or does not match the manifest")
        session = self._factory(model_path)
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "pixel_values" or [o.name for o in outputs] != ["logits"]:
            raise SegmenterError("segmenter_error", "segmenter ONNX inputs or outputs do not match the manifest")
        identity = {
            "model_id": manifest.get("model_id"),
            "run": manifest.get("run"),
            "release": manifest.get("release"),
            "model_sha256": model_sha,
            "supervision": manifest.get("supervision"),
            "input_size": config.size,
        }
        return Loaded(session=session, input_name=inputs[0].name, config=config, identity=identity)

    def status(self) -> dict[str, Any]:
        try:
            loaded = self._ensure_loaded()
        except SegmenterError as exc:
            return {"ready": False, "reason": exc.reason, "detail": str(exc), "identity": None}
        return {"ready": True, "reason": "ready", "detail": None, "identity": loaded.identity}

    def segment(self, rgb: np.ndarray) -> Segmentation:
        loaded = self._ensure_loaded()
        logits = self._run(loaded, rgb)
        probabilities = _sigmoid(logits)
        kept = largest_component(probabilities > THRESHOLD)
        if not kept.any():
            raise SegmenterError("empty_mask")
        height, width = rgb.shape[:2]
        full = np.asarray(
            Image.fromarray(np.where(kept, probabilities, 0.0).astype(np.float32), mode="F")
            .resize((width, height), Image.Resampling.BILINEAR)
        )
        full_mask = full > THRESHOLD
        self._check_area(full_mask)
        return Segmentation(payload=self._payload(full_mask, width, height, loaded), full_mask=full_mask)

    def _run(self, loaded: Loaded, rgb: np.ndarray) -> np.ndarray:
        size = loaded.config.size
        try:
            outputs = loaded.session.run(["logits"], {loaded.input_name: preprocess(rgb, loaded.config)})
            logits = np.asarray(outputs[0], dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - ONNX inference boundary
            raise SegmenterError("segmenter_error", "segmenter inference failed") from exc
        if logits.shape != (1, 1, size, size) or not np.isfinite(logits).all():
            raise SegmenterError("segmenter_error", "segmenter returned invalid logits")
        return logits[0, 0]

    @staticmethod
    def _check_area(mask: np.ndarray) -> None:
        if not mask.any():
            raise SegmenterError("empty_mask")
        area = float(mask.mean())
        if not MIN_AREA_FRACTION <= area <= MAX_AREA_FRACTION:
            raise SegmenterError("degenerate_mask")

    def _payload(self, full_mask: np.ndarray, width: int, height: int, loaded: Loaded) -> dict[str, Any]:
        raster_wh = _raster_size(width, height)
        raster = np.asarray(
            Image.fromarray(full_mask.astype(np.uint8) * 255, mode="L").resize(raster_wh, Image.Resampling.NEAREST)
        ) >= 128
        contour = simplify_contour(outer_contour(largest_component(raster)))
        return {
            "image_width": width,
            "image_height": height,
            "mask_width": raster_wh[0],
            "mask_height": raster_wh[1],
            "mask_png_base64": base64.b64encode(encode_mask_png(raster)).decode("ascii"),
            "contour": _contour_to_image(contour, raster_wh, (width, height)),
            "area_fraction": round(float(raster.mean()), PROBABILITY_DECIMALS),
            "threshold": THRESHOLD,
            "identity": dict(loaded.identity),
        }
