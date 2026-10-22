from __future__ import annotations

import base64
import importlib
import io
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

PROTOCOL = "B_s42_a1.0@224|cls4|balanced|tta"
EXPECTED_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
EXPECTED_FEATURE_DIM = 6151
N_D4 = 8


class F4ServiceError(RuntimeError):
    """F4 runtime is unavailable or does not satisfy its inference contract."""


def _load_e3(repo_root: Path) -> tuple[Any, Any, Any, Any]:
    source_root = repo_root / "wt-worker"
    package_root = source_root / "src"
    if not (package_root / "dermssl" / "e3_render.py").is_file():
        raise F4ServiceError(f"canonical E3 modules are missing under {package_root}")
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    try:
        return tuple(importlib.import_module(f"dermssl.{name}") for name in (
            "e3_render", "e3_variants", "e3_descriptors", "torch_v71"))
    except ImportError as exc:
        raise F4ServiceError("canonical E3 runtime dependencies are unavailable") from exc


class F4Service:
    """Compose canonical E3-F4 rendering, cls4 extraction and a portable probe."""

    def __init__(
        self,
        *,
        backbone: Any,
        segment_mask: Callable[[np.ndarray], np.ndarray],
        resize_crop: Callable[[np.ndarray, int], np.ndarray],
        extract_split: Callable[..., dict],
        artifact: dict[str, Any],
        device: Any,
        batch_size: int = 1,
        hair_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        repo_root: Path | None = None,
        renderer: Callable | None = None,
        describer: Callable | None = None,
        variant_builder: Callable | None = None,
        descriptor_module: Any | None = None,
        feature_dim: int = 3072,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.backbone = backbone
        self.segment_mask = segment_mask
        self.resize_crop = resize_crop
        self.extract_split = extract_split
        self.artifact = artifact
        self.device = device
        self.batch_size = batch_size
        self.hair_fn = hair_fn
        self.feature_dim = feature_dim
        if renderer is None or describer is None or variant_builder is None or descriptor_module is None:
            if repo_root is None:
                raise F4ServiceError("repo_root is required to load canonical E3 modules")
            modules = _load_e3(Path(repo_root))
            renderer = renderer or modules[0].render
            variant_builder = variant_builder or modules[1].concat_features
            descriptor_module = descriptor_module or modules[2]
            describer = describer or modules[0].describe
        self.renderer = renderer
        self.describer = describer
        self.variant_builder = variant_builder
        self.descriptor_module = descriptor_module

    def _validate_artifact(self) -> None:
        artifact = self.artifact
        try:
            dim = int(artifact["feature_dim"])
            classes = [str(value) for value in artifact["classes"]]
            mean = np.asarray(artifact["descriptor_mean"], dtype=float)
            std = np.asarray(artifact["descriptor_std"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise F4ServiceError("F4 probe artifact is incomplete") from exc
        if artifact.get("protocol") not in (None, PROTOCOL) or dim != 2 * self.feature_dim + 7:
            raise F4ServiceError("F4 probe artifact protocol or feature dimension mismatch")
        if classes != EXPECTED_CLASSES:
            raise F4ServiceError("F4 probe artifact classes do not match canonical HAM order")
        if (mean.shape != (7,) or std.shape != (7,) or np.any(std <= 0)
                or not np.isfinite(np.concatenate((mean, std))).all()):
            raise F4ServiceError("F4 probe descriptor statistics are invalid")
        try:
            scaler_mean = np.asarray(artifact["mean"], dtype=np.float64)
            scaler_scale = np.asarray(artifact["scale"], dtype=np.float64)
            coef = np.asarray(artifact["coef"], dtype=np.float64)
            intercept = np.asarray(artifact["intercept"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise F4ServiceError("F4 probe classifier parameters are incomplete") from exc
        if scaler_mean.shape != (dim,) or scaler_scale.shape != (dim,) or coef.shape != (7, dim) or intercept.shape != (7,):
            raise F4ServiceError("F4 probe classifier parameter shapes are invalid")
        if not np.isfinite(np.concatenate((scaler_mean, scaler_scale, coef.ravel(), intercept))).all():
            raise F4ServiceError("F4 probe classifier contains non-finite parameters")
        if np.any(scaler_scale <= 0):
            raise F4ServiceError("F4 probe scaler contains an invalid scale")

    def _validate_runtime(self) -> None:
        if not callable(self.hair_fn):
            raise F4ServiceError("F4 runtime requires the registered hair detector")

    def _score_views(self, views: np.ndarray) -> np.ndarray:
        views = np.asarray(views, dtype=np.float64)
        if views.shape != (N_D4, 1, int(self.artifact["feature_dim"])) or not np.isfinite(views).all():
            raise F4ServiceError(f"F4 features must have shape (8, 1, {self.artifact['feature_dim']})")
        artifact = self.artifact
        standardized = (views[:, 0, :] - artifact["mean"]) / artifact["scale"]
        logits = standardized @ np.asarray(artifact["coef"], dtype=np.float64).T
        logits += np.asarray(artifact["intercept"], dtype=np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    @staticmethod
    def _encode_mask(mask: np.ndarray) -> str:
        buffer = io.BytesIO()
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def predict(self, image: Image.Image) -> dict[str, Any]:
        self._validate_artifact()
        self._validate_runtime()
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        mask = np.asarray(self.segment_mask(rgb))
        if mask.shape != rgb.shape[:2] or mask.dtype != np.bool_:
            raise F4ServiceError("F4 segmenter must return a bool mask on the original image grid")
        try:
            conditions, fallbacks = self.renderer(
                rgb, mask, resize_crop=self.resize_crop, hair_fn=self.hair_fn)
            raw_descriptors = self.describer(rgb, mask, self.hair_fn)
            stats = {"mean": self.artifact["descriptor_mean"],
                     "std": self.artifact["descriptor_std"]}
            descriptors, _ = self.descriptor_module.apply_standardizer(stats, raw_descriptors[None])
            condition_features = {
                condition: self.extract_split(
                    self.backbone, conditions[condition][None], device=self.device,
                    batch_size=self.batch_size, n_views=N_D4)["cls4"]
                for condition in ("full", "crop_view")
            }
            views = self.variant_builder("F4", condition_features, descriptors)
        except Exception as exc:  # noqa: BLE001 - isolate heavy pipeline errors at service boundary
            raise F4ServiceError("canonical F4 feature extraction failed") from exc
        probabilities = self._score_views(views).mean(axis=0)
        fallbacks = set(fallbacks) & {"crop_view"}
        class_probabilities = dict(zip(self.artifact["classes"], probabilities.tolist(), strict=True))
        predicted_class = max(class_probabilities, key=class_probabilities.get)
        return {
            "pred": predicted_class,
            "confidence": class_probabilities[predicted_class],
            "classes": self.artifact["classes"],
            "probs": class_probabilities,
            "feature_dim": int(self.artifact["feature_dim"]),
            "inference_mode": "F4_D4_probability_mean",
            "views": N_D4,
            "fallback": bool(fallbacks),
            "fallback_conditions": sorted(fallbacks),
            "mask_png_base64": self._encode_mask(mask),
            "mask_width": int(mask.shape[1]),
            "mask_height": int(mask.shape[0]),
        }
