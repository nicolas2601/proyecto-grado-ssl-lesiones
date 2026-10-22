"""Serve F4clean only after a bound, successful twenty-case online golden."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

from .model_service import ModelService, ModelUnavailableError
from .f4_runtime import B71_SHA256, SEGMENTER_SHA256, EXPECTED_RECIPE, create_f4_service, file_sha256


class SegmentedModelService(ModelService):
    def __init__(self, weights_path: Path, manifest_path: Path, probe_path: Path, *,
                 f4_dir: Path, scientific_root: Path, classifier_manifest_path: Path | None = None,
                 device: str | None = None) -> None:
        super().__init__(weights_path, manifest_path, probe_path, device=device,
                         classifier_manifest_path=classifier_manifest_path)
        self.f4_dir = f4_dir
        self.scientific_root = scientific_root
        self._f4 = None

    def approved_f4_manifest(self) -> dict[str, Any] | None:
        try:
            manifest_path = self.f4_dir / "F4_classifier_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            receipt = json.loads((self.f4_dir / "F4_golden_verification.json").read_text())
            if any(manifest.get(key) != value for key, value in EXPECTED_RECIPE.items()):
                return None
            count, difference = receipt.get("n_cases", 0), receipt.get("max_abs_probability_difference")
            if (receipt.get("passed") is not True or not isinstance(count, int) or count < 20
                    or not isinstance(difference, (float, int)) or not math.isfinite(difference)
                    or difference < 0 or difference > 0.01):
                return None
            if not (self.f4_dir / "student_r3_final_best.pt").is_file():
                return None
            probe_sha = file_sha256(self.f4_dir / "F4_probe.npz")
            if (receipt.get("manifest_sha256") != file_sha256(manifest_path)
                    or receipt.get("probe_sha256") != probe_sha or manifest.get("probe_sha256") != probe_sha
                    or receipt.get("backbone_sha256") != B71_SHA256
                    or receipt.get("segmenter_sha256") != SEGMENTER_SHA256):
                return None
            return manifest
        except (OSError, ValueError, TypeError):
            return None

    def status(self) -> dict[str, Any]:
        result = super().status()
        manifest = self.approved_f4_manifest() if result["classification_ready"] else None
        result["segmentation_ready"] = manifest is not None
        if manifest:
            result["identity"] = {**(result.get("identity") or {}), "model_id": "B71_F4clean",
                                  "release": "F4clean float32, fixed exploratory recipe",
                                  "segmenter_sha256": SEGMENTER_SHA256,
                                  "segmented_probe_sha256": manifest["probe_sha256"]}
        return result

    def _load_f4(self):
        if self._f4 is None:
            if self.approved_f4_manifest() is None:
                raise ModelUnavailableError("segmentation has no verified online golden")
            import torch
            self._f4 = create_f4_service(self._load_backbone(), self.f4_dir,
                                         self.scientific_root, torch.device(self.device_name))
        return self._f4

    def predict(self, image: Image.Image, mode: str = "base") -> dict[str, Any]:
        if mode == "preview":
            result = super().predict(image, mode="base")
            return {**result, "inference_mode": "preview_CLS_without_segmentation", "approximate": True,
                    "identity": super().status().get("identity")}
        if mode == "base" and self.approved_f4_manifest() is not None:
            try:
                with self._inference_lock:
                    result = self._load_f4().predict(image)
                result["identity"] = self.status().get("identity")
                return result
            except ModelUnavailableError:
                raise
            except Exception as exc:
                raise ModelUnavailableError("segmented inference failed") from exc
        result = super().predict(image, mode=mode)
        result["identity"] = super().status().get("identity")
        result["segmentation_status"] = "not_verified"
        return result

    def unload(self) -> None:
        with self._inference_lock:
            self._f4 = None
            self._backbone = None
