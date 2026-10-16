from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .model_identity import describe_model, verify_model

class ModelServiceError(RuntimeError):
    """Base error for model readiness and inference failures."""


class ModelUnavailableError(ModelServiceError):
    """The published backbone cannot be loaded in the current environment."""


class ProbeNotReadyError(ModelServiceError):
    """The backbone exists, but no compatible classifier probe is configured."""


class ProbeArtifactError(ModelServiceError):
    """The classifier artifact is malformed or incompatible."""


def predict_probabilities(features: np.ndarray, artifact: dict[str, Any]) -> dict[str, Any]:
    """Apply a portable linear probe artifact to one frozen feature vector."""
    classes = [str(value) for value in artifact.get("classes", [])]
    if not classes:
        raise ProbeArtifactError("probe has no classes")

    vector = np.asarray(features, dtype=np.float32).reshape(-1)
    try:
        feature_dim = int(artifact["feature_dim"])
        mean = np.asarray(artifact["mean"], dtype=np.float32)
        scale = np.asarray(artifact["scale"], dtype=np.float32)
        coef = np.asarray(artifact["coef"], dtype=np.float32)
        intercept = np.asarray(artifact["intercept"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProbeArtifactError("probe is missing numeric parameters") from exc

    if vector.size != feature_dim:
        raise ProbeArtifactError(
            f"feature dimension {vector.size} does not match probe dimension {feature_dim}"
        )
    if mean.shape != (feature_dim,) or scale.shape != (feature_dim,):
        raise ProbeArtifactError("probe scaler shape does not match feature dimension")
    if coef.shape != (len(classes), feature_dim) or intercept.shape != (len(classes),):
        raise ProbeArtifactError("probe classifier shape does not match classes")
    if np.any(scale == 0) or not np.isfinite(np.concatenate([mean, scale, coef.ravel(), intercept])).all():
        raise ProbeArtifactError("probe contains invalid numeric values")

    standardized = (vector - mean) / scale
    logits = coef @ standardized + intercept
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    scores = {label: round(float(probability), 6) for label, probability in zip(classes, probabilities)}
    prediction_index = int(np.argmax(probabilities))
    return {
        "pred": classes[prediction_index],
        "confidence": round(float(probabilities[prediction_index]), 6),
        "classes": classes,
        "probs": scores,
        "feature_dim": feature_dim,
    }


class ModelService:
    """Lazy, frozen-backbone inference service for the local presentation demo."""

    def __init__(
        self,
        weights_path: Path,
        manifest_path: Path,
        probe_path: Path,
        device: str | None = None,
        classifier_manifest_path: Path | None = None,
    ) -> None:
        self.weights_path = weights_path
        self.manifest_path = manifest_path
        self.probe_path = probe_path
        self.classifier_manifest_path = classifier_manifest_path or probe_path.with_name(
            "ssl_proposed_vitb14_classifier_manifest.json"
        )
        self.device_name = device or os.getenv("DEMO_DEVICE", "cpu")
        self._backbone: Any | None = None
        self._probe: dict[str, Any] | None = None
        self._identity: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @classmethod
    def from_environment(cls, repo_root: Path) -> "ModelService":
        default_weights = repo_root / "demo" / "models" / "ssl_proposed_vitb14_backbone.pt"
        weights_path = Path(os.getenv("MODEL_WEIGHTS_PATH", default_weights))
        default_manifest = weights_path.with_name("ssl_proposed_vitb14_run.json")
        manifest_path = Path(os.getenv("MODEL_MANIFEST_PATH", default_manifest))
        probe_path = Path(
            os.getenv(
                "MODEL_PROBE_PATH",
                str(repo_root / "demo" / "models" / "ssl_proposed_vitb14_probe.npz"),
            )
        )
        classifier_manifest_path = Path(
            os.getenv(
                "MODEL_CLASSIFIER_MANIFEST_PATH",
                str(repo_root / "demo" / "models" / "ssl_proposed_vitb14_classifier_manifest.json"),
            )
        )
        return cls(weights_path, manifest_path, probe_path, classifier_manifest_path=classifier_manifest_path)

    def status(self) -> dict[str, Any]:
        weights_ready = self.weights_path.is_file()
        manifest_ready = self.manifest_path.is_file()
        probe_ready = self.probe_path.is_file()
        classifier_manifest_ready = self.classifier_manifest_path.is_file()
        classifier_manifest_valid = self._classifier_manifest_valid()
        manifest_valid = False
        if manifest_ready:
            try:
                manifest_valid = (
                    json.loads(self.manifest_path.read_text(encoding="utf-8")).get("backbone")
                    == "dinov2_vitb14"
                )
            except (OSError, json.JSONDecodeError):
                manifest_valid = False
        reason = "ready"
        if not weights_ready:
            reason = "weights_missing"
        elif not manifest_ready:
            reason = "manifest_missing"
        elif not manifest_valid:
            reason = "manifest_invalid"
        elif not probe_ready:
            reason = "probe_missing"
        elif not classifier_manifest_ready:
            reason = "classifier_manifest_missing"
        elif not classifier_manifest_valid:
            reason = "classifier_manifest_invalid"
        return {
            "backbone_ready": weights_ready and manifest_valid,
            "classification_ready": (
                weights_ready
                and manifest_valid
                and probe_ready
                and classifier_manifest_ready
                and classifier_manifest_valid
            ),
            "weights_path": str(self.weights_path),
            "manifest_path": str(self.manifest_path),
            "probe_path": str(self.probe_path),
            "classifier_manifest_path": str(self.classifier_manifest_path),
            "device": self.device_name,
            "reason": reason,
            "identity": self._identity or describe_model(self.classifier_manifest_path),
        }

    def _manifest(self) -> dict[str, Any]:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelUnavailableError("No se pudo leer el run.json del modelo") from exc
        if manifest.get("backbone") != "dinov2_vitb14":
            raise ModelUnavailableError("El run.json no corresponde a ViT-B/14")
        return manifest

    def _classifier_manifest_valid(self) -> bool:
        if not self.classifier_manifest_path.is_file():
            return False
        try:
            manifest = json.loads(self.classifier_manifest_path.read_text(encoding="utf-8"))
            return (
                manifest.get("backbone") == "dinov2_vitb14"
                and int(manifest["feature_extraction"]["dim"]) == 768
                and manifest["probe"]["file"] == self.probe_path.name
                and manifest["classes"] == ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return False

    def _load_backbone(self) -> Any:
        if self._backbone is not None:
            return self._backbone
        if not self.weights_path.is_file() or not self.manifest_path.is_file():
            raise ModelUnavailableError("Faltan el peso o el run.json del backbone")
        with self._lock:
            if self._backbone is not None:
                return self._backbone
            try:
                identity = verify_model(self.classifier_manifest_path, self.weights_path)
                import torch

                hub_dir = os.getenv("DEMO_HUB_DIR")
                model = torch.hub.load(
                    hub_dir or "facebookresearch/dinov2", self._manifest()["backbone"],
                    pretrained=False, **({"source": "local"} if hub_dir else {}),
                )
                state = torch.load(
                    self.weights_path,
                    map_location="cpu",
                    weights_only=True,
                )
                model.load_state_dict(state, strict=True)
                model.eval().to(self.device_name)
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
            except Exception as exc:  # noqa: BLE001 - boundary hides internals from HTTP clients
                raise ModelUnavailableError("No se pudo cargar el backbone publicado") from exc
            self._backbone = model
            self._identity = identity
            return model

    def _load_probe(self) -> dict[str, Any]:
        if self._probe is not None:
            return self._probe
        if not self.probe_path.is_file():
            raise ProbeNotReadyError(
                "El .pt publicado es un backbone; falta un probe de clasificacion compatible"
            )
        if not self.classifier_manifest_path.is_file():
            raise ProbeArtifactError("Falta el manifest del probe de clasificacion")
        try:
            classifier_manifest = json.loads(
                self.classifier_manifest_path.read_text(encoding="utf-8")
            )
            with np.load(self.probe_path, allow_pickle=False) as data:
                artifact = {
                    "classes": data["classes"].tolist(),
                    "mean": data["mean"].tolist(),
                    "scale": data["scale"].tolist(),
                    "coef": data["coef"].tolist(),
                    "intercept": data["intercept"].tolist(),
                    "feature_dim": int(data["feature_dim"]),
                }
            expected_dim = int(classifier_manifest["feature_extraction"]["dim"])
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            raise ProbeArtifactError("El probe no tiene el formato .npz esperado") from exc
        if artifact["feature_dim"] != expected_dim:
            raise ProbeArtifactError("El probe no coincide con la dimension del manifest")
        self._probe = artifact
        return artifact

    def _features(self, image: Image.Image) -> np.ndarray:
        with self._inference_lock:
            try:
                import torch
                from torchvision import transforms

                preprocess = transforms.Compose(
                    [
                        transforms.Resize(int(224 * 1.14)),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225),
                        ),
                    ]
                )
                tensor = preprocess(image.convert("RGB")).unsqueeze(0)
                with torch.no_grad():
                    features = self._load_backbone()(tensor.to(self.device_name))
                return features[0].float().cpu().numpy()
            except ModelServiceError:
                raise
            except Exception as exc:  # noqa: BLE001 - inference boundary
                raise ModelUnavailableError("La inferencia del backbone fallo") from exc

    def extract_features(self, image: Image.Image) -> dict[str, Any]:
        features = self._features(image)
        return {
            "feature_dim": int(features.shape[0]),
            "feature_norm": round(float(np.linalg.norm(features)), 6),
            "classification_ready": self.status()["classification_ready"],
        }

    def _features_tta(self, image: Image.Image) -> np.ndarray:
        """Average original, mirrored, and flipped views before the same probe."""
        views = (image, ImageOps.mirror(image), ImageOps.flip(image))
        return np.stack([self._features(view) for view in views]).mean(axis=0)

    def predict(self, image: Image.Image, mode: str = "base") -> dict[str, Any]:
        if mode == "base":
            features = self._features(image)
            result = predict_probabilities(features, self._load_probe())
            result.update({"inference_mode": "base", "views": 1})
            return result
        if mode == "tta":
            features = self._features_tta(image)
            result = predict_probabilities(features, self._load_probe())
            result.update({"inference_mode": "tta_3views", "views": 3})
            return result
        raise ProbeArtifactError(f"unsupported inference mode: {mode}")

    def predict_tta(self, image: Image.Image) -> dict[str, Any]:
        return self.predict(image, mode="tta")
