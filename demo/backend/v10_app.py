from __future__ import annotations

import asyncio
import io
import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from .model_service import ModelServiceError
from .request_limits import InferenceLimits
from .v10_report import V10ReportError, build_v10_report
from .v10_service import (
    MoCoArtifactError,
    MoCoModelService,
    MoCoServiceError,
    MoCoUnsupportedModeError,
)
from .v9_service import V9ModelService, V9ServiceError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MODEL_KEYS = ("MOCO_HAM7", "MOCO_PAD6", "A71", "B71")
DEFAULT_MODEL = "MOCO_HAM7"
FRONTEND_ORIGIN = "https://dermssl-phone.vercel.app"


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
        headers={"Cache-Control": "no-store"},
    )


async def _read_image(upload: UploadFile) -> Image.Image | JSONResponse:
    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        return _error(415, "UNSUPPORTED_MEDIA_TYPE", "Upload a JPEG, PNG, or WebP image.")
    content = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        return _error(413, "IMAGE_TOO_LARGE", "Image exceeds the 10 MiB limit.")
    try:
        image = Image.open(io.BytesIO(content))
        if image.width * image.height > MAX_IMAGE_PIXELS:
            return _error(413, "IMAGE_TOO_LARGE", "Image exceeds the 20 megapixel limit.")
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")
    except Image.DecompressionBombError:
        return _error(413, "IMAGE_TOO_LARGE", "Image exceeds the safe decompression limit.")
    except (UnidentifiedImageError, OSError, ValueError):
        return _error(400, "INVALID_IMAGE", "File does not contain a valid image.")


class _UnconfiguredMoCoService:
    def __init__(self, reason: str):
        self.reason = reason

    def status(self) -> dict[str, Any]:
        return {
            "backbone_ready": False,
            "classification_ready": False,
            "reason": self.reason,
            "release": "v10.0.0",
        }

    def predict(self, image: Image.Image, model_key: str, mode: str) -> dict[str, Any]:
        del image, model_key, mode
        raise MoCoArtifactError(self.reason)


def _default_moco_service(release_root: Path) -> Any:
    try:
        return MoCoModelService(
            bundle_dir=release_root,
            expected_manifest_sha256=os.getenv("MOCO_MANIFEST_SHA256"),
        )
    except MoCoArtifactError as exc:
        return _UnconfiguredMoCoService(str(exc))


def _default_legacy_services() -> dict[str, V9ModelService]:
    bundle_root = Path(os.getenv("DEMO_V9_MODEL_DIR", "/models/v9"))
    return {
        key: V9ModelService(bundle_root / key, expected_arm=key)
        for key in ("A71", "B71")
    }


def _public_status(service: Any) -> dict[str, Any]:
    return {key: value for key, value in service.status().items() if not key.endswith("_path")}


def create_app(
    repo_root: Path | None = None,
    moco_service: Any | None = None,
    legacy_services: dict[str, Any] | None = None,
    report_path: Path | None = None,
) -> FastAPI:
    del repo_root
    release_root = Path(os.getenv("MOCO_RELEASE_ROOT", "/models/v10"))
    report_file = report_path or release_root / "report.json"
    moco = moco_service if moco_service is not None else _default_moco_service(release_root)
    legacy = _default_legacy_services() if legacy_services is None else legacy_services
    services = {"MOCO_HAM7": moco, "MOCO_PAD6": moco, **legacy}
    inference_lock = threading.Lock()
    active_service: Any | None = None
    app = FastAPI(
        title="DermSSL research API",
        version="10.0.0",
        description="Research-only MoCo V10 and preserved V9 classifier endpoints.",
        docs_url=None,
    )
    app.state.model_services = services
    app.state.default_model = DEFAULT_MODEL
    app.add_middleware(InferenceLimits)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[FRONTEND_ORIGIN],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; "
            "script-src 'self'; style-src 'self'; img-src 'self' blob: data:; "
            "media-src 'self' blob:; connect-src 'self'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def activate_model(model_key: str) -> Any | None:
        nonlocal active_service
        service = services.get(model_key)
        if service is None:
            return None
        if service is not active_service:
            unloaded: set[int] = set()
            for other in services.values():
                if other is service or id(other) in unloaded or not hasattr(other, "unload"):
                    continue
                unloaded.add(id(other))
                other.unload()
            active_service = service
        return service

    def get_status(model_key: str) -> dict[str, Any]:
        with inference_lock:
            service = activate_model(model_key)
            if service is None:
                return {"backbone_ready": False, "classification_ready": False, "reason": "model unavailable"}
            return {**_public_status(service), "model_key": model_key}

    def run_model(model_key: str, image: Image.Image, mode: str) -> dict[str, Any]:
        with inference_lock:
            service = activate_model(model_key)
            if service is None:
                raise MoCoServiceError("selected model is unavailable")
            if model_key.startswith("MOCO_"):
                return service.predict(image, model_key=model_key, mode=mode)
            prediction = service.predict(image, mode=mode)
            prediction["model_key"] = model_key
            prediction["identity"] = service.status().get("identity")
            return prediction

    @app.get("/api/status")
    async def api_status(
        model: str = Query(default=DEFAULT_MODEL, pattern="^(MOCO_HAM7|MOCO_PAD6|A71|B71)$"),
    ) -> dict[str, Any]:
        return await asyncio.to_thread(get_status, model)

    @app.get("/api/models")
    async def api_models() -> dict[str, Any]:
        return {
            "default_model": DEFAULT_MODEL,
            "models": [
                {"model_key": "MOCO_HAM7", "release": "v10.0.0"},
                {"model_key": "MOCO_PAD6", "release": "v10.0.0"},
                {"model_key": "A71", "release": "v9.0.0"},
                {"model_key": "B71", "release": "v9.0.0"},
            ],
        }

    @app.get("/health", response_model=None)
    async def health() -> dict[str, Any] | JSONResponse:
        status = await asyncio.to_thread(get_status, DEFAULT_MODEL)
        if not status.get("classification_ready"):
            return _error(503, "MODEL_NOT_READY", "Default V10 model is not ready.")
        return {"status": "ready", "model_key": DEFAULT_MODEL, "release": "v10.0.0"}

    @app.get("/api/report", response_model=None)
    async def api_report() -> dict[str, Any] | JSONResponse:
        try:
            if isinstance(moco, _UnconfiguredMoCoService):
                raise V10ReportError("V10 manifest pin is not configured")
            expected_sha = None
            if isinstance(moco, MoCoModelService):
                expected_sha = await asyncio.to_thread(moco.verify_artifact, "report.json")
            report = await asyncio.to_thread(build_v10_report, report_file, expected_sha)
        except (MoCoArtifactError, V10ReportError):
            return _error(503, "REPORT_NOT_READY", "V10 research report is not available.")
        model_status = await asyncio.to_thread(get_status, DEFAULT_MODEL)
        return {
            **report,
            "model": model_status,
            "models": ["MOCO_HAM7", "MOCO_PAD6", "A71", "B71"],
        }

    @app.post("/api/features")
    async def api_features() -> JSONResponse:
        return _error(501, "FEATURES_UNSUPPORTED", "Feature-vector extraction is not supported.")

    @app.post("/api/predict", response_model=None)
    async def predict(
        file: UploadFile = File(...),
        model: str = Query(default=DEFAULT_MODEL, pattern="^(MOCO_HAM7|MOCO_PAD6|A71|B71)$"),
        mode: str = Query(default="base", pattern="^(base|tta)$"),
    ) -> dict[str, Any] | JSONResponse:
        if model.startswith("MOCO_") and mode == "tta":
            return _error(422, "UNSUPPORTED_INFERENCE_MODE", "TTA is not supported for MoCo V10.")
        image = await _read_image(file)
        if isinstance(image, JSONResponse):
            return image
        try:
            return await asyncio.to_thread(run_model, model, image, mode)
        except MoCoUnsupportedModeError:
            return _error(422, "UNSUPPORTED_INFERENCE_MODE", "TTA is not supported for MoCo V10.")
        except (MoCoArtifactError, MoCoServiceError, V9ServiceError, ModelServiceError):
            return _error(503, "MODEL_NOT_READY", "Selected model is not ready to classify.")

    return app


app = create_app()
