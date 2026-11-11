from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from .analyze_routes import register_analyze_routes
from .request_limits import InferenceLimits
from .segmenter_service import SegmenterService
from .v10_http import error_response, read_image
from .v10_report import V10ReportError, build_v10_report
from .v10_service import (
    MoCoArtifactError,
    MoCoModelService,
    MoCoServiceError,
    MoCoUnsupportedModeError,
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MODEL_KEYS = ("MOCO_HAM7", "MOCO_PAD6")
MODEL_PATTERN = "^(" + "|".join(MODEL_KEYS) + ")$"
DEFAULT_MODEL = "MOCO_HAM7"
FRONTEND_ORIGIN = "https://dermssl-phone.vercel.app"
DEFAULT_SEGMENTER_DIR = "/models/segmenter"


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


def _default_segmenter() -> SegmenterService:
    return SegmenterService(
        segmenter_dir=os.getenv("SEGMENTER_DIR", DEFAULT_SEGMENTER_DIR),
        expected_manifest_sha256=os.getenv("SEGMENTER_MANIFEST_SHA256"),
    )


def _public_status(service: Any) -> dict[str, Any]:
    return {key: value for key, value in service.status().items() if not key.endswith("_path")}


def _segmenter_summary(segmenter: Any) -> dict[str, Any]:
    status = segmenter.status()
    return {"ready": status["ready"], "reason": status["reason"], "identity": status.get("identity")}


def create_app(
    repo_root: Path | None = None,
    moco_service: Any | None = None,
    report_path: Path | None = None,
    segmenter_service: Any | None = None,
) -> FastAPI:
    del repo_root
    release_root = Path(os.getenv("MOCO_RELEASE_ROOT", "/models/v10"))
    report_file = report_path or release_root / "report.json"
    moco = moco_service if moco_service is not None else _default_moco_service(release_root)
    segmenter = segmenter_service if segmenter_service is not None else _default_segmenter()
    services = {key: moco for key in MODEL_KEYS}
    inference_lock = threading.Lock()
    app = FastAPI(
        title="DermSSL research API",
        version="10.2.0",
        description="Research-only MoCo V10 classifier, lesion segmentation and tone simulation endpoints.",
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

    def get_status(model_key: str) -> dict[str, Any]:
        with inference_lock:
            service = services.get(model_key)
            if service is None:
                return {"backbone_ready": False, "classification_ready": False, "reason": "model unavailable"}
            return {**_public_status(service), "model_key": model_key}

    def run_model(model_key: str, image: Image.Image, mode: str) -> dict[str, Any]:
        with inference_lock:
            service = services.get(model_key)
            if service is None:
                raise MoCoServiceError("selected model is unavailable")
            return service.predict(image, model_key=model_key, mode=mode)

    def run_exclusive(function: Any, *args: Any) -> Any:
        with inference_lock:
            return function(*args)

    async def read_upload(upload: UploadFile) -> Image.Image | JSONResponse:
        return await read_image(upload, MAX_UPLOAD_BYTES, MAX_IMAGE_PIXELS)

    @app.get("/api/status")
    async def api_status(
        model: str = Query(default=DEFAULT_MODEL, pattern=MODEL_PATTERN),
    ) -> dict[str, Any]:
        status = await asyncio.to_thread(get_status, model)
        return {**status, "segmenter": await asyncio.to_thread(_segmenter_summary, segmenter)}

    @app.get("/api/models")
    async def api_models() -> dict[str, Any]:
        return {
            "default_model": DEFAULT_MODEL,
            "models": [{"model_key": key, "release": "v10.0.0"} for key in MODEL_KEYS],
        }

    @app.get("/health", response_model=None)
    async def health() -> dict[str, Any] | JSONResponse:
        status = await asyncio.to_thread(get_status, DEFAULT_MODEL)
        if not status.get("classification_ready"):
            return error_response(503, "MODEL_NOT_READY", "Default V10 model is not ready.")
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
            return error_response(503, "REPORT_NOT_READY", "V10 research report is not available.")
        model_status = await asyncio.to_thread(get_status, DEFAULT_MODEL)
        return {
            **report,
            "model": model_status,
            "models": list(MODEL_KEYS),
        }

    @app.post("/api/features")
    async def api_features() -> JSONResponse:
        return error_response(501, "FEATURES_UNSUPPORTED", "Feature-vector extraction is not supported.")

    @app.post("/api/predict", response_model=None)
    async def predict(
        file: UploadFile = File(...),
        model: str = Query(default=DEFAULT_MODEL, pattern=MODEL_PATTERN),
        mode: str = Query(default="base", pattern="^(base|tta)$"),
    ) -> dict[str, Any] | JSONResponse:
        if mode == "tta":
            return error_response(422, "UNSUPPORTED_INFERENCE_MODE", "TTA is not supported for MoCo V10.")
        image = await read_upload(file)
        if isinstance(image, JSONResponse):
            return image
        try:
            return await asyncio.to_thread(run_model, model, image, mode)
        except MoCoUnsupportedModeError:
            return error_response(422, "UNSUPPORTED_INFERENCE_MODE", "TTA is not supported for MoCo V10.")
        except (MoCoArtifactError, MoCoServiceError):
            return error_response(503, "MODEL_NOT_READY", "Selected model is not ready to classify.")

    register_analyze_routes(
        app,
        run_model=run_model,
        run_exclusive=run_exclusive,
        read_upload=read_upload,
        segmenter=segmenter,
        default_model=DEFAULT_MODEL,
        model_pattern=MODEL_PATTERN,
    )
    return app


app = create_app()
