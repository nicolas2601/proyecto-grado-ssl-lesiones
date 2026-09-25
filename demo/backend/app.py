from __future__ import annotations

import asyncio
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .model_service import (
    ModelService,
    ModelServiceError,
    ModelUnavailableError,
    ProbeArtifactError,
    ProbeNotReadyError,
)
from .report import ReportDataError, build_report


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


async def _read_image(upload: UploadFile) -> tuple[Image.Image, str] | JSONResponse:
    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        return _error(415, "UNSUPPORTED_MEDIA_TYPE", "Sube una imagen JPEG, PNG o WebP.")
    content = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        return _error(413, "IMAGE_TOO_LARGE", "La imagen supera el limite de 10 MB.")
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
        return image.convert("RGB"), hashlib.sha256(content).hexdigest()
    except (UnidentifiedImageError, OSError) as exc:
        del exc
        return _error(400, "INVALID_IMAGE", "El archivo no contiene una imagen valida.")


def _build_golden_index(repo_root: Path) -> dict[str, dict[str, str]]:
    """Map exact golden bytes to their benchmark label without trusting filenames."""
    golden_root = repo_root / "demo" / "models"
    manifest_path = golden_root / "ssl_proposed_vitb14_golden.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    index: dict[str, dict[str, str]] = {}
    for case in manifest.get("cases", []):
        image_path = golden_root / "golden_images" / f"{case['image_id']}.jpg"
        try:
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        except (OSError, KeyError):
            continue
        index[digest] = {
            "image_id": case["image_id"],
            "true_class": case["true_class"],
        }
    return index


def create_app(repo_root: Path | None = None, service: ModelService | None = None) -> FastAPI:
    root = repo_root or Path(__file__).resolve().parents[2]
    frontend = root / "demo" / "frontend"
    try:
        report = build_report(root)
    except ReportDataError as exc:
        raise RuntimeError(str(exc)) from exc
    model_service = service or ModelService.from_environment(root)

    app = FastAPI(
        title="DermSSL · proposed-cluster",
        version="1.0.0",
        description="Demo local del pipeline SSL propuesto y sus resultados versionados.",
    )
    app.state.report = report
    app.state.model_service = model_service
    app.state.golden_index = _build_golden_index(root)
    app.mount("/static", StaticFiles(directory=frontend), name="static")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; "
            "script-src 'self'; style-src 'self'; img-src 'self' blob: data:; "
            "media-src 'self' blob:; connect-src 'self'"
        )
        return response

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(frontend / "index.html")

    @app.get("/api/report")
    async def get_report() -> dict[str, Any]:
        return {**app.state.report, "model": app.state.model_service.status()}

    @app.get("/api/status")
    async def get_status() -> dict[str, Any]:
        return app.state.model_service.status()

    @app.post("/api/features", response_model=None)
    async def extract_features(file: UploadFile = File(...)) -> dict[str, Any] | JSONResponse:
        image = await _read_image(file)
        if isinstance(image, JSONResponse):
            return image
        image, _ = image
        try:
            return await asyncio.to_thread(app.state.model_service.extract_features, image)
        except ModelUnavailableError as exc:
            del exc
            return _error(503, "MODEL_NOT_READY", "El backbone aun no esta disponible para esta demo.")
        except ModelServiceError as exc:
            del exc
            return _error(422, "FEATURE_EXTRACTION_FAILED", "No se pudo extraer la representacion.")

    @app.post("/api/predict", response_model=None)
    async def predict(
        file: UploadFile = File(...),
        mode: str = Query(default="base", pattern="^(base|tta)$"),
    ) -> dict[str, Any] | JSONResponse:
        image = await _read_image(file)
        if isinstance(image, JSONResponse):
            return image
        image, digest = image
        try:
            prediction = await asyncio.to_thread(app.state.model_service.predict, image, mode)
            benchmark = app.state.golden_index.get(digest)
            if benchmark:
                prediction["benchmark"] = benchmark
            return prediction
        except ProbeNotReadyError as exc:
            del exc
            return _error(
                503,
                "PROBE_NOT_READY",
                "El .pt publicado es un backbone congelado; falta configurar el probe de clasificacion.",
            )
        except (ModelUnavailableError, ProbeArtifactError) as exc:
            del exc
            return _error(503, "MODEL_NOT_READY", "El modelo no esta listo para clasificar.")
        except ModelServiceError as exc:
            del exc
            return _error(422, "PREDICTION_FAILED", "No se pudo procesar la imagen.")

    return app


app = create_app()
