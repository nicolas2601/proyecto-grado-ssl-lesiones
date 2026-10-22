from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
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
from .request_limits import InferenceLimits
from .v9_service import V9ModelService, V9ServiceError


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
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
        if image.width * image.height > MAX_IMAGE_PIXELS:
            return _error(413, "IMAGE_TOO_LARGE", "La imagen supera el limite de 20 megapixeles.")
        image.load()
        return image.convert("RGB"), hashlib.sha256(content).hexdigest()
    except Image.DecompressionBombError:
        return _error(413, "IMAGE_TOO_LARGE", "La imagen excede el limite seguro de descompresion.")
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


def create_app(repo_root: Path | None = None, service: ModelService | None = None,
               public_service: ModelService | None = None) -> FastAPI:
    root = repo_root or Path(__file__).resolve().parents[2]
    v9_bundle_dir = os.getenv("DEMO_V9_MODEL_DIR")
    if v9_bundle_dir and os.getenv("DEMO_API_ONLY") != "1":
        raise RuntimeError("DEMO_V9_MODEL_DIR requires DEMO_API_ONLY=1")
    frontend = Path(os.getenv("DEMO_FRONTEND_DIR", root / "demo" / "phone"))
    try:
        report = ({"source": {"release": "v9.0.0", "scope": "current_inference_runtime"},
                   "disclaimer": "Research only; model scores are not clinical probabilities."}
                  if v9_bundle_dir else build_report(root))
    except ReportDataError as exc:
        raise RuntimeError(str(exc)) from exc
    api_only = os.getenv("DEMO_API_ONLY") == "1"
    if v9_bundle_dir:
        services = {arm: V9ModelService(Path(v9_bundle_dir) / arm, expected_arm=arm)
                    for arm in ("A71", "B71")}
    else:
        model_service = service or ModelService.from_environment(root)
        public_root = Path(os.getenv("PUBLIC_MODEL_DIR", root / "demo" / "models" / "public"))
        public_manifest = public_root / "dinov2_public_vitb14_classifier_manifest.json"
        services = {"legacy": model_service, "public": public_service or ModelService(
            public_root / "dinov2_public_vitb14_backbone.pt", public_manifest,
            public_root / "dinov2_public_vitb14_probe.npz", classifier_manifest_path=public_manifest)}
        proposed_root = Path(os.getenv("PROPOSED_MODEL_DIR", root / "demo" / "models" / "proposed"))
        for arm, backbone_file in (("A71", "A71_backbone.pt"), ("B71", "B71_clean_Bstar_s42_backbone.pt")):
            manifest = proposed_root / f"{arm}_classifier_manifest.json"
            services[arm] = ModelService(proposed_root / backbone_file, manifest,
                                         proposed_root / f"{arm}_probe.npz", classifier_manifest_path=manifest)
        if api_only:
            services = {key: services[key] for key in ("A71", "B71")}
    model_lock = threading.Lock()
    default_model = os.getenv("DEMO_DEFAULT_MODEL", "B71" if api_only else "legacy")
    model_pattern = "^(A71|B71)$" if api_only else "^(legacy|public|A71|B71)$"
    if default_model not in services:
        raise RuntimeError("invalid default model")

    def public_status(key):
        return {name: value for name, value in services[key].status().items() if not name.endswith("_path")}

    def run_model(key, method, *args):
        with model_lock:
            for other in services.values():
                if isinstance(other, V9ModelService) and other is not services[key]:
                    other.unload()
                elif isinstance(other, ModelService) and other is not services[key]:
                    other._backbone = None
            return getattr(services[key], method)(*args)

    app = FastAPI(
        title="DermSSL research API",
        version="9.0.0" if v9_bundle_dir else "2.0.0",
        description="Experimental A71/B71 image classification. Research only, not a diagnostic device.",
        docs_url=None,
    )
    app.state.report = report
    app.state.model_service = services[default_model]
    app.state.model_services = services
    app.state.golden_index = _build_golden_index(root)
    app.add_middleware(InferenceLimits)
    origin = os.getenv("DEMO_FRONTEND_ORIGIN")
    if origin:
        app.add_middleware(CORSMiddleware, allow_origins=[origin], allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type"], allow_credentials=False)
    app.mount("/static", StaticFiles(directory=frontend), name="static")

    @app.get("/docs", include_in_schema=False)
    async def swagger():
        return get_swagger_ui_html(openapi_url=app.openapi_url, title="DermSSL research API",
                                   swagger_favicon_url="/static/icon.svg")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; "
            "script-src 'self'; style-src 'self'; img-src 'self' blob: data:; "
            "media-src 'self' blob:; connect-src 'self'"
        )
        if request.url.path == "/docs":
            # Swagger's generated page uses a CDN bundle and inline initialization.
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; "
                "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
                "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
                "img-src 'self' data:; connect-src 'self'"
            )
        return response

    @app.get("/", include_in_schema=False, response_model=None)
    async def index() -> FileResponse | JSONResponse:
        if api_only:
            return JSONResponse({"service": "DermSSL research API", "docs": "/docs"})
        return FileResponse(frontend / "index.html")

    @app.get("/api/report")
    async def get_report() -> dict[str, Any]:
        return {**app.state.report, "model": public_status(default_model),
                "models": {key: public_status(key) for key in services}}

    @app.get("/api/status")
    async def get_status() -> dict[str, Any]:
        return {**public_status(default_model), "model_key": default_model}

    @app.post("/api/features", response_model=None)
    async def extract_features(file: UploadFile = File(...)) -> dict[str, Any] | JSONResponse:
        if v9_bundle_dir:
            return _error(501, "FEATURES_UNSUPPORTED", "V9 entrega probabilidades de clasificación, no vectores CLS.")
        image = await _read_image(file)
        if isinstance(image, JSONResponse):
            return image
        image, _ = image
        try:
            return await asyncio.to_thread(run_model, default_model, "extract_features", image)
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
        model: str = Query(default=default_model, pattern=model_pattern),
    ) -> dict[str, Any] | JSONResponse:
        image = await _read_image(file)
        if isinstance(image, JSONResponse):
            return image
        image, digest = image
        try:
            prediction = await asyncio.to_thread(run_model, model, "predict", image, mode)
            prediction["model_key"] = model
            prediction["identity"] = services[model].status().get("identity")
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
        except V9ServiceError as exc:
            del exc
            return _error(503, "MODEL_NOT_READY", "El bundle V9 no esta listo para clasificar.")
        except ModelServiceError as exc:
            del exc
            return _error(422, "PREDICTION_FAILED", "No se pudo procesar la imagen.")

    return app


app = create_app()
