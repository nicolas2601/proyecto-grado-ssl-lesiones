"""POST /api/analyze and POST /api/simulate-tone, registered on the V10 FastAPI app.

Classification runs through the same ``run_model`` path as /api/predict. Segmentation and
photometry run afterwards under the same exclusive lock, so only one inference runs at a time.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse

from .analyze_pipeline import preprocessing_of, run_extras
from .tone_pipeline import simulate_for_image
from .tone_sim import ToneSimulationError
from .v10_http import error_response
from .v10_service import MoCoArtifactError, MoCoServiceError

ANALYZE_SCHEMA = "dermssl.analyze.v1"
ANALYZE_SCOPE = "research"
_MS_PER_SECOND = 1000


def _ms(seconds: float) -> int:
    return int(round(seconds * _MS_PER_SECOND))


def register_analyze_routes(
    app: FastAPI,
    *,
    run_model: Callable[[str, Any, str], dict[str, Any]],
    run_exclusive: Callable[..., Any],
    read_upload: Callable[[UploadFile], Awaitable[Any]],
    segmenter: Any,
    default_model: str,
    model_pattern: str,
) -> None:
    def _simulate(image: Any, mask_b64: str, delta_text: str) -> dict[str, Any]:
        return simulate_for_image(image, mask_b64, delta_text, preprocessing_of(segmenter))

    @app.post("/api/analyze", response_model=None)
    async def analyze(
        file: UploadFile = File(...),
        model: str = Query(default=default_model, pattern=model_pattern),
    ) -> dict[str, Any] | JSONResponse:
        image = await read_upload(file)
        if isinstance(image, JSONResponse):
            return image
        started = perf_counter()
        try:
            classification = await asyncio.to_thread(run_model, model, image, "base")
        except (MoCoArtifactError, MoCoServiceError):
            return error_response(503, "MODEL_NOT_READY", "Selected model is not ready to classify.")
        classified = perf_counter()
        extras = await asyncio.to_thread(run_exclusive, run_extras, segmenter, np.asarray(image))
        return {
            "schema": ANALYZE_SCHEMA,
            "scope": ANALYZE_SCOPE,
            "classification": classification,
            "segmentation": extras["segmentation"],
            "photometry": extras["photometry"],
            "timing_ms": {"classification": _ms(classified - started), **extras["timing_ms"]},
        }

    @app.post("/api/simulate-tone", response_model=None)
    async def simulate_tone_route(
        file: UploadFile = File(...),
        mask_png_base64: str = Form(...),
        delta_l: str = Form(...),
    ) -> dict[str, Any] | JSONResponse:
        image = await read_upload(file)
        if isinstance(image, JSONResponse):
            return image
        try:
            return await asyncio.to_thread(run_exclusive, _simulate, image, mask_png_base64, delta_l)
        except ToneSimulationError as exc:
            return error_response(422, exc.code, str(exc))
