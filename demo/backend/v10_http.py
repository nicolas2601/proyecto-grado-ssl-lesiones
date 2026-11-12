from __future__ import annotations

import io

from fastapi import UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
        headers={"Cache-Control": "no-store"},
    )


async def read_image(upload: UploadFile, max_bytes: int, max_pixels: int) -> Image.Image | JSONResponse:
    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        return error_response(415, "UNSUPPORTED_MEDIA_TYPE", "Upload a JPEG, PNG, or WebP image.")
    content = await upload.read(max_bytes + 1)
    if len(content) > max_bytes:
        return error_response(413, "IMAGE_TOO_LARGE", "Image exceeds the 10 MiB limit.")
    try:
        image = Image.open(io.BytesIO(content))
        if image.width * image.height > max_pixels:
            return error_response(413, "IMAGE_TOO_LARGE", "Image exceeds the 20 megapixel limit.")
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")
    except Image.DecompressionBombError:
        return error_response(413, "IMAGE_TOO_LARGE", "Image exceeds the safe decompression limit.")
    except (UnidentifiedImageError, OSError, ValueError):
        return error_response(400, "INVALID_IMAGE", "File does not contain a valid image.")
