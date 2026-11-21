"""Letterbox segmenter contract (manifest schema ``dermssl.segmenter.v2``).

The image keeps its aspect ratio: the long side is resized (PIL bilinear) to S, the short side to the
rounded proportional length, pixels are scaled to [0, 1], ImageNet-normalised and placed in the top-left
corner of an S x S canvas of zeros. The graph returns two-class logits; the foreground probability is the
channel-1 softmax over the valid (unpadded) region, upsampled bilinearly (``align_corners=False``) to the
original size and thresholded. No hair removal, no morphology.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from PIL import Image

from .segmenter_preprocess import Preprocessing, PreprocessError

LETTERBOX = "letterbox"
LETTERBOX_PATTERN = re.compile(r"^letterbox (\d+) bilinear, pad right-bottom$")
SCALE_STEP = "float32 /255"
NORMALIZE_STEP = "normalize imagenet"
SOFTMAX_ACTIVATION = "softmax over 2 channels, foreground channel 1, valid region only"
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
PIXEL_SCALE = np.float32(255.0)
ROW_CHUNK = 256
N_CLASSES = 2
FOREGROUND = 1


def parse_letterbox(input_block: dict[str, Any], output_block: Any, size: int) -> Preprocessing:
    steps = input_block.get("preprocessing_required")
    if not isinstance(steps, list) or len(steps) != 3:
        raise PreprocessError("letterbox preprocessing must list exactly three steps")
    match = LETTERBOX_PATTERN.fullmatch(steps[0]) if isinstance(steps[0], str) else None
    if match is None or int(match.group(1)) != size or steps[1:] != [SCALE_STEP, NORMALIZE_STEP]:
        raise PreprocessError("letterbox preprocessing contains an unknown, misplaced or mismatched step")
    shape = output_block.get("shape") if isinstance(output_block, dict) else None
    if (not isinstance(output_block, dict) or output_block.get("name") != "logits"
            or not isinstance(shape, list) or shape[1:] != [N_CLASSES, size, size]
            or output_block.get("activation") != SOFTMAX_ACTIVATION):
        raise PreprocessError("letterbox output block does not match the two-class softmax contract")
    return Preprocessing(size=size, hair=None, kind=LETTERBOX)


def letterbox_size(width: int, height: int, size: int) -> tuple[int, int]:
    """Valid (width, height) of the resized image inside the S x S canvas."""
    scale = size / max(width, height)
    return max(1, min(size, round(width * scale))), max(1, min(size, round(height * scale)))


def preprocess_letterbox(rgb: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = rgb.shape[:2]
    valid_w, valid_h = letterbox_size(width, height, size)
    resized = Image.fromarray(rgb, mode="RGB").resize((valid_w, valid_h), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / PIXEL_SCALE
    canvas = np.zeros((1, 3, size, size), dtype=np.float32)
    canvas[0, :, :valid_h, :valid_w] = np.transpose((pixels - IMAGENET_MEAN) / IMAGENET_STD, (2, 0, 1))
    return canvas, (valid_w, valid_h)


def foreground_probability(logits: np.ndarray, valid_wh: tuple[int, int]) -> np.ndarray:
    """Channel-1 softmax of (1, 2, S, S) logits, cropped to the valid region."""
    valid_w, valid_h = valid_wh
    region = np.asarray(logits, dtype=np.float32)[0, :, :valid_h, :valid_w]
    shifted = np.exp(region - region.max(axis=0, keepdims=True))
    return (shifted[FOREGROUND] / shifted.sum(axis=0)).astype(np.float32)


def _axis(out_len: int, in_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = np.float32(in_len) / np.float32(out_len)
    coords = (np.arange(out_len, dtype=np.float32) + np.float32(0.5)) * scale - np.float32(0.5)
    coords = np.clip(coords, np.float32(0.0), np.float32(in_len - 1))
    low = np.floor(coords).astype(np.int64)
    return low, np.minimum(low + 1, in_len - 1), coords - low.astype(np.float32)


def upsample_threshold(probability: np.ndarray, width: int, height: int, threshold: float) -> np.ndarray:
    """Bilinear (align_corners=False) upsample to (height, width) then >= threshold, in row chunks."""
    in_h, in_w = probability.shape
    y0, y1, wy = _axis(height, in_h)
    x0, x1, wx = _axis(width, in_w)
    wx = wx[None, :]
    mask = np.empty((height, width), dtype=bool)
    for start in range(0, height, ROW_CHUNK):
        rows = slice(start, start + ROW_CHUNK)
        top_rows, bottom_rows, weight = probability[y0[rows]], probability[y1[rows]], wy[rows][:, None]
        top = top_rows[:, x0] * (1.0 - wx) + top_rows[:, x1] * wx
        bottom = bottom_rows[:, x0] * (1.0 - wx) + bottom_rows[:, x1] * wx
        mask[rows] = (top * (1.0 - weight) + bottom * weight).astype(np.float32) >= np.float32(threshold)
    return mask


def letterbox_mask(logits: np.ndarray, valid_wh: tuple[int, int], width: int, height: int,
                   threshold: float) -> np.ndarray:
    return upsample_threshold(foreground_probability(logits, valid_wh), width, height, threshold)
