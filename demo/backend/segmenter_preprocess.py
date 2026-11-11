"""Segmenter pre- and postprocessing, as the release scripts run them.

Preprocessing comes only from the manifest ``input`` block. Known steps are: PIL bilinear resize to S,
hair removal (cv2 black-hat + Telea inpaint, shaped=False, parameters parsed from the step text), and
float32 / 255. Any other step fails closed. Postprocessing is sigmoid -> uint8 -> PIL bilinear resize
-> /255 -> >= threshold, with no morphology and no largest-component step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

RESIZE_PATTERN = re.compile(r"^resize (\d+)(?:x\1)? bilinear$")
HAIR_PATTERN = re.compile(
    r"^hair removal: cv2 black-hat \(kernel (\d+) ellipse, threshold (\d+)\) \+ Telea inpaint radius (\d+), "
    r"shaped=False$"
)
SCALE_STEP = "float32 /255"
LOGIT_CLIP = 60.0
PROBABILITY_SCALE = 255.0


class PreprocessError(ValueError):
    """The manifest declares a preprocessing contract this runtime does not implement."""


@dataclass(frozen=True)
class HairParams:
    kernel: int
    threshold: int
    radius: int


@dataclass(frozen=True)
class Preprocessing:
    size: int
    hair: HairParams | None


def parse_preprocessing(input_block: dict[str, Any], size: int) -> Preprocessing:
    steps = input_block.get("preprocessing_required")
    if not isinstance(steps, list) or not all(isinstance(step, str) for step in steps):
        raise PreprocessError("segmenter input.preprocessing_required must be a list of strings")
    if input_block.get("resize") != "bilinear":
        raise PreprocessError("segmenter input resize must be bilinear")
    resize_steps = [step for step in steps if step.startswith("resize ")]
    hair_steps = [step for step in steps if step.startswith("hair removal")]
    scale_steps = [step for step in steps if step == SCALE_STEP]
    known = len(resize_steps) + len(hair_steps) + len(scale_steps)
    if known != len(steps) or len(resize_steps) != 1 or len(scale_steps) != 1 or len(hair_steps) > 1:
        raise PreprocessError("segmenter preprocessing contains an unknown or repeated step")
    resize = RESIZE_PATTERN.fullmatch(resize_steps[0])
    if resize is None or int(resize.group(1)) != size:
        raise PreprocessError("segmenter resize step does not match the input size")
    return Preprocessing(size=size, hair=_parse_hair(hair_steps[0]) if hair_steps else None)


def _parse_hair(step: str) -> HairParams:
    match = HAIR_PATTERN.fullmatch(step)
    if match is None:
        raise PreprocessError("segmenter hair removal step is not one this runtime implements")
    kernel, threshold, radius = (int(group) for group in match.groups())
    return HairParams(kernel=kernel, threshold=threshold, radius=radius)


def require_opencv(config: Preprocessing) -> None:
    if config.hair is None:
        return
    try:
        import cv2  # noqa: F401 - availability check only
    except ImportError as exc:
        raise PreprocessError("OpenCV is required by the segmenter hair removal step") from exc


def remove_hair(rgb: np.ndarray, params: HairParams) -> np.ndarray:
    import cv2

    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.kernel, params.kernel))
    response = cv2.morphologyEx(grey, cv2.MORPH_BLACKHAT, element)
    mask = (response > params.threshold).astype(np.uint8)
    if not mask.any():
        return rgb.copy()
    return cv2.inpaint(np.ascontiguousarray(rgb), mask, params.radius, cv2.INPAINT_TELEA)


def preprocess(rgb: np.ndarray, config: Preprocessing) -> np.ndarray:
    resized = Image.fromarray(rgb, mode="RGB").resize((config.size, config.size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.uint8)
    if config.hair is not None:
        pixels = remove_hair(pixels, config.hair)
    chw = np.transpose(pixels.astype(np.float32) / np.float32(PROBABILITY_SCALE), (2, 0, 1))
    return np.ascontiguousarray(chw[None], dtype=np.float32)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -LOGIT_CLIP, LOGIT_CLIP)))).astype(np.float32)


def probability_to_mask(logits: np.ndarray, width: int, height: int, threshold: float) -> np.ndarray:
    """Reference postprocess: sigmoid, uint8 (truncating), bilinear to the original size, /255, >= threshold."""
    probability = (sigmoid(logits) * PROBABILITY_SCALE).astype(np.uint8)
    full = Image.fromarray(probability, mode="L").resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(full, dtype=np.float32) / np.float32(PROBABILITY_SCALE) >= threshold
