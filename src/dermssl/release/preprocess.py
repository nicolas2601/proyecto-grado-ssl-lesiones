"""Evaluation preprocessing of the v7 classifier, in numpy and PIL only.

Reproduces ``torchvision`` ``Resize(255, BILINEAR)`` (shorter side to 255,
aspect kept) + ``CenterCrop(224)`` + ``ToTensor`` + ImageNet ``Normalize``,
the transform the evaluated features were extracted with
(``scripts/cluster_frozen_bank.py::eval_transform``). Standalone on purpose:
the file is copied verbatim into the release, so it imports neither torch nor
the ``dermssl`` package.

    from preprocess import preprocess_batch
    batch = preprocess_batch(["lesion.jpg"])   # float32, shape (1, 3, 224, 224)
"""

from __future__ import annotations

import numpy as np
from PIL import Image

RESIZE = 255  # int(224 * 1.14), as in the bank
CROP = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PREPROCESSING = {
    "input": "RGB image, any size",
    "resize_shorter_side": RESIZE,
    "interpolation": "PIL.Image.BILINEAR",
    "center_crop": CROP,
    "crop_offset": "int(round((size - 224) / 2))",
    "scale": "uint8 / 255 in float32",
    "mean": list(MEAN),
    "std": list(STD),
    "layout": "NCHW float32",
}


def resized_size(width: int, height: int, size: int = RESIZE) -> tuple[int, int]:
    """torchvision ``Resize(int)``: the shorter side becomes ``size``, aspect kept."""
    if width <= height:
        return size, int(size * height / width)
    return int(size * width / height), size


def _center_crop(image: Image.Image, size: int = CROP) -> Image.Image:
    """torchvision ``CenterCrop``: offsets rounded with ``int(round(margin / 2))``."""
    width, height = image.size
    top = int(round((height - size) / 2.0))
    left = int(round((width - size) / 2.0))
    return image.crop((left, top, left + size, top + size))


def preprocess_image(image: Image.Image) -> np.ndarray:
    """One PIL image -> float32 ``(3, 224, 224)`` normalised array."""
    rgb = image.convert("RGB")
    resized = rgb.resize(resized_size(*rgb.size), Image.BILINEAR)
    array = np.asarray(_center_crop(resized), dtype=np.float32) / np.float32(255)
    array = (array - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def preprocess_batch(paths) -> np.ndarray:
    """Image files -> float32 ``(N, 3, 224, 224)`` batch."""
    arrays = []
    for path in paths:
        with Image.open(path) as image:
            arrays.append(preprocess_image(image))
    return np.stack(arrays)
