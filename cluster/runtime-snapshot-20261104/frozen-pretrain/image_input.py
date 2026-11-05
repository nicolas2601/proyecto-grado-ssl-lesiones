"""Canonical orientation shared by training, reference extraction and phone inference."""
from PIL import Image, ImageOps


def load_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")
