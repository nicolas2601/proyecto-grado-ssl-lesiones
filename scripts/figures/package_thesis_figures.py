"""Package the 17 approved thesis figures as named JPEGs and a vector-only SVG ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from PIL import Image, ImageChops

DATASETS = ("ham10000", "isic2019", "isic2020", "pad_ufes_20")
ORIENTATIONS = ("vertical", "horizontal")
STEMS = (
    *(f"fig_obj1_diagnosis_{dataset}_{orientation}" for dataset in DATASETS for orientation in ORIENTATIONS),
    *(f"fig_obj1_pad_fst_registered_{orientation}" for orientation in ORIENTATIONS),
    *(f"fig_obj1_ita_{dataset}" for dataset in DATASETS[:3]),
    "fig1_ita", "fig_obj1_pad_fst_ita", "fig_isic2018_unet_vs_reference", "fig2_models",
)
ZIP_NAME = "thesis_figures_svg.zip"
MANIFEST_NAME = "thesis_figures_manifest.json"
JPEG_QUALITY = 97
MAX_MEAN_ERROR = 5.0
SVG_TAG = "{http://www.w3.org/2000/svg}svg"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _source_files(source: Path) -> list[tuple[str, Path, Path]]:
    if not source.is_dir():
        raise ValueError(f"Source directory missing: {source}")
    pairs = []
    for stem in STEMS:
        png, svg = source / f"{stem}.png", source / f"{stem}.svg"
        if png.is_symlink() or svg.is_symlink():
            raise ValueError(f"unsafe link in source: {stem}")
        if not png.is_file() or not svg.is_file():
            raise ValueError(f"missing PNG/SVG pair: {stem}")
        try:
            if ET.parse(svg).getroot().tag != SVG_TAG:
                raise ValueError(f"Invalid SVG root: {svg.name}")
        except ET.ParseError as exc:
            raise ValueError(f"Invalid SVG: {svg.name}") from exc
        pairs.append((stem, png, svg))
    return pairs


def _mean_error(original: Image.Image, jpeg: Image.Image) -> float:
    histogram = ImageChops.difference(original, jpeg).histogram()
    return sum(index % 256 * n for index, n in enumerate(histogram)) / (original.width * original.height * 3)


def _jpeg_from_png(png: Path, destination: Path) -> float:
    with Image.open(png) as image:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        original = Image.alpha_composite(background, rgba).convert("RGB")
        original.save(destination, format="JPEG", quality=JPEG_QUALITY, subsampling=0, optimize=True)
    with Image.open(destination) as jpeg:
        error = _mean_error(original, jpeg.convert("RGB"))
    if error > MAX_MEAN_ERROR:
        raise ValueError(f"JPEG differs too much from PNG: {png.name} mean error {error:.2f}")
    return round(error, 3)


def build(source_dir: Path, output_dir: Path) -> dict:
    """Refuse missing pairs and existing destinations before creating any deliverables."""
    source, output = Path(source_dir), Path(output_dir)
    pairs = _source_files(source)
    if output.exists():
        raise FileExistsError(f"Destination already exists: {output}")
    output.mkdir(parents=True)
    figures = []
    for stem, png, svg in pairs:
        jpg = output / f"{stem}.jpg"
        error = _jpeg_from_png(png, jpg)
        figures.append({"stem": stem, "png_sha256": _sha256(png.read_bytes()),
                        "svg_sha256": _sha256(svg.read_bytes()), "jpg_sha256": _sha256(jpg.read_bytes()),
                        "jpeg_mean_error": error})
    archive_path = output / ZIP_NAME
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for stem, _, svg in pairs:
            archive.writestr(f"{stem}.svg", svg.read_bytes())
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None or archive.namelist() != [f"{stem}.svg" for stem in STEMS]:
            raise ValueError("SVG archive integrity or names differ")
    result = {"figures": figures, "zip_sha256": _sha256(archive_path.read_bytes())}
    (output / MANIFEST_NAME).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.source_dir, args.output_dir)
    print(f"{len(report['figures'])} named JPEGs, {ZIP_NAME} SHA256 {report['zip_sha256']}")
