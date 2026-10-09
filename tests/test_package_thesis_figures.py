"""The WhatsApp JPEGs and SVG archive retain one vetted name per figure."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import package_thesis_figures as package  # noqa: E402


def _staged(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    for stem in package.STEMS:
        im = Image.new("RGB", (32, 24), "#fbfbfa")
        for x in range(6, 20):
            for y in range(4, 12):
                im.putpixel((x, y), (63, 104, 165))
        im.save(source / f"{stem}.png")
        (source / f"{stem}.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="24"></svg>', encoding="utf-8"
        )
    return source


def test_packages_seventeen_named_jpegs_and_only_matching_svgs(tmp_path):
    source = _staged(tmp_path)
    result = package.build(source, tmp_path / "deliver")
    assert len(result["figures"]) == len(package.STEMS) == 17
    names = [row["stem"] for row in result["figures"]]
    assert names == list(package.STEMS)
    for name in names:
        with Image.open(tmp_path / "deliver" / f"{name}.jpg") as image:
            assert image.format == "JPEG" and image.size == (32, 24)
        assert next(row for row in result["figures"] if row["stem"] == name)["jpeg_mean_error"] < 5
    with zipfile.ZipFile(tmp_path / "deliver" / package.ZIP_NAME) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == [f"{name}.svg" for name in package.STEMS]
        assert all(archive.read(f"{name}.svg") == (source / f"{name}.svg").read_bytes() for name in names)
    saved = json.loads((tmp_path / "deliver" / package.MANIFEST_NAME).read_text())
    assert saved == result
    assert not any(path.suffix == ".png" for path in (tmp_path / "deliver").iterdir())


def test_missing_or_invalid_svg_aborts_before_writing(tmp_path):
    source = _staged(tmp_path)
    (source / f"{package.STEMS[0]}.svg").unlink()
    with pytest.raises(ValueError, match="missing"):
        package.build(source, tmp_path / "deliver")
    assert not (tmp_path / "deliver").exists()
    (source / f"{package.STEMS[0]}.svg").write_text("not xml", encoding="utf-8")
    with pytest.raises(ValueError, match="SVG"):
        package.build(source, tmp_path / "deliver")


def test_existing_destination_or_unsafe_link_is_not_overwritten(tmp_path):
    source = _staged(tmp_path)
    destination = tmp_path / "deliver"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        package.build(source, destination)
    destination.rmdir()
    (source / f"{package.STEMS[0]}.svg").unlink()
    (source / f"{package.STEMS[0]}.svg").symlink_to(tmp_path / "outside.svg")
    with pytest.raises(ValueError, match="unsafe"):
        package.build(source, destination)
