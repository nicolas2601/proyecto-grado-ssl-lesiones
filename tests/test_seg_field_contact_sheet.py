"""Visual audit sheet of the dermoscope field: grouping, sampling and drawing."""

import json

import numpy as np
import pytest
from PIL import Image

import seg_paths  # noqa: F401
import cache_io
import field_contact_sheet as sheet
import field_roi

RESOLUTION = 32


def _vignetted(radius=13, inside=200, outside=10):
    yy, xx = np.mgrid[:RESOLUTION, :RESOLUTION]
    center = (RESOLUTION - 1) / 2.0
    circle = (yy - center) ** 2 + (xx - center) ** 2 <= radius**2
    image = np.full((RESOLUTION, RESOLUTION, 3), outside, dtype=np.uint8)
    image[circle] = inside
    return image


def _full_frame():
    return np.full((RESOLUTION, RESOLUTION, 3), 180, dtype=np.uint8)


def _build_cache(tmp_path, circular=3, full=2):
    """A cache with two datasets: one vignetted, one already cropped."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    images, records = [], []
    for index in range(circular):
        images.append(_vignetted())
        records.append(cache_io.ImageRecord("dermato", f"VIG_{index}", f"dermato/VIG_{index}.png"))
    for index in range(full):
        images.append(_full_frame())
        records.append(cache_io.ImageRecord("recortado", f"CROP_{index}", f"r/CROP_{index}.png"))
    stack = (
        np.stack(images)
        if images
        else np.zeros((0, RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
    )

    cache = cache_io.create_cache(cache_dir / "images.npy", stack.shape, np.uint8)
    cache[:] = stack
    cache.flush()
    field = cache_io.create_cache(
        cache_dir / "field.npy", (len(stack), RESOLUTION, RESOLUTION), np.uint8
    )
    field_roi.compute_field_cache(stack, field)
    field.flush()
    cache_io.write_manifest(cache_dir / "manifest.csv", records, ["train"] * len(records))
    return cache_dir


def _run(tmp_path, **kwargs):
    cache_dir = kwargs.pop("cache_dir", None) or _build_cache(tmp_path)
    output = tmp_path / "field.png"
    argv = ["--cache-dir", str(cache_dir), "--output", str(output)]
    for name, value in kwargs.items():
        argv += [f"--{name.replace('_', '-')}", str(value)]
    payload = sheet.build_sheet(sheet.parse_args(argv))
    return output, payload


def test_the_method_follows_the_valid_fraction():
    assert sheet.method_of(1.0) == "full_frame"
    assert sheet.method_of(0.9995) == "full_frame"
    assert sheet.method_of(0.5) == "circular"


def test_one_group_per_dataset_and_method(tmp_path):
    _, payload = _run(tmp_path)
    assert sorted(payload["groups"]) == ["dermato/circular", "recortado/full_frame"]
    assert payload["groups"]["dermato/circular"]["sampled"] == ["VIG_0", "VIG_1", "VIG_2"]
    assert payload["groups"]["recortado/full_frame"]["n"] == 2


def test_the_sheet_has_one_row_per_group(tmp_path):
    output, payload = _run(tmp_path, per_group=4)
    with Image.open(output) as image:
        width, height = image.size
    assert width == 4 * RESOLUTION
    assert height == len(payload["groups"]) * (RESOLUTION + sheet.CAPTION_HEIGHT)


def test_per_group_caps_the_sample_and_keeps_it_reproducible(tmp_path):
    cache_dir = _build_cache(tmp_path, circular=6, full=1)
    first = _run(tmp_path, cache_dir=cache_dir, per_group=2, seed=7)[1]
    second = _run(tmp_path, cache_dir=cache_dir, per_group=2, seed=7)[1]
    sampled = first["groups"]["dermato/circular"]["sampled"]
    assert len(sampled) == 2
    assert second["groups"]["dermato/circular"]["sampled"] == sampled
    other = _run(tmp_path, cache_dir=cache_dir, per_group=2, seed=8)[1]
    assert other["groups"]["dermato/circular"]["n"] == 6


def test_the_field_boundary_is_painted_in_red(tmp_path):
    output, _ = _run(tmp_path)
    with Image.open(output) as image:
        painted = np.asarray(image.convert("RGB"))
    red = (painted[..., 0] == 255) & (painted[..., 1] == 0) & (painted[..., 2] == 0)
    assert red.sum() > RESOLUTION  # a full ring, not a stray pixel


def test_the_boundary_is_two_pixels_thick_and_inside_the_field():
    valid = np.zeros((16, 16), dtype=bool)
    valid[4:12, 4:12] = True
    ring = sheet.boundary(valid)
    assert ring[4, 4] and ring[5, 5]
    assert not ring[6, 6]
    assert not ring[3, 3]


def test_a_full_frame_keeps_its_boundary_at_the_image_border():
    ring = sheet.boundary(np.ones((8, 8), dtype=bool))
    assert ring[0, 0] and ring[1, 1] and not ring[4, 4]


def test_the_caption_names_the_dataset_the_image_and_the_fraction():
    entry = {
        "dataset": "dermato",
        "image_id": "VIG_0",
        "method": "circular",
        "valid_fraction": 0.4567,
    }
    assert sheet.caption(entry) == "dermato VIG_0 circular valid=0.457"


def test_the_json_sits_next_to_the_png_and_records_the_run(tmp_path):
    output, payload = _run(tmp_path, seed=3, per_group=5)
    saved = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved == payload
    assert saved["seed"] == 3 and saved["per_group"] == 5


def test_the_sheet_reads_the_cache_only(tmp_path):
    source = (seg_paths.COMUN / "datos" / "field_contact_sheet.py").read_text(encoding="utf-8")
    assert "Image.open(" not in source
    for forbidden in ("mask", "label", "diagnos"):
        assert forbidden not in source.lower()


def test_an_empty_cache_is_rejected_instead_of_writing_an_empty_sheet(tmp_path):
    cache_dir = _build_cache(tmp_path, circular=0, full=0)
    with pytest.raises(ValueError, match="sin imagenes"):
        _run(tmp_path, cache_dir=cache_dir)
