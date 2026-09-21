"""Tests for the one-time image cache builder (no masks, no labels)."""

import json

import numpy as np
import pytest
from PIL import Image

import seg_paths  # noqa: F401
import cache_io
import prepare_cache


def _make_pool(tmp_path, count=6, name="ham"):
    root = tmp_path / name
    root.mkdir()
    for index in range(count):
        Image.new("RGB", (20, 12), (index * 20, 50, 90)).save(root / f"ISIC_{index:04d}.jpg")
    return root


def _args(tmp_path, *extra):
    return prepare_cache.parse_args(
        ["--output-dir", str(tmp_path / "cache"), "--resolution", "8", *extra]
    )


def test_cache_is_built_with_manifest_and_info(tmp_path):
    root = _make_pool(tmp_path)
    args = _args(tmp_path, "--root", f"ham10000={root}", "--holdout-fraction", "0.2")
    info = prepare_cache.run(args)

    images = cache_io.open_cache(tmp_path / "cache" / "images.npy")
    rows = cache_io.read_manifest(tmp_path / "cache" / "manifest.csv")
    assert images.shape == (6, 8, 8, 3) and images.dtype == np.uint8
    assert len(rows) == 6
    assert sum(r["split"] == "val" for r in rows) == 1
    assert info["count"] == 6 and info["resolution"] == 8
    saved = json.loads((tmp_path / "cache" / "cache_info.json").read_text(encoding="utf-8"))
    assert saved["reads_masks_or_labels"] is False
    assert saved["per_dataset"] == {"ham10000": 6}


def test_cache_rows_match_the_resized_source_images(tmp_path):
    root = _make_pool(tmp_path, count=3)
    prepare_cache.run(_args(tmp_path, "--root", f"ham10000={root}"))
    images = cache_io.open_cache(tmp_path / "cache" / "images.npy")
    rows = cache_io.read_manifest(tmp_path / "cache" / "manifest.csv")
    for row in rows:
        expected = cache_io.load_resized_rgb(row["path"], 8)
        assert np.array_equal(images[row["index"]], expected)


def test_limit_caps_the_number_of_images(tmp_path):
    root = _make_pool(tmp_path)
    info = prepare_cache.run(_args(tmp_path, "--root", f"ham10000={root}", "--limit", "4"))
    assert info["count"] == 4


def test_duplicate_ids_across_roots_are_reported(tmp_path):
    first = _make_pool(tmp_path, count=3, name="a")
    second = _make_pool(tmp_path, count=3, name="b")
    info = prepare_cache.run(
        _args(tmp_path, "--root", f"ham10000={first}", "--root", f"isic2019={second}")
    )
    assert info["count"] == 3 and info["duplicates_dropped"] == 3


def test_synthetic_mode_needs_no_roots(tmp_path):
    info = prepare_cache.run(_args(tmp_path, "--synthetic", "10"))
    assert info["count"] == 10
    assert info["per_dataset"] == {"synthetic": 10}


def test_no_images_is_an_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        prepare_cache.run(_args(tmp_path, "--root", f"x={empty}"))


def test_roots_or_synthetic_is_required(tmp_path):
    with pytest.raises(SystemExit):
        prepare_cache.run(_args(tmp_path))


def test_cache_also_writes_the_field_of_view_mask(tmp_path):
    root = _make_pool(tmp_path)
    info = prepare_cache.run(_args(tmp_path, "--root", f"ham10000={root}"))

    field = cache_io.open_cache(tmp_path / "cache" / "field.npy")
    assert field.shape == (6, 8, 8) and field.dtype == np.uint8
    assert set(np.unique(np.asarray(field))) <= {0, 1}
    stats = json.loads((tmp_path / "cache" / "field_stats.json").read_text(encoding="utf-8"))
    assert stats["images"] == 6
    assert stats["black_threshold"] == prepare_cache.field_roi.DEFAULT_BLACK_THRESHOLD
    assert info["field"]["images"] == 6
    assert info["reads_masks_or_labels"] is False


def test_field_flags_are_forwarded_and_can_be_disabled(tmp_path):
    root = _make_pool(tmp_path)
    info = prepare_cache.run(
        _args(
            tmp_path,
            "--root",
            f"ham10000={root}",
            "--field-black-threshold",
            "0.2",
            "--field-margin-fraction",
            "0.1",
        )
    )
    assert info["field"]["black_threshold"] == 0.2
    assert info["field"]["margin_fraction"] == 0.1

    other = tmp_path / "second"
    info = prepare_cache.run(
        prepare_cache.parse_args(
            ["--output-dir", str(other / "cache"), "--resolution", "8", "--root", f"ham10000={root}", "--no-field"]
        )
    )
    assert info["field"] is None
    assert not (other / "cache" / "field.npy").exists()
