"""Tests for the pseudo-mask generator script (saliency array -> uint8 masks)."""

import json

import numpy as np
import pytest

import seg_paths  # noqa: F401
import cache_io
import make_pseudo_masks as mpm


def _blob(size=32, radius=8.0):
    yy, xx = np.mgrid[:size, :size]
    distance = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    return np.clip(1 - distance / (radius * 1.6), 0, 1).astype(np.float16)


def _write_saliency(tmp_path, arrays):
    path = tmp_path / "saliency.npy"
    np.save(path, np.stack(arrays))
    return path


def _run(tmp_path, saliency_path, *extra):
    args = mpm.parse_args(
        ["--saliency", str(saliency_path), "--output-dir", str(tmp_path / "out"), *extra]
    )
    return mpm.run(args)


def test_ccam_source_orients_and_normalizes(tmp_path):
    saliency = _write_saliency(tmp_path, [_blob(), 1.0 - _blob()])
    stats = _run(tmp_path, saliency, "--source", "ccam")
    masks = cache_io.open_cache(tmp_path / "out" / "pseudo_masks.npy")
    assert masks.shape == (2, 32, 32) and masks.dtype == np.uint8
    assert masks[0, 16, 16] == cache_io.FOREGROUND
    assert masks[1, 16, 16] == cache_io.FOREGROUND  # sign fixed by orientation
    assert stats["excluded"] == 0


def test_student_source_keeps_probabilities_as_they_are(tmp_path):
    flat = np.full((32, 32), 0.5, dtype=np.float16)
    saliency = _write_saliency(tmp_path, [_blob(), flat])
    stats = _run(tmp_path, saliency, "--source", "student")
    masks = cache_io.open_cache(tmp_path / "out" / "pseudo_masks.npy")
    assert masks[0, 16, 16] == cache_io.FOREGROUND
    assert np.all(masks[1] == cache_io.IGNORE)
    assert stats["excluded"] == 1


def test_stats_file_records_thresholds_and_source(tmp_path):
    saliency = _write_saliency(tmp_path, [_blob()])
    _run(tmp_path, saliency, "--source", "ccam", "--low", "0.3", "--high", "0.7")
    saved = json.loads((tmp_path / "out" / "pseudo_masks_stats.json").read_text(encoding="utf-8"))
    assert saved["source"] == "ccam" and saved["low"] == 0.3 and saved["high"] == 0.7
    assert saved["reads_masks_or_labels"] is False


def test_invalid_thresholds_fail_with_a_clear_exit(tmp_path):
    saliency = _write_saliency(tmp_path, [_blob()])
    with pytest.raises(SystemExit):
        _run(tmp_path, saliency, "--low", "0.8", "--high", "0.2")


def _write_field(tmp_path, size=32, radius=13):
    yy, xx = np.mgrid[:size, :size]
    center = (size - 1) / 2.0
    disk = (yy - center) ** 2 + (xx - center) ** 2 <= radius**2
    path = tmp_path / "field.npy"
    np.save(path, np.stack([disk, disk]).astype(np.uint8))
    return path


def test_the_field_restricts_candidates_and_is_recorded(tmp_path):
    saliency = _write_saliency(tmp_path, [_blob(), _blob()])
    field_path = _write_field(tmp_path)
    stats = _run(tmp_path, saliency, "--source", "student", "--field", str(field_path))
    masks = cache_io.open_cache(tmp_path / "out" / "pseudo_masks.npy")
    assert masks[0, 16, 16] == cache_io.FOREGROUND
    assert masks[0, 0, 0] == cache_io.BACKGROUND  # outside the field
    saved = json.loads((tmp_path / "out" / "pseudo_masks_stats.json").read_text(encoding="utf-8"))
    assert saved["field"] == str(field_path)
    assert saved["field_used"] is True
    assert stats["excluded_by_area"] == 0


def test_the_area_cap_excludes_masks_that_cover_the_whole_field(tmp_path):
    saturated = np.ones((32, 32), dtype=np.float16)
    saliency = _write_saliency(tmp_path, [saturated, saturated])
    field_path = _write_field(tmp_path)
    stats = _run(
        tmp_path,
        saliency,
        "--source",
        "student",
        "--field",
        str(field_path),
        "--max-area-fraction",
        "0.6",
    )
    assert stats["excluded_by_area"] == 2
    saved = json.loads((tmp_path / "out" / "pseudo_masks_stats.json").read_text(encoding="utf-8"))
    assert saved["max_area_fraction"] == 0.6


def test_without_a_field_the_stats_say_so(tmp_path):
    saliency = _write_saliency(tmp_path, [_blob()])
    stats = _run(tmp_path, saliency, "--source", "student")
    assert stats["field_used"] is False
    saved = json.loads((tmp_path / "out" / "pseudo_masks_stats.json").read_text(encoding="utf-8"))
    assert saved["field"] is None
    assert saved["max_area_fraction"] == mpm.um.DEFAULT_MAX_AREA_FRACTION


def _write_images(tmp_path, count=2, size=32, box=(15, 18)):
    """Flat skin with a violet ink square, in the ``(N, H, W, 3)`` cache layout."""
    images = np.full((count, size, size, 3), (200, 150, 140), dtype=np.uint8)
    low, high = box
    images[:, low:high, low:high] = (90, 60, 160)
    path = tmp_path / "images.npy"
    np.save(path, images)
    return path


def test_the_image_cache_turns_ink_marks_into_ignored_pixels(tmp_path):
    saliency = _write_saliency(tmp_path, [_blob(), _blob()])
    images = _write_images(tmp_path)
    stats = _run(tmp_path, saliency, "--source", "student", "--images", str(images))
    masks = cache_io.open_cache(tmp_path / "out" / "pseudo_masks.npy")
    assert stats["excluded"] == 0
    assert masks[0, 16, 16] == cache_io.IGNORE
    assert masks[0, 16, 20] == cache_io.FOREGROUND
    assert stats["mean_artifact_fraction"] > 0.0
    saved = json.loads((tmp_path / "out" / "pseudo_masks_stats.json").read_text(encoding="utf-8"))
    assert saved["artifact_suppression"] is True
    assert saved["images"] == str(images)
    assert saved["reads_masks_or_labels"] is False


def test_artifact_suppression_can_be_turned_off_from_the_command_line(tmp_path):
    saliency = _write_saliency(tmp_path, [_blob(), _blob()])
    images = _write_images(tmp_path)
    stats = _run(
        tmp_path, saliency, "--source", "student",
        "--images", str(images), "--no-artifact-suppression",
    )
    masks = cache_io.open_cache(tmp_path / "out" / "pseudo_masks.npy")
    assert masks[0, 16, 16] == cache_io.FOREGROUND
    assert stats["mean_artifact_fraction"] == 0.0
    saved = json.loads((tmp_path / "out" / "pseudo_masks_stats.json").read_text(encoding="utf-8"))
    assert saved["artifact_suppression"] is False


def test_without_an_image_cache_nothing_is_suppressed(tmp_path):
    saliency = _write_saliency(tmp_path, [_blob()])
    stats = _run(tmp_path, saliency, "--source", "student")
    assert stats["mean_artifact_fraction"] == 0.0
    saved = json.loads((tmp_path / "out" / "pseudo_masks_stats.json").read_text(encoding="utf-8"))
    assert saved["images"] is None and saved["artifact_suppression"] is False


def test_an_image_cache_that_does_not_match_the_saliency_is_rejected(tmp_path):
    saliency = _write_saliency(tmp_path, [_blob(), _blob()])
    images = _write_images(tmp_path, count=1)
    with pytest.raises(SystemExit):
        _run(tmp_path, saliency, "--source", "student", "--images", str(images))
