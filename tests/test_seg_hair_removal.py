"""S2 (docs/26 3.4, docs/27 Fase 2): black-hat + TELEA hair removal inside the cache stage."""

import numpy as np
import pytest

import seg_paths  # noqa: F401

cv2 = pytest.importorskip("cv2")
import hair_removal  # noqa: E402

SKIN = (200, 150, 130)


def _skin(size=64):
    return np.full((size, size, 3), SKIN, dtype=np.uint8)


def _with_hair(image):
    hairy = image.copy()
    cv2.line(hairy, (0, 10), (63, 50), (20, 15, 10), 1)
    cv2.line(hairy, (5, 60), (60, 5), (25, 20, 15), 1)
    return hairy


def test_thin_dark_lines_are_detected_and_filled():
    hairy = _with_hair(_skin())
    clean, fraction = hair_removal.remove_hair(hairy)
    line = np.any(hairy != _skin(), axis=2)
    assert 0 < fraction < 0.2
    before = np.abs(hairy[line].astype(int) - SKIN).mean()
    after = np.abs(clean[line].astype(int) - SKIN).mean()
    assert after < before / 4


def test_hairless_image_is_returned_unchanged():
    image = _skin()
    clean, fraction = hair_removal.remove_hair(image)
    assert fraction == 0.0
    assert np.array_equal(clean, image)


def test_a_large_dark_lesion_is_not_treated_as_hair():
    image = _skin()
    cv2.circle(image, (32, 32), 14, (60, 35, 30), -1)
    clean, fraction = hair_removal.remove_hair(image)
    assert fraction < 0.02
    assert np.abs(clean[32, 32].astype(int) - (60, 35, 30)).max() <= 5


def test_input_is_not_modified_in_place():
    hairy = _with_hair(_skin())
    copy = hairy.copy()
    hair_removal.remove_hair(hairy)
    assert np.array_equal(hairy, copy)


def test_cache_stage_applies_it_and_records_the_setting(tmp_path):
    import json

    from PIL import Image

    import cache_io
    import prepare_cache

    root = tmp_path / "ham"
    root.mkdir()
    Image.fromarray(_with_hair(_skin())).save(root / "ISIC_0001.png")
    Image.fromarray(_skin()).save(root / "ISIC_0002.png")
    built = {}
    for flag in ([], ["--hair-removal"]):
        out = tmp_path / ("clean" if flag else "plain")
        prepare_cache.run(prepare_cache.parse_args(
            ["--output-dir", str(out), "--resolution", "64", "--root", f"ham10000={root}", *flag]
        ))
        built[bool(flag)] = (np.asarray(cache_io.open_cache(out / "images.npy")),
                             json.loads((out / "cache_info.json").read_text()))
    plain, clean = built[False][0], built[True][0]
    assert not np.array_equal(plain, clean)
    assert built[False][1]["hair_removal"] is None
    info = built[True][1]["hair_removal"]
    assert info["kernel"] == hair_removal.KERNEL and 0 < info["mean_fraction"] < 0.2
