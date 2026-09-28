"""The ITA can read the cache pixels (images.npy) instead of the source files.

The cache is where S2(a) removes hair (docs/26 §3.4); the source files never
change, so without this option a hair-free cache cannot reach the estimate.
"""

import numpy as np
import pytest
from PIL import Image

import seg_paths  # noqa: F401
import fst_pseudo_manifest as fpm
from dermssl import fitzpatrick

SOURCE_RGB = (180, 130, 100)
CACHE_RGB = (120, 80, 60)


def _write_fixture(tmp_path):
    image_path = tmp_path / "images" / "img-a.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), SOURCE_RGB).save(image_path)
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        f"index,dataset,image_id,path,split\n0,ham10000,img-a,{image_path},train\n",
        encoding="utf-8",
    )
    masks = np.zeros((1, 8, 8), dtype=np.uint8)
    masks[0, 2:5, 2:5] = 1
    masks_path = tmp_path / "pseudo_masks.npy"
    np.save(masks_path, masks)
    return manifest_path, masks_path


def _write_cache_images(tmp_path, shape=(1, 8, 8, 3), dtype=np.uint8):
    images = np.zeros(shape, dtype=dtype)
    images[...] = CACHE_RGB
    path = tmp_path / "images.npy"
    np.save(path, images)
    return path


def _run(tmp_path, manifest_path, masks_path, *extra):
    return fpm.run(
        fpm.parse_args(
            ["--manifest", str(manifest_path), "--pseudo-masks", str(masks_path),
             "--output-json", str(tmp_path / "fst.json"), *extra]
        )
    )


def _expected_ita(rgb, masks_path):
    pseudo = np.load(masks_path)[0]
    image = np.full((8, 8, 3), rgb, dtype=np.uint8).astype(np.float32) / 255.0
    return fitzpatrick.estimate_fst(image, mask=pseudo == 1, border_mask=pseudo == 255).ita


def test_with_images_the_estimate_reads_the_cache_pixels(tmp_path):
    manifest_path, masks_path = _write_fixture(tmp_path)
    images_path = _write_cache_images(tmp_path)
    output = _run(tmp_path, manifest_path, masks_path, "--images", str(images_path))
    assert output["records"][0]["ITA"] == _expected_ita(CACHE_RGB, masks_path)
    assert output["images_used"] == str(images_path)


def test_without_images_the_source_file_is_still_read(tmp_path):
    manifest_path, masks_path = _write_fixture(tmp_path)
    output = _run(tmp_path, manifest_path, masks_path)
    assert output["records"][0]["ITA"] == _expected_ita(SOURCE_RGB, masks_path)
    assert output["images_used"] is None


def test_cache_images_that_do_not_match_the_masks_are_rejected(tmp_path):
    manifest_path, masks_path = _write_fixture(tmp_path)
    images_path = _write_cache_images(tmp_path, shape=(1, 4, 4, 3))
    with pytest.raises(ValueError, match="images"):
        _run(tmp_path, manifest_path, masks_path, "--images", str(images_path))


def test_cache_images_must_be_uint8(tmp_path):
    manifest_path, masks_path = _write_fixture(tmp_path)
    images_path = _write_cache_images(tmp_path, dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        _run(tmp_path, manifest_path, masks_path, "--images", str(images_path))
