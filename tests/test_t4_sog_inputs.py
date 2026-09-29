"""docs/31: the input (identity or Shades of Gray p = 6) is part of every cache key.

Arrays and unit results with and without SoG must never be mixed: the decoded
array's file name carries the input, and so does the run fingerprint that
``v71_run._check_cached`` compares. Identity keeps the v7.1 names and
fingerprint byte for byte, so nothing already on the pod is invalidated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from dermssl import feature_cache as fc  # noqa: E402
from dermssl import t4_sog  # noqa: E402
from dermssl.pretrain import color_constancy as cc  # noqa: E402

FIT, HOLD = ["ISIC_1", "ISIC_2"], ["ISIC_3"]


def test_identity_keeps_the_v71_fingerprint():
    assert t4_sog.run_fingerprint(FIT, HOLD, t4_sog.IDENTITY) == fc.fingerprint_ids([*FIT, "|", *HOLD])


def test_each_input_has_its_own_fingerprint():
    prints = {name: t4_sog.run_fingerprint(FIT, HOLD, name) for name in t4_sog.INPUTS}
    assert len(set(prints.values())) == len(t4_sog.INPUTS)


def test_the_registered_input_is_sog_with_p6():
    assert t4_sog.INPUTS == ("identity", "sog_p6")
    img = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    assert np.array_equal(t4_sog.preprocess(img, "sog_p6"), cc.shades_of_gray_uint8(img, p=6))
    assert t4_sog.preprocess(img, "identity") is img


def test_an_unknown_input_is_refused():
    with pytest.raises(ValueError, match="entrada"):
        t4_sog.run_fingerprint(FIT, HOLD, "sog_p5")
    with pytest.raises(ValueError, match="entrada"):
        t4_sog.preprocess(np.zeros((2, 2, 3), np.uint8), "gray_world")


@pytest.fixture()
def images(tmp_path):
    pytest.importorskip("torchvision")
    from PIL import Image

    rng = np.random.default_rng(1)
    index = {}
    for image_id in FIT:
        pixels = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
        pixels[..., 0] = np.clip(pixels[..., 0].astype(int) + 60, 0, 255)  # a red cast to remove
        path = tmp_path / f"{image_id}.png"
        Image.fromarray(pixels).save(path)
        index[image_id] = path
    return index


def test_the_store_applies_sog_after_the_bank_geometry(images, tmp_path):
    import v71_io

    plain = v71_io.ImageStore(tmp_path / "in", images, workers=1).array(FIT, 28, "fit")
    sog = v71_io.ImageStore(tmp_path / "in", images, workers=1, input_name="sog_p6").array(FIT, 28, "fit")
    assert not np.array_equal(plain, sog)
    for k in range(len(FIT)):
        assert np.array_equal(sog[k], cc.shades_of_gray_uint8(np.asarray(plain[k]), p=6))


def test_the_store_never_reads_the_other_inputs_file(images, tmp_path):
    import v71_io

    v71_io.ImageStore(tmp_path / "in", images, workers=1).array(FIT, 28, "fit")
    v71_io.ImageStore(tmp_path / "in", images, workers=1, input_name="sog_p6").array(FIT, 28, "fit")
    names = sorted(p.name for p in (tmp_path / "in").glob("*.npy"))
    assert names == sorted([f"fit_28_{fc.fingerprint_ids(FIT)}.npy",
                            f"fit_sog_p6_28_{fc.fingerprint_ids(FIT)}.npy"])
