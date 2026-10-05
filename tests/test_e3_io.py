"""The real-data adapters of e3_io on a synthetic jpg, a synthetic mask array and a stub extractor."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import e3_io  # noqa: E402


def write_case(tmp_path, mask_shape=(40, 60)):
    Image.fromarray(np.full((40, 60, 3), 120, dtype=np.uint8)).save(tmp_path / "a.jpg", quality=100)
    masks = np.zeros((2,) + mask_shape, dtype=np.uint8)
    masks[1, 10:20, 20:40] = e3_io.LESION
    masks[1, 0:5, 0:5] = 255  # uncertainty band: NOT lesion
    return {"x": tmp_path / "a.jpg"}, masks


def test_load_item_returns_rgb_and_a_boolean_lesion_mask_where_255_is_not_lesion(tmp_path):
    index, masks = write_case(tmp_path)
    image, mask = e3_io.make_load_item(index, masks, {"x": 1})("x")
    assert image.shape == (40, 60, 3) and image.dtype == np.uint8
    assert mask.dtype == bool and mask.sum() == 10 * 20 and not mask[:5, :5].any()


def test_load_item_refuses_a_mask_on_another_grid(tmp_path):
    index, masks = write_case(tmp_path, mask_shape=(20, 30))
    with pytest.raises(ValueError):
        e3_io.make_load_item(index, masks, {"x": 1})("x")


def test_the_extract_wrapper_asks_for_the_cls4_part_with_the_requested_views(tmp_path, monkeypatch):
    from dermssl import torch_v71 as tv

    seen = {}

    def fake(model, images, *, device, batch_size, n_views):
        seen.update(batch_size=batch_size, n_views=n_views, n=len(images))
        return {part: np.zeros((n_views, len(images), 3)) for part in tv.PARTS} | {"cls4": np.ones((n_views, len(images), 5))}

    monkeypatch.setattr(tv, "extract_split", fake)
    out = e3_io.make_extract_fn("model", "cpu", 16)(np.zeros((4, 8, 8, 3), np.uint8), 8)
    assert out.shape == (8, 4, 5) and seen == {"batch_size": 16, "n_views": 8, "n": 4}


def test_the_resize_crop_is_the_b71_torchvision_geometry():
    from dermssl import torch_v71 as tv

    image = np.random.default_rng(0).integers(0, 255, (45, 60, 3), dtype=np.uint8)
    out = e3_io.make_resize_crop()(image, 224)
    assert out.shape == (224, 224, 3) and (out == tv.resize_crop_uint8(Image.fromarray(image), 224)).all()


def test_the_protocol_name_is_the_registered_b71_config():
    from dermssl import v71_protocol as vp

    assert e3_io.PROTOCOL == vp.config_id("B_s42_a1.0", 224, "cls4", "balanced", True)
