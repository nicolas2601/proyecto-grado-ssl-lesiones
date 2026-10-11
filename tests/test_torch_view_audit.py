"""Run on CPU with the real torch/torchvision pair before exporting real images."""

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from dermssl.pretrain.torch_data import SimSiamPairDataset
from dermssl.pretrain.torch_view_audit import audit_item

pytestmark = pytest.mark.torch


@pytest.mark.parametrize("tone_view", [False, True])
def test_observer_preserves_dataset_tensors_rng_and_mask_alignment(tmp_path, tone_view):
    yy, xx = np.mgrid[:64, :64]
    mask = (xx - 32)**2 + (yy - 32)**2 < 15**2
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[mask] = 255
    original, shifted = tmp_path / "original.png", tmp_path / "shifted.png"
    Image.fromarray(rgb).save(original)
    Image.fromarray(rgb // 2).save(shifted)
    dataset = SimSiamPairDataset([(original, shifted)], 32, (("identity", 1.0),),
                                seed=42, hue_jitter=0.02, tone_view=tone_view)
    torch.manual_seed(42)
    expected = dataset[0]
    torch.manual_seed(99)
    before = torch.get_rng_state().clone()
    actual, records = audit_item(dataset, 0, mask)
    assert torch.equal(before, torch.get_rng_state())
    for a, b in zip(actual, expected):
        assert torch.equal(a, b)
    assert len(records) == len(actual)
    if tone_view:
        assert records[0]["crop_tlhw"] == records[2]["crop_tlhw"]
        np.testing.assert_array_equal(records[0]["mask"], records[2]["mask"])


def test_replayed_geometry_matches_a_binary_image_without_photometry(tmp_path):
    from dermssl.pretrain.torch_view_audit import GeometryObserver
    from dermssl.pretrain.torch_data import _geometric
    mask = np.zeros((80, 96), dtype=bool)
    mask[15:40, 20:65] = True
    observer = GeometryObserver(_geometric(32, (0.2, 1.0)), mask)
    actual = observer(Image.fromarray(mask.astype(np.uint8) * 255))
    # Image interpolation differs at edges; interior foreground must coincide.
    replayed = observer.records[0]["mask"]
    values = np.asarray(actual)
    assert replayed.shape == values.shape
    assert not np.any((values == 255) & ~replayed)
