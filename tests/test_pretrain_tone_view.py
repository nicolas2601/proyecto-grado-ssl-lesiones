"""docs/28: the tone term compares view 1 with THE SAME crop, tone-shifted.

Opus review P2: comparing independent crops asked the backbone for
invariance to framing as well as tone. The third view must replay view 1's
crop, flips and photometric draws on the tone-shifted image, and must leave
the two SimSiam views exactly as they were.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from dermssl.pretrain import torch_data  # noqa: E402

CC = (("identity", 1.0),)


def _image(path, seed, size=(96, 80)):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (*size[::-1], 3), dtype=np.uint8)).save(path)
    return path


def _dataset(pairs, tone_view):
    return torch_data.SimSiamPairDataset(pairs, 64, CC, seed=3, hue_jitter=0.02, tone_view=tone_view)


def test_with_the_same_image_the_tone_view_is_view_one_exactly(tmp_path):
    """Identical files isolate geometry: any crop or flip mismatch shows up."""
    img = _image(tmp_path / "a.png", 0)
    torch.manual_seed(0)
    v1, v2, tone = _dataset([(img, img)], tone_view=True)[0]
    assert torch.equal(v1, tone)
    assert not torch.equal(v1, v2)


def test_the_simsiam_views_are_untouched_by_the_tone_view(tmp_path):
    original, shifted = _image(tmp_path / "o.png", 0), _image(tmp_path / "s.png", 1)
    torch.manual_seed(0)
    v1_off, v2_off = _dataset([(original, shifted)], tone_view=False)[0]
    torch.manual_seed(0)
    v1_on, v2_on, _ = _dataset([(original, shifted)], tone_view=True)[0]
    assert torch.equal(v1_off, v1_on) and torch.equal(v2_off, v2_on)


def test_a_different_image_gives_a_different_tone_view(tmp_path):
    original, shifted = _image(tmp_path / "o.png", 0), _image(tmp_path / "s.png", 1)
    torch.manual_seed(0)
    v1, _, tone = _dataset([(original, shifted)], tone_view=True)[0]
    assert not torch.equal(v1, tone)


def test_mismatched_sizes_are_refused(tmp_path):
    original = _image(tmp_path / "o.png", 0, size=(96, 80))
    shifted = _image(tmp_path / "s.png", 1, size=(80, 96))
    with pytest.raises(ValueError, match="mismo crop"):
        _dataset([(original, shifted)], tone_view=True)[0]
