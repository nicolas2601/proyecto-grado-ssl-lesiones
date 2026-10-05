"""T7's darkening augmentation (docs/27): photometric, eligible samples only, never brightens."""

from __future__ import annotations

import pytest
import torch

import seg_paths  # noqa: F401
import gpu_augment as ga


def _batch(count: int = 64, size: int = 8) -> torch.Tensor:
    return torch.rand(count, 3, size, size, generator=torch.Generator().manual_seed(0))


def _gen(seed: int = 1) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_ineligible_samples_are_untouched_and_eligible_ones_darken():
    images = _batch()
    eligible = torch.arange(64) % 2 == 0
    out = ga.random_darken(images, _gen(), eligible, probability=1.0)
    assert torch.equal(out[~eligible], images[~eligible])
    changed = (out[eligible] != images[eligible]).flatten(1).any(dim=1)
    assert changed.float().mean() > 0.9


def test_the_mean_luminance_never_goes_up_and_values_stay_in_range():
    images = _batch()
    out = ga.random_darken(images, _gen(), torch.ones(64, dtype=torch.bool), probability=1.0)
    luminance = lambda x: ga._grayscale(x).mean(dim=(1, 2, 3))  # noqa: E731
    assert (luminance(out) <= luminance(images) + 1e-6).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_probability_zero_and_identity_ranges_are_the_identity():
    images = _batch()
    everyone = torch.ones(64, dtype=torch.bool)
    assert torch.equal(ga.random_darken(images, _gen(), everyone, probability=0.0), images)
    same = ga.random_darken(images, _gen(), everyone, probability=1.0, gamma_range=(1.0, 1.0),
                            brightness_range=(1.0, 1.0), contrast_range=(1.0, 1.0))
    assert torch.allclose(same, images, atol=1e-6)


def test_the_registered_defaults_and_the_gate_rate():
    images = _batch(count=4000, size=2)
    out = ga.random_darken(images, _gen(), torch.ones(4000, dtype=torch.bool))
    rate = (out != images).flatten(1).any(dim=1).float().mean().item()
    assert rate == pytest.approx(0.5, abs=0.03)
    assert ga.DARKEN_DEFAULTS == {"probability": 0.5, "gamma_range": (1.0, 2.2),
                                  "brightness_range": (0.85, 1.0), "contrast_range": (0.85, 1.0)}


def test_it_is_deterministic_and_consumes_a_fixed_number_of_draws():
    images = _batch()
    half, none = torch.arange(64) < 32, torch.zeros(64, dtype=torch.bool)
    first, second = _gen(), _gen()
    assert torch.equal(ga.random_darken(images, first, half), ga.random_darken(images, _gen(), half))
    ga.random_darken(images, second, none)
    assert torch.equal(torch.rand(3, generator=first), torch.rand(3, generator=second))
