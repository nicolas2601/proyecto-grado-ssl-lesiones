"""Unit tests for the Derm-TriplePrior modules:
1. SuperpixelAffinityLoss (differentiability, gradient flow, and shape correctness).
2. PeripheralRingNormalizer (mask construction, color and token baseline extraction).
"""
import numpy as np
import pytest
import torch

from dermssl.segmentation_methods.dermlip_slic.tripleprior import (
    PeripheralRingNormalizer,
    SuperpixelAffinityLoss,
)


def test_superpixel_affinity_loss_gradient_flow():
    """Verify that SuperpixelAffinityLoss computes accurate variance and backpropagates gradients."""
    loss_fn = SuperpixelAffinityLoss(max_superpixels=50)

    # Synthetic batch of 2 images, 32x32 resolution
    # Create logits and get probabilities with grad enabled
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    probs = torch.sigmoid(logits)

    # Synthetic superpixel labels: 4 quadrants per image
    labels = np.zeros((2, 32, 32), dtype=np.int32)
    labels[:, :16, :16] = 0
    labels[:, :16, 16:] = 1
    labels[:, 16:, :16] = 2
    labels[:, 16:, 16:] = 3

    loss = loss_fn(probs, labels)
    assert loss.item() >= 0.0
    assert loss.requires_grad

    # Backward pass must succeed
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert logits.grad.shape == (2, 1, 32, 32)


def test_superpixel_affinity_loss_zero_on_constant():
    """When probabilities inside each superpixel are perfectly uniform, loss must be ~0."""
    loss_fn = SuperpixelAffinityLoss()

    probs = torch.zeros(1, 32, 32, requires_grad=True)
    with torch.no_grad():
        # Quadrant 0 has value 0.2, Quadrant 1 has value 0.8
        probs[0, :16, :] = 0.2
        probs[0, 16:, :] = 0.8

    labels = np.zeros((1, 32, 32), dtype=np.int32)
    labels[0, :16, :] = 0
    labels[0, 16:, :] = 1

    loss = loss_fn(probs, labels)
    assert pytest.approx(loss.item(), abs=1e-6) == 0.0


def test_peripheral_ring_normalizer():
    """Verify peripheral ring extraction on synthetic skin image."""
    normalizer = PeripheralRingNormalizer(ring_margin_frac=0.1)
    h, w = 100, 100
    mask = normalizer.create_peripheral_ring_mask(h, w)

    # Perimeter should be True, center must be False
    assert bool(mask[0, 0]) is True
    assert bool(mask[50, 50]) is False
    assert bool(mask[99, 99]) is True

    # Test baseline extraction
    img = np.ones((h, w, 3), dtype=np.uint8) * 150
    # Set center to different color (e.g. lesion = 20)
    img[20:80, 20:80] = 20

    color_mean, token_mean = normalizer.extract_skin_baseline(img)
    # Mean color of peripheral ring must be exactly 150, ignoring the center
    assert np.allclose(color_mean, [150.0, 150.0, 150.0])
    assert token_mean is None

    # Test with dense tokens
    tokens = torch.randn(512, 14, 14)
    _, token_mean = normalizer.extract_skin_baseline(img, dense_tokens=tokens)
    assert token_mean is not None
    assert token_mean.shape == (512, 1)
    # Should be normalized
    norm = torch.norm(token_mean, p=2).item()
    assert pytest.approx(norm, abs=1e-4) == 1.0
