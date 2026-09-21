"""Torch tests for the CCAM head and its contrastive loss."""

import math

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.torch

import seg_paths  # noqa: F401,E402
import ccam_engine as ce  # noqa: E402


def _two_pattern_features(batch=6, size=8, seed=0):
    """Foreground and background carry different channel patterns."""
    generator = torch.Generator().manual_seed(seed)
    fg_pattern = torch.tensor([1.0, 0.0, 1.0, 0.0]).view(1, 4, 1, 1)
    bg_pattern = torch.tensor([0.0, 1.0, 0.0, 1.0]).view(1, 4, 1, 1)
    truth = torch.zeros(batch, 1, size, size)
    truth[:, :, 2:6, 2:6] = 1.0
    noise = torch.rand(batch, 4, size, size, generator=generator) * 0.1
    features = truth * fg_pattern + (1 - truth) * bg_pattern + noise
    return features, truth


def test_rank_weights_decay_exponentially_with_similarity_rank():
    similarity = torch.tensor([[1.0, 0.9, 0.5, 0.1], [0.9, 1.0, 0.4, 0.2],
                               [0.5, 0.4, 1.0, 0.3], [0.1, 0.2, 0.3, 1.0]])
    weights = ce.rank_weights(similarity, alpha=0.5)
    assert weights[0, 1] == pytest.approx(1.0)  # most similar off-diagonal partner
    assert weights[0, 2] == pytest.approx(math.exp(-0.5))
    assert weights[0, 3] == pytest.approx(math.exp(-1.0))
    assert not weights.requires_grad


def test_ccam_loss_is_finite_scalar_with_gradient_to_the_activation():
    features, _ = _two_pattern_features()
    logits = torch.randn(6, 1, 8, 8, requires_grad=True)
    loss = ce.ccam_loss(features, torch.sigmoid(logits), alpha=0.25)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_ccam_loss_prefers_the_true_partition_over_a_random_one():
    features, truth = _two_pattern_features()
    perfect = ce.ccam_loss(features, truth.clamp(0.02, 0.98), alpha=0.25)
    random_map = torch.rand(6, 1, 8, 8, generator=torch.Generator().manual_seed(9))
    assert perfect < ce.ccam_loss(features, random_map, alpha=0.25)


def test_ccam_loss_survives_degenerate_all_foreground_activation():
    features, _ = _two_pattern_features()
    loss = ce.ccam_loss(features, torch.ones(6, 1, 8, 8), alpha=0.25)
    assert torch.isfinite(loss)


def test_ccam_loss_needs_at_least_two_images():
    with pytest.raises(ValueError):
        ce.ccam_loss(torch.rand(1, 4, 4, 4), torch.rand(1, 1, 4, 4), alpha=0.25)


@pytest.mark.parametrize("kernel", [1, 3])
def test_head_outputs_a_probability_map_of_the_feature_resolution(kernel):
    head = ce.CCAMHead(in_channels=4, kernel_size=kernel)
    out = head(torch.rand(3, 4, 5, 5)).detach()
    assert out.shape == (3, 1, 5, 5)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_head_learns_to_separate_two_patterns_in_a_few_steps():
    torch.manual_seed(0)
    features, truth = _two_pattern_features(batch=8)
    head = ce.CCAMHead(in_channels=4, kernel_size=1)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.05)
    first = last = None
    for step in range(60):
        loss = ce.ccam_loss(features, head(features), alpha=0.25)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        first = loss.item() if step == 0 else first
        last = loss.item()
    assert last < first
    with torch.no_grad():
        activation = head.eval()(features)
    inside = activation[truth > 0.5].mean()
    outside = activation[truth < 0.5].mean()
    assert abs(float(inside - outside)) > 0.3  # sign is arbitrary, contrast is not
