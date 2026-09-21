"""Torch tests for the student loss (confident pixels only)."""

import pytest

import seg_paths  # noqa: F401

torch = pytest.importorskip("torch")
import student_engine as se  # noqa: E402

pytestmark = pytest.mark.torch


def _target(values):
    return torch.tensor(values, dtype=torch.float32).view(1, 1, 2, 2)


def test_loss_is_small_when_logits_agree_with_the_confident_pixels():
    target = _target([[0.0, 1.0], [255.0, 1.0]])
    logits = torch.tensor([[-9.0, 9.0], [0.0, 9.0]]).view(1, 1, 2, 2)
    loss, _ = se.masked_bce_dice_loss(logits, target)
    assert float(loss) < 0.01


def test_ignored_pixels_have_no_influence_and_no_gradient():
    target = _target([[0.0, 1.0], [255.0, 1.0]])
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    loss, _ = se.masked_bce_dice_loss(logits, target)
    loss.backward()
    assert float(logits.grad[0, 0, 1, 0]) == 0.0
    assert float(logits.grad[0, 0, 0, 0]) != 0.0


def test_fully_ignored_samples_contribute_nothing_and_do_not_produce_nan():
    target = torch.full((2, 1, 2, 2), 255.0)
    target[0] = _target([[0.0, 1.0], [0.0, 1.0]])[0]
    logits = torch.randn(2, 1, 2, 2, requires_grad=True)
    loss, valid = se.masked_bce_dice_loss(logits, target)
    loss.backward()
    assert torch.isfinite(loss) and int(valid) == 1
    assert float(logits.grad[1].abs().sum()) == 0.0


def test_batch_of_only_ignored_samples_gives_zero_loss():
    loss, valid = se.masked_bce_dice_loss(torch.randn(1, 1, 2, 2), torch.full((1, 1, 2, 2), 255.0))
    assert float(loss) == 0.0 and int(valid) == 0


def test_loss_decreases_when_the_prediction_moves_toward_the_target():
    target = _target([[0.0, 1.0], [0.0, 1.0]])
    bad, _ = se.masked_bce_dice_loss(torch.tensor([[5.0, -5.0], [5.0, -5.0]]).view(1, 1, 2, 2), target)
    good, _ = se.masked_bce_dice_loss(torch.tensor([[-5.0, 5.0], [-5.0, 5.0]]).view(1, 1, 2, 2), target)
    assert good < bad


def test_summed_losses_support_distributed_validation_reduction():
    target = torch.cat([_target([[0.0, 1.0], [0.0, 1.0]])] * 2)
    logits = torch.zeros(2, 1, 2, 2)
    total, valid = se.summed_masked_loss(logits, target)
    mean, _ = se.masked_bce_dice_loss(logits, target)
    assert float(total) / int(valid) == pytest.approx(float(mean))
