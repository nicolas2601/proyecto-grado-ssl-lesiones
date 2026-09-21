"""Test-time augmentation for the student: flips averaged back in place."""

import numpy as np
import pytest

import seg_paths  # noqa: F401

torch = pytest.importorskip("torch")
import eval_student  # noqa: E402


class _PositionModel(torch.nn.Module):
    """Logit depends on the column only, so flips change the answer."""

    def forward(self, batch):
        _, _, height, width = batch.shape
        ramp = torch.linspace(-3.0, 3.0, width).view(1, 1, 1, width).expand(batch.shape[0], 1, height, width)
        return ramp.clone()


def test_predict_masks_without_tta_keeps_the_raw_prediction():
    images = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    probs = eval_student.predict_masks(_PositionModel(), images, torch.device("cpu"), 2)
    assert probs.shape == (2, 8, 8)
    assert probs[0, 0, 0] < 0.1 and probs[0, 0, -1] > 0.9


def test_predict_masks_with_tta_averages_the_flipped_views():
    images = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    device = torch.device("cpu")
    plain = eval_student.predict_masks(_PositionModel(), images, device, 1)
    tta = eval_student.predict_masks(_PositionModel(), images, device, 1, tta=True)
    # A horizontal flip mirrors the ramp, so the average of both views is symmetric.
    assert np.allclose(tta[0], tta[0][:, ::-1], atol=1e-6)
    assert not np.allclose(plain[0], plain[0][:, ::-1])
    expected = (plain[0] + plain[0][:, ::-1]) / 2.0  # vertical flips do not change a column ramp
    assert np.allclose(tta[0], expected, atol=1e-6)
