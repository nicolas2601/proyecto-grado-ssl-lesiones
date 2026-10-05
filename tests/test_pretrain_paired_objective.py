"""The paired arm must not add its own loss to itself.

With v1 = original and v2 = tone-shifted, the SimSiam loss over that pair
ALREADY is the original-to-transformation consistency. Adding a separately
named "consistency" term computed from the same four tensors doubled the
gradient of the proposed arm and made it incomparable with the plain one.
"""

import pytest

torch = pytest.importorskip("torch")

from dermssl.pretrain import torch_simsiam  # noqa: E402


def test_the_paired_objective_weights_the_loss_instead_of_doubling_it():
    ssl_loss = torch.tensor(-0.7785277807218088)
    loss, consistency = torch_simsiam.paired_objective(ssl_loss, weight=1.0)
    assert torch.equal(loss, ssl_loss)
    assert torch.equal(consistency, ssl_loss)


def test_the_weight_scales_the_paired_objective():
    ssl_loss = torch.tensor(-0.5)
    loss, _ = torch_simsiam.paired_objective(ssl_loss, weight=0.25)
    assert loss.item() == pytest.approx(-0.125)


def test_the_objective_is_never_the_sum_of_the_loss_with_itself():
    """Regression guard on the exact defect: the total was 2x the ssl loss."""
    ssl_loss = torch.tensor(-0.7785277807218088)
    loss, consistency = torch_simsiam.paired_objective(ssl_loss, weight=1.0)
    assert loss.item() != pytest.approx((ssl_loss + consistency).item())
