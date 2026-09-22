"""DDP wrapping: the reducer must tolerate parameters the forward never uses."""

import pytest

torch = pytest.importorskip("torch")

from dermssl.pretrain import torch_loop  # noqa: E402


def test_ddp_is_wrapped_with_a_static_graph(monkeypatch):
    """DINOv2 registers parameters a plain forward never touches (mask_token),
    and DDP's reducer raises "Expected to have finished reduction in the prior
    iteration" unless it knows the graph is the same every step. static_graph
    buys that tolerance without the per-iteration traversal of
    find_unused_parameters."""
    captured = {}

    class FakeDDP:
        def __init__(self, module, **kwargs):
            captured["module"] = module
            captured["kwargs"] = kwargs

    monkeypatch.setattr(torch_loop, "DDP", FakeDDP)
    sentinel = object()
    torch_loop._wrap_ddp(sentinel, local_rank=1, device=torch.device("cuda"))

    assert captured["module"] is sentinel
    assert captured["kwargs"]["static_graph"] is True
    assert captured["kwargs"]["device_ids"] == [1]


def test_ddp_on_cpu_passes_no_device_ids(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        torch_loop, "DDP", lambda module, **kwargs: captured.update(kwargs) or module
    )
    torch_loop._wrap_ddp(object(), local_rank=0, device=torch.device("cpu"))

    assert captured["device_ids"] is None
    assert captured["static_graph"] is True


# ------------------------------------------------- gradient norm is measured
def test_the_clipped_gradient_norm_is_returned_for_logging():
    """Without the returned norm, 'the clipping absorbed the doubling' stays
    an argument instead of a measurement."""
    model = torch.nn.Linear(4, 2)
    loss = model(torch.ones(1, 4)).sum()
    loss.backward()
    norm = torch_loop._clip_gradients(model.parameters(), max_norm=3.0)
    assert isinstance(norm, float)
    assert norm >= 0.0


def test_the_norm_reflects_a_gradient_over_the_threshold():
    model = torch.nn.Linear(4, 2)
    with torch.no_grad():
        model.weight.fill_(1.0)
    loss = (model(torch.full((1, 4), 100.0)).sum()) * 100.0
    loss.backward()
    norm = torch_loop._clip_gradients(model.parameters(), max_norm=3.0)
    assert norm > 3.0
