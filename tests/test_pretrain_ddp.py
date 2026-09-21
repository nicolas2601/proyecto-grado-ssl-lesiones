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
