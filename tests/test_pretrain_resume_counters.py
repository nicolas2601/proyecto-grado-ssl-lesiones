"""A resumed run must keep counting where it left off.

The budget survives a restart through ``elapsed_wall_seconds``, but the run
manifest also reports how many steps and images the arm actually saw. If
those counters restart at zero, a run that was interrupted at 80 percent
reports only the last 20 percent, and the epoch count in the thesis is wrong
by five times - silently, because the number still looks plausible.
"""

import pytest

torch = pytest.importorskip("torch")

from dermssl.pretrain import torch_loop  # noqa: E402


def _saved(tmp_path, **extra):
    path = tmp_path / "resume.pt"
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    torch_loop._save_checkpoint(path, model, optimizer, 12.5, {0.1}, **extra)
    return torch.load(path, map_location="cpu", weights_only=True)


def test_the_checkpoint_records_the_counters(tmp_path):
    state = _saved(tmp_path, step=417, images_seen=53376)
    assert state["step"] == 417
    assert state["images_seen"] == 53376
    assert state["elapsed_wall_seconds"] == 12.5


def test_the_counters_default_to_zero_for_a_checkpoint_written_before_this_fix(tmp_path):
    state = _saved(tmp_path)
    assert torch_loop._counters_from(state) == (0, 0)


def test_the_counters_are_read_back_from_the_checkpoint(tmp_path):
    state = _saved(tmp_path, step=417, images_seen=53376)
    assert torch_loop._counters_from(state) == (417, 53376)
