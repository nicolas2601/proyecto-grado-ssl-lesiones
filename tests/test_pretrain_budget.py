"""Equal-compute is the thesis novelty: the budget math must be exact."""

import pytest

from dermssl.pretrain import budget as bg


def make_budget(hours=10.0, gpus=4, fractions=(0.25, 0.5, 1.0)):
    return bg.ComputeBudget(gpu_hours=hours, n_gpus=gpus, fractions=fractions)


def test_gpu_hours_scale_with_gpu_count():
    b = make_budget(hours=10.0, gpus=4)
    # 1 wall-clock hour on 4 GPUs = 4 GPU-hours.
    assert b.gpu_hours_elapsed(wall_seconds=3600) == pytest.approx(4.0)


def test_fraction_done_is_clamped_to_one():
    b = make_budget(hours=10.0, gpus=4)
    assert b.fraction_done(wall_seconds=0) == 0.0
    assert b.fraction_done(wall_seconds=4500) == pytest.approx(0.5)
    assert b.fraction_done(wall_seconds=999_999) == 1.0


def test_should_stop_only_at_exhaustion():
    b = make_budget(hours=1.0, gpus=1)
    assert b.should_stop(wall_seconds=3599) is False
    assert b.should_stop(wall_seconds=3600) is True


def test_due_checkpoints_fire_once_and_in_order():
    b = make_budget(hours=1.0, gpus=1, fractions=(0.25, 0.5, 1.0))
    saved: set[float] = set()

    due = b.due_checkpoints(wall_seconds=0.6 * 3600, already_saved=saved)
    assert due == [0.25, 0.5]

    saved.update(due)
    assert b.due_checkpoints(wall_seconds=0.7 * 3600, already_saved=saved) == []
    assert b.due_checkpoints(wall_seconds=3600, already_saved=saved) == [1.0]


def test_checkpoint_tag_is_filename_safe():
    assert bg.checkpoint_tag(0.25) == "ckpt_c025"
    assert bg.checkpoint_tag(1.0) == "ckpt_c100"


def test_invalid_construction_is_loud():
    with pytest.raises(ValueError):
        make_budget(hours=0)
    with pytest.raises(ValueError):
        make_budget(gpus=0)
