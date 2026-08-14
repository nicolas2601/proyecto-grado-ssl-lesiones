"""Schedules are driven by compute fraction, not epochs - same axis as the
equal-compute curves, so two arms at 50% budget are at the same LR phase."""

import pytest

from dermssl.pretrain import schedules as sc


def test_lr_warms_up_linearly():
    assert sc.lr_at(0.0, base_lr=1.0, warmup_fraction=0.1) == 0.0
    assert sc.lr_at(0.05, base_lr=1.0, warmup_fraction=0.1) == pytest.approx(0.5)
    assert sc.lr_at(0.1, base_lr=1.0, warmup_fraction=0.1) == pytest.approx(1.0)


def test_lr_decays_with_cosine_to_the_floor():
    peak = sc.lr_at(0.1, base_lr=1.0, warmup_fraction=0.1)
    mid = sc.lr_at(0.55, base_lr=1.0, warmup_fraction=0.1)
    end = sc.lr_at(1.0, base_lr=1.0, warmup_fraction=0.1, floor=0.001)
    assert peak > mid > end
    assert end == pytest.approx(0.001)


def test_lr_never_negative_and_clamps_past_one():
    assert sc.lr_at(1.5, base_lr=1.0, warmup_fraction=0.1) >= 0.0


def test_ema_momentum_rises_from_base_to_one():
    assert sc.ema_momentum_at(0.0, base=0.996) == pytest.approx(0.996)
    assert sc.ema_momentum_at(1.0, base=0.996) == pytest.approx(1.0)
    mid = sc.ema_momentum_at(0.5, base=0.996)
    assert 0.996 < mid < 1.0


def test_ema_momentum_is_monotonic():
    values = [sc.ema_momentum_at(f / 20, base=0.996) for f in range(21)]
    assert values == sorted(values)
