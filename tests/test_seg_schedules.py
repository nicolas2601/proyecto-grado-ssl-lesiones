"""Pure tests for the learning-rate schedule and state-dict key helpers."""

import pytest

import seg_paths  # noqa: F401
import ckpt_io
import schedules


def test_lr_warms_up_linearly_then_decays_with_cosine():
    base, total, warmup = 1.0, 100, 10
    assert schedules.cosine_lr(0, total, base, warmup) == pytest.approx(0.1)
    assert schedules.cosine_lr(9, total, base, warmup) == pytest.approx(1.0)
    assert schedules.cosine_lr(10, total, base, warmup) == pytest.approx(1.0)
    midpoint = schedules.cosine_lr(55, total, base, warmup)
    assert 0.4 < midpoint < 0.6
    assert schedules.cosine_lr(99, total, base, warmup) < 0.01


def test_lr_without_warmup_starts_at_base():
    assert schedules.cosine_lr(0, 50, 0.5, 0) == pytest.approx(0.5)


def test_lr_rejects_non_positive_total():
    with pytest.raises(ValueError):
        schedules.cosine_lr(0, 0, 1.0, 0)


def test_strip_prefixes_handles_wrappers_and_module_prefix():
    state = {"module.encoder_q.conv1.weight": 1, "module.encoder_q.fc.bias": 2, "other": 3}
    cleaned = ckpt_io.strip_state_dict_prefixes(state, ("module.", "encoder_q."))
    assert cleaned == {"conv1.weight": 1, "fc.bias": 2, "other": 3}


def test_unwrap_state_dict_prefers_known_wrapper_keys():
    assert ckpt_io.unwrap_state_dict({"state_dict": {"a": 1}}) == {"a": 1}
    assert ckpt_io.unwrap_state_dict({"model": {"a": 2}}) == {"a": 2}
    assert ckpt_io.unwrap_state_dict({"a": 3}) == {"a": 3}
