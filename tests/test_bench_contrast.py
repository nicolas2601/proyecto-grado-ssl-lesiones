"""Paired-contrast record: flip, reading and the consistency of the committed JSON it is loaded from."""

from __future__ import annotations

import pytest

from dermssl.bench import contrast as K

REC = {"metric": "macro_f1", "point_arm": 0.7, "point_control": 0.6, "point_delta": 0.1, "ci95_low": 0.02,
       "ci95_high": 0.18, "excludes_zero": True, "direction": "arm_better"}


def _make(**over):
    return K.from_record({**REC, **over}, arm="B71", control="X", unit="lesion_id", source="a.json")


def test_flip_negates_and_reverses_the_interval_and_swaps_the_winner():
    flip = K.flipped(_make())
    assert (flip.arm, flip.control) == ("X", "B71")
    assert (flip.point_arm, flip.point_control) == (0.6, 0.7)
    assert (flip.delta, flip.low, flip.high) == (-0.1, -0.18, -0.02)
    assert flip.direction == "control_better" and flip.excludes_zero
    assert K.flipped(flip) == _make()


def test_reading_names_a_winner_only_when_the_interval_excludes_zero():
    assert K.reading(_make()) == "B71 mejor (el IC excluye 0)"
    assert K.reading(K.flipped(_make())) == "B71 mejor (el IC excluye 0)"
    tie = _make(excludes_zero=False, direction="indistinguishable", ci95_low=-0.01)
    assert "sin diferencia demostrable" in K.reading(tie) and "mejor" not in K.reading(tie).replace("demostrable", "")


def test_numbers_use_decimal_comma_and_explicit_sign():
    assert K.interval(_make()) == "+0,1000 [+0,0200; +0,1800]"
    assert "." not in K.row(_make()).replace("a.json", "")


@pytest.mark.parametrize("bad", ["", "x"])
def test_flip_rejects_an_unknown_direction(bad):
    with pytest.raises(KeyError):
        K.flipped(_make(direction=bad))
