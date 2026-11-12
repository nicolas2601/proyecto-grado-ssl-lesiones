from __future__ import annotations

import pytest

from demo.backend.runtime_threads import DEFAULT_INTRA_THREADS, intra_op_threads


def test_default_is_two_threads_when_unset(monkeypatch):
    monkeypatch.delenv("DERM_ORT_INTRA_THREADS", raising=False)
    assert intra_op_threads() == DEFAULT_INTRA_THREADS == 2


def test_override_within_bounds_is_honoured(monkeypatch):
    monkeypatch.setenv("DERM_ORT_INTRA_THREADS", "8")
    assert intra_op_threads() == 8


@pytest.mark.parametrize("value", ["0", "9", "many", "2.5"])
def test_out_of_range_or_malformed_override_fails_closed(monkeypatch, value):
    monkeypatch.setenv("DERM_ORT_INTRA_THREADS", value)
    with pytest.raises(ValueError):
        intra_op_threads()
