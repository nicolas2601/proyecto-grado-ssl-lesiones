"""Characterisation: ITA histogram data must reconcile (valid + abstention = total) and use contiguous 10 degree bins."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))
sys.path.insert(0, str(REPO / "src"))

import fig_char_ita_distribution as figure  # noqa: E402


def test_valid_plus_abstention_equals_total_in_every_panel():
    for name, item in figure.load_all(REPO).items():
        assert item["n_valid"] + item["n_abstention"] == item["n_total"], name
        assert sum(item["counts"].values()) == item["n_valid"], name


def test_bins_are_contiguous_and_ten_degrees_wide():
    assert set(np.diff(figure.BIN_EDGES)) == {figure.BIN_WIDTH_DEG}
    assert figure.BIN_EDGES[0] == -90 and figure.BIN_EDGES[-1] == 90


def test_pad_scope_is_the_fst_registered_subset():
    pad = figure.load_pad(REPO)
    assert pad["n_total"] == 1494 and pad["n_excluded_without_fst"] == 804


def test_build_exports_and_declares_pad_scope(tmp_path):
    assert {p.suffix for p in figure.build(REPO, tmp_path)} == {".pdf", ".svg", ".png"}
    trace = json.loads((tmp_path / "fig_char_ita_distribution.traceability.json").read_text(encoding="utf-8"))
    assert "fototipo registrado" in trace["pad_scope"]
