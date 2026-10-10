"""D1/D2: PAD figures must reproduce the committed E2 evaluation exactly."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig_d1_d2_pad as figure  # noqa: E402


def test_primary_contrast_is_the_registered_d3_minus_d2():
    doc = figure.load(REPO)["doc"]
    row = figure.contrast(doc, "D3", "D2")
    assert row["excludes_zero"] and row["ci95_low"] < row["point_delta"] < row["ci95_high"]
    assert row == doc["primary_contrast"]


def test_d3_minus_d0_is_negative_with_ci_below_zero():
    row = figure.contrast(figure.load(REPO)["doc"], "D3", "D0")
    assert row["ci95_high"] < 0


def test_missing_contrast_aborts():
    with pytest.raises(ValueError):
        figure.contrast({"paired": []}, "D3", "D2")


def test_builds_export_all_formats_and_flag_missing_data(tmp_path):
    for builder, stem in ((figure.build_d1, "fig_d1_pad_d0_d3"), (figure.build_d2, "fig_d2_pad_contrasts")):
        assert {p.suffix for p in builder(REPO, tmp_path)} == {".pdf", ".svg", ".png"}
        trace = json.loads((tmp_path / f"{stem}.traceability.json").read_text(encoding="utf-8"))
        assert trace["n_images"] == 567 and trace["missing"]
