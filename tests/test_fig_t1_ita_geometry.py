from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))
sys.path.insert(0, str(REPO / "src"))

import fig_t1_ita_geometry as figure  # noqa: E402
from dermssl.fitzpatrick import MIN_B_FOR_ITA, ita_degrees  # noqa: E402


def test_t1_formula_and_reporting_guard_match_domain_source():
    assert figure.ITA_FORMULA == "ITA = arctan((L* − 50) / b*) × 180 / π"
    assert figure.MIN_B_FOR_ITA == MIN_B_FOR_ITA == 5.0
    assert ita_degrees(60.0, 10.0) == pytest.approx(45.0)
    assert ita_degrees(40.0, 10.0) == pytest.approx(-45.0)


def test_t1_diagram_is_schematic_and_has_no_numeric_sample_points():
    import matplotlib.pyplot as plt

    fig, axis = figure.draw()
    assert len(fig.axes) == 1
    assert not axis.get_xticks().size
    assert not axis.get_yticks().size
    text = " ".join(item.get_text() for item in fig.texts + axis.texts)
    assert "ITA > 0" in text
    assert "ITA < 0" in text
    assert "Esquema conceptual" in text
    plt.close(fig)


def test_t1_exports_vector_preview_and_source_hashes(tmp_path):
    paths = figure.build(REPO, tmp_path)

    assert {path.suffix for path in paths} == {".pdf", ".svg", ".png"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    note = (tmp_path / "fig_t1_ita_geometry.note.txt").read_text(encoding="utf-8")
    assert "mediana b* ≥ 5" in note
    assert "no constituye validación clínica" in note
    trace = json.loads((tmp_path / "fig_t1_ita_geometry.traceability.json").read_text(encoding="utf-8"))
    source = REPO / "src/dermssl/fitzpatrick.py"
    assert trace["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        figure.build(REPO, tmp_path)
