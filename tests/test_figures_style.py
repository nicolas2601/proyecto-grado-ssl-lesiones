"""The shared BeautifulFigures-style helpers: palette, grid, top legend, font fallback and the three exports."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import beautiful as bf  # noqa: E402


def test_palette_is_the_purple_teal_one_of_the_reference_script():
    assert bf.FILLS == ("#9671bd", "#7e7e7e", "#77b5b6")
    assert bf.EDGES == ("#6a408d", "#4e4e4e", "#378d94")


def test_font_size_follows_the_figure_width_and_the_font_has_a_monospace_fallback():
    bf.apply_style(width=10)
    assert plt.rcParams["font.size"] == 20
    assert bf.FONT_FAMILY in ("Courier New", "DejaVu Sans Mono")
    bf.apply_style(width=16)
    assert plt.rcParams["font.size"] == 32


def test_axes_get_major_and_minor_grids_below_the_data():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    bf.style_axes(ax)
    fig.canvas.draw()
    assert ax.get_axisbelow() is True
    major = ax.xaxis.get_major_ticks()[0].gridline
    minor = ax.xaxis.get_minor_ticks()[0].gridline
    assert (major.get_linewidth(), major.get_alpha()) == (pytest.approx(0.75), pytest.approx(0.25))
    assert (minor.get_linewidth(), minor.get_alpha()) == (pytest.approx(0.25), pytest.approx(0.15))
    plt.close(fig)


def test_legend_sits_above_the_plot_horizontal_and_without_frame():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="uno")
    ax.plot([0, 1], [1, 0], label="dos")
    leg = bf.top_legend(ax)
    assert leg.get_frame_on() is False and leg._ncols == 2
    plt.close(fig)


def test_save_all_writes_pdf_svg_and_a_300_dpi_png(tmp_path):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1])
    paths = bf.save_all(fig, "demo", tmp_path)
    assert {p.suffix for p in paths} == {".pdf", ".svg", ".png"} and all(p.is_file() for p in paths)
    png = next(p for p in paths if p.suffix == ".png")
    assert round(Image.open(png).info["dpi"][0]) == 300  # the tight bounding box trims margins, so the size is not fixed


@pytest.mark.parametrize("value,decimals,expected", [(0.5, 2, "0,50"), (-0.0123, 3, "−0,012"), (1234567, 0, "1.234.567")])
def test_spanish_number_format_uses_decimal_comma_and_a_real_minus(value, decimals, expected):
    assert bf.fmt_es(value, decimals) == expected
