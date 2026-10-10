from __future__ import annotations

import sys
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import thesis_style as style  # noqa: E402


def test_style_constants_match_thesis_brief():
    assert style.FIGURE_WIDTH_CM == 17.3
    assert style.LABEL_PT == 11
    assert style.PANEL_TITLE_PT == 12
    assert style.FONT_FAMILY in {"Courier New", "Nimbus Mono PS"}
    assert style.COLORS == {
        "background": "#FFFFFF",
        "bar_light": "#F4EDE6",
        "accent": "#8A6C58",
        "text": "#222222",
        "frame": "#AAA39E",
        "abstention": "#FAF9F7",
    }
    assert style.ITA_COLORS == (
        "#FEF3E0", "#F2C598", "#CE8B51", "#B5723F", "#91552D", "#53311B",
    )
    assert style.LINE_WIDTHS == {"frame": 0.6, "bar": 0.7, "curve": 1.1}


def test_exports_keep_physical_width_and_create_all_formats(tmp_path):
    import matplotlib.pyplot as plt
    import xml.etree.ElementTree as ET

    fig, ax = plt.subplots(figsize=(style.FIGURE_WIDTH_IN, 2.0))
    ax.plot([0, 1], [0, 1], color=style.COLORS["accent"])
    ax.set_xlabel("Etiqueta de prueba")
    paths = style.save_figure(fig, "sample", tmp_path)
    assert {path.suffix for path in paths} == {".pdf", ".svg", ".png"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)

    svg = ET.parse(tmp_path / "sample.svg").getroot()
    width_pt = float(svg.attrib["width"].removesuffix("pt"))
    assert width_pt == pytest.approx(style.FIGURE_WIDTH_CM / 2.54 * 72, abs=0.02)
    svg_text = (tmp_path / "sample.svg").read_text(encoding="utf-8")
    assert "<text" in svg_text
    assert all(line == line.rstrip() for line in svg_text.splitlines())
    from PIL import Image

    with Image.open(tmp_path / "sample.png") as png:
        assert png.width == round(style.FIGURE_WIDTH_CM / 2.54 * style.DPI)
    plt.close(fig)


def test_rejects_existing_figure_outputs(tmp_path):
    import matplotlib.pyplot as plt

    fig, _ = plt.subplots(figsize=(style.FIGURE_WIDTH_IN, 2.0))
    style.save_figure(fig, "sample", tmp_path)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        style.save_figure(fig, "sample", tmp_path)
    plt.close(fig)
