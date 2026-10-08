"""Figure 1 (ITA per dataset): every number is recomputed from the raw JSON, and build() writes the three exports."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from matplotlib import colors as mcolors

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig1_ita as fig1  # noqa: E402

DATASETS = ("ham10000", "isic2019", "isic2020")
BUCKETS = ("I", "II", "III", "IV", "V", "VI")
E5 = REPO / "resultados" / "banco_completo" / "e5"


def _raw(dataset: str) -> dict:
    return json.loads((E5 / f"e5_fitzpatrick_{dataset}.json").read_text())


def _rows(dataset: str) -> list[dict]:
    pi = _raw(dataset)["per_image"]
    return [dict(zip(pi["columns"], row)) for row in pi["rows"]]


def test_the_script_reads_the_three_right_source_files():
    assert [p.name for p in fig1.source_files(REPO)] == [f"e5_fitzpatrick_{d}.json" for d in DATASETS]
    assert all(p.parent == E5 and p.is_file() for p in fig1.source_files(REPO))


@pytest.mark.parametrize("dataset", DATASETS)
def test_counts_match_the_raw_json(dataset):
    rows = _rows(dataset)
    got = fig1.figure_data(REPO)[dataset]
    ok = [r for r in rows if r["roi_status"] == "ok"]
    assert got["n_total"] == len(rows)
    assert got["n_ita"] == len(ok) == sum(r["roi_ita"] is not None for r in rows)
    assert got["bucket_counts"] == {b: sum(r["roi_bucket"] == b for r in rows) for b in BUCKETS}
    assert got["n_abstention"] == len(rows) - len(ok)
    assert sum(got["bucket_counts"].values()) + got["n_abstention"] == len(rows)
    assert got["abstention_by_reason"] == {
        s: sum(r["roi_status"] == s for r in rows) for s in {r["roi_status"] for r in rows} - {"ok"}
    }


@pytest.mark.parametrize("dataset", DATASETS)
def test_fractions_and_quantiles_match_an_independent_computation(dataset):
    rows = _rows(dataset)
    got = fig1.figure_data(REPO)[dataset]
    itas = sorted(r["roi_ita"] for r in rows if r["roi_status"] == "ok")
    n = len(itas)
    assert got["fractions"]["abstention"] == pytest.approx((len(rows) - n) / len(rows))
    for b in BUCKETS:
        assert got["fractions"][b] == pytest.approx(sum(r["roi_bucket"] == b for r in rows) / len(rows))
    assert sum(got["fractions"].values()) == pytest.approx(1.0)
    q = got["quantiles"]
    assert q["min"] == itas[0] and q["max"] == itas[-1]
    mid = (itas[n // 2 - 1] + itas[n // 2]) / 2 if n % 2 == 0 else itas[n // 2]
    assert q["median"] == pytest.approx(mid)
    assert q["q25"] <= q["median"] <= q["q75"]


def test_thresholds_come_from_the_json_and_buckets_agree_with_them():
    data = fig1.figure_data(REPO)
    assert data["thresholds"] == _raw("ham10000")["fst_thresholds"] == [40, 23, 12, 0, -25]
    for dataset in DATASETS:
        for r in _rows(dataset):
            if r["roi_status"] == "ok":
                assert r["roi_bucket"] == fig1.bucket_of(r["roi_ita"], data["thresholds"])


def test_bucket_on_a_threshold_goes_to_the_darker_one():
    th = [40, 23, 12, 0, -25]
    assert [fig1.bucket_of(v, th) for v in (41, 40, 23, 12, 0, -25, -26)] == ["I", "II", "III", "IV", "V", "VI", "VI"]


def test_bucket_colours_are_stable_sequential_cream_to_dark_brown():
    fills, edges = fig1.bucket_palette()
    assert dict(zip(fig1.BUCKETS, map(mcolors.to_hex, fills))) == {
        "I": "#c2a68e", "II": "#b88a64", "III": "#ad713b",
        "IV": "#89562f", "V": "#664126", "VI": "#3b2819",
    }
    assert len(set(map(mcolors.to_hex, fills))) == len(fig1.BUCKETS)
    assert all(mcolors.to_hex(edge) != mcolors.to_hex(fill) for edge, fill in zip(edges, fills))
    assert all(_luminance(a) > _luminance(b) for a, b in zip(fills, fills[1:]))
    assert fig1.ABSTENTION_FILL == "#7e7e7e" and fig1.ABSTENTION_EDGE == "#4e4e4e"


def _luminance(value):
    channels = mcolors.to_rgb(value)
    linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in channels]
    return sum(weight * channel for weight, channel in zip((0.2126, 0.7152, 0.0722), linear))


def test_direct_labels_remain_readable_on_every_bucket_and_abstention():
    fills, _ = fig1.bucket_palette()
    for fill in [*fills, fig1.ABSTENTION_FILL]:
        text = fig1._text_color(fill)
        lights = sorted((_luminance(fill), _luminance(text)))
        assert (lights[1] + 0.05) / (lights[0] + 0.05) >= 4.5


def test_ita_bands_are_subtle_and_labels_do_not_treat_a_dataset_as_one_bucket():
    import matplotlib.pyplot as plt

    data = fig1.figure_data(REPO)
    fig, ax = plt.subplots()
    fills, _ = fig1.bucket_palette()
    fig1._panel_distribution(ax, data, fills)
    bands = ax.patches[:6]
    assert len(bands) == 6
    assert all(patch.get_alpha() <= 0.18 for patch in bands)
    violin_edges = [collection.get_edgecolor()[0] for collection in ax.collections]
    violin_fills = [collection.get_facecolor()[0] for collection in ax.collections]
    assert violin_edges and all(np.allclose(edge[:3], mcolors.to_rgb(fig1.NEUTRAL_VIOLIN_EDGE)) for edge in violin_edges)
    assert all(np.ptp(fill[:3]) < 1e-6 for fill in violin_fills)  # monochrome gray; ITA classes are encoded by the bands
    assert all(label.get_text() not in fig1.DATASET_LABELS.values() for label in ax.texts)
    assert [label.get_text() for label in ax.texts] == list(fig1.BUCKETS)
    assert all((1.0 + 0.05) / (_luminance(label.get_color()) + 0.05) >= 4.5 for label in ax.texts)
    assert "no validada" in fig1.NOTE and "convención visual" in fig1.NOTE
    plt.close(fig)


def test_abstention_uses_gray_not_a_skin_tone_and_legend_names_all_buckets():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    fig1._panel_buckets(ax, fig1.figure_data(REPO), *fig1.bucket_palette())
    legend = [text.get_text() for text in ax.get_legend().get_texts()]
    assert legend == [*fig1.BUCKETS, fig1.ABSTENTION]
    assert mcolors.to_hex(ax.patches[-1].get_facecolor()) == fig1.ABSTENTION_FILL
    assert all(ax.patches[index].get_hatch() == fig1.ABSTENTION_HATCH for index in (6, 13, 20))
    assert ax.get_legend().legend_handles[-1].get_hatch() == fig1.ABSTENTION_HATCH
    plt.close(fig)


def test_build_writes_pdf_svg_and_png(tmp_path):
    paths = fig1.build(REPO, tmp_path)
    assert sorted(p.name for p in paths) == ["fig1_ita.pdf", "fig1_ita.png", "fig1_ita.svg"]
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)
    assert "no validada de fototipo" in next(p for p in paths if p.suffix == ".svg").read_text()
    assert all(line == line.rstrip() for line in next(p for p in paths if p.suffix == ".svg").read_text().splitlines())


def test_pdf_svg_and_png_show_the_same_seven_legend_colours_and_plot(tmp_path):
    fig1.build(REPO, tmp_path)
    if shutil.which("pdftoppm") and shutil.which("rsvg-convert"):
        subprocess.run(["pdftoppm", "-f", "1", "-singlefile", "-scale-to-x", "1200", "-scale-to-y", "-1",
                        "-png", str(tmp_path / "fig1_ita.pdf"), str(tmp_path / "pdf")], check=True)
        subprocess.run(["rsvg-convert", "-w", "1200", "-o", str(tmp_path / "svg.png"),
                        str(tmp_path / "fig1_ita.svg")], check=True)
    else:
        try:
            import cairosvg
            import fitz
        except ImportError as exc:
            pytest.fail(f"Cross-format parity needs pdftoppm+rsvg-convert OR PyMuPDF+CairoSVG: {exc}")
        with fitz.open(tmp_path / "fig1_ita.pdf") as pdf:
            page = pdf[0]
            scale = 1200 / page.rect.width
            (tmp_path / "pdf.png").write_bytes(page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).tobytes("png"))
        cairosvg.svg2png(url=str(tmp_path / "fig1_ita.svg"), write_to=str(tmp_path / "svg.png"), output_width=1200)
    formats = [tmp_path / "fig1_ita.png", tmp_path / "pdf.png", tmp_path / "svg.png"]
    height = Image.open(formats[1]).height
    assert Image.open(formats[2]).size == (1200, height)
    images = [np.asarray(Image.open(path).convert("RGB").resize((1200, height), Image.Resampling.LANCZOS),
                         dtype=np.int16) for path in formats]
    swatches = [*fig1.BUCKET_FILLS, fig1.ABSTENTION_FILL]
    for image in images:
        for hex_color in swatches:
            rgb = np.array(mcolors.to_rgb(hex_color)) * 255
            assert np.count_nonzero(np.max(np.abs(image - rgb), axis=2) < 15) >= 300, hex_color
    for one, two in ((0, 1), (0, 2), (1, 2)):
        assert np.abs(images[one] - images[two]).mean() < 12, (formats[one], formats[two])
