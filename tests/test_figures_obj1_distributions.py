"""Objective 1 distributions count original classes and registered PAD FST separately."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig_obj1_distributions as figure  # noqa: E402


def _e5(tmp_path: Path, dataset: str, labels: list[str]) -> None:
    source = tmp_path / "resultados" / "banco_completo" / "e5"
    source.mkdir(parents=True, exist_ok=True)
    rows = [[f"id-{index}", label, "ok" if index % 2 else "ring_too_small"]
            for index, label in enumerate(labels)]
    (source / f"e5_fitzpatrick_{dataset}.json").write_text(json.dumps({
        "per_image": {"columns": ["image_id", "label", "roi_status"], "rows": rows},
    }), encoding="utf-8")


def test_e5_counts_original_labels_even_when_ita_is_missing(tmp_path):
    _e5(tmp_path, "isic2019", ["nv", "nv", "ak", "scc"])
    assert figure.count_e5(tmp_path, "isic2019") == {"NV": 2, "AK": 1, "SCC": 1}
    _e5(tmp_path, "isic2020", ["benign", "malignant"])
    assert figure.count_e5(tmp_path, "isic2020") == {"BENIGN": 1, "MALIGNANT": 1}


def test_pad_keeps_native_diagnoses_and_missing_fst_separate():
    records = [SimpleNamespace(diagnostic="NEV", fst=1, fst_status="ok"),
               SimpleNamespace(diagnostic="ACK", fst=None, fst_status="missing"),
               SimpleNamespace(diagnostic="ACK", fst=6, fst_status="ok")]
    diagnoses, phototypes = figure.count_pad(records)
    assert diagnoses == {"NEV": 1, "ACK": 2}
    assert phototypes == {"I": 1, "II": 0, "III": 0, "IV": 0, "V": 0, "VI": 1, "missing": 1}
    records[1].fst_status = "out_of_range"
    with pytest.raises(ValueError, match="FST"):
        figure.count_pad(records)


def test_frozen_sources_reconcile_before_export():
    expected = {"ham10000": 10015, "isic2019": 25331, "isic2020": 33126}
    for dataset, total in expected.items():
        counts = figure.count_e5(REPO, dataset)
        figure.validate_diagnoses(dataset, counts)
        assert sum(counts.values()) == total
    with pytest.raises(ValueError, match="total"):
        figure.validate_diagnoses("isic2020", {"BENIGN": 2, "MALIGNANT": 1})
    with pytest.raises(ValueError, match="classes"):
        figure.validate_diagnoses("ham10000", {"FABRICATED": 10015})


def test_pad_fst_rejects_invalid_values_and_mismatched_total():
    with pytest.raises(ValueError, match="FST total"):
        figure.validate_fst({"I": 1, "missing": 0}, total=2)
    with pytest.raises(ValueError, match="FST bucket"):
        figure.validate_fst({"I": 1, "VII": 1}, total=2)


def test_vertical_labels_show_counts_and_horizontal_ticks_stay_sparse():
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, (vertical, horizontal) = plt.subplots(1, 2)
    colors = [("#BE8277", "#9F5D50"), ("#A58B7B", "#725043")]
    figure._draw_vertical(vertical, ["BCC", "NV"], [845, 244], colors, 1089)
    figure._draw_horizontal(horizontal, ["BCC", "NV"], [845, 244], colors, 1089)
    assert [label.get_text() for label in vertical.texts] == ["845", "244"]
    assert vertical.get_yscale() == "log"
    assert isinstance(horizontal.xaxis.get_major_locator(), MaxNLocator)
    plt.close(fig)


def test_binary_horizontal_caption_does_not_overlap_axis_label():
    import matplotlib.pyplot as plt

    with matplotlib.rc_context():
        figure.bf.apply_style(width=11)
        matplotlib.rcParams.update({"font.size": 12, "axes.labelsize": 12})
        fig = figure._diagnosis_chart("isic2020", {"BENIGN": 32542, "MALIGNANT": 584}, "horizontal")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        caption = fig.texts[0].get_window_extent(renderer)
        xlabel = fig.axes[0].xaxis.label.get_window_extent(renderer)
        assert caption.y1 + 8 < xlabel.y0
        assert caption.x1 < fig.bbox.x1
        plt.close(fig)


def test_fst_horizontal_percentages_disclose_recorded_only_denominator():
    import matplotlib.pyplot as plt

    counts = {"I": 153, "II": 876, "III": 392, "IV": 62, "V": 10, "VI": 1, "missing": 804}
    fig = figure._fst_chart(counts, "horizontal")
    caption = fig.texts[0].get_text()
    assert "Porcentajes" in caption
    assert "1.494" in caption
    assert "804 sin FST excluidos" in caption
    plt.close(fig)


def test_vertical_and_horizontal_figures_export_separately_without_overwriting(tmp_path):
    diagnoses = {dataset: {label: index + 1 for index, label in enumerate(labels)}
                 for dataset, labels in figure.CLASSES.items()}
    diagnoses["pad_ufes_20"] = {"NEV": 2, "ACK": 1}
    fst = {bucket: 0 for bucket in figure.FST_BUCKETS}
    fst.update({"I": 2, "missing": 1})
    paths = figure.render(diagnoses, fst, tmp_path)
    assert len(paths) == 30
    assert len({path.stem for path in paths}) == 10
    assert {path.suffix for path in paths} == {".png", ".pdf", ".svg"}
    assert {"fig_obj1_diagnosis_ham10000_vertical", "fig_obj1_diagnosis_ham10000_horizontal",
            "fig_obj1_pad_fst_registered_vertical", "fig_obj1_pad_fst_registered_horizontal"} <= {
                path.stem for path in paths
            }
    assert all(path.stat().st_size for path in paths)
    with pytest.raises(FileExistsError):
        figure.render(diagnoses, fst, tmp_path)


def test_diagnosis_grid_exports_equal_top_and_bottom_sheets_with_legible_type(tmp_path):
    diagnoses = {dataset: figure.count_e5(REPO, dataset) for dataset in figure.ita.DATASETS}
    diagnoses["pad_ufes_20"] = figure.pad.EXPECTED_DIAGNOSTIC_COUNTS
    paths = figure.render_diagnosis_grid(diagnoses, tmp_path)
    assert len(paths) == 6
    assert {path.suffix for path in paths} == {".pdf", ".svg", ".png"}
    assert {path.stem for path in paths} == {
        "fig_obj1_diagnosis_top", "fig_obj1_diagnosis_bottom",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    top_svg = (tmp_path / "fig_obj1_diagnosis_top.svg").read_text(encoding="utf-8")
    bottom_svg = (tmp_path / "fig_obj1_diagnosis_bottom.svg").read_text(encoding="utf-8")
    assert "HAM10000" in top_svg and "ISIC 2019" in top_svg
    assert "ISIC 2020" in bottom_svg and "PAD-UFES-20" in bottom_svg
    assert "font-size: 28.0px" not in top_svg + bottom_svg
    assert "#ffffff" in top_svg and "#ffffff" in bottom_svg
    with pytest.raises(FileExistsError):
        figure.render_diagnosis_grid(diagnoses, tmp_path)
