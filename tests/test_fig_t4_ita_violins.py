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

import fig_t4_ita_violins as figure  # noqa: E402


EXPECTED = {
    "ham10000": (10015, 4539, 5476, 0),
    "isic2019": (25331, 12558, 12773, 753),
    "isic2020": (33126, 21973, 11153, 1269),
}


def test_t4_loads_only_valid_e5_ita_and_reconciles_abstentions():
    data = figure.figure_data(REPO)

    assert set(data["datasets"]) == set(EXPECTED)
    for dataset, expected in EXPECTED.items():
        panel = data["datasets"][dataset]
        assert (panel["n_total"], panel["n_valid"], panel["n_abstention"],
                panel["n_fallback_no_mask_border"]) == expected
        assert len(panel["ita_values"]) == panel["n_valid"]
        assert all(-90 <= value <= 90 for value in panel["ita_values"])
        assert panel["procedure"] == "E5 perilesional ROI estimator"


def test_t4_reports_valid_no_mask_fallback_separately_from_abstentions():
    data = figure.figure_data(REPO)["datasets"]

    assert data["isic2019"]["n_fallback_no_mask_border"] == 753
    assert data["isic2019"]["n_abstention"] == 12773
    assert data["isic2020"]["n_fallback_no_mask_border"] == 1269
    assert data["isic2020"]["n_abstention"] == 11153
    assert data["ham10000"]["n_fallback_no_mask_border"] == 0


def test_t4_density_profile_is_deterministic_and_uses_all_values():
    centers, density = figure.density_profile([-10.0, 0.0, 0.0, 10.0])

    assert len(centers) == len(density) == 181
    assert density.max() == pytest.approx(1.0)
    assert density[89] == pytest.approx(density[91])
    assert density[80] == pytest.approx(density[100])


def test_t4_draw_uses_neutral_violins_and_separates_ita_groups():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    fig, axis = figure.draw(figure.figure_data(REPO))
    assert len(fig.axes) == 1
    assert list(axis.get_yticks()) == [-90, -25, 0, 12, 23, 40, 90]
    assert len(axis.collections) == 3
    expected_face = to_rgba(figure.style.COLORS["bar_light"])
    for collection, panel in zip(axis.collections, figure.figure_data(REPO)["datasets"].values(), strict=True):
        assert to_rgba(collection.get_facecolor()[0]) == expected_face
        ordinates = collection.get_paths()[0].vertices[:, 1]
        assert ordinates.min() >= min(panel["ita_values"]) - 4 * figure.SMOOTH_SIGMA_DEG
        assert ordinates.max() <= max(panel["ita_values"]) + 4 * figure.SMOOTH_SIGMA_DEG
    labels = [text.get_text() for text in axis.texts]
    assert all(group in labels for group in ("I", "II", "III", "IV", "V", "VI"))
    plt.close(fig)


def test_t4_exports_three_datasets_with_procedure_counts_and_hashes(tmp_path):
    outputs = figure.build(REPO, tmp_path)

    assert {path.suffix for path in outputs} == {".pdf", ".svg", ".png"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    note = (tmp_path / "fig_t4_ita_violins.note.txt").read_text(encoding="utf-8")
    caption = (tmp_path / "fig_t4_ita_violins.caption.txt").read_text(encoding="utf-8")
    assert "ROI perilesional" in caption
    assert "PAD-UFES-20" in note and "Otsu" in note and "T3" in note
    assert "no equivalen a fototipos clínicos validados" in note
    trace = json.loads((tmp_path / "fig_t4_ita_violins.traceability.json").read_text(encoding="utf-8"))
    assert set(trace["datasets"]) == set(EXPECTED)
    for dataset, panel in trace["datasets"].items():
        assert "ita_values" not in panel
        source = REPO / panel["source"]["path"]
        assert panel["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert panel["n_abstention"] == EXPECTED[dataset][2]
    with pytest.raises(FileExistsError):
        figure.build(REPO, tmp_path)
