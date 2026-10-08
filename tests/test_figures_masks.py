"""Figure 3 (segmentation masks): the data layer reads the committed per_image.csv files and picks the
qualitative cases by Dice percentile, never by eye. Every expectation is recomputed here from the raw CSV."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig3_masks as f3  # noqa: E402

RESULTS = REPO / "plan-segmentacion/camino-b-cnn/resultados"
EXPECTED_CSV = {
    ("student_r3", "ham10000"): RESULTS / "2026-09-21-final/evaluacion_ham10000/per_image.csv",
    ("student_r3", "isic2018"): RESULTS / "2026-09-21-final/evaluacion_isic2018/per_image.csv",
    ("A2", "ham10000"): RESULTS / "2026-09-28-s4-a2/evaluacion_ham10000/per_image.csv",
    ("A2", "isic2018"): RESULTS / "2026-09-28-s4-a2/evaluacion_isic2018/per_image.csv",
}


def _dice(path: Path) -> dict[str, float]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["image_id"]: float(row["dice"]) for row in csv.DictReader(handle)}


def test_the_sources_are_the_four_documented_per_image_csv_files():
    assert {key: REPO / rel for key, rel in f3.SOURCES.items()} == EXPECTED_CSV
    assert all(path.is_file() for path in EXPECTED_CSV.values())


@pytest.mark.parametrize("dataset", ["ham10000", "isic2018"])
def test_cases_are_the_images_nearest_to_p90_p50_p10_of_the_mean_dice(dataset):
    r3, a2 = _dice(EXPECTED_CSV[("student_r3", dataset)]), _dice(EXPECTED_CSV[("A2", dataset)])
    ids = sorted(set(r3) & set(a2))
    mean = np.array([(r3[i] + a2[i]) / 2 for i in ids])
    cases = [c for c in f3.select_cases(REPO) if c["dataset"] == dataset]
    assert [(c["role"], c["percentile"]) for c in cases] == [("bueno", 90), ("mediano", 50), ("malo", 10)]
    for case in cases:
        target = np.percentile(mean, case["percentile"])
        best = ids[int(np.argmin(np.abs(mean - target)))]
        assert case["image_id"] == best
        assert case["dice"] == {"student_r3": r3[best], "A2": a2[best]}


def test_the_committed_cases_file_is_what_the_selection_produces():
    committed = json.loads((REPO / f3.CASES_JSON).read_text(encoding="utf-8"))
    assert committed == f3.select_cases(REPO)


def test_distribution_summary_matches_the_csv_and_ids_are_shared_between_segmenters():
    data = f3.figure_data(REPO)
    for (segmenter, dataset), path in EXPECTED_CSV.items():
        values = np.array(list(_dice(path).values()))
        row = data["dice"][dataset][segmenter]
        assert row["n"] == len(values) and row["mean"] == pytest.approx(values.mean())
        assert row["median"] == pytest.approx(np.median(values)) and row["p10"] == pytest.approx(np.percentile(values, 10))
    for dataset in ("ham10000", "isic2018"):
        assert set(_dice(EXPECTED_CSV[("student_r3", dataset)])) == set(_dice(EXPECTED_CSV[("A2", dataset)]))


def _synthetic_npz(path: Path) -> None:
    """Six fake cases (test only, never committed): a disc as the lesion and shifted discs as the predictions."""
    yy, xx = np.mgrid[:224, :224]
    disc = lambda dx: ((yy - 112) ** 2 + (xx - 112 - dx) ** 2 < 50**2).astype(np.uint8)  # noqa: E731
    arrays = {}
    for case in f3.select_cases(REPO):
        base = f"{case['dataset']}__{case['image_id']}"
        arrays[f"{base}__image"] = np.full((224, 224, 3), 200, dtype=np.uint8)
        arrays[f"{base}__reference"] = disc(0)
        arrays[f"{base}__student_r3"], arrays[f"{base}__A2"] = disc(8), disc(-8)
    np.savez_compressed(path, **arrays)


def test_build_requires_real_pod_cases_unless_panel_a_only_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="panel b requires verified pod cases"):
        f3.build(REPO, tmp_path, cases_npz=tmp_path / "missing.npz")
    result = f3.build(REPO, tmp_path, cases_npz=tmp_path / "missing.npz", panel_a_only=True)
    assert result["panel_b"] is False
    assert {p.suffix for p in result["paths"]} == {".pdf", ".svg", ".png"} and all(p.is_file() for p in result["paths"])


def test_build_with_the_pod_cases_adds_the_qualitative_panel(tmp_path):
    npz = tmp_path / "casos.npz"
    _synthetic_npz(npz)
    result = f3.build(REPO, tmp_path, cases_npz=npz)
    assert result["panel_b"] is True and all(p.is_file() for p in result["paths"])


def test_a_cases_file_that_misses_a_case_is_refused(tmp_path):
    npz = tmp_path / "casos.npz"
    _synthetic_npz(npz)
    kept = {k: v for k, v in np.load(npz).items() if "isic2018" not in k}
    np.savez_compressed(npz, **kept)
    with pytest.raises(f3.CasesMismatchError):
        f3.build(REPO, tmp_path, cases_npz=npz)


def test_figure_3_keeps_the_width_and_text_size_of_figure_1_so_the_typography_matches_on_the_page():
    import fig1_ita as f1  # noqa: PLC0415

    assert f3.WIDTH == f1.WIDTH_IN and f3.TEXT_PT == f1.TEXT_PT
    assert f3.NOTE_PT < f3.TEXT_PT  # annotations stay one step below ticks, labels and legend
