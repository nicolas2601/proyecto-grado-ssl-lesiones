"""PAD phototypes and ITA share IDs but not the same measurement."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig_obj1_pad_ita as figure  # noqa: E402


def _source(tmp_path: Path, records: list[dict], missing: int = 1) -> Path:
    path = tmp_path / "pad.json"
    path.write_text(json.dumps({"n_images": len(records) + missing, "n_without_fst": missing,
                                "records": records}), encoding="utf-8")
    return path


def _record(image_id: str, fst: int, ita: float | None) -> dict:
    return {"img_id": image_id, "actual_fst": fst,
            "masked": {"status": "ok" if ita is not None else "ita_indeterminate", "ita": ita}}


def test_ita_is_grouped_only_for_same_image_with_valid_fst_and_masked_estimate(tmp_path):
    source = _source(tmp_path, [_record("a", 1, 42.0), _record("b", 1, None),
                                _record("c", 6, -30.0)])
    data = figure.load_data(source)
    assert data["n_total"] == 4
    assert data["n_missing_fst"] == 1
    assert data["fst_counts"] == {"I": 2, "II": 0, "III": 0, "IV": 0, "V": 0, "VI": 1}
    assert data["ita_by_fst"]["I"] == [42.0]
    assert data["ita_by_fst"]["VI"] == [-30.0]
    assert data["n_ita_unavailable"] == 1


@pytest.mark.parametrize("records,missing", [
    ([_record("a", 1, 42), _record("a", 2, 43)], 0),
    ([_record("a", 7, 42)], 0),
    ([_record("a", 1, float("nan"))], 0),
])
def test_invalid_join_or_ita_aborts(records, missing, tmp_path):
    with pytest.raises(ValueError):
        figure.load_data(_source(tmp_path, records, missing))


def test_ita_histogram_leaves_headroom_above_peak():
    import matplotlib.pyplot as plt
    import numpy as np

    data = figure.load_data(figure.SOURCE)
    peak = sum(np.histogram(data["ita_by_fst"][bucket], bins=figure.ITA_BINS)[0]
               for bucket in figure.BUCKETS).max()
    fig, ax = plt.subplots()
    figure._ita_histograms(ax, data)
    assert ax.get_ylim()[1] > peak
    plt.close(fig)


def test_figure_and_plot_panels_use_pure_white_background():
    import matplotlib.pyplot as plt

    data = figure.load_data(figure.SOURCE)
    fig, (phototypes, histogram) = plt.subplots(1, 2, facecolor=figure.BACKGROUND)
    figure._phototypes(phototypes, data)
    figure._ita_histograms(histogram, data)
    assert figure.BACKGROUND == "#FFFFFF"
    assert figure.PANEL == "#FFFFFF"
    assert all(tuple(round(channel * 255) for channel in ax.get_facecolor()[:3]) == (255, 255, 255)
               for ax in (phototypes, histogram))
    plt.close(fig)


def test_build_exports_three_formats_and_reconciles_counts(tmp_path):
    source = _source(tmp_path, [_record("a", 1, 42.0), _record("b", 2, 23.0)])
    paths = figure.build(source, tmp_path / "output")
    assert {path.suffix for path in paths} == {".pdf", ".svg", ".png"}
    assert all(path.stat().st_size > 0 for path in paths)
    summary = json.loads((tmp_path / "output" / "fig_obj1_pad_fst_ita.summary.json").read_text())
    assert summary["n_ita_valid"] == 2
    assert summary["n_missing_fst"] == 1
