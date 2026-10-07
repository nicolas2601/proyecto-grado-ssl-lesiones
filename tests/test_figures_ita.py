"""Figure 1 (ITA per dataset): every number is recomputed from the raw JSON, and build() writes the three exports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

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


def test_build_writes_pdf_svg_and_png(tmp_path):
    paths = fig1.build(REPO, tmp_path)
    assert sorted(p.name for p in paths) == ["fig1_ita.pdf", "fig1_ita.png", "fig1_ita.svg"]
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)
    assert "no validada de fototipo" in next(p for p in paths if p.suffix == ".svg").read_text()
