"""scripts/build_segmentation_table.py: registered thresholds, verdicts, rendering."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_segmentation_table as bst  # noqa: E402


def test_direction_says_hd95_is_lower_is_better():
    assert "menor es mejor" in bst.DIRECTION["hd95"]
    assert "↓" in bst.DIRECTION["hd95"]
    assert "mayor es mejor" in bst.DIRECTION["dice"]
    assert "mayor es mejor" in bst.DIRECTION["iou"]


def test_registered_thresholds_are_the_ones_in_docs27():
    assert bst.S1_COVERAGE_BAND == (0.21, 0.27)
    assert bst.MIN_COVERAGE_GAIN == 0.02
    assert bst.HAIR_MARKED_MAX == 0.50
    assert bst.HAIR_PAINTED_MAX == 0.03
    assert bst.PRECISION_MAX_ABS_DELTA == 0.005


@pytest.mark.parametrize("coverage, expected", [(0.24, bst.MET), (0.21, bst.MET),
                                                (0.13376, bst.NOT_MET), (0.30, bst.NOT_MET)])
def test_s1_verdict(coverage, expected):
    assert bst.s1_verdict(coverage) == expected


@pytest.mark.parametrize("coverage, expected", [(0.151, bst.MET), (0.13328, bst.NOT_MET)])
def test_coverage_gain_verdict(coverage, expected):
    assert bst.gain_verdict(coverage, r0=0.13) == expected


@pytest.mark.parametrize("marked, painted, ok", [(0.49, 0.029, True), (0.50, 0.01, False),
                                                 (0.10, 0.03, False), (0.825, 0.0103, False)])
def test_hair_sanity(marked, painted, ok):
    assert bst.hair_sanity(marked, painted) is ok


def test_s2prime_is_not_interpreted_when_the_sanity_fails():
    verdict = bst.s2prime_verdict(0.20, r0=0.13, marked=0.825, painted=0.01)
    assert verdict == bst.NOT_INTERPRETED


def test_s2prime_reads_the_gain_when_the_sanity_passes():
    assert bst.s2prime_verdict(0.16, r0=0.13, marked=0.2, painted=0.01) == bst.MET
    assert bst.s2prime_verdict(0.131, r0=0.13, marked=0.2, painted=0.01) == bst.NOT_MET


def test_precision_control_verdict():
    assert bst.precision_valid(0.121653, 0.121636) is True
    assert bst.precision_valid(0.13, 0.12) is False


def test_missing_tta_summary_is_omitted(tmp_path):
    real = bst.STUDENT["student_r3"]
    paths = {"student_r3": real, "student_r3 + TTA": {"isic2018": tmp_path / "nope.json"}}
    loaded = bst.load_student(paths)
    assert list(loaded) == ["student_r3"]


def test_committed_data_reproduces_the_numbers_in_docs27():
    text = bst.render(bst.load_sources())
    for number in ("0,13045", "0,13376", "0,13328", "0,13119", "48.200 / 58.457", "82,5 %",
                   "+0,0033", "+0,0028", "+0,0007", "−0,000016"):
        assert number in text, number
    assert bst.NOT_INTERPRETED in text
    assert "↓ menor es mejor" in text


def test_every_student_row_shows_hd95_mean_and_median_labelled():
    src = bst.load_sources()
    text = bst.render(src)
    header = [ln for ln in text.splitlines() if ln.startswith("| Variante | Referencia |")][0]
    assert "HD95 media [IC 95 %]" in header
    assert "HD95 mediana" in header
    for variant, refs in src["student"].items():
        for ref, summary in refs.items():
            row = [ln for ln in text.splitlines()
                   if ln.startswith(f"| {variant} | {bst.REFERENCES[ref]} |")]
            assert len(row) == 1
            hd95 = summary["metrics"]["hd95"]
            assert bst._num(hd95["median"], 2) in row[0]
            assert bst._num(hd95["mean"], 2) in row[0]
