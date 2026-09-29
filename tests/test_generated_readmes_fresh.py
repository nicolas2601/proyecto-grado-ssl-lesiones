"""The C1 and v7 result tables say "--check lo compara en CI": this is that check.

Both READMEs are rendered from committed JSON. A hand-edited number, or a
JSON updated without regenerating the table, fails here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_c1_readme  # noqa: E402
import build_classifier_table  # noqa: E402
import build_closing_table  # noqa: E402
import build_final_eval_table  # noqa: E402
import build_seed_summary  # noqa: E402
import build_segmentation_table  # noqa: E402

V7_DIR = REPO / "resultados" / "clasificacion" / "v7"


def test_c1_readme_matches_its_json():
    assert build_c1_readme.main(["--check"]) == 0


@pytest.mark.parametrize("folder", [V7_DIR, V7_DIR / "s43", V7_DIR / "s44"], ids=["seed42", "seed43", "seed44"])
def test_v7_readme_matches_its_json(folder):
    doc = folder / "final_eval.json"
    assert build_final_eval_table.main([str(doc), "--out", str(folder / "README.md"), "--check"]) == 0


def test_seed_summary_matches_every_seed_json():
    assert build_seed_summary.main(["--check"]) == 0


def test_classifier_table_matches_its_sources():
    assert build_classifier_table.main(["--check"]) == 0


def test_segmentation_table_matches_its_sources():
    assert build_segmentation_table.main(["--check"]) == 0


def test_closing_table_matches_its_sources():
    assert build_closing_table.main(["--check"]) == 0
