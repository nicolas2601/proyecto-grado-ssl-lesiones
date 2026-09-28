"""The C1 and v7 result tables say "--check lo compara en CI": this is that check.

Both READMEs are rendered from committed JSON. A hand-edited number, or a
JSON updated without regenerating the table, fails here.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_c1_readme  # noqa: E402
import build_final_eval_table  # noqa: E402

V7_DIR = REPO / "resultados" / "clasificacion" / "v7"


def test_c1_readme_matches_its_json():
    assert build_c1_readme.main(["--check"]) == 0


def test_v7_readme_matches_its_json():
    doc = V7_DIR / "final_eval.json"
    assert build_final_eval_table.main([str(doc), "--out", str(V7_DIR / "README.md"), "--check"]) == 0
