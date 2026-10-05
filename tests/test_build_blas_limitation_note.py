"""docs/31 carries the BLAS-threads limitation in a generated region AFTER the registered block.

The region is filled by ``scripts/build_blas_limitation_note.py`` from the same
renderer as the v7.1 README; ``--check`` fails on a stale region, and the
registered T4 block stays byte-identical to its registration commit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_blas_limitation_note as bn  # noqa: E402

from dermssl import probe_blas_limitation as pbl  # noqa: E402
from dermssl import t4_sog  # noqa: E402
from dermssl.pad_ufes_guard import registered_block  # noqa: E402

DOC = REPO / t4_sog.REGISTRATION_DOC
START, END = "<!-- registro-t4-sog:inicio -->", "<!-- registro-t4-sog:fin -->"


def _doc(region: str = "viejo") -> str:
    return (f"# t\n\n{START}\nbloque\n{END}\n\n## Notas\n\n"
            f"{pbl.REGION_START}\n{region}\n{pbl.REGION_END}\n\ncola\n")


def test_fill_region_only_touches_the_region():
    out = pbl.fill_region(_doc(), ["nuevo", "", "dos"])
    assert out == _doc("nuevo\n\ndos")


def test_fill_region_requires_one_region_after_the_registered_block():
    with pytest.raises(ValueError):
        pbl.fill_region("sin marcadores\n", ["x"])
    inside = f"{START}\n{pbl.REGION_START}\n{pbl.REGION_END}\n{END}\n"
    with pytest.raises(ValueError, match="después"):
        pbl.fill_region(inside, ["x"])


def test_check_detects_a_stale_region(tmp_path):
    doc = tmp_path / "31.md"
    doc.write_text(_doc(), encoding="utf-8")
    assert bn.main(["--doc", str(doc), "--check"]) == 1
    assert bn.main(["--doc", str(doc)]) == 0
    assert bn.main(["--doc", str(doc), "--check"]) == 0
    assert registered_block(doc.read_text(encoding="utf-8"), START, END) == f"{START}\nbloque\n{END}"


def test_committed_docs31_region_matches_the_render():
    assert bn.main(["--check"]) == 0
    text = DOC.read_text(encoding="utf-8")
    assert pbl.paragraph(bn.facts()) in text
    assert text.index(END) < text.index(pbl.REGION_START)


def test_the_registered_block_is_unchanged_since_registration():
    shown = subprocess.run(["git", "-C", str(REPO), "show", f"{t4_sog.REGISTRATION_COMMIT}:{t4_sog.REGISTRATION_DOC}"],
                           capture_output=True, text=True)
    if shown.returncode != 0:
        pytest.skip("registration commit not in this clone")
    now = registered_block(DOC.read_text(encoding="utf-8"), START, END)
    assert now == registered_block(shown.stdout, START, END)
