"""The closing table's limitations section: measured facts, not experiments.

The BLAS-thread note is the one function that already renders it for the
v7.1 README and docs/31; the probe-convergence and C1 thread rows read their
own diagnostics, and the perturbation tests pin each number to its field.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_closing_table as ct  # noqa: E402
import closing_rows_limitations as lim  # noqa: E402

from dermssl import probe_blas_limitation as pbl  # noqa: E402


def _rendered() -> str:
    return ct.render(ct.build_rows())


def _copy_sources(tmp_path: Path) -> None:
    for source in lim.sources():
        (tmp_path / source).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / source, tmp_path / source)


def _edit(tmp_path: Path, path: Path, keys: tuple, value) -> None:
    doc = json.loads((tmp_path / path).read_text(encoding="utf-8"))
    node = doc
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value
    (tmp_path / path).write_text(json.dumps(doc), encoding="utf-8")


def _text(title_fragment: str, root: Path = REPO) -> str:
    return next(text for title, text, _ in lim.limitation_rows(root) if title_fragment in title)


def test_the_section_follows_the_closing_table_and_precedes_the_sdiag_detail():
    text = _rendered()
    assert text.index("## Tabla de cierre") < text.index(lim.HEADING) < text.index("## Detalle de S-diag")


def test_the_blas_row_is_the_shared_note_verbatim():
    facts = pbl.load_facts(REPO / lim.T4_DIR, json.loads((REPO / lim.V71_EVAL).read_text(encoding="utf-8")))
    assert ct._cell(pbl.paragraph(facts)) in _rendered()


def test_the_convergence_row_reads_its_json(tmp_path):
    _copy_sources(tmp_path)
    doc = json.loads((REPO / lim.CONVERGENCE).read_text(encoding="utf-8"))
    summary = doc["summary"]
    assert f"{summary['n_hit_max_iter']}/{summary['n_fits']}" in _text("Convergencia")
    _edit(tmp_path, lim.CONVERGENCE, ("summary", "n_convergence_warning"), 7)
    assert "avisos de convergencia: 7" in _text("Convergencia", tmp_path)


def test_the_c1_thread_row_reads_its_json(tmp_path):
    _copy_sources(tmp_path)
    summary = json.loads((REPO / lim.C1_THREADS).read_text(encoding="utf-8"))["summary"]
    text = _text("C1-v7")
    assert f"fijar C1 en {summary['pin_to']} hilos" in text
    assert ", ".join(summary["reproducing_settings"]) in text
    _edit(tmp_path, lim.C1_THREADS, ("summary", "per_setting", "1", "max_abs_delta_macro_f1"), 0.4321)
    assert "0,4321" in _text("C1-v7", tmp_path)


def test_limitation_rows_carry_no_verdict_and_link_their_data():
    for title, text, fuente in lim.limitation_rows(REPO):
        assert "**" not in text
        assert fuente.startswith("[`")
