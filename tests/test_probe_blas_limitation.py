"""The BLAS-threads limitation note is rendered from committed diagnostics, never typed.

Every number of the note (sensitivity, threads, probe_jobs, reproduction delta,
convergence counts, SSL effect and the T4 P1 threshold) comes from the JSONs of
commits 0a691eb / 278efd2, from ``v71_final_eval.json`` and from
``t4_sog.P1_MAX_ABS_DELTA``. Perturbing a JSON value must change the text.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from dermssl import probe_blas_limitation as pbl
from dermssl import t4_sog

REPO = Path(__file__).resolve().parents[1]
T4_DIR = REPO / "resultados" / "clasificacion" / "t4"
V71_EVAL = REPO / "resultados" / "clasificacion" / "v71" / "v71_final_eval.json"


def _v71() -> dict:
    return json.loads(V71_EVAL.read_text(encoding="utf-8"))


def _copy_t4(tmp_path: Path) -> Path:
    for rel in (pbl.REPRO_DIR, pbl.CONVERGENCE_DIR):
        shutil.copytree(T4_DIR / rel, tmp_path / rel)
    return tmp_path


def _edit(path: Path, edit) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    edit(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_the_committed_facts_read_off_the_diagnostics():
    facts = pbl.load_facts(T4_DIR, _v71())
    assert facts["threads"] == (4, 8) and facts["probe_jobs"] == (10, 6)
    assert facts["sensitivity"] == pytest.approx(0.013968646472301227)
    assert facts["reference_vs_published"] == 0.0
    assert facts["changed_cells"] == (48, 48) and facts["cells_per_unit"] == 60
    assert facts["n_fits"] == 30 and facts["n_hit_max_iter"] == 0
    assert facts["n_iter"] == (165, 862) and facts["max_iter"] == 2000
    assert facts["p1"] == t4_sog.P1_MAX_ABS_DELTA


def test_the_paragraph_carries_the_numbers_in_spanish_notation():
    text = pbl.paragraph(pbl.load_facts(T4_DIR, _v71()))
    for piece in ("hasta 0,014 de macro-F1", "(4 vs 8 hilos; probe_jobs 10 vs 6", "máx. |Δ| 0)",
                  "+0,091 [+0,050; +0,132]", "~6,5×", "0/30 alcanzan max_iter=2000", "n_iter 165–862",
                  "(≤ 0,01)", "0a691eb", "278efd2", "48 de 60 celdas"):
        assert piece in text, piece


def test_perturbing_the_sensitivity_changes_the_text(tmp_path):
    t4 = _copy_t4(tmp_path)
    before = pbl.paragraph(pbl.load_facts(t4, _v71()))
    _edit(t4 / pbl.REPRO_DIR / pbl.THREADS_RUN,
          lambda d: d["a_vs_b"]["max_abs_delta"].__setitem__("macro_f1", 0.02))
    after = pbl.paragraph(pbl.load_facts(t4, _v71()))
    assert before != after and "hasta 0,020 de macro-F1" in after and "~4,5×" in after


def test_perturbing_the_convergence_changes_the_text(tmp_path):
    t4 = _copy_t4(tmp_path)
    _edit(t4 / pbl.CONVERGENCE_DIR / pbl.CONVERGENCE_FILE, lambda d: d["fits"][0].__setitem__("n_iter", [1999]))
    _edit(t4 / pbl.CONVERGENCE_DIR / pbl.CONVERGENCE_FILE, lambda d: d["summary"].__setitem__("n_hit_max_iter", 1))
    text = pbl.paragraph(pbl.load_facts(t4, _v71()))
    assert "n_iter 165–1999" in text and "1/30 alcanzan" in text


def test_the_ssl_effect_comes_from_the_v71_paired_row():
    doc = _v71()
    for row in doc["paired"]:
        if (row["arm"], row["control"], row["metric"]) == ("A71", "public_A71", "macro_f1"):
            row["point_delta"] = 0.1
    assert "macro-F1 +0,100 [" in pbl.paragraph(pbl.load_facts(T4_DIR, doc))


def test_an_unreproduced_reference_run_is_refused(tmp_path):
    t4 = _copy_t4(tmp_path)
    _edit(t4 / pbl.REPRO_DIR / pbl.THREADS_RUN,
          lambda d: d["a_vs_published"]["max_abs_delta"].__setitem__("macro_f1", 0.003))
    with pytest.raises(ValueError, match="publicada"):
        pbl.load_facts(t4, _v71())


def test_the_p1_threshold_is_the_one_registered_in_docs31():
    doc = (REPO / t4_sog.REGISTRATION_DOC).read_text(encoding="utf-8")
    assert f"|Δ macro-F1| ≤ {pbl.number(t4_sog.P1_MAX_ABS_DELTA)}" in doc


def test_available_needs_every_diagnostic(tmp_path):
    assert pbl.available(T4_DIR)
    assert not pbl.available(tmp_path)
