"""Prose that reports the per-image ITA estimate must say it is not validated FST.

The per-image "FST" of this project is ITA with the fixed Groh cuts
(src/dermssl/fitzpatrick.py). Against expert FST it does not beat a trivial
baseline: 28,5-30,8 % vs 32,9 % on Fitzpatrick17k-C (docs/20) and 22,9 % vs
51,3 % on PAD-UFES-20 (Alipour et al. 2026, PMC13459198). Every document that
reports a result from it carries the note, and the `tone_tercile` strata
(terciles of the skin median L*) are called luminance terciles, never tone.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LABEL = "estimación ITA no validada (no es FST)"
CITED = ("docs/20", "28,5", "30,8", "32,9", "PMC13459198", "22,9", "51,3")

NOTE_DOCS = (
    "docs/20-esquema-fst-manifest.md",
    "docs/22-pipeline-propuesto-segmentacion-tono-clasificador.md",
    "docs/23-modelo-descargable-y-uso.md",
    "docs/24-analisis-final-resultados-y-escenarios-ada.md",
    "docs/26-plan-correccion-segmentacion-y-clasificador.md",
    "docs/27-plan-v7-mejoras-ssl.md",
    "docs/plan-segmentacion-sin-etiquetas.md",
    "resultados/README.md",
    "resultados/segmentacion/README.md",
    "resultados/tono/fst_diagnostico/README.md",
    "resultados/tono/fst_diagnostico_v2/README.md",
    "plan-segmentacion/README.md",
    "plan-segmentacion/comun/evaluacion/README.md",
    "plan-segmentacion/camino-b-cnn/resultados/README.md",
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-21-final/fst_diagnostico/README.md",
    "plan-segmentacion/camino-b-cnn/resultados/2026-09-21-final/fst_diagnostico_v2/README.md",
)

TONE_TERCILE = re.compile(r"tercil(es)?\s+(de|por|del)\s+(tono|luminosidad)", re.IGNORECASE)


def _scanned() -> list[Path]:
    paths = [*REPO.glob("docs/**/*.md"), *REPO.glob("resultados/**/*.md"),
             *REPO.glob("plan-segmentacion/**/README.md")]
    return sorted(p for p in paths if p.is_file())


@pytest.mark.parametrize("doc", NOTE_DOCS)
def test_a_doc_that_reports_the_estimate_carries_the_unvalidated_note(doc):
    text = (REPO / doc).read_text(encoding="utf-8")
    assert LABEL in text, f"{doc} reports the ITA estimate without saying it is not FST"
    missing = [c for c in CITED if c not in text]
    assert not missing, f"{doc}: the note does not cite {missing}"


def test_lightness_terciles_are_never_called_tone():
    offenders = [f"{p.relative_to(REPO)}:{i}" for p in _scanned()
                 for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
                 if TONE_TERCILE.search(line)]
    assert not offenders, f"terciles of L* called tone/lightness instead of luminancia: {offenders}"
