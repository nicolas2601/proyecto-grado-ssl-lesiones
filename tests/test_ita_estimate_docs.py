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
BARE_FST = re.compile(r"\bFST\b")

# A pre-registration is never edited afterwards, not even a word: its old
# labels stay, and a fixed errata line follows the block, outside it.
ERRATA = ("Errata (2026-09-29): el rótulo 'FST' en este bloque se refiere a la estimación ITA no "
          "validada (ver docs/20, PMC13459198), y 'terciles de tono' se lee como 'terciles de luminancia'")
_MARK = re.compile(r"registrad", re.IGNORECASE)
_BOLD = re.compile(r"\*\*((?:\\\*|[^*])+?)\*\*")
_HEADING = re.compile(r"^(#{1,6})\s")
_CONTINUATION = re.compile(r"^(\s+\S|[-*+]\s|\d+[.)]\s)")
_HTML_START = re.compile(r"^<!--\s*[\w-]*registro[\w-]*:inicio\s*-->")
_HTML_END = re.compile(r"^<!--\s*[\w-]*registro[\w-]*:fin\s*-->")


def _next_filled(lines: list[str], i: int) -> int:
    while i < len(lines) and not lines[i].strip():
        i += 1
    return i


def _section_end(lines: list[str], i: int, level: int) -> int:
    """A heading that says "registrado" owns everything up to the next heading of its level."""
    j = i + 1
    while j < len(lines) and lines[j].strip() != ERRATA:
        head = _HEADING.match(lines[j])
        if head and len(head.group(1)) <= level:
            break
        j += 1
    return j


def _html_end(lines: list[str], i: int) -> int:
    j = i + 1
    while j < len(lines) and not _HTML_END.match(lines[j]):
        j += 1
    return min(j + 1, len(lines))


def _run_end(lines: list[str], i: int) -> int:
    """A paragraph or list: blank lines only continue it into more list items or indented lines."""
    j, is_list = i, False
    while True:
        while j < len(lines) and lines[j].strip() and not _HEADING.match(lines[j]) \
                and lines[j].strip() != ERRATA:
            is_list |= bool(_CONTINUATION.match(lines[j]))
            j += 1
        k = _next_filled(lines, j)
        if j == k or k == len(lines) or not is_list or not _CONTINUATION.match(lines[k]):
            return j
        j = k


def registered_blocks(text: str) -> list[tuple[int, int, str]]:
    """``(start, end, body)`` of every block that says it was registered, by its own marker."""
    lines, out, i = text.splitlines(), [], 0
    while (i := _next_filled(lines, i)) < len(lines):
        head = _HEADING.match(lines[i])
        if head:
            end = _section_end(lines, i, len(head.group(1))) if _MARK.search(lines[i]) else i + 1
            marked = end > i + 1
        elif _HTML_START.match(lines[i]):
            end, marked = _html_end(lines, i), True
        else:
            end = _run_end(lines, i) or i + 1
            marked = any(_MARK.search(b) for line in lines[i:end] for b in _BOLD.findall(line))
        if marked:
            out.append((i, end, "\n".join(lines[i:end])))
        i = max(end, i + 1) if not head or marked else i + 1
    return out


def _errata_follows(text: str, end: int) -> bool:
    lines = text.splitlines()
    k = _next_filled(lines, end)
    return k < len(lines) and lines[k].strip() == ERRATA


def unannotated_labels(text: str) -> list[int]:
    """Start line (1-based) of registered blocks that use the old labels without the errata."""
    return [start + 1 for start, end, body in registered_blocks(text)
            if (BARE_FST.search(body.replace(LABEL, "")) or TONE_TERCILE.search(body))
            and not _errata_follows(text, end)]


def _excused_lines(text: str) -> set[int]:
    return {n for start, end, _ in registered_blocks(text) if _errata_follows(text, end)
            for n in range(start + 1, end + 1)}


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
    offenders = []
    for p in _scanned():
        text = p.read_text(encoding="utf-8")
        excused = _excused_lines(text)
        offenders += [f"{p.relative_to(REPO)}:{i}" for i, line in enumerate(text.splitlines(), 1)
                      if TONE_TERCILE.search(line) and i not in excused and line.strip() != ERRATA]
    assert not offenders, f"terciles of L* called tone/lightness instead of luminancia: {offenders}"


@pytest.mark.parametrize("doc", NOTE_DOCS)
def test_a_registered_block_keeps_the_old_labels_only_with_the_errata_after_it(doc):
    bad = unannotated_labels((REPO / doc).read_text(encoding="utf-8"))
    assert not bad, f"{doc}: registered blocks at lines {bad} say FST/tercil de tono without the errata"


_PARAGRAPH = "Texto.\n\n**Método (registrado el 2026-09-28).** El FST por imagen.\n- sigue la lista\n"


@pytest.mark.parametrize("text", [
    _PARAGRAPH,
    "- **Predicción registrada:** los terciles de tono no cambian.\n\n  tabla indentada\n",
    "### 5.2 Algo (registrado el 2026-09-28)\n\nEl FST.\n\n#### 5.2.1 sub\n\nMás.\n\n## 6. Otra\n",
    "<!-- registro-x:inicio -->\nEl FST.\n<!-- registro-x:fin -->\n",
], ids=["bold-paragraph", "list-item", "heading-section", "html-markers"])
def test_a_registered_block_with_old_labels_and_no_errata_is_flagged(text):
    assert unannotated_labels(text), "the registered block was not detected"
    fixed = unannotated_labels(text.rstrip("\n").split("\n\n## 6.")[0] + f"\n\n{ERRATA}\n")
    assert not fixed, "the errata right after the block must excuse it"


def test_the_errata_must_follow_the_block_not_sit_elsewhere():
    text = f"{ERRATA}\n\n{_PARAGRAPH}\nOtro párrafo.\n"
    assert unannotated_labels(text) == [5]


def test_an_unregistered_block_is_not_excused_by_an_errata():
    text = f"Resultado: el FST por imagen.\n\n{ERRATA}\n"
    assert registered_blocks(text) == []
