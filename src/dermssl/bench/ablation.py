"""TABLA_ABLACION.md and its LaTeX twin: all our methods in one document (docs/33, section 6)."""

from __future__ import annotations

from pathlib import Path

from dermssl.bench import ablation_data, glossary, tables
from dermssl.bench.rows import DATASETS

SCRIPT = "scripts/build_bench_ablation.py"
NAME = "TABLA_ABLACION"
INTRO = ("Todos los números son de la partición de val de cada dataset y salen de un JSON del repositorio. F1-F5 y "
         "F4 son exploratorios: no cambian el propuesto (B71 sin segmentación). D0-D3 esperan a E2.")


from dermssl.display_nomenclature import context_presentation  # noqa: E402


@context_presentation("bench")
def render_markdown(repo: Path) -> str:
    lines = ["# Banco completo: ablación de nuestros métodos", "", INTRO, "",
             *glossary.preamble(repo, DATASETS, SCRIPT)]
    for table in ablation_data.tables(repo):
        lines += tables.markdown(table)
    return "\n".join(lines)


@context_presentation("bench")
def render_latex(repo: Path) -> str:
    return tables.latex_document(ablation_data.tables(repo), SCRIPT)


def outputs(repo: Path) -> dict[Path, str]:
    return {Path(f"{NAME}.md"): render_markdown(repo), Path("latex") / f"{NAME}.tex": render_latex(repo)}
