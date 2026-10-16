"""Build resultados/TABLA_CIERRE.md: professor's and Pau's requests -> what was done -> result.

Every number is read from a committed JSON; every verdict is decided by the
rules in ``closing_table_rules.py``. A registered item without a data file
in the repo renders as pending. ``--check`` fails in CI if the table drifts.

    python scripts/build_closing_table.py            # write
    python scripts/build_closing_table.py --check    # CI
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import closing_rows_classifier as clf  # noqa: E402
import closing_rows_exploratory as exrows  # noqa: E402
import closing_rows_limitations as limits  # noqa: E402
import closing_rows_segmentation as segrows  # noqa: E402
from closing_table_rules import (  # noqa: E402, F401
    BETTER, DISTINGUISHABLE, DONE, ITA_LABEL, ITA_NOTE, LUMINANCE_NOTE, LUMINANCE_TERCILES, MEETS_THRESHOLD, MET, NOT_MET, PENDING, REPO, SAME,
    WORSE, Row, RowSpec, below_without_overlap, build_row,
    mask_helps, paired_verdict, s4_threshold_met, seeds_verdict)

TARGET = REPO / "resultados" / "TABLA_CIERRE.md"
A2_VS_STUDENT = clf.A2_VS_STUDENT
HEADER = ["| # | Origen | Pedido | Qué se hizo | Resultado | Veredicto | Datos |",
          "|---:|---|---|---|---|---|---|"]


def specs() -> list[RowSpec]:
    return clf.rows() + segrows.rows() + exrows.rows()


def build_rows(root: Path = REPO) -> list[Row]:
    return [build_row(s, root) for s in specs()]


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def table(rows: list[Row]) -> list[str]:
    lines = list(HEADER)
    for i, row in enumerate(rows, 1):
        cells = (row.origen, row.pedido, row.hecho, row.resultado, f"**{row.veredicto}**", row.fuente)
        lines.append(f"| {i} | " + " | ".join(_cell(c) for c in cells) + " |")
    return lines


def rules() -> list[str]:
    return [
        "## Cómo se decide el veredicto", "",
        f"- **{MET} / {NOT_MET}**: el ítem tiene una predicción o un criterio registrado antes del dato "
        "(docs/26, docs/27, docs/28), y el script lo evalúa con esa regla.",
        f"- **{DONE}**: el pedido es una acción sin predicción numérica; el artefacto que la prueba está "
        "commiteado, pero no hay una mejora medida contra un umbral.",
        f"- **{MEETS_THRESHOLD} (…)**: se cumple el umbral registrado, pero no es una mejora; el paréntesis, "
        "calculado desde los datos, dice por qué (por ejemplo, la cobertura supera el umbral porque el método "
        "sobre-segmenta).",
        f"- **{BETTER} / {SAME} / {WORSE}**: comparación sin umbral registrado; decide el IC 95 % pareado. "
        f"**{BETTER}** o **{WORSE}** solo si el intervalo excluye el cero; si lo incluye, **{SAME}**.",
        "- **T1–T7**: bloques exploratorios registrados antes del dato (docs/27, docs/28, docs/30, docs/31). "
        "El veredicto es el que escribe el JSON de cada experimento; aquí se copia, no se recalcula, y una "
        f"fila cuyos veredictos no coinciden falla. **{DISTINGUISHABLE}** es el veredicto intermedio de T2: "
        "el IC excluye el cero, pero no queda entero bajo el margen registrado (ver la fila).",
        f"- **{PENDING}**: el ítem está registrado pero su archivo de datos no está en el repositorio. "
        "No se completa a mano.",
        "- Los IC son percentiles de bootstrap. Los de C1 y S4 son pareados sobre las mismas imágenes; "
        "los de S-diag contra student_r3 no (no hay comparación pareada de S-diag en el repo), así que "
        "ahí se exige que los intervalos no se solapen. Las estadísticas de pseudo-máscaras y de la "
        f"{ITA_LABEL} no traen IC y se leen contra el umbral registrado.",
        f"- **{ITA_LABEL}**: la estimación por imagen del proyecto es el ITA con los umbrales fijos de Groh "
        "(`src/dermssl/fitzpatrick.py`), no un fototipo validado. Toda fila que lo usa lleva esta nota, "
        f"agregada por regla: «{ITA_NOTE}»",
        f"- **{LUMINANCE_TERCILES}**: los terciles salen de la mediana de L\\* de la piel (`tone_tercile`), "
        "no del ITA ni de un fototipo. Toda fila que los usa lleva, también por regla, "
        f"«{LUMINANCE_NOTE}» Una fila que usa ambos lleva las dos notas.",
        "- En Dice e IoU mayor es mejor; en HD95 y ECE, menor es mejor.",
    ]


from dermssl.display_nomenclature import presentation  # noqa: E402


@presentation
def render(rows: list[Row], root: Path = REPO) -> str:
    detail = segrows.sdiag_detail({p: json.loads((root / p).read_text(encoding="utf-8"))
                                   for p in segrows.SDIAG_ALL + tuple(segrows.STUDENT.values())})
    lines = ["# Cierre: pedidos del profesor y plan de Pau → qué se hizo → resultado", "",
             "Generado por `scripts/build_closing_table.py` desde JSON commiteados. No editar a mano: "
             "`--check` lo compara en CI. Los documentos de origen son "
             "[docs/26](../docs/26-plan-correccion-segmentacion-y-clasificador.md), "
             "[docs/27](../docs/27-plan-v7-mejoras-ssl.md) y [docs/28](../docs/28-rediseno-brazo-propuesto-v7.md).",
             "", *rules(), "", "## Tabla de cierre", "", *table(rows), "", *limits.section(root, _cell), "",
             "## Detalle de S-diag (los 18 resúmenes)", "",
             "Contra student_r3 en la misma referencia; «por debajo, sin solape» = el límite superior del IC "
             "de S-diag queda bajo el límite inferior del de student_r3.", "", *detail]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render(build_rows())
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != text:
            print(f"{TARGET} esta desactualizado: corre scripts/build_closing_table.py")
            return 1
        return 0
    TARGET.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
