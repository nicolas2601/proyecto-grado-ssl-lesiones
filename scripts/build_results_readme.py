"""Write the classification tables of resultados/README.md from the bank output.

Transcribing numbers by hand is how a report drifts from its data: a rerun
changes bank.json and the prose keeps the old figure, and nobody notices
until someone checks. Here the tables between the markers are generated from
``bank.json`` and ``strata.json`` every time, so the document cannot disagree
with the run that produced it.

    python scripts/build_results_readme.py --bank salidas/tone/bank \\
        --readme resultados/README.md

``--check`` exits non-zero when the file would change, which is what CI runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BEGIN = "<!-- TABLA-CLASIFICACION:INICIO -->"
END = "<!-- TABLA-CLASIFICACION:FIN -->"
HEADLINE_PROBE = "logreg"
FULL_FRACTION = 1.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bank", type=Path, required=True, help="Directory holding bank.json and strata.json")
    parser.add_argument("--readme", type=Path, required=True, help="Markdown file with the markers")
    parser.add_argument("--check", action="store_true", help="Fail instead of writing when the file would change")
    return parser.parse_args(argv)


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "-"
    return f"{number:.{digits}f}".replace(".", ",")


def _rows_by_backbone(rows: list[dict], probe: str, fraction: float) -> dict[str, float]:
    """Mean of the headline metric per backbone, averaged over seeds."""
    collected: dict[str, list[float]] = {}
    for row in rows:
        if row.get("probe") != probe or float(row.get("fraction", -1)) != fraction:
            continue
        if row.get("error"):
            continue
        value = row.get("macro_f1")
        if value is None:
            continue
        collected.setdefault(row["backbone"], []).append(float(value))
    return {name: sum(values) / len(values) for name, values in collected.items() if values}


def classification_table(bank: dict) -> list[str]:
    """One row per backbone: the headline metric at both label regimes."""
    rows = bank.get("rows", [])
    full = _rows_by_backbone(rows, HEADLINE_PROBE, FULL_FRACTION)
    if not full:
        return ["_Sin filas de clasificacion en `bank.json`._"]
    scarce = _rows_by_backbone(rows, HEADLINE_PROBE, 0.1)
    balanced = _rows_by_backbone(rows, "logreg_balanced", FULL_FRACTION)

    lines = [
        "| Backbone | macro-F1 (logreg, 100 %) | macro-F1 (logreg, 10 %) | macro-F1 (balanced, 100 %) |",
        "|---|---|---|---|",
    ]
    for name in sorted(full, key=lambda key: full[key], reverse=True):
        lines.append(
            f"| `{name}` | {_fmt(full[name])} | {_fmt(scarce.get(name))} | {_fmt(balanced.get(name))} |"
        )
    return lines


def strata_table(strata: dict) -> list[str]:
    """Per backbone: the tone gap and whether the intervals actually separate."""
    if not strata:
        return ["_Sin estratos por tono en `strata.json`._"]
    lines = [
        "| Backbone | Brecha macro-F1 | Estrato peor | Intervalos separados |",
        "|---|---|---|---|",
    ]
    for name, entry in sorted(strata.items()):
        established = entry.get("gap_established")
        verdict = {True: "si", False: "no", None: "sin datos"}[established]
        lines.append(
            f"| `{name}` | {_fmt(entry.get('gap_macro_f1'))} | "
            f"{entry.get('worst_label') or '-'} | {verdict} |"
        )
    lines += [
        "",
        "Un `no` significa que los intervalos de los estratos extremos se solapan:",
        "esa brecha NO se cita como efecto de tono por grande que sea el numero.",
        "`sin datos` no es lo mismo que ausencia de brecha.",
    ]
    return lines


def render(bank: dict, strata: dict) -> str:
    lines = [
        BEGIN,
        "",
        "### Rendimiento",
        "",
        *classification_table(bank),
        "",
        "### Equidad por tono de piel",
        "",
        *strata_table(strata),
        "",
        f"_Generado desde `bank.json` por `scripts/build_results_readme.py`; "
        f"modo de la corrida: {bank.get('mode', 'desconocido')}._",
        "",
        END,
    ]
    return "\n".join(lines)


def replace_block(markdown: str, block: str) -> str:
    start = markdown.find(BEGIN)
    end = markdown.find(END)
    if start == -1 or end == -1:
        raise SystemExit(f"faltan los marcadores {BEGIN} / {END} en el documento")
    return markdown[:start] + block + markdown[end + len(END):]


def main() -> None:
    args = parse_args()
    bank = json.loads((args.bank / "bank.json").read_text(encoding="utf-8"))
    strata_path = args.bank / "strata.json"
    strata = json.loads(strata_path.read_text(encoding="utf-8")) if strata_path.is_file() else {}

    current = args.readme.read_text(encoding="utf-8")
    updated = replace_block(current, render(bank, strata))

    if args.check:
        if current != updated:
            print("resultados/README.md esta desactualizado respecto de bank.json", file=sys.stderr)
            raise SystemExit(1)
        print("las tablas coinciden con bank.json")
        return

    args.readme.write_text(updated, encoding="utf-8")
    print(f"tablas actualizadas en {args.readme}")


if __name__ == "__main__":
    main()
