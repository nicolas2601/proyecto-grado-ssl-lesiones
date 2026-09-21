"""Tabla comparativa de varias evaluaciones, a partir de sus summary.json.

Cada corrida de ``evaluate_segmentation`` deja un ``summary.json``. Este script
los junta en una sola tabla Markdown (y su JSON equivalente) para el informe,
sin volver a tocar imagenes ni predicciones: lee numeros ya calculados y los
formatea. Los datos del informe se generan, no se transcriben a mano.

Cuando una corrida trae estratos por tono de piel se agregan las columnas del
peor bucket y la brecha de Dice, y se escribe una segunda tabla por corrida con
el Dice de cada bucket que alcanzo el minimo de imagenes.

Ejemplo:
    python plan-segmentacion/comun/evaluacion/build_results_table.py \\
        --run ssl=salidas/evaluacion_ssl/summary.json \\
        --run supervisado=salidas/evaluacion_sup/summary.json \\
        --output resultados/tabla.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

_PLAN_ROOT = Path(__file__).resolve().parents[2]
for _extra in (_PLAN_ROOT.parent / "src", _PLAN_ROOT / "comun" / "nucleo"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import cache_io  # noqa: E402
from dermssl.cli import build_spanish_parser  # noqa: E402

BASE_COLUMNS = (
    "run",
    "n",
    "dice_mean",
    "dice_median",
    "dice_p10",
    "iou_mean",
    "sensitivity_mean",
    "specificity_mean",
    "hd95_mean",
    "pred_coverage_mean",
)
STRATA_COLUMNS = ("worst_bucket", "gap_dice")
POOL_COLUMNS = ("in_pool_dice_mean", "out_of_pool_dice_mean")
BUCKET_COLUMNS = ("bucket", "n", "dice_mean", "dice_median")
DECIMALS = 3


def _stat(metrics: Mapping, metric: str, statistic: str) -> float | None:
    """One statistic of one metric, or None when the run did not report it."""
    entry = metrics.get(metric)
    if not isinstance(entry, Mapping):
        return None
    value = entry.get(statistic)
    return None if value is None else float(value)


def _qualifying_buckets(strata: Mapping) -> dict[str, dict]:
    """Buckets that reached the minimum count; the rest are not comparable."""
    buckets = {}
    for key, entry in sorted(strata.get("buckets", {}).items()):
        if entry.get("status") != "OK":
            continue
        buckets[key] = {
            "n": int(entry.get("n", 0)),
            "dice_mean": _stat(entry, "dice", "mean"),
            "dice_median": _stat(entry, "dice", "median"),
        }
    return buckets


def _pool_dice(strata: Mapping, key: str) -> float | None:
    """Dice mean of one aggregate; older runs did not report these entries."""
    entry = strata.get(key)
    return _stat(entry, "dice", "mean") if isinstance(entry, Mapping) else None


def load_run(name: str, path: Path) -> dict:
    """Flatten one ``summary.json`` into the row this report needs."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    strata = payload.get("strata") or {}
    count = payload.get("meta", {}).get("n_images")
    gap = strata.get("gap_dice")
    return {
        "run": name,
        "n": int(count if count is not None else (_stat(metrics, "dice", "n") or 0)),
        "dice_mean": _stat(metrics, "dice", "mean"),
        "dice_median": _stat(metrics, "dice", "median"),
        "dice_p10": _stat(metrics, "dice", "p10"),
        "iou_mean": _stat(metrics, "iou", "mean"),
        "sensitivity_mean": _stat(metrics, "sensitivity", "mean"),
        "specificity_mean": _stat(metrics, "specificity", "mean"),
        "hd95_mean": _stat(metrics, "hd95", "mean"),
        "pred_coverage_mean": _stat(metrics, "pred_coverage", "mean"),
        "worst_bucket": strata.get("worst_bucket"),
        "gap_dice": None if gap is None else float(gap),
        "in_pool_dice_mean": _pool_dice(strata, "in_pool"),
        "out_of_pool_dice_mean": _pool_dice(strata, "out_of_pool"),
        "buckets": _qualifying_buckets(strata),
    }


def _cell(value) -> str:
    """Three decimals for a number, an empty cell for a metric nobody reported."""
    if value is None:
        return ""
    if isinstance(value, bool) or isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{DECIMALS}f}"
    return str(value)


def _table(columns: Sequence[str], rows: Sequence[Sequence]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines += ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows]
    return lines


def build_markdown(records: Sequence[Mapping]) -> str:
    """Headline table plus one per-bucket table for every stratified run."""
    columns = list(BASE_COLUMNS)
    if any(record["buckets"] for record in records):
        columns += list(STRATA_COLUMNS)
    if any(record[name] is not None for record in records for name in POOL_COLUMNS):
        columns += list(POOL_COLUMNS)
    lines = ["# Resultados de segmentacion", ""]
    lines += _table(columns, [[record[name] for name in columns] for record in records])

    for record in records:
        if not record["buckets"]:
            continue
        lines += ["", f"### {record['run']}", ""]
        rows = [
            [key, entry["n"], entry["dice_mean"], entry["dice_median"]]
            for key, entry in sorted(record["buckets"].items())
        ]
        lines += _table(BUCKET_COLUMNS, rows)
    return "\n".join(lines) + "\n"


def build_table(args: argparse.Namespace) -> dict:
    """Write the Markdown table and its JSON twin; returns the JSON payload."""
    records = [load_run(name, path) for name, path in args.runs]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(records), encoding="utf-8")
    payload = {"runs": records, "output": str(output)}
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_spanish_parser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NOMBRE=RUTA", help="Corrida a comparar: NOMBRE=ruta/al/summary.json; se repite")
    parser.add_argument("--output", required=True, help="Archivo Markdown de salida; el JSON se escribe al lado")
    args = parser.parse_args(argv)
    try:
        args.runs = [cache_io.parse_root_spec(spec) for spec in args.run]
    except ValueError as error:
        parser.error(str(error))
    missing = [str(path) for _, path in args.runs if not path.is_file()]
    if missing:
        parser.error(f"no existe el resumen: {missing[0]}")
    return args


def main() -> None:
    payload = build_table(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
