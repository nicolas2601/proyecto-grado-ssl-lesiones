"""Diagnóstico A/B: compara el ITA sin máscara (imagen completa) contra el ITA
con la mascara actual (Otsu - el U-Net primario de PR3 todavia no corrio
sobre datos reales, asi que ``estimate_fst`` cae siempre en su fallback),
sobre un pool de imagenes real (HAM10000/ISIC2019/ISIC2020).

No necesita GPU ni torch: ``estimate_fst_whole_image`` y ``estimate_fst``
(via su fallback Otsu en ``pretrain.lesion_guided``) son pura numpy. Corre
igual de bien local como en Kaggle - a diferencia de
``scripts/kaggle_fst_manifest.py``, que si necesita una sesion GPU.

Por cada imagen calcula las dos estimaciones y escribe una fila con ambas al
CSV de salida. Al final imprime un resumen: distribucion de buckets por
corrida, tasa de status != "ok" por corrida, y tasa de cambio de bucket
entre A y B - las tres senales que decidimos mirar antes de confiar en las
labels de FST para entrenar CIRCLe.

    python scripts/diagnostico_fst_comparativa.py \\
        --image-root /kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_1 \\
        --dataset-id ham10000 \\
    --output-json diagnostico_fst_ham10000.json

Smoke (valida el cableado, pocas imagenes):
    python scripts/diagnostico_fst_comparativa.py ... --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MANIFEST_COLUMNS = (
    "dataset_id",
    "image_id",
    "ita_sin_mascara",
    "bucket_sin_mascara",
    "status_sin_mascara",
    "ita_con_mascara",
    "bucket_con_mascara",
    "status_con_mascara",
    "mask_source",
    "bucket_cambio",
    "artifact_sin_mascara",
    "artifact_con_mascara",
    "fallback_sin_mascara",
    "fallback_con_mascara",
    "confidence_sin_mascara",
    "confidence_con_mascara",
    "mask_producer",
    "mask_version",
    "mask_sha256",
)

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42

SMOKE_MAX_IMAGES = 20
IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image-root", required=True,
                         help="Carpeta con las imágenes del conjunto de datos")
    parser.add_argument("--dataset-id", required=True,
                         help="Etiqueta del conjunto de datos para el reporte; p. ej., ham10000")
    parser.add_argument("--output-csv", default="",
                        help="Exportación CSV heredada opcional")
    parser.add_argument("--output-json", default="",
                        help="Artefacto JSON principal; valor predeterminado: diagnostico_fst.json")
    parser.add_argument("--limit", type=int, default=0,
                         help="0 = todas las imágenes encontradas")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Ejecución reducida (<= {SMOKE_MAX_IMAGES} imágenes) para validar el cableado",
    )
    return parser.parse_args(argv)


def image_id(path: Path) -> str:
    return path.stem


def find_images(root: Path) -> list[Path]:
    """Recursivo (``**/patron``), no solo el nivel superior de ``root`` -
    HAM10000 en Kaggle trae las imagenes repartidas en subcarpetas
    (part_1/part_2), mismo criterio que ``scripts/build_baseline_tests.py``
    usa para encontrar imagenes bajo /kaggle/input.

    Deduplicado por ``image_id`` (el stem del archivo), quedandose con la
    primera ruta en orden alfabetico: el dataset de HAM10000 en Kaggle trae
    las mismas imagenes dos veces bajo carpetas con distinto casing
    (``HAM10000_images_part_1`` y ``ham10000_images_part_1``) - sin esto,
    cada imagen se contaria y procesaria dos veces."""
    paths: list[Path] = []
    for pattern in IMAGE_GLOBS:
        paths.extend(root.glob(f"**/{pattern}"))
    paths.sort()

    seen: set[str] = set()
    deduped: list[Path] = []
    for path in paths:
        stem = image_id(path)
        if stem in seen:
            continue
        seen.add(stem)
        deduped.append(path)
    return deduped


def build_row(dataset_id: str, path: Path, whole, masked) -> dict:
    """``whole``/``masked`` son ``FSTEstimate`` (de ``estimate_fst_whole_image``
    y ``estimate_fst`` respectivamente). ``bucket_cambio`` solo tiene sentido
    cuando las dos corridas dieron un bucket real - si alguna fallo
    (status != "ok"), queda en blanco en vez de comparar None contra un
    numero, que mezclaria "cambio de bucket" con "una corrida fallo"."""
    both_ok = whole.status == "ok" and masked.status == "ok"
    bucket_cambio = (whole.bucket != masked.bucket) if both_ok else ""
    return {
        "dataset_id": dataset_id,
        "image_id": image_id(path),
        "ita_sin_mascara": whole.ita,
        "bucket_sin_mascara": whole.bucket if whole.bucket is not None else "",
        "status_sin_mascara": whole.status,
        "ita_con_mascara": masked.ita,
        "bucket_con_mascara": masked.bucket if masked.bucket is not None else "",
        "status_con_mascara": masked.status,
        "mask_source": "otsu_fallback" if masked.fallback_used else "unet",
        "bucket_cambio": bucket_cambio,
        "artifact_sin_mascara": whole.artifact_detected,
        "artifact_con_mascara": masked.artifact_detected,
        "fallback_sin_mascara": whole.fallback_used,
        "fallback_con_mascara": masked.fallback_used,
        "confidence_sin_mascara": whole.confidence,
        "confidence_con_mascara": masked.confidence,
        "mask_producer": masked.mask_producer,
        "mask_version": masked.mask_version,
        "mask_sha256": masked.mask_sha256,
    }


def run(args: argparse.Namespace) -> list[dict]:
    """Import perezoso de PIL/numpy - mismo patron que
    ``kaggle_fst_manifest.py``: ``parse_args`` sola nunca toca una
    dependencia pesada."""
    import numpy as np
    from PIL import Image

    from dermssl import fitzpatrick, run_environment

    image_root = Path(args.image_root)
    paths = find_images(image_root)
    if args.smoke:
        paths = paths[:SMOKE_MAX_IMAGES]
    elif args.limit > 0:
        paths = paths[: args.limit]

    if not paths:
        raise SystemExit(f"no se encontraron imagenes en {image_root}")

    rows = []
    for path in paths:
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

        whole = fitzpatrick.estimate_fst_whole_image(img)
        masked = fitzpatrick.estimate_fst(img)

        rows.append(build_row(args.dataset_id, path, whole, masked))

    summary = print_summary(args.dataset_id, rows, smoke=args.smoke)
    if args.output_csv:
        write_csv(rows, Path(args.output_csv))
        print(f"Se escribieron {len(rows)} filas CSV heredadas en {args.output_csv}")
    output_json = (
        Path(args.output_json)
        if args.output_json
        else Path(args.output_csv).with_suffix(".json")
        if args.output_csv
        else Path("diagnostico_fst.json")
    )
    manifest = build_diagnostic_manifest(args.dataset_id, rows, smoke=args.smoke)
    write_diagnostic_manifest({**manifest, "environment": run_environment.snapshot(args)}, output_json)
    print(f"Se escribieron {len(rows)} registros JSON de comparación FST en {output_json}")
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_ci_bucket_cambio(comparables: list[dict],
                                 n_bootstrap: int = N_BOOTSTRAP,
                                 seed: int = BOOTSTRAP_SEED) -> list[float] | None:
    """IC95 bootstrap sobre ``bucket_cambio_pct`` - mismo patron que
    ``ic95_bootstrap`` en ``results_{modelo}.json`` (percentiles 2.5/97.5
    remuestreando con reemplazo), aplicado a la tasa de cambio de bucket en
    vez de a accuracy. Semilla fija (misma convencion que el resto del
    proyecto) para que la corrida sea reproducible."""
    if not comparables:
        return None
    rng = random.Random(seed)
    n = len(comparables)
    flags = [1 if r["bucket_cambio"] else 0 for r in comparables]
    pcts = sorted(100 * sum(rng.choices(flags, k=n)) / n for _ in range(n_bootstrap))
    lo = pcts[int(0.025 * n_bootstrap)]
    hi = pcts[int(0.975 * n_bootstrap) - 1]
    return [round(lo, 1), round(hi, 1)]


def _transition_matrix(comparables: list[dict]) -> dict:
    """Para las filas comparables, cuenta a que bucket_con_mascara fue a
    parar cada bucket_sin_mascara - el equivalente a una matriz de
    confusion aca: no hay ground truth, pero si dos mediciones para
    comparar entre si."""
    matrix: dict[int, Counter] = {}
    for r in comparables:
        matrix.setdefault(r["bucket_sin_mascara"], Counter())[r["bucket_con_mascara"]] += 1
    return {
        str(origen): {str(destino): c for destino, c in sorted(destinos.items())}
        for origen, destinos in sorted(matrix.items())
    }


def _bucket_cambio_rate(rows: list[dict]) -> dict:
    if not rows:
        return {"total": 0, "cambia": 0, "pct": None}
    cambia = sum(1 for r in rows if r["bucket_cambio"])
    return {"total": len(rows), "cambia": cambia, "pct": round(100 * cambia / len(rows), 1)}


def _artifact_breakdown(comparables: list[dict]) -> dict:
    """Cruza bucket_cambio con si CUALQUIERA de las dos corridas detecto un
    artefacto (pelo/regla/marcador de tinta) - prueba directa de la
    hipotesis de que los artefactos explican buena parte del cambio de
    bucket, en vez de dejarla como sospecha sin verificar."""
    con = [r for r in comparables if r["artifact_sin_mascara"] or r["artifact_con_mascara"]]
    sin = [r for r in comparables if not (r["artifact_sin_mascara"] or r["artifact_con_mascara"])]
    return {"con_artefacto": _bucket_cambio_rate(con), "sin_artefacto": _bucket_cambio_rate(sin)}


def summary_dict(dataset_id: str, rows: list[dict], smoke: bool = False) -> dict:
    """Mismo formato que ``results_{modelo}{sufijo}.json`` en
    ``scripts/build_baseline_tests.py`` (metadata + metricas, ``json.dumps``
    con ``indent=2``) - para que el resumen de este diagnostico se vea y se
    consuma igual que el resto de los resultados del proyecto."""
    n = len(rows)
    status_sin = Counter(r["status_sin_mascara"] for r in rows)
    status_con = Counter(r["status_con_mascara"] for r in rows)
    bucket_sin = Counter(
        r["bucket_sin_mascara"] for r in rows if r["status_sin_mascara"] == "ok"
    )
    bucket_con = Counter(
        r["bucket_con_mascara"] for r in rows if r["status_con_mascara"] == "ok"
    )
    comparables = [r for r in rows if r["bucket_cambio"] != ""]
    cambios = sum(1 for r in comparables if r["bucket_cambio"])
    pct = round(100 * cambios / len(comparables), 1) if comparables else None

    return {
        "dataset": dataset_id,
        "n": n,
        "smoke": smoke,
        "status_sin_mascara": dict(status_sin),
        "status_con_mascara": dict(status_con),
        "buckets_sin_mascara": dict(sorted(bucket_sin.items())),
        "buckets_con_mascara": dict(sorted(bucket_con.items())),
        "comparables": len(comparables),
        "bucket_cambio": cambios,
        "bucket_cambio_pct": pct,
        "bucket_cambio_ic95_bootstrap": _bootstrap_ci_bucket_cambio(comparables),
        "transicion_buckets": _transition_matrix(comparables),
        "bucket_cambio_por_artefacto": _artifact_breakdown(comparables),
    }


def write_summary_json(summary: dict, path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _fst_payload(row: dict, prefix: str) -> dict:
    """``ita`` viaja como ``None`` cuando es NaN: ``json.dumps`` serializa
    ``float("nan")`` como el token literal ``NaN``, que no es JSON valido
    (RFC 8259) y rompe cualquier parser estricto (visor de JupyterLab,
    JSON.parse del navegador, jq), aunque Python lo pueda releer sin
    quejarse."""
    bucket = row[f"bucket_{prefix}"]
    ita = row[f"ita_{prefix}"]
    return {
        "ita": ita if ita == ita else None,
        "bucket": bucket if bucket != "" else "unavailable",
        "status": row[f"status_{prefix}"],
        "fallback_used": row[f"fallback_{prefix}"],
        "confidence": row[f"confidence_{prefix}"],
        "artifact_detected": row[f"artifact_{prefix}"],
    }


def build_diagnostic_manifest(dataset_id: str, rows: list[dict], smoke: bool = False) -> dict:
    """Return the JSON-primary full-image versus masked FST contract."""
    summary = summary_dict(dataset_id, rows, smoke=smoke)
    records = []
    for row in rows:
        masked = _fst_payload(row, "con_mascara")
        masked["mask_provenance"] = {
            "producer": row["mask_producer"] or row["mask_source"],
            "version": row["mask_version"],
            "sha256": row["mask_sha256"],
        }
        records.append(
            {
                "dataset": row["dataset_id"],
                "image_id": row["image_id"],
                "whole_image": _fst_payload(row, "sin_mascara"),
                "masked": masked,
                "bucket_changed": row["bucket_cambio"],
            }
        )
    return {"schema_version": "1.0", **summary, "records": records}


def write_diagnostic_manifest(manifest: dict, path: Path) -> None:
    """Persist the primary FST diagnostic artifact as deterministic JSON."""
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_summary(dataset_id: str, rows: list[dict], smoke: bool = False) -> dict:
    summary = summary_dict(dataset_id, rows, smoke=smoke)
    print(f"\n=== resumen {dataset_id} (n={summary['n']}) ===")
    print(f"status sin mascara: {summary['status_sin_mascara']}")
    print(f"status con mascara: {summary['status_con_mascara']}")
    print(f"buckets sin mascara (solo ok): {summary['buckets_sin_mascara']}")
    print(f"buckets con mascara (solo ok): {summary['buckets_con_mascara']}")
    if summary["comparables"]:
        print(f"cambia el bucket entre corridas: {summary['bucket_cambio']}/{summary['comparables']} "
              f"({summary['bucket_cambio_pct']}%)")
    else:
        print("cambia el bucket entre corridas: sin filas comparables (ninguna con status ok en ambas)")
    return summary


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
