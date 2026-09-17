"""Validacion externa provisional (sin GPU, sin torch) del diagnostico FST:
compara ``estimate_fst_whole_image`` y ``estimate_fst`` (fallback Otsu, PR3
todavia no corrio su U-Net sobre datos reales) contra Fitzpatrick17k-C -
version limpia y con splits estandarizados de Abhishek, Jain & Hamarneh
("Investigating the Quality of DermaMNIST and Fitzpatrick17k Dermatological
Image Datasets", Nature Scientific Data 2025, Zenodo 10.5281/zenodo.11101337).

El Fitzpatrick17k crudo tiene ruido de etiquetado conocido (el propio paper
reporta 803 pares de imagenes casi identicas con FST distinto); por eso este
script cruza contra la version -C, no contra un mirror sin depurar. El cruce
es por ``md5hash`` (mismo nombre de archivo que usan los mirrors crudos:
``{md5hash}.jpg``).

Esto es contraste externo, no verdad clinica - documentado como tal en
docs/superpowers y openspec/changes/fst-domain-adaptation-circle.

    python scripts/validar_fitzpatrick17k.py \\
        --image-root /ruta/a/finalfitz17k \\
        --labels-csv /ruta/a/Fitzpatrick17k-C.csv \\
        --output-json validacion_fitzpatrick17k.json

Smoke (valida el cableado, pocas imagenes):
    python scripts/validar_fitzpatrick17k.py ... --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SMOKE_MAX_IMAGES = 20


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image-root", required=True,
                         help="Carpeta con las imagenes nombradas {md5hash}.jpg")
    parser.add_argument("--labels-csv", required=True,
                         help="Fitzpatrick17k-C.csv (columnas md5hash, fst)")
    parser.add_argument("--output-json", required=True,
                         help="Artefacto JSON con metricas y registros por imagen")
    parser.add_argument("--limit", type=int, default=0,
                         help="0 = todas las imagenes con FST valido")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Ejecucion reducida (<= {SMOKE_MAX_IMAGES} imagenes) para validar el cableado",
    )
    return parser.parse_args(argv)


def load_ground_truth(csv_path: Path) -> dict[str, int]:
    """Fitzpatrick17k-C usa fst=0 para fototipo desconocido/faltante - excluido
    porque no es un ground truth utilizable, no un fototipo real."""
    ground_truth: dict[str, int] = {}
    with Path(csv_path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row["fst"]
            if raw in ("", "0"):
                continue
            ground_truth[row["md5hash"]] = int(raw)
    return ground_truth


def find_image(image_root: Path, md5hash: str) -> Path | None:
    candidate = Path(image_root) / f"{md5hash}.jpg"
    return candidate if candidate.is_file() else None


def evaluate_image(img, actual_fst: int) -> dict:
    """Import perezoso de numpy/dermssl - mismo patron que
    ``diagnostico_fst_comparativa.run``: ``parse_args`` sola nunca toca una
    dependencia pesada."""
    from dermssl import fitzpatrick

    whole = fitzpatrick.estimate_fst_whole_image(img)
    masked = fitzpatrick.estimate_fst(img)

    return {
        "actual_fst": actual_fst,
        "whole_image": {
            "predicted_bucket": whole.bucket,
            "status": whole.status,
            "ita": whole.ita if whole.ita == whole.ita else None,
        },
        "masked": {
            "predicted_bucket": masked.bucket,
            "status": masked.status,
            "ita": masked.ita if masked.ita == masked.ita else None,
            "fallback_used": masked.fallback_used,
        },
    }


def accuracy_metrics(pairs: list[tuple[int, int | None]]) -> dict:
    """``pairs`` es (actual_fst, predicted_bucket); ``predicted_bucket`` es
    None cuando el estimador no llego a ``status == "ok"``. Esas filas se
    excluyen de accuracy/off-by-one/confusion (no se pueden comparar contra
    nada), pero se cuentan en ``n_unavailable`` para no esconder cobertura -
    mismo criterio que ``bucket_cambio`` en diagnostico_fst_comparativa.py:
    nunca mezclar "el estimador fallo" con "el estimador se equivoco"."""
    evaluable = [(actual, predicted) for actual, predicted in pairs if predicted is not None]
    n_evaluable = len(evaluable)
    n_unavailable = len(pairs) - n_evaluable

    confusion: dict[int, dict[int, int]] = {}
    for actual, predicted in evaluable:
        confusion.setdefault(actual, {})[predicted] = confusion.setdefault(
            actual, {}
        ).get(predicted, 0) + 1

    if n_evaluable == 0:
        accuracy = None
        off_by_one_accuracy = None
    else:
        exact = sum(1 for actual, predicted in evaluable if actual == predicted)
        off_by_one = sum(1 for actual, predicted in evaluable if abs(actual - predicted) <= 1)
        accuracy = exact / n_evaluable
        off_by_one_accuracy = off_by_one / n_evaluable

    return {
        "n_evaluable": n_evaluable,
        "n_unavailable": n_unavailable,
        "accuracy": accuracy,
        "off_by_one_accuracy": off_by_one_accuracy,
        "confusion": {
            str(actual): {str(predicted): count for predicted, count in sorted(row.items())}
            for actual, row in sorted(confusion.items())
        },
    }


def run(args: argparse.Namespace) -> list[dict]:
    import numpy as np
    from PIL import Image

    ground_truth = load_ground_truth(args.labels_csv)
    items = sorted(ground_truth.items())
    if args.smoke:
        items = items[:SMOKE_MAX_IMAGES]
    elif args.limit > 0:
        items = items[: args.limit]

    image_root = Path(args.image_root)
    rows: list[dict] = []
    for md5hash, actual_fst in items:
        path = find_image(image_root, md5hash)
        if path is None:
            rows.append({
                "md5hash": md5hash,
                "actual_fst": actual_fst,
                "status": "image_missing",
                "whole_image": None,
                "masked": None,
            })
            continue
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        row = evaluate_image(img, actual_fst)
        row["md5hash"] = md5hash
        row["status"] = "evaluated"
        rows.append(row)

    report = build_report(rows, ground_truth_total=len(ground_truth))
    Path(args.output_json).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Se evaluaron {report['n_images_found']}/{report['n_ground_truth']} imagenes "
        f"({report['n_images_missing']} faltantes)."
    )
    print(f"Accuracy sin mascara: {report['metrics']['whole_image']['accuracy']}")
    print(f"Accuracy con mascara: {report['metrics']['masked']['accuracy']}")
    return rows


def build_report(rows: list[dict], ground_truth_total: int) -> dict:
    found = [row for row in rows if row["status"] == "evaluated"]
    missing = [row for row in rows if row["status"] == "image_missing"]

    whole_pairs = [(row["actual_fst"], row["whole_image"]["predicted_bucket"]) for row in found]
    masked_pairs = [(row["actual_fst"], row["masked"]["predicted_bucket"]) for row in found]

    return {
        "schema_version": "1.0",
        "source": "Fitzpatrick17k-C (Abhishek, Jain & Hamarneh, Nature Scientific Data 2025; "
                  "Zenodo 10.5281/zenodo.11101337) - contraste externo provisional, no verdad clinica",
        "n_ground_truth": ground_truth_total,
        "n_images_found": len(found),
        "n_images_missing": len(missing),
        "metrics": {
            "whole_image": accuracy_metrics(whole_pairs),
            "masked": accuracy_metrics(masked_pairs),
        },
        "records": rows,
    }


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
