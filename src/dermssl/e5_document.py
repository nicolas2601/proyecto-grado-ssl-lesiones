"""E5 (docs/33): the result document of one dataset (descriptive, NOT validated).

Pure: records in, a JSON-safe dict out. Every threshold and bound it states is read from the
estimator modules, never typed again, so the document cannot drift from the code that produced it.
The per-image table is columnar (one row per image, in task order, floats rounded to
``ROUND_DIGITS``) to keep a 33k-image dataset in a few megabytes.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from dermssl import e5_analyze as an
from dermssl import e5_counts as ec
from dermssl import e5_ita as ita
from dermssl import e5_roi as roi
from dermssl import e3_geometry as geo
from dermssl import fitzpatrick as fz

ROUND_DIGITS = 4
SCHEMA = "e5_fitzpatrick/1"
PLAN_DOC = "docs/33-banco-completo-multidataset.md"
CAVEAT = (
    "Descriptivo y NO está validado: los buckets I-VI son la salida del estimador ITA con umbrales fijos, no "
    "el fototipo Fitzpatrick. Ese estimador acierta 22,9 % frente a 51,3 % de la baseline de mayoría en "
    "PAD-UFES-20 (Alipour et al., Skin Research and Technology 32(8):e70374, 2026, PMC13459198; docs/20, "
    "docs/30) y 28,5-30,8 % frente a 32,9 % en Fitzpatrick17k-C. Los terciles de L* son luminancia (mediana de "
    "L* de la piel), no es tono de piel, y son relativos a cada dataset: «oscuro» en uno no equivale a "
    "«oscuro» en otro. Ningún conteo de este archivo cambia el resultado primario del banco (docs/33, sección 5)."
)
CANONICAL_DECODE = (
    "PIL draft (escalado DCT del JPEG) + resize bilineal con antialias hasta que el lado corto mida "
    "exactamente el lado canónico si lo supera; las imágenes más chicas no se tocan; RGB float32 en [0, 1], "
    "sin constancia de color. Solo CPU."
)
ROI_RULE = (
    "Anillo fuera de la máscara de lesión (dilatación de Chebyshev de ancho 25 % del diámetro equivalente, "
    "mínimo 3 px, el mismo anillo de la variante F5 de E3) sin los píxeles de pelo, reflejos, saturados o del "
    "borde negro; mediana de L* y b* de lo que queda y el mismo ITA y umbrales que el estimador de Otsu."
)
COLUMNS = ("image_id", "split", "label", "tercile", "otsu_bucket", "otsu_status", "otsu_ita", "otsu_L",
           "otsu_b", "whole_bucket", "whole_status", "whole_ita", "roi_bucket", "roi_status", "roi_ita",
           "roi_L", "roi_n_px", "roi_valid_fraction")


class MissingHairDetector(ValueError):
    """The dataset has masks, so the ring excludes hair, and no detector description was given."""


@dataclass(frozen=True)
class DatasetSpec:
    """What the runner needs to know about one dataset (built by ``scripts/e5_data.py``)."""

    name: str
    classes: tuple[str, ...]
    splits: tuple[str, ...]
    tasks: list[an.Task]
    provenance: dict
    masks: dict | None  # manifest facts (producer, sha256) when the dataset has masks, else None


def _round(value):
    return round(value, ROUND_DIGITS) if isinstance(value, float) else value


def _row(rec: dict) -> list:
    o, w, r = rec["otsu"], rec["whole_image"], rec["roi"]
    values = (rec["image_id"], rec["split"], rec["label"], rec["tercile"], o["bucket"], o["status"], o["ita"],
              o["L_median"], o["b_median"], w["bucket"], w["status"], w["ita"], r["bucket"], r["status"],
              r["ita"], r["L_median"], r["n_px"], r["valid_fraction"])
    return [_round(v) for v in values]


def per_image_rows(records: Sequence[dict]) -> tuple[list[str], list[list]]:
    """Column names and one row per record."""
    return list(COLUMNS), [_row(r) for r in records]


def with_terciles(records: Sequence[dict]) -> tuple[list[dict], dict]:
    """Copies of the records with the tercile of this dataset's skin L* (Otsu estimator), and the cuts."""
    labels, cuts = ita.assign_terciles([r["otsu"]["L_median"] for r in records])
    return [{**r, "tercile": t} for r, t in zip(records, labels, strict=True)], cuts


def _roi_definition(spec: DatasetSpec, hair_detector: dict | None) -> dict:
    if spec.masks is None:
        return {"applicable": False, "reason": "el dataset no tiene máscaras de lesión: el estimador por anillo "
                "no aplica y se abstiene en todas las imágenes (no se inventan máscaras)"}
    if hair_detector is None:
        raise MissingHairDetector(f"{spec.name} has masks: the hair detector description is required")
    return {"applicable": True, "reason": "", "rule": ROI_RULE, "ring_width_fraction": geo.RING_WIDTH_FRACTION,
            "ring_min_width_px": geo.MIN_RING_WIDTH, "saturation_level": roi.SATURATION_LEVEL,
            "specular_min_L": roi.SPECULAR_MIN_L, "specular_max_chroma": roi.SPECULAR_MAX_CHROMA,
            "min_valid_px": roi.MIN_RING_VALID_PX, "min_valid_fraction": roi.MIN_RING_VALID_FRACTION,
            "abstains_on": list(roi.STATUSES[1:]), "hair_detector": hair_detector,
            "choices_not_registered": "ancho del anillo, umbrales de reflejo/saturación y mínimos de píxeles"}


def _estimator_definitions(spec: DatasetSpec, hair_detector: dict | None) -> dict:
    return {
        "otsu": {"applicable": True, "rule": "fitzpatrick.estimate_fst(img): lesión por Otsu, piel = el resto "
                 "sin borde negro; es el estimador de referencia (comparable entre datasets)"},
        "whole_image": {"applicable": True, "rule": "fitzpatrick.estimate_fst_whole_image(img): todos los "
                        "píxeles cuentan, sin máscara"},
        "roi": _roi_definition(spec, hair_detector),
    }


def build_document(spec: DatasetSpec, records: Sequence[dict], *, code_revision: str, smoke: int | None,
                   hair_detector: dict | None) -> dict:
    """The JSON of one dataset: definitions, caveat, descriptive counts and the per-image table."""
    estimators = _estimator_definitions(spec, hair_detector)  # first: a missing detector fails before any work
    final, cuts = with_terciles(records)
    report = ec.build_report(final, classes=spec.classes, splits=spec.splits, tercile_cuts=cuts,
                             roi_applicable=spec.masks is not None,
                             roi_reason=estimators["roi"]["reason"])
    columns, rows = per_image_rows(final)
    return {
        "schema": SCHEMA, "experiment": "E5", "plan_doc": PLAN_DOC, "descriptive": True, "validated": False,
        "caveat": CAVEAT, "dataset": spec.name, "smoke_limit": smoke, "code_revision": code_revision,
        "classes": list(spec.classes), "splits": list(spec.splits), "provenance": spec.provenance,
        "masks": spec.masks,
        "canonical_input": {"short_side": ita.CANONICAL_SHORT_SIDE, "decode": CANONICAL_DECODE},
        "fst_thresholds": list(fz.FST_THRESHOLDS), "min_b_for_ita": fz.MIN_B_FOR_ITA,
        "abstention": {"min_skin_fraction": fz.MIN_SKIN_FRACTION, "min_b_for_ita": fz.MIN_B_FOR_ITA,
                       "mask_area_bounds": [fz.MIN_MASK_AREA_FRACTION, fz.MAX_MASK_AREA_FRACTION],
                       "rule": "sin bucket (abstain) cuando no hay piel suficiente, la máscara es degenerada, "
                       "b* < mínimo o el archivo no se pudo leer; nunca se fuerza un bucket"},
        "estimators": estimators, "report": report,
        "unreadable": [{"image_id": r["image_id"], "error": r["error"]} for r in final if "error" in r],
        "per_image": {"columns": columns, "rows": rows},
    }


def dump_document(document: dict) -> str:
    """Indented JSON with one image per line in ``per_image.rows``; NaN is an error, never written."""
    head = {k: v for k, v in document.items() if k != "per_image"}
    table = document["per_image"]
    rows = ",\n".join("   " + json.dumps(row, ensure_ascii=False, allow_nan=False) for row in table["rows"])
    columns = json.dumps(table["columns"], ensure_ascii=False, allow_nan=False)
    body = json.dumps(head, indent=1, ensure_ascii=False, allow_nan=False)[:-2]
    return f'{body},\n "per_image": {{\n  "columns": {columns},\n  "rows": [\n{rows}\n  ]\n }}\n}}\n'
