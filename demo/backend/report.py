from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ReportDataError(RuntimeError):
    """Raised when the sealed result package cannot be read safely."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportDataError(f"No se pudo leer el resultado versionado: {path}") from exc


def _headline_rows(bank: dict[str, Any]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for row in bank.get("rows", []):
        if row.get("probe") != "logreg" or row.get("fraction") != 1.0 or row.get("seed") != 0:
            continue
        rows.setdefault(row["backbone"], {})
        for metric in ("macro_f1", "balanced_accuracy", "auc", "ece", "accuracy"):
            if metric in row:
                rows[row["backbone"]][metric] = round(float(row[metric]), 4)
    return rows


def _contrast_rows(bootstrap: dict[str, Any]) -> dict[str, dict[str, Any]]:
    contrasts: dict[str, dict[str, Any]] = {}
    for row in bootstrap.get("results", []):
        key = f"{row['arm']}|{row['control']}|{row['metric']}"
        contrasts[key] = {
            "arm": row["arm"],
            "control": row["control"],
            "metric": row["metric"],
            "delta": round(float(row["point_delta"]), 4),
            "ci95": [round(float(row["ci95_low"]), 4), round(float(row["ci95_high"]), 4)],
            "excludes_zero": bool(row["excludes_zero"]),
            "direction": row["direction"],
        }
    return contrasts


def _find_contrast(
    contrasts: dict[str, dict[str, Any]], arm: str, control: str, metric: str
) -> dict[str, Any]:
    key = f"{arm}|{control}|{metric}"
    try:
        return contrasts[key]
    except KeyError as exc:
        raise ReportDataError(f"Falta contraste sellado: {key}") from exc


def build_report(repo_root: Path) -> dict[str, Any]:
    """Build the page payload from the latest committed result artifacts."""
    result_root = repo_root / "resultados"
    site_data_path = result_root / "site_data.json"
    bank_path = result_root / "clasificacion" / "bank.json"
    strata_path = result_root / "clasificacion" / "strata.json"
    bootstrap_path = result_root / "clasificacion" / "bootstrap.json"

    site_data = _read_json(site_data_path)
    bank = _read_json(bank_path)
    strata = _read_json(strata_path)
    bootstrap = _read_json(bootstrap_path)
    classification = _headline_rows(bank)
    contrasts = _contrast_rows(bootstrap)
    site_classification = site_data["classification"]
    headline_table = [
        row
        for row in site_classification["rows"]
        if row.get("probe") == site_classification["headline_probe"]
        and row.get("fraction") == 1.0
        and row.get("seed") == 0
    ]

    required = {
        "ssl_proposed_vitb14",
        "ssl_plain_vitb14",
        "dinov2_public_vitb14",
    }
    if not required.issubset(classification):
        missing = sorted(required.difference(classification))
        raise ReportDataError(f"Faltan brazos en el banco sellado: {missing}")

    proposed_vs_plain_f1 = _find_contrast(
        contrasts, "ssl_proposed_vitb14", "ssl_plain_vitb14", "macro_f1"
    )
    proposed_vs_plain_balanced = _find_contrast(
        contrasts, "ssl_proposed_vitb14", "ssl_plain_vitb14", "balanced_accuracy"
    )
    proposed_vs_public_auc = _find_contrast(
        contrasts, "ssl_proposed_vitb14", "dinov2_public_vitb14", "auc"
    )
    proposed_vs_public_ece = _find_contrast(
        contrasts, "ssl_proposed_vitb14", "dinov2_public_vitb14", "ece"
    )

    proposed_strata = strata["ssl_proposed_vitb14"]
    plain_strata = strata["ssl_plain_vitb14"]

    site_meta = site_data["meta"]
    datasets = site_data["datasets"]
    training = site_data["training"]
    segmentation_site = site_data["segmentation"]
    isic_site = segmentation_site["isic2018_external"]
    ham_site = segmentation_site["ham10000_in_pool"]
    tone_site = segmentation_site["tone_equity"]

    return {
        "source": {
            "release": site_meta["release"].rsplit("/", 1)[-1],
            "bank": "resultados/clasificacion/bank.json",
            "bootstrap": "resultados/clasificacion/bootstrap.json",
            "strata": "resultados/clasificacion/strata.json",
            "site_data": "resultados/site_data.json",
            "protocol": "logreg al 100 %, primer pliegue StratifiedGroupKFold por lesion_id",
        },
        "site_meta": site_meta,
        "datasets": datasets,
        "method": {
            "pool_images": datasets["pool_total"],
            "tone_inputs": datasets["pool_valid"],
            "tone_inputs_pct": datasets["pool_valid_pct"],
            "quarantined": datasets["quarantined"],
            "backbone": "DINOv2 ViT-B/14 + SimSiam de dominio",
            "proposed_view": "imagen original + copia con desplazamiento de tono y lesion protegida por pseudo-mascara",
            "label_free_pretraining": True,
        },
        "segmentation": {
            "isic2018": {
                "role": "externo",
                "n": isic_site["n_images"],
                "dice": round(float(isic_site["dice"]["mean"]), 4),
                "ci95": [isic_site["dice"]["ci95_low"], isic_site["dice"]["ci95_high"]],
                "hd95_median_px": isic_site["hd95"]["median"],
            },
            "ham10000": {
                "role": "en distribucion",
                "n": ham_site["n_images"],
                "dice": round(float(ham_site["dice"]["mean"]), 4),
                "ci95": [ham_site["dice"]["ci95_low"], ham_site["dice"]["ci95_high"]],
                "hd95_median_px": ham_site["hd95"]["median"],
            },
            "tone_equity": tone_site,
            "note": segmentation_site["note"],
        },
        "training": training,
        "classification": classification,
        "classification_meta": {
            "headline_probe": site_classification["headline_probe"],
            "strata_probe": site_classification["strata_probe"],
            "split": site_classification["split"],
        },
        "classification_table": headline_table,
        "contrasts": site_data["contrasts"]["results"],
        "contrast_meta": {
            "n_resamples": site_data["contrasts"]["n_resamples"],
            "rng_seed": site_data["contrasts"]["rng_seed"],
            "paired": site_data["contrasts"]["paired"],
        },
        "classification_tone_equity": site_data["classification_tone_equity"],
        "downloads": site_data["downloads"],
        "classifier_contract": site_data["classifier"],
        "disclaimer": site_meta["disclaimer"],
        "framing": site_meta["framing"],
        "classification_verdict": {
            "global": "loss",
            "proposed_vs_plain_macro_f1": proposed_vs_plain_f1,
            "proposed_vs_plain_balanced_accuracy": proposed_vs_plain_balanced,
            "proposed_vs_public_auc": proposed_vs_public_auc,
            "proposed_vs_public_ece": proposed_vs_public_ece,
        },
        "tone": {
            "classification_gap_plain_vitb14": round(float(plain_strata["gap_macro_f1"]), 4),
            "classification_gap_proposed_vitb14": round(
                float(proposed_strata["gap_macro_f1"]), 4
            ),
            "classification_gap_established": bool(proposed_strata["gap_established"]),
            "worst_label_proposed": proposed_strata["worst_label"],
            "segmentation_gap_ham10000": round(float(tone_site["ham10000"]["gap_dice"]), 4),
            "segmentation_gap_isic2018": round(float(tone_site["isic2018"]["gap_dice"]), 4),
            "proxy_note": "Los terciles de L* son una particion relativa del pool, no fototipos clinicos.",
        },
        "claims": {
            "wins": [
                "La adaptacion de dominio mejora AUC y ECE frente a DINOv2 publico en ViT-B; los intervalos bootstrap excluyen cero.",
                "La segmentacion sin etiquetas es utilizable como paso previo; la lectura honesta es la evaluacion externa.",
                "El pipeline deja trazabilidad por run.json, presupuesto, hash de particion y bootstrap pareado.",
            ],
            "losses": [
                "El emparejamiento por tono pierde macro-F1 y balanced accuracy frente a su control plain en ViT-S y ViT-B.",
                "No se puede afirmar un efecto de tono en clasificacion: los intervalos por estrato se solapan.",
                "El checkpoint publicado es un backbone congelado; no contiene una cabeza de clasificacion.",
            ],
        },
        "next_steps": [
            {
                "title": "Separar las señales del objetivo",
                "detail": "Usar tres vistas: dos augmentaciones de la imagen original y una transformada de tono. Asi la invariancia a augmentacion y la invariancia a tono dejan de competir dentro de una sola perdida.",
            },
            {
                "title": "Hacer variable el desplazamiento",
                "detail": "Probar magnitud y signo aleatorios por imagen, con limites fijados antes de abrir el test; hoy el desplazamiento va en una sola direccion.",
            },
            {
                "title": "Reexportar pseudo-mascaras",
                "detail": "Generar las predicciones de la ronda 3 para que el transformador de tono use la salida final y no la ronda 2.",
            },
            {
                "title": "Comparar a igual numero de muestras",
                "detail": "El brazo pareado procesa aproximadamente 3 % menos muestras a igual presupuesto GPU; repetir con una condicion de muestras igualada separa ese efecto del objetivo de tono.",
            },
            {
                "title": "Subir resolucion con presupuesto explicito",
                "detail": "224 px es el compromiso actual. Una prueba a mayor resolucion debe fijar el costo y el numero de seeds antes de evaluar.",
            },
            {
                "title": "Validar fuera del pool y por paciente",
                "detail": "Hacer validacion externa en fototipos altos y resolver la agrupacion por paciente en ISIC 2019 antes de extrapolar resultados clinicos.",
            },
        ],
        "caveats": [
            "No es un dispositivo de diagnostico ni sustituye a un dermatologo.",
            "La evaluacion de clasificacion usa HAM10000 y su split por lesion_id; no es validacion clinica externa.",
            "Al 100 % de etiquetas, las tres semillas repiten la misma corrida; la incertidumbre publicada es bootstrap del test.",
            "El pool es mayoritariamente de piel clara y no permite afirmar rendimiento en fototipos altos reales.",
        ],
    }
