"""Confusion matrices of the self-supervised models (docs/33, E4): data, manifest, README, PNG.

Only DINOv2 S and B and SimSiam (frozen and fine-tuned) plus A71/B71. Counts come from the
``matriz_confusion`` of the committed baseline JSON; v7.1 stores none, so A71/B71 are the
argmax of the committed probabilities (which reproduce the stored accuracy exactly).
"""

from __future__ import annotations

import json
from pathlib import Path

from dermssl.bench import glossary, isic_rows, metrics, pending as pend
from dermssl.bench.rows import (DATASETS, MODEL_NAMES, PENDING, PLAN_DOC, V71_JSON, V71_NPZ, V71_ROWS,
                                _load, v71_derived)

SSL_MODELS = ("dinov2_vits14", "dinov2_vitb14", "simsiam_r50")
CONDITIONS = ("congelado", "fine-tuning")
OUT = Path("confusion")
V71_ARMS = {"B71": "B71 (propuesto)", "A71": "A71 (ablación sin término de tono)"}
NO_DIFF = ("No se genera figura de diferencias frente a B71: B71 se evalúa sobre la partición v6/v7 (n = 2.004) y "
           "los baselines sobre la de Kaggle (n = 2.014); no comparten partición ni imágenes, así que una resta "
           "celda a celda no tiene sentido. En ISIC 2019 e ISIC 2020 B71-limpio sí comparte partición con los "
           "baselines, pero tampoco se genera la figura de diferencias (decisión de alcance).")
# (folder, cell, model label, condition, origin) of the ISIC E1 cells that get a matrix
CANON_ORIGIN = "matriz_confusion de la celda E1 (cómputo canónico: batch 64 + autocast)"
SENS_ORIGIN = "matriz_confusion de la celda E1 (sensibilidad: batch 16, transductivo)"
ISIC_CELLS = (
    (isic_rows.CANONICAL_DIR, "B71__clean", "B71-limpio (canónico)", CONDITIONS[0], CANON_ORIGIN),
    (isic_rows.CANONICAL_DIR, "A71__clean", "A71-limpio (canónico)", CONDITIONS[0], CANON_ORIGIN),
    (isic_rows.CANONICAL_DIR, "B71__original", "B71 original (canónico, transductivo)", CONDITIONS[0], CANON_ORIGIN),
    (isic_rows.CANONICAL_DIR, "A71__original", "A71 original (canónico, transductivo)", CONDITIONS[0], CANON_ORIGIN),
    (isic_rows.SENSITIVITY_DIR, "B71__original", "B71 original (sensibilidad: batch 16, transductivo)", CONDITIONS[0],
     SENS_ORIGIN),
    (isic_rows.SENSITIVITY_DIR, "A71__original", "A71 original (sensibilidad: batch 16, transductivo)", CONDITIONS[0],
     SENS_ORIGIN),
)


def _artifact(repo: Path, ds_key: str, model: str, condition: str, cm: list[list[int]], source: list[str],
              origin: str) -> dict:
    facts = glossary.dataset_facts(repo, DATASETS[ds_key])
    classes = [glossary.CLASSES_ES.get(c, c) for c in facts["classes"]]
    return {"dataset": DATASETS[ds_key].title, "modelo": model, "condicion": condition, "clases": classes,
            "n": sum(map(sum, cm)), "conteos": cm, "normalizada_por_fila": metrics.row_normalized(cm),
            "mcc": metrics.mcc_from_confusion(cm), "origen": origin, "fuente": source}


def _baseline_cell(repo: Path, ds_key: str, model: str, condition: str) -> dict | None:
    ds = DATASETS[ds_key]
    frozen = condition == CONDITIONS[0]
    folder = ds.baselines if frozen else ds.finetune
    if folder is None:
        return None
    rel = f"{folder}/results_{model}{'' if frozen else '_ft'}{ds.suffix}.json"
    if not (repo / rel).is_file():
        return None
    probe = _load(repo, rel)["probes"]["logreg" if frozen else "finetune"]
    return _artifact(repo, ds_key, MODEL_NAMES[model], condition, probe["matriz_confusion"], [rel],
                     "matriz_confusion del JSON")


def _v71_cell(repo: Path, key: str) -> dict:
    cm, values = v71_derived(repo, key)
    art = _artifact(repo, "ham10000", V71_ARMS[key], CONDITIONS[0], cm, [V71_JSON, V71_NPZ],
                    "argmax de las probabilidades commiteadas (el JSON v7.1 no trae matriz_confusion)")
    art["verificacion"] = {"accuracy_json": _load(repo, V71_JSON)["results"][key]["accuracy"],
                           "accuracy_desde_matriz": sum(cm[i][i] for i in range(len(cm))) / art["n"]}
    return art


def _isic_cell(repo: Path, ds_key: str, folder: str, cell: str, model: str, origin: str) -> dict | None:
    rel = isic_rows.cell_path(folder, ds_key, cell)
    if not (repo / rel).is_file():
        return None
    result = _load(repo, rel)["result"]
    art = _artifact(repo, ds_key, model, CONDITIONS[0], result["metrics"]["matriz_confusion"], [rel], origin)
    art["verificacion"] = {"n_val_json": result["n_val"], "n_desde_matriz": art["n"]}
    return art


def slug(art: dict) -> str:
    base = art["modelo"].split(" (")[0].lower().replace(" ", "_").replace("/", "").replace("-", "")
    variant = "_batch16" if "batch 16" in art["modelo"] else ""
    return f"{base}{variant}__{art['condicion'].replace('-', '')}"


def rel_json(art: dict) -> Path:
    return OUT / art["dataset"].replace(" ", "").lower() / f"{slug(art)}.json"


def artifacts(repo: Path) -> list[dict]:
    out = []
    for ds_key in DATASETS:
        for model in SSL_MODELS:
            out += [a for c in CONDITIONS if (a := _baseline_cell(repo, ds_key, model, c))]
        if ds_key == "ham10000":
            out += [_v71_cell(repo, key) for key, _, _ in V71_ROWS if key in V71_ARMS]
            continue
        out += [a for f, c, m, _, o in ISIC_CELLS if (a := _isic_cell(repo, ds_key, f, c, m, o))]
    return out


def _pending(dataset: str, model: str, why: pend.Pending, condition: str = CONDITIONS[0]) -> dict:
    return {"dataset": dataset, "modelo": model, "condicion": condition, "estado": PENDING, "motivo": why.reason,
            "eta": why.eta, "fuente": PLAN_DOC}


def _missing(repo: Path, ds_key: str, cell: str) -> bool:
    return not (repo / isic_rows.cell_path(isic_rows.CANONICAL_DIR, ds_key, cell)).is_file()


def pending(repo: Path) -> list[dict]:
    """Cells without a JSON on disk; derived from the files, so nothing stale can stay listed."""
    ft = DATASETS["isic2020"].finetune
    out = [_pending(DATASETS["isic2020"].title, MODEL_NAMES[m], pend.missing(f"{ft}/results_{m}_ft_isic2020.json"),
                    CONDITIONS[1]) for m in SSL_MODELS if _baseline_cell(repo, "isic2020", m, CONDITIONS[1]) is None]
    ham = DATASETS["ham10000"].title
    out += [_pending(ham, f"{key}-limpio (canónico)", pend.HAM_CLEAN) for key in V71_ARMS]
    for ds in (d for d in DATASETS.values() if d.key != "ham10000"):
        for cell, label in ((f"{key}__clean", f"{key}-limpio (canónico)") for key in V71_ARMS):
            if _missing(repo, ds.key, cell):
                out.append(_pending(ds.title, label, pend.missing(isic_rows.cell_path(isic_rows.CANONICAL_DIR, ds.key, cell))))
        for key in V71_ARMS:
            if _missing(repo, ds.key, f"{key}__original"):
                rel = isic_rows.cell_path(isic_rows.CANONICAL_DIR, ds.key, f"{key}__original")
                out.append(_pending(ds.title, f"{key} original (canónico, transductivo)", pend.missing(rel)))
    return out


def manifest(arts: list[dict], waiting: list[dict]) -> dict:
    return {"incluidas": [{"json": rel_json(a).as_posix(), "png": rel_json(a).with_suffix(".png").as_posix(),
                           "dataset": a["dataset"], "modelo": a["modelo"], "condicion": a["condicion"],
                           "origen": a["origen"], "fuente": a["fuente"]} for a in arts],
            "pendientes": waiting, "figura_de_diferencias": NO_DIFF}


def dump(doc: dict) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
