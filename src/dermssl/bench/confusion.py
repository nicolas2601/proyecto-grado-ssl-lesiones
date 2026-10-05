"""Confusion matrices of the self-supervised models (docs/33, E4): data, manifest, README, PNG.

Only DINOv2 S and B and SimSiam (frozen and fine-tuned) plus A71/B71. Counts come from the
``matriz_confusion`` of the committed baseline JSON; v7.1 stores none, so A71/B71 are the
argmax of the committed probabilities (which reproduce the stored accuracy exactly).
"""

from __future__ import annotations

import json
from pathlib import Path

from dermssl.bench import glossary, metrics
from dermssl.bench.rows import (DATASETS, MODEL_NAMES, NOT_RUN, PENDING, PLAN_DOC, V71_JSON, V71_NPZ, V71_ROWS,
                                _load, v71_derived)

SSL_MODELS = ("dinov2_vits14", "dinov2_vitb14", "simsiam_r50")
CONDITIONS = ("congelado", "fine-tuning")
OUT = Path("confusion")
V71_ARMS = {"B71": "B71 (propuesto)", "A71": "A71 (ablación sin término de tono)"}
NO_DIFF = ("No se genera figura de diferencias frente a B71: B71 se evalúa sobre la partición v6/v7 (n = 2.004) y "
           "los baselines sobre la de Kaggle (n = 2.014); no comparten partición ni imágenes, así que una resta "
           "celda a celda no tiene sentido. Se hará cuando exista B71 sobre la misma partición que un baseline.")


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


def slug(art: dict) -> str:
    base = art["modelo"].split(" (")[0].lower().replace(" ", "_").replace("/", "").replace("-", "")
    return f"{base}__{art['condicion'].replace('-', '')}"


def rel_json(art: dict) -> Path:
    return OUT / art["dataset"].replace(" ", "").lower() / f"{slug(art)}.json"


def artifacts(repo: Path) -> list[dict]:
    out = []
    for ds_key in DATASETS:
        for model in SSL_MODELS:
            out += [a for c in CONDITIONS if (a := _baseline_cell(repo, ds_key, model, c))]
        if ds_key == "ham10000":
            out += [_v71_cell(repo, key) for key, _, _ in V71_ROWS if key in V71_ARMS]
    return out


def _pending(dataset: str, model: str) -> dict:
    return {"dataset": dataset, "modelo": model, "condicion": CONDITIONS[0], "estado": PENDING,
            "motivo": "E1 de docs/33", "fuente": PLAN_DOC}


def pending() -> list[dict]:
    """Cells the plan will add; listed so a missing matrix is never silent."""
    out = [{"dataset": DATASETS["isic2020"].title, "modelo": MODEL_NAMES[m], "condicion": CONDITIONS[1],
            "estado": NOT_RUN, "motivo": "G1.3 de docs/33 (fine-tuning en ISIC 2020)", "fuente": PLAN_DOC}
           for m in SSL_MODELS]
    for ds in DATASETS.values():
        for key in V71_ARMS:
            out.append(_pending(ds.title, f"{key}-limpio"))
            if ds.key != "ham10000":
                out.append(_pending(ds.title, f"{key} original (transductivo)"))
    return out


def manifest(arts: list[dict]) -> dict:
    return {"incluidas": [{"json": rel_json(a).as_posix(), "png": rel_json(a).with_suffix(".png").as_posix(),
                           "dataset": a["dataset"], "modelo": a["modelo"], "condicion": a["condicion"],
                           "origen": a["origen"], "fuente": a["fuente"]} for a in arts],
            "pendientes": pending(), "figura_de_diferencias": NO_DIFF}


def dump(doc: dict) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
