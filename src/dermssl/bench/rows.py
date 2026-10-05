"""Row model of the bench tables and the loaders that read committed result JSON.

Nothing is typed in by hand except labels: every number comes from a JSON under
``results/`` or ``resultados/``. A row that cannot exist yet is a *pending* row with
no values and a reason, never a blank or an invented number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dermssl.bench import metrics

REPO = Path(__file__).resolve().parents[3]
PLAN_DOC = "docs/33-banco-completo-multidataset.md"
V71_JSON = "resultados/clasificacion/v71/v71_final_eval.json"
V71_NPZ = "resultados/clasificacion/v71/v71_final_eval.probas.npz"
METRICS = ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "auc", "ece", "mcc")
JSON_KEYS = {"auc": "auc_ovr_macro"}
FROZEN, FINETUNE = "congelados (sin fine-tuning)", "con fine-tuning"
OK, PENDING, NOT_RUN = "ok", "pendiente", "no ejecutado todavia"
MODELS = ("alexnet", "vgg19", "resnet50", "efficientnet_b0", "vit_b16", "simsiam_r50",
          "dinov2_vits14", "dinov2_vitb14")
MODEL_NAMES = {"alexnet": "AlexNet", "vgg19": "VGG19", "resnet50": "ResNet-50",
               "efficientnet_b0": "EfficientNet-B0", "vit_b16": "ViT-B/16",
               "simsiam_r50": "SimSiam ResNet-50", "dinov2_vits14": "DINOv2 ViT-S/14",
               "dinov2_vitb14": "DINOv2 ViT-B/14"}
SUPERVISED = "supervisado (ImageNet)"
PRETRAINING = {"simsiam_r50": "auto-supervisado (SimSiam, ImageNet)",
               "dinov2_vits14": "auto-supervisado (DINOv2 público)",
               "dinov2_vitb14": "auto-supervisado (DINOv2 público)"}
OWN_B = "auto-supervisado propio (DINOv2 + SimSiam anclado + término de tono, B*)"
OWN_A = "auto-supervisado propio (DINOv2 + SimSiam anclado sin término de tono, A*)"
PUBLIC = "auto-supervisado (DINOv2 público)"
V71_ROWS = (("B71", "B71 (propuesto)", OWN_B), ("A71", "A71 (ablación sin término de tono)", OWN_A),
            ("public_B71", "DINOv2 público, protocolo de B71", PUBLIC),
            ("public_A71", "DINOv2 público, protocolo de A71", PUBLIC),
            ("public71", "DINOv2 público, su mejor protocolo v7.1", PUBLIC))
OPTIMISTIC = "época elegida en val (sesgo optimista)"


@dataclass(frozen=True)
class Value:
    point: float
    ci: tuple[float, float] | None = None


@dataclass(frozen=True)
class Dataset:
    key: str
    title: str
    suffix: str
    baselines: str
    finetune: str | None


DATASETS = {
    "ham10000": Dataset("ham10000", "HAM10000", "", "results/baselines", "results/finetune/ham10000"),
    "isic2019": Dataset("isic2019", "ISIC 2019", "_isic2019", "results/baselines_isic2019",
                        "results/finetune/isic2019"),
    "isic2020": Dataset("isic2020", "ISIC 2020", "_isic2020", "results/baselines_isic2020", None),
}


@dataclass(frozen=True)
class Row:
    dataset: str
    block: str
    name: str
    pretraining: str
    finetune: bool
    transductive: bool
    status: str
    sources: tuple[str, ...]
    n_val: int | None = None
    values: dict[str, Value] = field(default_factory=dict)
    note: str = ""
    segmentation: str = "no"


def _load(repo: Path, rel: str) -> dict:
    return json.loads((repo / rel).read_text(encoding="utf-8"))


def _ci(record: dict, key: str) -> tuple[float, float] | None:
    interval = record.get("ic95_bootstrap", {}).get(key)
    return (interval[0], interval[1]) if interval else None


def probe_values(record: dict) -> dict[str, Value]:
    """Values of one probe record; MCC is exact from its confusion matrix."""
    out = {m: Value(record[JSON_KEYS.get(m, m)], _ci(record, m)) for m in METRICS if m != "mcc"}
    out["mcc"] = Value(metrics.mcc_from_confusion(record["matriz_confusion"]))
    return out


def _pretraining(model: str) -> str:
    return PRETRAINING.get(model, SUPERVISED)


def _ft_note(data: dict) -> str:
    if not data.get("truncada"):
        return OPTIMISTIC
    return f"{OPTIMISTIC}; mejor época = última ({data['mejor_epoca']}/{data['epocas_corridas']}), curva sin converger"


def baseline_rows(repo: Path, ds: Dataset) -> list[Row]:
    """The 8 frozen baselines (probe ``logreg``) and, when they exist, their fine-tuned twins."""
    out = []
    for model in MODELS:
        rel = f"{ds.baselines}/results_{model}{ds.suffix}.json"
        data = _load(repo, rel)
        out.append(Row(ds.key, FROZEN, MODEL_NAMES[model], _pretraining(model), False, False, OK, (rel,),
                       data["n_val"], probe_values(data["probes"]["logreg"])))
        if ds.finetune is None:
            continue
        rel = f"{ds.finetune}/results_{model}_ft{ds.suffix}.json"
        data = _load(repo, rel)
        out.append(Row(ds.key, FINETUNE, MODEL_NAMES[model], _pretraining(model), True, False, OK, (rel,),
                       data["n_val"], probe_values(data["probes"]["finetune"]), _ft_note(data)))
    return out


def v71_derived(repo: Path, key: str) -> tuple[list[list[int]], dict[str, Value]]:
    """Confusion matrix of a v7.1 arm from its stored probabilities, and its metric values.

    Accuracy, balanced accuracy, macro-F1, AUC and ECE come from the JSON (with its CI);
    precision, recall and MCC are not stored there, so they come from the argmax of the
    committed probabilities (which reproduce the stored accuracy and macro-F1 exactly).
    """
    doc, npz = _load(repo, V71_JSON), np.load(repo / V71_NPZ)
    cm = metrics.confusion_from_probabilities(npz["y_true"], npz[key])
    precision, recall = metrics.macro_precision_recall(cm)
    stored = doc["results"][key]
    out = {m: Value(stored[m], tuple(stored["ci95"][m]) if m in stored["ci95"] else None)
           for m in ("accuracy", "balanced_accuracy", "macro_f1", "auc", "ece")}
    out["macro_precision"], out["macro_recall"] = Value(precision), Value(recall)
    out["mcc"] = Value(metrics.mcc_from_confusion(cm))
    return cm, out


def proposed_rows_ham(repo: Path) -> list[Row]:
    n_eval = _load(repo, V71_JSON)["n_eval"]
    return [Row("ham10000", FROZEN, name, pre, False, False, OK, (V71_JSON, V71_NPZ), n_eval,
                v71_derived(repo, key)[1], "probe logreg C1; precisión, recall y MCC desde las probabilidades")
            for key, name, pre in V71_ROWS]


def pending_rows(ds: Dataset) -> list[Row]:
    """Rows the plan (docs/33, E1 and G1.3) will add; listed explicitly so nothing looks forgotten."""
    clean = (("B71-limpio (B* s42 limpio, config fija)", OWN_B), ("A71-limpio (sopa A* s42+s43 limpios)", OWN_A))
    out = [Row(ds.key, FROZEN, name, pre, False, False, PENDING, (PLAN_DOC,), note="E1 de docs/33")
           for name, pre in clean]
    if ds.key != "ham10000":
        out += [Row(ds.key, FROZEN, f"{name} original", pre, False, True, PENDING, (PLAN_DOC,),
                    note="transductivo: imágenes de val en el pool SSL, sin etiquetas (E1)")
                for name, pre in (("B71", OWN_B), ("A71", OWN_A))]
    if ds.finetune is None:
        out.append(Row(ds.key, FINETUNE, "8 baselines con fine-tuning", "varios", True, False, NOT_RUN,
                       (PLAN_DOC,), note="G1.3 de docs/33; optimista: época elegida en val (receta existente)"))
    return out


def sort_rows(rows: list[Row]) -> list[Row]:
    """Per block: rows with values by macro-F1 descending (ties by name), then pending ones in given order."""
    out = []
    for block in (FROZEN, FINETUNE):
        mine = [r for r in rows if r.block == block]
        done = sorted((r for r in mine if r.status == OK), key=lambda r: (-r.values["macro_f1"].point, r.name))
        out += done + [r for r in mine if r.status != OK]
    return out


def dataset_rows(repo: Path, ds: Dataset) -> list[Row]:
    rows = baseline_rows(repo, ds) + (proposed_rows_ham(repo) if ds.key == "ham10000" else [])
    return sort_rows(rows + pending_rows(ds))
