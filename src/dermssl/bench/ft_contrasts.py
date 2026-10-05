"""Paired contrasts of B71-limpio (canonical) against the ISIC 2020 fine-tuned baselines.

The fine-tuning per-image probabilities live in ``ft_isic2020/probas/<model>.probas.npz`` (arrays
``val_ids``, ``y``, ``proba`` (N, 2), ``groups``). Images are paired BY ID, never by position: if the
two id sets differ the run aborts. The bootstrap is the one of ``isic_eval`` (``scoring.contrasts``:
1.000 resamples, seed 20260923, grouped by ``patient_id``); only four metrics are kept. A model
without its npz has no contrast: it is listed as pending with its reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dermssl import paired_bootstrap as pb
from dermssl.bench import isic_rows, pending
from dermssl.bench.contrast import METRIC_ES, from_record, interval, num, reading
from dermssl.bench.rows import FT_ISIC2020_DIR, MODEL_NAMES, MODELS
from dermssl.bench.tables import Table
from dermssl.isic_eval import runner, scoring

PROBAS_DIR = f"{FT_ISIC2020_DIR}/probas"
CONTRASTS_DIR = f"{FT_ISIC2020_DIR}/contrasts"
ARM_CELL = "B71__clean"
ARM_LABEL = "B71-limpio (canónico)"
METRICS = ("macro_f1", "balanced_accuracy", "auc", "mcc")
N_RESAMPLES = 1000
POST_HOC = "análisis posterior a la ejecución (post-hoc), no parte del registro original"
LABEL_NOTE = ("El fine-tuning usa la mejor época elegida en val (optimista); B71-limpio no tiene fine-tuning ni "
              "selección de época en val. La época se tomó de `mejor_epoca` del JSON del fine-tuning (el .pth no la "
              "trae); el sha256 del checkpoint está en el sidecar. Pareo por id de imagen, bootstrap por `patient_id`, "
              "1.000 remuestras, semilla 20260923.")
P_DEFINITION = ("p bootstrap aproximado = 2·min(P(Δ≤0), P(Δ≥0)) sobre las 1.000 remuestras pareadas, con piso 1/1000 "
                "(no es un p exacto). Holm sobre los 8 modelos de cada métrica; un modelo ausente cuenta como p = 1.")
SIDECAR = f"{PROBAS_DIR}/ft_isic2020_probas.sidecar.json"
PAIRED = "pareado"


class AlignmentError(ValueError):
    """The two probability files do not cover the same val images (or disagree on label or patient)."""


def _by_id(loaded: dict, name: str) -> dict:
    ids = np.asarray(loaded["val_ids"]).astype(str)
    if len(set(ids)) != len(ids):
        raise AlignmentError(f"{name}: hay ids de val repetidos")
    order = np.argsort(ids, kind="stable")
    return {k: np.asarray(loaded[k])[order] for k in ("val_ids", "y", "proba", "groups") if k in loaded}


def align(arm: dict, control: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(y, arm proba, control proba, groups) with both sorted by id; aborts if the sets differ."""
    a, c = _by_id(arm, "brazo"), _by_id(control, "control")
    if not np.array_equal(a["val_ids"].astype(str), c["val_ids"].astype(str)):
        raise AlignmentError("los ids de val de B71-limpio y del fine-tuning no son el mismo conjunto de imágenes")
    if not np.array_equal(a["y"], c["y"]):
        raise AlignmentError("las etiquetas difieren entre B71-limpio y el fine-tuning para los mismos ids")
    if "groups" in a and "groups" in c and not np.array_equal(a["groups"].astype(str), c["groups"].astype(str)):
        raise AlignmentError("los pacientes difieren entre B71-limpio y el fine-tuning para los mismos ids")
    return a["y"], a["proba"], c["proba"], a["groups"].astype(str)


def compute(arm: dict, control: dict, control_name: str, *, n_resamples: int = N_RESAMPLES) -> dict:
    """The paired record (4 metrics, grouped by patient) of ``arm - control`` after id alignment."""
    y, pa, pc, groups = align(arm, control)
    records = [r for r in scoring.contrasts(ARM_CELL, control_name, y, pa, pc, groups=groups, n_resamples=n_resamples)
               if r["metric"] in METRICS]
    for rec in records:
        deltas = pb._paired_deltas(y_true=y, arm_proba=pa, control_proba=pc,
                                   metric_fn=scoring._metric_fn(rec["metric"]), n_resamples=n_resamples,
                                   seed=scoring.SEED, groups=groups, control_metric_fn=None)[2]
        rec["p_boot"] = p_bootstrap(deltas)
    return {"paired": records, "unit": "patient_id", "n_images": int(len(y)), "p_definition": P_DEFINITION}


def p_bootstrap(deltas: np.ndarray) -> float:
    """2·min(P(Δ≤0), P(Δ≥0)), capped at 1 and floored at 1/n_resamples."""
    n = len(deltas)
    return float(min(1.0, max(2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0)), 1 / n)))


def holm(pvalues: list[float | None], family: int = len(MODELS)) -> list[float | None]:
    """Holm-adjusted p of the present values in a family of ``family`` tests (absent ones count as p = 1)."""
    present = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    out: list[float | None] = [None] * len(pvalues)
    running = 0.0
    for rank, (p, i) in enumerate(present):
        running = max(running, min(1.0, (family - rank) * p))
        out[i] = running
    return out


def _arm(repo: Path) -> tuple[dict, str]:
    rel = isic_rows.cell_path(isic_rows.CANONICAL_DIR, "isic2020", ARM_CELL)
    sha = json.loads((repo / rel).read_text(encoding="utf-8"))["result"]["probas_sha256"]
    return runner.load_probas(repo / rel.replace(".json", ".probas.npz"), sha), sha


def probas_path(model: str) -> str:
    return f"{PROBAS_DIR}/{model}.probas.npz"


def contrast_path(model: str) -> str:
    return f"{CONTRASTS_DIR}/{ARM_CELL}__vs__ft_{model}.json"


def _file_sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_all(repo: Path, out_root: Path | None = None) -> list[str]:
    """Write the contrast JSON of every paired model whose npz exists (recompute only if missing or stale)."""
    arm, arm_sha = _arm(repo)
    side, written = sidecar(repo), []
    for model in MODELS:
        npz = repo / probas_path(model)
        if not npz.is_file() or unpaired_reason(side, model):
            continue
        target = (out_root or repo) / contrast_path(model)
        if model not in stale_models(repo):  # committed JSON is fresh: reuse it, the bootstrap is slow
            if target != repo / contrast_path(model):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text((repo / contrast_path(model)).read_text(encoding="utf-8"), encoding="utf-8")
            written.append(model)
            continue
        record = compute(arm, runner.load_probas(npz, ""), f"ft_{model}")
        doc = {"arm": ARM_CELL, "control": f"ft_{model}", "arm_probas_sha256": arm_sha,
               "control_probas_sha256": _file_sha(npz), "n_resamples": N_RESAMPLES, "metrics": list(METRICS),
               "result": record}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        written.append(model)
    return written


def stale_models(repo: Path, out_root: Path | None = None) -> list[str]:
    """Models whose npz exists but whose contrast JSON is missing or was built from other probabilities."""
    out, side = [], sidecar(repo)
    for model in MODELS:
        npz, js = repo / probas_path(model), (out_root or repo) / contrast_path(model)
        if not npz.is_file() or unpaired_reason(side, model):
            continue
        if not js.is_file() or json.loads(js.read_text(encoding="utf-8"))["control_probas_sha256"] != _file_sha(npz):
            out.append(model)
    return out


def sidecar(repo: Path) -> dict:
    """The combined sidecar of the reconstructed probabilities ({} if the file does not exist yet)."""
    path = repo / SIDECAR
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def unpaired_reason(side: dict, model: str) -> str | None:
    """Why ``model`` must not enter the paired contrast; None only if the sidecar says ``pareado``.

    A model missing from the sidecar, or with any other status, is never paired by default.
    """
    entry = next((m for m in side.get("models", []) if m.get("model") == model), None)
    if entry is None:
        return "el modelo no aparece en el sidecar de las probabilidades"
    if entry.get("status") == PAIRED:
        return None
    return entry.get("motivo") or entry.get("reason") or str(entry.get("status", "sin estado"))


def label_note(repo: Path) -> str:
    """The sidecar's own label for the reconstructed probabilities, plus the optimism and epoch notes."""
    side = sidecar(repo)
    label = side.get("label", "probas de val reconstruidas por inferencia (sin sidecar todavía)")
    identity = side.get("checkpoint_identity_note")
    return f"Probas: {label}. " + (f"{identity}. " if identity else "") + LABEL_NOTE


def _rows_for(repo: Path, model: str, label: str, by_metric: dict[str, list]) -> None:
    js = repo / contrast_path(model)
    record = json.loads(js.read_text(encoding="utf-8"))["result"]
    for rec in record["paired"]:
        by_metric.setdefault(rec["metric"], [None] * len(MODELS))[MODELS.index(model)] = (
            from_record(rec, arm=ARM_LABEL, control=label, unit=record["unit"], source=contrast_path(model)),
            rec["p_boot"])


def entries(repo: Path) -> tuple[dict[str, list], list[tuple[str, str]], list[tuple[str, str]]]:
    """(by metric: [(contrast, p) per model slot], pending [(label, why)], unpaired [(label, reason)])."""
    by_metric: dict[str, list] = {}
    waiting, unpaired, side = [], [], sidecar(repo)
    for model in MODELS:
        label = f"{MODEL_NAMES[model]} (fine-tuning)"
        if not (repo / probas_path(model)).is_file():
            waiting.append((label, pending.FT_PROBAS.note))
        elif reason := unpaired_reason(side, model):
            unpaired.append((label, reason))
        elif (repo / contrast_path(model)).is_file():
            _rows_for(repo, model, label, by_metric)
        else:
            waiting.append((label, "las probabilidades existen pero el contraste no se ha calculado: "
                            "ejecutar `scripts/build_ft_contrasts.py`"))
    return by_metric, waiting, unpaired


def contrasts(repo: Path) -> list:
    """Every paired contrast present (for the multiplicity count)."""
    return [e[0] for slots in entries(repo)[0].values() for e in slots if e]


def table(repo: Path) -> Table:
    """One row per (model, metric) with the interval, the approximate bootstrap p and its Holm adjustment."""
    by_metric, waiting, unpaired = entries(repo)
    rows = []
    for metric in METRICS:
        slots = by_metric.get(metric, [None] * len(MODELS))
        adjusted = holm([e[1] if e else None for e in slots])
        rows += [[e[0].control, METRIC_ES[metric], interval(e[0]), "sí" if e[0].excludes_zero else "no",
                  reading(e[0]), num(e[1]), num(adj)] for e, adj in zip(slots, adjusted) if e]
    notes = [f"{label}: pendiente; {why}." for label, why in waiting]
    notes += [f"{label}: no pareado: {why}; solo comparación de intervalos." for label, why in unpaired]
    note = " ".join([f"Sección: {POST_HOC}.", label_note(repo), P_DEFINITION, *notes])
    return Table("B71-limpio contra el fine-tuning (pareado por id, ISIC 2020)",
                 ["Control (fine-tuning)", "Métrica", "Δ (B71-limpio − FT) [IC 95 %]", "El IC excluye 0", "Lectura",
                  "p bootstrap", "p Holm"], rows, note, "ft-contrasts")
