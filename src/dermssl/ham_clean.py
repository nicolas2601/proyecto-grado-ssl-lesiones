"""HAM10000 clean-weights sensitivity (docs/33): pure rules, paths and labels, no torch.

The release publishes CLEAN weights (SSL pool without the val ids) but the registered HAM10000 row of the
proposed model comes from the ORIGINAL weights. ``scripts/ham_clean_run.py`` evaluates B71-limpio and
A71-limpio on the C1 val with the exact v7.1 protocol and the canonical compute (batch 64 + CUDA autocast);
nothing is selected here: configurations and operating thresholds come from the original holdout selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dermssl import c1_split_identity as si
from dermssl import final_eval as fe
from dermssl.isic_eval import protocol as pr

DIR = "resultados/banco_completo/ham_clean"
RESULT_JSON = f"{DIR}/ham_clean_val.json"
RESULT_NPZ = f"{DIR}/ham_clean_val.probas.npz"
REHEARSAL_JSON = "ham_clean_holdout_rehearsal.json"
EXTRACT_JSON = "ham_clean_extract.json"
CLEAN, ORIGINAL, PUBLIC = ("B71_clean", "A71_clean"), ("B71", "A71"), ("public_B71", "public_A71")
CONTRASTS = (("B71_clean", "public_B71"), ("B71_clean", "B71"), ("A71_clean", "public_A71"), ("A71_clean", "A71"),
             ("B71_clean", "A71_clean"))
CONTRAST_METRICS = (*fe.METRICS, "mel_recall", "mel_auc", "mcc")  # mcc appended LAST: other draws unchanged
# sha256 of the released clean finals (file simsiam_dinov2_backbone_final.pt, 346388719 bytes each).
RELEASE_SHA256 = {
    "B_s42": "a392ebe8eedcc519a6131e3930bed5ea232d4be7084010f82740e55cadbe39c1",  # v7clean/final_B_s42
    "A_s42": "9a4b88f415da66349646fb68a2b3ad3ac50845f04b1179f1aa871639c0c5e2c8",  # v7clean/final_A_s42
    "A_s43": "7994ffa54ad08b7aedfbe0a3e0bcda02d125e6903aea75a862238c3d39b9bcfc",  # v7clean/final_A_s43
}
GROUPED_BY = "lesion_id"
CANONICAL_BATCH = 64
NEEDED_FINALS = (("B", "s42"), ("A", "s42"), ("A", "s43"))  # B71-limpio = B s42; A71-limpio = soup A s42 + s43
LABEL = ("analisis de sensibilidad posterior; no es seleccion; la seleccion se hizo sobre el holdout y el "
         "val_read.lock de E3 no se reinterpreta")
THRESHOLD_ASSUMPTION = ("los umbrales de operacion del holdout se ajustaron con los pesos ORIGINALES y se aplican "
                        "sin cambios a los pesos limpios (supuesto declarado)")
REHEARSAL_WARNING = ("ensayo de plomeria sobre el holdout: descartado, no se reporta ni se usa para decidir nada")


class HamCleanError(RuntimeError):
    """A precondition of the clean-weights run does not hold."""


PROBE_JOBS, DECODE_WORKERS = 10, 16  # defaults shared by ham_clean_run.py and scripts/pod/ham_clean.sh


def check_finals_exist(paths: dict) -> None:
    """Every final .pt must be on disk before the lock is taken."""
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise HamCleanError(f"faltan finales: {missing}")


def check_run_identity(manifests: dict) -> None:
    """``run.json`` seed of ``<arm>_s<seed>`` must be that seed: a swapped or copied folder does not pass."""
    for name, run in manifests.items():
        want = int(name.rsplit("_s", 1)[1])
        if run.get("seed") != want:
            raise HamCleanError(f"{name}: el run.json trae seed {run.get('seed')!r}, la carpeta es de la seed {want}")


def check_distinct_weights(shas: dict) -> None:
    """Two finals with the same bytes are a copied folder, not two seeds."""
    seen: dict = {}
    for name, sha in shas.items():
        if sha in seen:
            raise HamCleanError(f"{name} y {seen[sha]} tienen pesos identicos (sha256 {sha[:12]}): carpeta copiada")
        seen[sha] = name


def check_release_hashes(shas: dict, pins: dict | None = None) -> None:
    """Each final must be byte-identical to the released file; names which one differs from which hash."""
    pins = RELEASE_SHA256 if pins is None else pins
    wrong = [f"v7clean/final_{name}: sha256 {shas[name]} != release {pins[name]}"
             for name in pins if shas.get(name) != pins[name]]
    if wrong:
        raise HamCleanError("la final no es la publicada (release): " + "; ".join(wrong))


def check_images(ids, index: dict) -> None:
    """Every image to decode has a file (an undecodable surprise must not appear after the lock)."""
    missing = [i for i in ids if i not in index]
    if missing:
        raise HamCleanError(f"{len(missing)} imagenes sin archivo, p.ej. {missing[:3]}")


def require_cuda(available: bool) -> None:
    if not available:
        raise HamCleanError("CUDA no disponible: la val exige el cómputo canonico (autocast cuda)")


def declarations(*, rehearsal: bool) -> dict:
    """What the result says about itself: label, declared assumption and rehearsal flag."""
    out = {"label": LABEL, "assumptions": [THRESHOLD_ASSUMPTION], "rehearsal": rehearsal}
    return {**out, "discard": True, "warning": REHEARSAL_WARNING} if rehearsal else out


def clean_entries(original_entries: dict) -> dict:
    """Clean labels -> config ids: the ORIGINAL selection's B71 / A71, which must be the registered ones."""
    registered = {"B71": pr.B71, "A71": pr.A71}
    for label, cid in registered.items():
        if original_entries[label] != cid:
            raise HamCleanError(f"{label}: la seleccion original trae {original_entries[label]!r}, "
                                f"el protocolo registrado es {cid!r}")
    return {"B71_clean": original_entries["B71"], "A71_clean": original_entries["A71"]}


def check_reference(ref: dict, y_ref, *, split: str, eval_ids, y, entries: dict) -> None:
    """The original/public probabilities pair with the clean ones only on identical split, ids, labels, configs."""
    if ref.get("split") != split:
        raise HamCleanError(f"la referencia es del split {ref.get('split')!r}, no {split!r}")
    if ref.get("eval_ids_sha256") != si.ids_sha256(list(eval_ids)):
        raise HamCleanError("los ids evaluados no son los de la referencia: no hay pareo posible")
    if not np.array_equal(np.asarray(y_ref), np.asarray(y)):
        raise HamCleanError("las etiquetas de la referencia no coinciden con las evaluadas")
    for label in (*ORIGINAL, *PUBLIC):
        if ref["entries"].get(label) != entries[label]:
            raise HamCleanError(f"{label}: la referencia usa {ref['entries'].get(label)!r}, "
                                f"la seleccion {entries[label]!r}")


def compute_block(device_type: str, batch_size: int, *, rehearsal: bool) -> dict:
    """Canonical compute is batch 64 + CUDA autocast; the real val read refuses anything else."""
    canonical = device_type == "cuda" and batch_size == CANONICAL_BATCH
    if not canonical and not rehearsal:
        raise HamCleanError(f"cómputo no canonico: device={device_type}, batch={batch_size}; "
                            f"la val exige cuda y batch {CANONICAL_BATCH} (autocast cuda)")
    return {"device": device_type, "batch_size": batch_size, "autocast": "cuda" if device_type == "cuda" else "none",
            "canonical": canonical}


def contrast_probas(clean: dict, reference: dict) -> dict:
    """Every array a contrast reads, checked to share one shape."""
    probas = {**clean, **{label: reference[label] for label in (*ORIGINAL, *PUBLIC)}}
    shapes = {name: np.shape(p) for name, p in probas.items()}
    if len(set(shapes.values())) != 1:
        raise HamCleanError(f"forma distinta entre probabilidades: {shapes}")
    return probas


def measured(repo: Path) -> bool:
    """True only for the real, canonical val read: not a rehearsal, split val, canonical compute, probabilities on disk."""
    path = Path(repo) / RESULT_JSON
    if not path.is_file() or not (Path(repo) / RESULT_NPZ).is_file():
        return False
    doc = json.loads(path.read_text(encoding="utf-8"))
    return (doc.get("rehearsal") is False and doc.get("split") == "val"
            and (doc.get("compute") or {}).get("canonical") is True)
