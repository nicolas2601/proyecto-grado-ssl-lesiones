"""HAM10000 clean-weights sensitivity (docs/33): pure rules, paths and labels, no torch.

The release publishes CLEAN weights (SSL pool without the val ids) but the registered HAM10000 row of the
proposed model comes from the ORIGINAL weights. ``scripts/ham_clean_run.py`` evaluates B71-limpio and
A71-limpio on the C1 val with the exact v7.1 protocol and the canonical compute (batch 64 + CUDA autocast);
nothing is selected here: configurations and operating thresholds come from the original holdout selection.
"""

from __future__ import annotations

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
