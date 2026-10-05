"""ISIC 2019 / ISIC 2020 rows of the bench: the E1 cells under ``resultados/banco_completo/isic_eval``.

Canonical rows use batch 64 + autocast (the compute of v7.1). The batch-16 run of the original
B71/A71 is a SENSITIVITY block: non-canonical compute and transductive (docs/33, errata).
A cell that is not on disk yet is a pending row with its reason and ETA, and becomes a
measured row on its own when its JSON appears.
"""

from __future__ import annotations

from pathlib import Path

from dermssl.bench import metrics, pending
from dermssl.bench.rows import FROZEN, OK, OWN_A, OWN_B, PENDING, PLAN_DOC, PUBLIC, SENSITIVITY, Row, Value, _load

ISIC_EVAL = "resultados/banco_completo/isic_eval"
CANONICAL_DIR = f"{ISIC_EVAL}/limpio_canonico_bs64"
SENSITIVITY_DIR = f"{ISIC_EVAL}/originals_bs16_transductivo"
CANONICAL_NOTE = "cómputo canónico (batch 64 + autocast)"
CANONICAL_CONTRASTS = ("B71__clean__vs__B71__original", "A71__clean__vs__A71__original", "B71__clean__vs__A71__clean",
                       "B71__clean__vs__PUBLIC_B71", "A71__clean__vs__PUBLIC_A71")
SENSITIVITY_NOTE = "sensibilidad: cómputo no canónico (batch 16)"
VALUE_KEYS = ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "auc", "ece", "mcc")
# (cell, label, pretraining, transductive)
CANONICAL_CELLS = (
    ("B71__clean", "B71-limpio (canónico)", OWN_B, False),
    ("A71__clean", "A71-limpio (canónico)", OWN_A, False),
    ("PUBLIC_B71", "DINOv2 público, protocolo de B71 (canónico)", PUBLIC, False),
    ("PUBLIC_A71", "DINOv2 público, protocolo de A71 (canónico)", PUBLIC, False),
    ("B71__original", "B71 original (canónico)", OWN_B, True),
    ("A71__original", "A71 original (canónico)", OWN_A, True),
)
SENSITIVITY_CELLS = (
    ("B71__original", "B71 original (batch 16)", OWN_B, True),
    ("A71__original", "A71 original (batch 16)", OWN_A, True),
    ("PUBLIC_B71", "DINOv2 público, protocolo de B71 (batch 16)", PUBLIC, False),
    ("PUBLIC_A71", "DINOv2 público, protocolo de A71 (batch 16)", PUBLIC, False),
)


def cell_path(folder: str, ds_key: str, cell: str) -> str:
    return f"{folder}/{ds_key}/{cell}.json"


def cell_values(record: dict) -> dict[str, Value]:
    """Metric values of one E1 cell; the CI of every published metric comes from the cell."""
    stored = record["metrics"]
    ci = stored.get("ci95", {})
    out = {k: Value(stored[k], tuple(ci[k]) if k in ci else None) for k in VALUE_KEYS if k in stored}
    if "mcc" not in out:
        out["mcc"] = Value(metrics.mcc_from_confusion(stored["matriz_confusion"]))
    return out


def _measured(repo: Path, rel: str, block: str, name: str, pre: str, trans: bool, note: str) -> Row:
    result = _load(repo, rel)["result"]
    probas = rel.replace(".json", ".probas.npz")
    return Row(result["dataset"], block, name, pre, False, trans, OK, (rel, probas), result["n_val"],
               cell_values(result), note)


def _pending_row(ds_key: str, block: str, name: str, pre: str, trans: bool, why: pending.Pending) -> Row:
    return Row(ds_key, block, name, pre, False, trans, PENDING, (PLAN_DOC,), note=why.note)


def canonical_rows(repo: Path, ds_key: str) -> list[Row]:
    out = []
    for cell, name, pre, trans in CANONICAL_CELLS:
        rel = cell_path(CANONICAL_DIR, ds_key, cell)
        if (repo / rel).is_file():
            out.append(_measured(repo, rel, FROZEN, name, pre, trans, CANONICAL_NOTE))
        else:
            out.append(_pending_row(ds_key, FROZEN, name, pre, trans, pending.missing(rel)))
    return out


def sensitivity_rows(repo: Path, ds_key: str) -> list[Row]:
    return [_measured(repo, cell_path(SENSITIVITY_DIR, ds_key, cell), SENSITIVITY, name, pre, trans,
                      SENSITIVITY_NOTE + ("; transductivo" if trans else ""))
            for cell, name, pre, trans in SENSITIVITY_CELLS]


def isic_rows(repo: Path, ds_key: str) -> list[Row]:
    return canonical_rows(repo, ds_key) + sensitivity_rows(repo, ds_key)
