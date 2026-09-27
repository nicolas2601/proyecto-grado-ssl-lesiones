"""Recording, and verifying, which images C1 ran on.

Why this module exists
----------------------
The bank's guarantee that two runs saw the same images is its cache key, which
embeds ``fingerprint_ids``: a sha256 of the ORDERED id list truncated to 16
hex. That is enough to make a stale cache miss, but it is short, implicit, and
only covers the val side.

There is no "split_hash of the bank". That name exists in the repo
(``pool_manifest.split_hash``, ``run_stage2.compute_pool_split_hash``) and
hashes the PRE-TRAINING pool -- a different subsystem. Recording it as though
it identified this split would put a number in the manifest that looks like
provenance and is not, which is the same defect as a fallback counter that is
structurally zero.

So C1 records three values, each named for what it actually is, plus a note
saying the short one is short.

Order matters throughout: features are stored aligned to the id order, so two
splits with identical membership and different order are different splits, and
every row of one would be scored against the other's labels.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from dermssl.feature_cache import fingerprint_ids

#: How many hex chars the cache keeps. Quoted in the note so nobody reads the
#: short value as a truncation of a different construction.
FINGERPRINT_LENGTH = 16


def ids_sha256(ids: Sequence[str]) -> str:
    """Full, untruncated digest of an ordered id list.

    Built exactly like :func:`dermssl.feature_cache.fingerprint_ids` so the
    first 16 characters ARE that fingerprint -- one construction, two lengths,
    and a test that fails if they ever drift.
    """
    if not len(ids):
        raise ValueError(
            "lista de ids vacia: su digest es una constante que coincidiria con "
            "cualquier otro split vacio, asi que no identifica nada"
        )
    digest = hashlib.sha256()
    for identifier in ids:
        digest.update(str(identifier).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def split_identity(*, train_ids: Sequence[str], val_ids: Sequence[str]) -> dict:
    """The three provenance values, under their real names."""
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise ValueError(
            f"{len(overlap)} imagenes en ambos lados del split (p.ej. {sorted(overlap)[:3]}): "
            "el probe se entrenaria con las imagenes con las que se lo evalua"
        )

    return {
        "c1_train_ids_sha256": ids_sha256(train_ids),
        "c1_val_ids_sha256": ids_sha256(val_ids),
        "feature_cache_ids_fingerprint": fingerprint_ids(val_ids),
        "fingerprint_note": (
            f"feature_cache_ids_fingerprint son los primeros {FINGERPRINT_LENGTH} hex del "
            "sha256 de los val_ids; es el valor que nombra los archivos del cache del "
            "banco v6, no un hash completo"
        ),
        "n_train": int(len(train_ids)),
        "n_val": int(len(val_ids)),
    }


def _compare_side(name: str, stored: object, rebuilt: Sequence[str]) -> list[str]:
    if stored is None:
        return [
            f"el indice guardado no tiene {name}: no se puede comparar ese lado, y darlo "
            "por igual seria el pase vacuo que esta verificacion existe para evitar"
        ]
    stored_list = list(stored)
    rebuilt_list = list(rebuilt)
    if stored_list == rebuilt_list:
        return []

    reasons = []
    missing = set(stored_list) - set(rebuilt_list)
    extra = set(rebuilt_list) - set(stored_list)
    if missing or extra:
        reasons.append(
            f"{name}: {len(missing)} ids del indice no estan en el split reconstruido y "
            f"{len(extra)} sobran (guardado {len(stored_list)}, reconstruido "
            f"{len(rebuilt_list)})"
        )
    else:
        reasons.append(
            f"{name}: mismos ids en distinto ORDEN. Los features se guardan alineados a "
            "ese orden, asi que cada fila quedaria contra la etiqueta de otra imagen"
        )
    return reasons


def compare_to_index(
    stored_index: Mapping[str, object],
    *,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
) -> tuple[bool, list[str]]:
    """``(agrees, reasons)`` against an existing ``val_index.json``.

    BOTH sides are compared. The bank gate only ever reads ``val_ids``, so a
    difference in train would pass unnoticed there -- and train is what the
    probe is FIT on.
    """
    reasons = _compare_side("train_ids", stored_index.get("train_ids"), train_ids)
    reasons += _compare_side("val_ids", stored_index.get("val_ids"), val_ids)
    return not reasons, reasons


def compare_to_index_detailed(
    stored_index: Mapping[str, object],
    *,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
) -> dict:
    """Separate "these disagree" from "this could not be compared".

    Measured on the pod: the real ``val_index.json`` holds
    ``['n_classes', 'val_ids', 'val_order_sha256', 'y_true']``. It describes
    the evaluation set and never claimed to describe train, so reporting the
    absent ``train_ids`` as a MISMATCH would turn a missing field into the
    claim "the bank and C1 ran on different images" -- far more than the facts
    support, and exactly the kind of overstated alarm that gets a real one
    ignored later.

    ``agrees`` therefore covers only the sides that could actually be
    compared; what could not be is listed separately, with what to do about it.
    """
    mismatches: list[str] = []
    not_comparable: list[str] = []

    for name, rebuilt in (("train_ids", train_ids), ("val_ids", val_ids)):
        stored = stored_index.get(name)
        if stored is None:
            not_comparable.append(name)
            continue
        mismatches += _compare_side(name, stored, rebuilt)

    # The file already carries a digest of the val order. Ignoring evidence
    # that is sitting right there would be a choice to know less.
    recorded_digest = stored_index.get("val_order_sha256")
    if recorded_digest is None:
        digest_verdict = "el indice no trae val_order_sha256"
    elif recorded_digest == ids_sha256(val_ids):
        digest_verdict = "coincide"
    else:
        # NOT a mismatch on its own: the ids may match while the digest was
        # built another way (different separator, sorted, encoded otherwise).
        # Worth saying, not worth calling a different split over.
        digest_verdict = (
            "el indice trae val_order_sha256 pero no coincide con el nuestro; si los "
            "val_ids SI coinciden, es otra construccion del hash, no otro split"
        )

    advice = ""
    if not_comparable:
        advice = (
            f"el indice no describe {', '.join(not_comparable)}; C1 escribe su propio "
            "indice completo y registra los sha256 de ambos lados, para que la proxima "
            "verificacion pueda comparar todo"
        )

    return {
        "agrees": not mismatches,
        "mismatches": mismatches,
        "not_comparable": not_comparable,
        "val_order_sha256": digest_verdict,
        "advice": advice,
    }


def shared_lesions(
    *,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    lesion_by_id: Mapping[str, str],
) -> list[str]:
    """Lesion ids present on both sides, sorted. Empty is the expected answer.

    ``StratifiedGroupKFold`` groups by lesion and should make this impossible,
    and ``build_split`` already asserts it. Verified again here because the
    cost of being wrong is that the same lesion, photographed twice, trains
    and evaluates the probe -- which inflates every number without breaking
    anything.

    An id with no known lesion raises: skipping it would let the check pass by
    knowing less.
    """
    train_lesions = {lesion_by_id[i] for i in train_ids}
    val_lesions = {lesion_by_id[i] for i in val_ids}
    return sorted(train_lesions & val_lesions)
