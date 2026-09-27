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
import re
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


def identify_digest(recorded: str, ids: Sequence[str]) -> str | None:
    """Which known construction produced ``recorded`` for ``ids``, if any.

    Measured on the pod: the bank's ``val_order_sha256`` is sha256 of
    ``"\\n".join(ids)`` -- separator BETWEEN ids -- while
    :func:`ids_sha256` appends ``"\\n"`` after EACH id. Same list, same order,
    one trailing byte apart.

    Identifying it matters: it turns a digest nobody could explain into a
    second, independent confirmation that the stored index and the rebuilt
    split hold the same sequence. Both constructions are order-sensitive, so
    a match is evidence about the ORDER and not just the membership.

    Returns ``None`` rather than guessing. An unrecognised digest is a fact to
    report, not one to explain away.
    """
    known = {
        "join_newline": hashlib.sha256("\n".join(map(str, ids)).encode("utf-8")).hexdigest(),
        "trailing_newline": ids_sha256(ids),
    }
    for name, value in known.items():
        if recorded == value:
            return name
    return None


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
    else:
        construction = identify_digest(str(recorded_digest), val_ids)
        if construction == "trailing_newline":
            digest_verdict = "coincide"
        elif construction is not None:
            # Same ids, same ORDER, a different way of hashing them. Both
            # constructions are order-sensitive, so this confirms the sequence.
            digest_verdict = (
                f"coincide con la construccion {construction!r} sobre los mismos val_ids "
                "en el mismo orden (el hash difiere solo en el separador)"
            )
        else:
            digest_verdict = (
                "el indice trae val_order_sha256 y no coincide con ninguna construccion "
                "conocida sobre estos val_ids; queda reportado sin explicar"
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


#: ``v1__<backbone>__<split>__r<res>__<tag>__<fingerprint>``; the fingerprint is
#: the last field, which is what makes the filenames usable as evidence.
_CACHE_NAME = re.compile(
    r"^v\d+__(?P<backbone>.+?)__(?P<split>train|val)__r\d+__[^_]+__(?P<fingerprint>[0-9a-f]+)\."
)


def verify_against_cache_names(
    filenames: Sequence[str],
    *,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
) -> dict:
    """Check the rebuilt split against the fingerprints in the v6 cache names.

    The stored ``val_index.json`` never described train, so there was nothing
    to compare that side against. But the cache names each file after
    ``fingerprint_ids`` of the ids it holds, and it has a train file per
    backbone -- evidence already on disk, for the side that had none.

    Two things are checked per split, and they are different claims:

    * every backbone's file carries the SAME fingerprint (if not, the bank
      itself was built on two different splits and no contrast across
      backbones means anything), and
    * that shared fingerprint is the one the rebuilt split produces.

    ``verified`` is ``None`` when there are no files for a split: missing
    evidence is not contrary evidence, and reporting it as a failure would
    put a real problem and an absent file in the same bucket.
    """
    found: dict[str, set[str]] = {"train": set(), "val": set()}
    counts = {"train": 0, "val": 0}

    for name in filenames:
        match = _CACHE_NAME.match(name)
        if match is None:
            continue
        split = match.group("split")
        found[split].add(match.group("fingerprint"))
        counts[split] += 1

    report = {}
    for split, ids in (("train", train_ids), ("val", val_ids)):
        fingerprints = sorted(found[split])
        expected = fingerprint_ids(ids)
        if not fingerprints:
            verified = None
        else:
            verified = len(fingerprints) == 1 and fingerprints[0] == expected
        report[split] = {
            "verified": verified,
            "expected_fingerprint": expected,
            "fingerprints_found": fingerprints,
            "n_files": counts[split],
        }
    return report


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
