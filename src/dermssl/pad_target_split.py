"""PAD-UFES-20 split by patient into ``target_adapt`` and ``target_test`` (docs/33 section 4).

``target_adapt`` feeds SSL adaptation (images only, no labels); ``target_test`` is
locked and read once, by the single evaluation. Rules inherited from the plan:

* a patient is entirely on one side, so no lesion key ``(patient_id, lesion_id)``
  spans both;
* images are deduplicated by ``img_id`` and by content sha256 before the split;
* stratified by diagnostic as far as the grouping allows, deterministic by seed.

The proportion is NOT fixed by docs/33 or docs/30: 50 % of the patients go to
``target_adapt`` (``ADAPT_FRACTION``) as a declared assumption, not a registered rule.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path

SEED = 20260923
ADAPT_FRACTION = 0.5
LOCK_NAME = "target_test_read.lock"
SPLIT_VERSION = 1
Row = dict
HashOf = Callable[[Row], str]


class TargetSplitError(ValueError):
    """The rows cannot be split as the plan requires."""


class TargetTestPinError(RuntimeError):
    """The target_test list does not match the hash recorded when it was built."""


class TargetTestAlreadyReadError(RuntimeError):
    """target_test was already read (the lock exists)."""


def ids_sha256(ids: Iterable[str]) -> str:
    """sha256 of the sorted ids joined by newlines: independent of list order."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def dedup(rows: list[Row], hash_of: HashOf) -> tuple[list[Row], dict]:
    """Drop repeated ``img_id`` and byte-identical copies; the smallest ``img_id`` is kept."""
    kept, seen_ids, by_hash = [], set(), {}
    report = {"n_input": len(rows), "dropped_duplicate_img_id": [], "dropped_duplicate_content": []}
    for row in sorted(rows, key=lambda r: r["img_id"]):
        if row["img_id"] in seen_ids:
            report["dropped_duplicate_img_id"].append(row["img_id"])
            continue
        seen_ids.add(row["img_id"])
        digest = hash_of(row)
        if digest in by_hash:
            report["dropped_duplicate_content"].append({"img_id": row["img_id"], "same_as": by_hash[digest]})
            continue
        by_hash[digest] = row["img_id"]
        kept.append(row)
    report["n_kept"] = len(kept)
    return kept, report


def _patient_order(by_patient: dict[str, list[Row]], totals: Counter, seed: int) -> list[str]:
    """Patients with the rarest diagnostic first, shuffled by seed within each rarity."""
    rarity = {dx: rank for rank, dx in enumerate(sorted(totals, key=lambda d: (totals[d], d)))}
    shuffled = sorted(by_patient)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled, key=lambda p: min(rarity[r["diagnostic"]] for r in by_patient[p]))


def _unfilled(counts: Counter, target: dict[str, float], assigned: Counter) -> float:
    """Share of each diagnostic's target still open on a side, weighted by the patient's images."""
    return sum(n * (target[dx] - assigned[dx]) / target[dx] for dx, n in counts.items())


def split_by_patient(rows: list[Row], *, seed: int = SEED,
                     adapt_fraction: float = ADAPT_FRACTION) -> tuple[list[Row], list[Row]]:
    """``(target_adapt rows, target_test rows)``, whole patients, greedy by diagnostic deficit."""
    if not 0.0 < adapt_fraction < 1.0:
        raise TargetSplitError(f"adapt_fraction {adapt_fraction} fuera de (0, 1): fraction invalida")
    by_patient: dict[str, list[Row]] = {}
    for row in rows:
        by_patient.setdefault(row["patient_id"], []).append(row)
    if len(by_patient) < 2:
        raise TargetSplitError("se necesitan al menos 2 patients para dividir")
    totals = Counter(r["diagnostic"] for r in rows)
    targets = {"adapt": {d: adapt_fraction * n for d, n in totals.items()},
               "test": {d: (1 - adapt_fraction) * n for d, n in totals.items()}}
    assigned = {"adapt": Counter(), "test": Counter()}
    sides: dict[str, list[Row]] = {"adapt": [], "test": []}
    for patient in _patient_order(by_patient, totals, seed):
        counts = Counter(r["diagnostic"] for r in by_patient[patient])
        score = {s: _unfilled(counts, targets[s], assigned[s]) for s in sides}
        if score["adapt"] != score["test"]:
            side = max(score, key=score.get)
        else:
            side = "adapt" if len(sides["adapt"]) <= len(sides["test"]) else "test"
        sides[side] += by_patient[patient]
        assigned[side].update(counts)
    return sides["adapt"], sides["test"]


def check_disjoint(adapt: list[Row], test: list[Row], hash_of: HashOf) -> None:
    """Refuse any image, patient, lesion key or content hash present on both sides."""
    checks = {
        "img_id": lambda r: r["img_id"],
        "patient_id": lambda r: r["patient_id"],
        "lesion key": lambda r: (r["patient_id"], r["lesion_id"]),
        "content": hash_of,
    }
    for name, key in checks.items():
        shared = {key(r) for r in adapt} & {key(r) for r in test}
        if shared:
            raise TargetSplitError(f"{len(shared)} {name} en ambos lados, p. ej. {sorted(map(str, shared))[:3]}")


def _side(rows: list[Row]) -> dict:
    ids = sorted(r["img_id"] for r in rows)
    return {"n_images": len(ids), "n_patients": len({r["patient_id"] for r in rows}),
            "n_lesions": len({(r["patient_id"], r["lesion_id"]) for r in rows}),
            "by_diagnostic": dict(sorted(Counter(r["diagnostic"] for r in rows).items())),
            "patient_ids": sorted({r["patient_id"] for r in rows}), "img_ids": ids}


def build_document(rows: list[Row], hash_of: HashOf, *, manifest_sha256: str, seed: int = SEED,
                   adapt_fraction: float = ADAPT_FRACTION) -> dict:
    """Dedup, split, verify disjointness and describe both sides; nothing is written."""
    kept, report = dedup(rows, hash_of)
    adapt, test = split_by_patient(kept, seed=seed, adapt_fraction=adapt_fraction)
    check_disjoint(adapt, test, hash_of)
    sides = {"target_adapt": _side(adapt), "target_test": _side(test)}
    return {"split_version": SPLIT_VERSION, "seed": seed, "adapt_fraction": adapt_fraction,
            "fraction_source": "assumption: neither docs/33 nor docs/30 fixes it",
            "manifest_sha256": manifest_sha256, "dedup": report, **sides,
            "sha256": {name: ids_sha256(side["img_ids"]) for name, side in sides.items()}}


def _load_pinned(split_path: Path, side: str) -> tuple[list[str], str]:
    doc = json.loads(Path(split_path).read_text(encoding="utf-8"))
    ids = doc[side]["img_ids"]
    if ids_sha256(ids) != doc["sha256"][side]:
        raise TargetTestPinError(f"la lista {side} de {split_path} no coincide con su sha256 registrado")
    return ids, doc["sha256"][side]


def read_target_adapt_ids(split_path: Path) -> list[str]:
    """The adaptation ids (free to read, pinned to the recorded hash)."""
    return _load_pinned(split_path, "target_adapt")[0]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TargetTestLock:
    """``target_test_read.lock``: created with ``O_EXCL``, never removed by code."""

    def __init__(self, directory: Path, clock: Callable[[], str] = _utc_now):
        self.path = Path(directory) / LOCK_NAME
        self._clock = clock

    def acquire(self, target_test_sha256: str) -> None:
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise TargetTestAlreadyReadError(
                f"target_test ya se leyo ({self.path}); la evaluacion unica no se repite") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"target_test_sha256": target_test_sha256, "read_at": self._clock()}, handle)


def read_target_test_ids(split_path: Path, lock: TargetTestLock) -> list[str]:
    """The locked ids. Verifies the pin first (a corrupt file does not burn the lock), then locks."""
    ids, digest = _load_pinned(split_path, "target_test")
    lock.acquire(digest)
    return ids
