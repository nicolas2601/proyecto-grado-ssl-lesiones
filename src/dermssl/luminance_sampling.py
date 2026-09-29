"""T7 (docs/27): luminance terciles per cache row and the fixed-budget oversampled epoch.

The bins are ``tone_tercile`` of the published manifest: terciles of the skin
median L* over the pool, luminance and not skin tone. The dark tercile is
oversampled by an integer factor ``k`` without changing the budget: each
epoch is the multiset of the training rows plus ``k - 1`` extra copies of the
dark ones, permuted with ``default_rng([seed, epoch])`` and cut to the number
of samples A2 drew. With ``k = 1`` the permutation is the one
``sampling.shard_batches`` draws itself, so A2's batches are reproduced
exactly (tests pin it). Pure numpy and csv.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from dermssl import pad_ufes_guard as pg

REGISTRATION_COMMIT = "d24a0f252e6c06a6bce5c85cf01de52139a64ae5"
REGISTRATION_DOC = "docs/27-plan-v7-mejoras-ssl.md"
BLOCK_START = "<!-- registro-t7-reentreno-luminancia:inicio -->"
BLOCK_END = "<!-- registro-t7-reentreno-luminancia:fin -->"
#: The code that implements T7's training changes; it must postdate the registration.
ANCHOR = "src/dermssl/luminance_sampling.py"
MANIFEST_SHA256 = "57df1e65e88d3b54ba530d6e026010bfc8ce25d1437e50b4ffe37cfd7d4f87da"
PSEUDO_R0_SHA256 = "8c9e47978070baa7bfa9f0174a80eb4e4236d45737801abf690804ae1eb0fc87"
CACHE_MANIFEST_SHA256 = "a83f927ab798b6f7a0614c3923fda77d082f887d10fdbe73416e34e9574ba71a"
REGISTERED_K = 2
#: The committed r0 teacher check that must precede any T7 GPU job (docs/27, T7).
TEACHER_CHECK_DIR = "plan-segmentacion/camino-b-cnn/resultados/2026-09-29-t7-r0-check"
TEACHER_CHECK_JSON, TEACHER_CHECK_ROWS = "t7_r0_tercile_check.json", "r0_per_image.csv"
#: The committed result (bbbcd4a; block moved to its sidecar in 01b414f; re-run with reference_masks_sha256,
#: identical analysis and rows): a later commit cannot swap it silently.
TEACHER_CHECK_SHA256 = {TEACHER_CHECK_JSON: "3f6e5443446c3c02d227181ff7e7d64b0e5b3730d6522731d8afded292e55163",
                        TEACHER_CHECK_ROWS: "4dc4875f1f2288f191bdfe3ffe79676e362154dee22461fdb641f41ddc528f4e"}
#: A2's published HAM10000 per-image Dice (its SHA256SUMS entry), the other side of the addendum rule.
A2_HAM_PER_IMAGE = "plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a2/evaluacion_ham10000/per_image.csv"
A2_HAM_PER_IMAGE_SHA256 = "f464932812aa9e9f67a065f165f12cc3945e86b10228e9a045a89ff284eeef78"
#: Written outside the block by the dated addendum "T7 no cumple por techo del maestro".
ADDENDUM_MARKER = "<!-- adenda-t7-techo-maestro -->"
DARK = "dark"
DARKENED = ("medium", "light")
UNAVAILABLE = "unavailable"


def check_registration(repo: Path) -> str:
    return pg.check_registration(repo, commit=REGISTRATION_COMMIT, doc=REGISTRATION_DOC, script=ANCHOR,
                                 start=BLOCK_START, end=BLOCK_END)


class TeacherGateError(ValueError):
    """The r0 teacher check is missing, ran on other inputs, or triggered an addendum not yet committed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def terciles_by_row(rows: Iterable[Mapping], manifest: Path, expected_sha256: str) -> np.ndarray:
    """``tone_tercile`` for every cache row (by ``index``); ids absent from the manifest are unavailable."""
    digest = _sha256(Path(manifest))
    if digest != expected_sha256:
        raise ValueError(f"el manifiesto {manifest} tiene sha256 {digest[:12]}, no el registrado")
    with open(manifest, newline="") as handle:
        by_id = {r["image_id"]: r["tone_tercile"] for r in csv.DictReader(handle)}
    rows = list(rows)
    labels = np.full(max(int(r["index"]) for r in rows) + 1, UNAVAILABLE, dtype=object)
    for row in rows:
        labels[int(row["index"])] = by_id.get(row["image_id"], UNAVAILABLE)
    return labels.astype(str)


def darken_eligible(labels: np.ndarray) -> np.ndarray:
    """Rows the darkening augmentation may touch: the medium and light terciles only."""
    return np.isin(np.asarray(labels), DARKENED)


def oversampled_epoch(train: np.ndarray, is_dark: np.ndarray, k: int, n_samples: int, seed: int,
                      epoch: int) -> np.ndarray:
    """The epoch's sample order: train plus ``k - 1`` copies of the dark rows, permuted, cut to ``n_samples``."""
    train, is_dark = np.asarray(train), np.asarray(is_dark, dtype=bool)
    if k < 1:
        raise ValueError(f"k tiene que ser un entero >= 1, no {k}")
    if is_dark.shape != train.shape:
        raise ValueError(f"is_dark tiene longitud {is_dark.size}, train {train.size}")
    pool = np.concatenate([train, *([train[is_dark]] * (k - 1))])
    if n_samples > pool.size:
        raise ValueError(f"se piden {n_samples} muestras y el multiconjunto tiene {pool.size}")
    order = np.random.default_rng([seed, epoch]).permutation(pool.size)
    return pool[order][:n_samples]


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def recomputed_addendum(rows: list[dict], a2_dice: Mapping[str, float]) -> bool:
    """The registered rule from the rows: |G(r0)| >= |G(A2)|, point estimates, form (a), same kept images."""
    kept = [r for r in rows if r["excluded"] == "False" and r["tone_tercile"] in (DARK, *DARKENED)]
    dark = [r for r in kept if r["tone_tercile"] == DARK]
    rest = [r for r in kept if r["tone_tercile"] != DARK]
    if not dark or not rest:
        raise TeacherGateError("el chequeo del maestro no tiene imágenes en el oscuro o en el resto")
    mean = lambda group, value: float(np.mean([value(r) for r in group]))  # noqa: E731
    teacher = mean(dark, lambda r: float(r["dice_binary"])) - mean(rest, lambda r: float(r["dice_binary"]))
    a2 = mean(dark, lambda r: a2_dice[r["image_id"]]) - mean(rest, lambda r: a2_dice[r["image_id"]])
    return abs(teacher) >= abs(a2)


def check_teacher_gate(repo: Path) -> dict:
    """The registered teacher check ran on the registered inputs, its addendum flag is the one its hashed
    rows imply, and the addendum is committed if triggered."""
    repo = Path(repo)
    result = repo / TEACHER_CHECK_DIR
    if not (result / TEACHER_CHECK_JSON).is_file():
        raise TeacherGateError(f"no está el chequeo del maestro en {TEACHER_CHECK_DIR}: se corre antes de la GPU")
    for name, pinned in TEACHER_CHECK_SHA256.items():
        if _sha256(result / name) != pinned:
            raise TeacherGateError(f"{name} no es el resultado comprometido del chequeo del maestro")
    doc = json.loads((result / TEACHER_CHECK_JSON).read_text(encoding="utf-8"))
    provenance = doc["provenance"]
    expected = {"registration_commit": REGISTRATION_COMMIT, "manifest_sha256": MANIFEST_SHA256,
                "pseudo_r0_sha256": PSEUDO_R0_SHA256, "cache_manifest_sha256": CACHE_MANIFEST_SHA256,
                "rows_sha256": _sha256(result / TEACHER_CHECK_ROWS)}
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise TeacherGateError(f"el chequeo del maestro no coincide en {key}")
    reference = provenance.get("reference_masks_sha256")
    if not isinstance(reference, str) or not re.fullmatch(r"[0-9a-f]{64}", reference):
        raise TeacherGateError("el chequeo del maestro no fija reference_masks_sha256 (sha256 hex de 64)")
    if _sha256(repo / A2_HAM_PER_IMAGE) != A2_HAM_PER_IMAGE_SHA256:
        raise TeacherGateError("el per_image.csv de A2 en HAM10000 no es el publicado")
    a2_dice = {r["image_id"]: float(r["dice"]) for r in _read_csv(repo / A2_HAM_PER_IMAGE)}
    rows = _read_csv(result / TEACHER_CHECK_ROWS)
    ids = [r["image_id"] for r in rows]
    if len(ids) != len(set(ids)) or not set(ids) <= a2_dice.keys():
        raise TeacherGateError("filas del chequeo del maestro duplicadas o fuera de la evaluación de A2")
    addendum = recomputed_addendum(rows, a2_dice)
    if addendum != bool(doc["analysis"]["addendum_required"]):
        raise TeacherGateError("addendum_required del JSON no coincide con lo recalculado desde las filas")
    if addendum and ADDENDUM_MARKER not in (repo / REGISTRATION_DOC).read_text(encoding="utf-8"):
        raise TeacherGateError("el chequeo del maestro exige la adenda fechada y no está en el registro")
    return {"addendum_required": addendum, "rows_sha256": expected["rows_sha256"]}
