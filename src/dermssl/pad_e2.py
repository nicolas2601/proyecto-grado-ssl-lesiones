"""E2 of docs/33: a HAM-fitted probe, frozen, applied to PAD ``target_test`` BCC/MEL/NV.

Arms D0 (public DINOv2), D1 (A* s42 clean), D2 (D1 + 2000 steps on the pool) and D3
(D2 + PAD ``target_adapt`` images). Primary contrast: paired D3 - D2 macro-F1, patient
grouped; "mejora" only if its interval excludes 0. The model-facing step (features and
probe) is injected as ``predict`` so the protocol around it is testable without a GPU:

1. every manifest image is re-hashed (no lock);
2. predictions are computed from images alone and frozen atomically with their sha256 (no lock);
3. only then is target_test read through the lock (once, ``target_test_read.lock``);
4. the labels are read and the metrics computed. The lock covers 3-4 only.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from dermssl import pad_target_split as ts
from dermssl import pad_ufes as pu
from dermssl import pad_ufes_files as pf
from dermssl import pad_ufes_readings as pr
from dermssl import paired_bootstrap as pb
from dermssl import v71_metrics as vm
from dermssl.isic_eval import scoring

ARMS = ("D0", "D1", "D2", "D3")
PRIMARY = ("D3", "D2")
CONTRASTS = (PRIMARY, ("D2", "D1"), ("D1", "D0"), ("D3", "D1"), ("D3", "D0"), ("D2", "D0"))
REHEARSAL_ARMS = ("D0", "D1")
REHEARSAL_CONTRASTS = (("D1", "D0"),)
REHEARSAL_WARNING = ("ENSAYO DE PLOMERIA sobre target_adapt: la metrica NO se reporta ni se usa para elegir ni "
                     "decidir nada; se descarta. El E2 real corre despues sobre target_test con el lock.")
READING = "primary"
BATCH_SIZE = 64
N_RESAMPLES, RNG_SEED = scoring.N_RESAMPLES, scoring.SEED
#: run.json fields D2 and D3 must share: they may differ only in the pool.
RUN_FIELDS = ("code_revision", "exclude_ids_sha256", "seed", "split_hash", "method", "backbone", "initial_weights",
              "max_steps", "precision", "hyperparameters", "adaptation_levers", "checkpoint_fractions")
Predict = Callable[[list[str], dict[str, Path]], dict[str, np.ndarray]]


class E2Error(ValueError):
    """The E2 protocol was not met."""


class NonCanonicalComputeError(E2Error):
    """Features were not extracted with batch 64 and CUDA autocast (docs/33 canonical compute)."""


class RunMismatchError(E2Error):
    """The D runs do not share the recipe the contrast assumes."""


class RehearsalOutputError(E2Error):
    """A rehearsal result may only be written under a ``*rehearsal*`` directory, never under ``resultados/``."""


class ImageChangedError(E2Error):
    """A target_test file no longer matches the manifest sha256."""


class NonFiniteResamplesError(E2Error):
    """A bootstrap resample gave a non-finite metric; the interval would be silently biased."""


def check_canonical_compute(device_type: str, batch_size: int) -> None:
    if device_type != "cuda" or batch_size != BATCH_SIZE:
        raise NonCanonicalComputeError(
            f"computo canonico: CUDA con autocast y batch {BATCH_SIZE}; recibido {device_type}, batch {batch_size}")


def check_d_runs(runs: dict) -> None:
    """D2 and D3 share every recipe field; D1, D2 and D3 share the clean exclusion list."""
    for arm in ("D1", "D2", "D3"):
        if arm not in runs:
            raise RunMismatchError(f"falta el run.json de {arm}")
    for field in RUN_FIELDS:
        if runs["D2"].get(field) != runs["D3"].get(field):
            raise RunMismatchError(f"{field}: D2 {runs['D2'].get(field)!r} != D3 {runs['D3'].get(field)!r}")
    exclusions = {arm: runs[arm].get("exclude_ids_sha256") for arm in ("D1", "D2", "D3")}
    if not all(exclusions.values()) or len(set(exclusions.values())) != 1:
        raise RunMismatchError(f"exclude_ids_sha256 distinto o ausente entre D1, D2 y D3: {exclusions}")


def _bootstrap(y: np.ndarray, proba: np.ndarray, groups: np.ndarray, n_resamples: int) -> dict:
    draw, rng = pb.index_sampler(len(y), groups), np.random.RandomState(RNG_SEED)
    draws = [scoring.bundle(y[i], proba[i]) for i in (draw(rng) for _ in range(n_resamples))]
    bad = {m: n for m in scoring.HEADLINE if (n := sum(not np.isfinite(d[m]) for d in draws))}
    if bad:
        raise NonFiniteResamplesError(f"remuestras con valor no finito, de {n_resamples}: {bad}")
    return {m: vm.percentile_interval([d[m] for d in draws]) for m in scoring.HEADLINE}


def _arm(y, proba, groups, classes, n_resamples: int) -> dict:
    return {**scoring.bundle(y, proba), **scoring.per_class(y, proba, classes),
            "ci95": _bootstrap(y, proba, groups, n_resamples),
            "bootstrap": {"n_resamples": n_resamples, "seed": RNG_SEED, "unit": "patient_id"}}


def evaluate(ham_labels, groups, probas7: dict, *, n_resamples: int = N_RESAMPLES, arms: tuple = ARMS,
             contrasts: tuple = CONTRASTS) -> dict:
    """Arms and paired contrasts on the primary reading; ``probas7`` are the probe's seven HAM columns.

    The defaults are the real E2 (D0..D3, D3 - D2 primary); the rehearsal passes two arms and no D3 - D2.
    """
    mask, y = pr.targets(ham_labels, READING)
    g, classes = np.asarray(groups)[mask], list(pu.READINGS[READING])
    raw = {arm: np.asarray(probas7[arm])[mask] for arm in arms}
    probas = {arm: pr.restrict(p, READING) for arm, p in raw.items()}
    scored = {arm: _arm(y, probas[arm], g, classes, n_resamples) for arm in arms}  # before the contrasts: typed abort
    paired = [row for arm, control in contrasts
              for row in scoring.contrasts(arm, control, y, probas[arm], probas[control], groups=g,
                                           n_resamples=n_resamples)]
    result = {"reading": READING, "classes": classes, "n_images": int(len(y)), "n_patients": int(len(set(g))),
              "bootstrap": {"n_resamples": n_resamples, "rng_seed": RNG_SEED, "paired": True, "unit": "patient_id"},
              "arms": scored, "paired": paired}
    if PRIMARY in contrasts:
        primary = next(r for r in paired if (r["arm"], r["control"], r["metric"]) == (*PRIMARY, "macro_f1"))
        result |= {"primary_contrast": primary, "prediction_d3_minus_d2_met": primary["direction"] == "arm_better"}
    return result | {"outside_rate": {arm: pr.outside_rate(raw[arm], READING) for arm in arms}}


def _verified_paths(rows: list[dict], pad_root: Path, only: set[str] | None = None) -> dict[str, Path]:
    """Manifest images re-hashed (``only`` restricts them); a changed file aborts before the lock is taken."""
    paths = {}
    for row in rows:
        if only is not None and row["img_id"] not in only:
            continue
        path = Path(pad_root) / row["path"]
        if pf.sha256_of(path) != row["sha256"]:
            raise ImageChangedError(f"{row['img_id']}: el archivo cambio desde el manifiesto")
        paths[row["img_id"]] = path
    return paths


def _atomic_write(path: Path, write: Callable) -> None:
    """``write(handle)`` into ``<path>.tmp``, then rename: a death never leaves a half file."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        write(handle)
    os.replace(tmp, path)


def _freeze(out: Path, ids: list[str], probas7: dict, fingerprint: dict | None = None) -> str:
    """``<out>.probas.npz`` (atomic) and its sha256; with a fingerprint also ``<out>.frozen.json``.

    Written before any label is read. The record lets a relaunch reuse the predictions only when
    the weights, split, manifest and code are the ones that produced them.
    """
    path = out.with_suffix(".probas.npz")
    _atomic_write(path, lambda h: np.savez_compressed(h, ids=np.array(ids), **probas7))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if fingerprint is not None:
        record = {"probas_sha256": sha, "fingerprint": fingerprint}
        _atomic_write(out.with_suffix(".frozen.json"), lambda h: h.write(json.dumps(record, sort_keys=True).encode()))
    return sha


def _load_frozen(out: Path, ids: list[str], fingerprint: dict) -> tuple[dict, str] | None:
    """Earlier frozen predictions, only if their sha256 and fingerprint match; otherwise None (recompute)."""
    npz, record_path = out.with_suffix(".probas.npz"), out.with_suffix(".frozen.json")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        sha = hashlib.sha256(npz.read_bytes()).hexdigest()
        if record["probas_sha256"] != sha or record["fingerprint"] != json.loads(json.dumps(fingerprint)):
            return None
        saved = dict(np.load(npz))
        if list(saved.pop("ids")) != list(ids):
            return None
        _check_shapes(saved, len(ids))
    except (OSError, ValueError, KeyError, E2Error):
        return None
    return saved, sha


def _frozen_predictions(out: Path, ids: list[str], paths: dict, predict: Predict, fingerprint: dict):
    reused = _load_frozen(out, ids, fingerprint)
    if reused is not None:
        return reused
    probas7 = predict(ids, paths)
    _check_shapes(probas7, len(ids))
    return probas7, _freeze(out, ids, probas7, fingerprint)


def _check_shapes(probas7: dict, n: int, arms: tuple = ARMS) -> None:
    bad = {arm: np.shape(probas7.get(arm)) for arm in arms if np.shape(probas7.get(arm)) != (n, len(pr.HAM_CLASSES))}
    if bad:
        raise E2Error(f"probabilidades con forma distinta de ({n}, {len(pr.HAM_CLASSES)}): {bad}")


def run_single_evaluation(split_path: Path, rows: list[dict], pad_root: Path, predict: Predict, out: Path, *,
                          n_resamples: int = N_RESAMPLES, provenance: dict | None = None,
                          fingerprint: dict | None = None) -> dict:
    """The real E2, with the lock as late as possible. Returns the document; writes only the frozen files.

    1. every manifest image is re-hashed (no lock);
    2. predictions come from images alone and are frozen atomically with their sha256 (no lock),
       or reused when an earlier freeze has the same fingerprint and sha256;
    3. only then ``target_test_read.lock`` is taken (``O_EXCL``);
    4. labels are read and the metrics computed. The lock covers 3-4: seconds, not the GPU time.
    """
    split_path, out = Path(split_path), Path(out)
    lock = ts.TargetTestLock(split_path.parent)
    if lock.path.exists():
        raise ts.TargetTestAlreadyReadError(f"target_test ya se leyo ({lock.path}); la evaluacion unica no se repite")
    verified = _verified_paths(rows, pad_root)
    ids = ts.target_test_image_ids(split_path)
    fingerprint = {**(fingerprint or {}), "target_test_sha256": ts.ids_sha256(ids)}
    probas7, probas_sha = _frozen_predictions(out, ids, {i: verified[i] for i in ids}, predict, fingerprint)
    ts.read_target_test_ids(split_path, lock)
    by_id = {row["img_id"]: row for row in rows}
    labels = np.array([by_id[i]["ham_label"] for i in ids])
    groups = np.array([by_id[i]["patient_id"] for i in ids])
    return {**(provenance or {}), "target_test_sha256": ts.ids_sha256(ids), "n_target_test_images": len(ids),
            "probas_sha256": probas_sha, **evaluate(labels, groups, probas7, n_resamples=n_resamples)}


def check_rehearsal_output(out) -> None:
    """The result goes under a ``*rehearsal*`` directory with ``rehearsal`` in its name, never under ``resultados``."""
    path = Path(out)
    if "rehearsal" not in path.name or "rehearsal" not in path.parent.name or "resultados" in path.parts:
        raise RehearsalOutputError(
            f"{out}: un ensayo solo se escribe en un directorio *rehearsal* con 'rehearsal' en el nombre, "
            "y nunca dentro de resultados/")


def run_rehearsal(split_path: Path, rows: list[dict], pad_root: Path, predict: Predict, out: Path, *,
                  n_resamples: int = N_RESAMPLES, provenance: dict | None = None) -> dict:
    """Plumbing rehearsal on ``target_adapt``: D0 and D1, same pipeline, no lock, ``target_test`` untouched."""
    out = Path(out)
    check_rehearsal_output(out)
    ids = ts.read_target_adapt_ids(split_path)
    paths = _verified_paths(rows, pad_root, only=set(ids))
    probas7 = predict(ids, {i: paths[i] for i in ids})
    _check_shapes(probas7, len(ids), REHEARSAL_ARMS)
    probas_sha = _freeze(out, ids, probas7)
    by_id = {row["img_id"]: row for row in rows}
    labels = np.array([by_id[i]["ham_label"] for i in ids])
    groups = np.array([by_id[i]["patient_id"] for i in ids])
    marks = {"rehearsal": True, "split": "target_adapt", "discard": True, "warning": REHEARSAL_WARNING}
    return {**(provenance or {}), **marks, "target_adapt_sha256": ts.ids_sha256(ids),
            "n_target_adapt_images": len(ids), "probas_sha256": probas_sha,
            **evaluate(labels, groups, probas7, n_resamples=n_resamples, arms=REHEARSAL_ARMS,
                       contrasts=REHEARSAL_CONTRASTS)}
