"""E5 (docs/33): one image -> one record, and the process pool that maps it over a dataset.

A record carries the two global ITA estimates (``e5_ita``), the perilesional-ring estimate
(``e5_roi``) and the identity of the image. Terciles are NOT in it: they are cut on the whole
dataset afterwards. An image that cannot be decoded becomes an ``unreadable`` record (counted, never
dropped); a mask that does not sit on the image grid aborts the run, because a misaligned ring would
be read as skin without any sign of it.

The pool is memory-safe by construction: workers return small dicts, the masks are an ``np.load``
memory map opened once per worker, the hair detector is built once per worker, and ``map`` keeps the
input order, so the result does not depend on ``workers`` or on the chunk size.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dermssl import e5_counts, e5_ita, e5_roi

LESION = 1  # c1_mask_export.LESION_LABEL: 255 (uncertain) is NOT lesion
DEFAULT_CHUNK = 32
DEFAULT_PROGRESS_EVERY = 2000


class MaskGridError(ValueError):
    """The mask is missing or is not on the grid of its image."""


class PoolFileError(FileNotFoundError):
    """A pool image or mask the manifest says exists could not be read."""


class HairDetectorUnavailable(RuntimeError):
    """E5 excludes hair from the ring; without the detector the estimator would be a different one."""


@dataclass(frozen=True)
class Task:
    """One image to analyse; ``mask_pos`` indexes the mask array, ``None`` when the dataset has no masks."""

    image_id: str
    path: str
    split: str
    label: str
    mask_pos: int | None = None  # row in the native mask array (HAM10000)
    pool_image: str | None = None  # 224 x 224 pool image (ISIC), the same grid as its mask
    pool_mask: str | None = None  # PNG, true (> 0) = lesion; only for an ``eligible`` pool mask
    mask_state: str | None = None  # c1_native | eligible | quarantine | absent | None (no masks at all)
    mask_sha256: str | None = None  # the manifest's hash of the mask file, recorded against the bytes read


def _nucleo_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "plan-segmentacion" / "comun" / "nucleo"


def load_hair_fn() -> e5_roi.HairFn:
    """The repo's elongated-hair detector (cv2); a typed error when it cannot be imported."""
    try:
        sys.path.insert(0, str(_nucleo_dir()))
        import cv2  # noqa: F401, PLC0415
        import hair_removal  # noqa: PLC0415
    except ImportError as exc:
        raise HairDetectorUnavailable(f"cv2 / hair_removal not importable: {exc}") from exc
    return lambda rgb: hair_removal.elongated_hair_mask(rgb).astype(bool)


def hair_parameters() -> dict:
    """The constants of the detector that runs, read from its module (for the result document)."""
    load_hair_fn()
    import hair_removal  # noqa: PLC0415

    names = ("KERNEL", "THRESHOLD", "MIN_HAIR_LENGTH", "MAX_MEAN_WIDTH")
    return {"module": "plan-segmentacion/comun/nucleo/hair_removal.py", "function": "elongated_hair_mask",
            **{n: getattr(hair_removal, n) for n in names}}


PERILESIONAL, BORDER = "perilesional_roi", "no_mask_border"
CANONICAL, POOL = "canonical", "pool_224"


def _roi_record(est: e5_roi.RoiEstimate, *, estimator: str | None, state: str | None, image: str | None,
                sha_ok: bool | None = None) -> dict:
    return {"estimator": estimator, "mask_state": state, "roi_image": image, "mask_sha_ok": sha_ok,
            "status": est.status, "bucket": None if est.bucket is None else e5_ita.bucket_label(est.bucket),
            "ita": est.ita, "L_median": est.L_median, "b_median": est.b_median, "n_px": est.n_px,
            "ring_px": est.ring_px, "valid_fraction": est.valid_fraction}


def _lesion(task: Task, masks, shape: tuple[int, int]) -> np.ndarray:
    if masks is None:
        raise MaskGridError(f"{task.image_id}: has mask position {task.mask_pos} but no mask array was given")
    mask = np.asarray(masks[task.mask_pos]) == LESION
    if mask.shape != shape:
        raise MaskGridError(f"{task.image_id}: mask {mask.shape} and image {shape} are not on the same grid")
    return mask


def _read_pool(task: Task, path: str, mode: str) -> np.ndarray:
    from PIL import Image  # noqa: PLC0415

    try:
        with Image.open(path) as image:
            return np.asarray(image.convert(mode))
    except (OSError, ValueError) as exc:
        raise PoolFileError(f"{task.image_id}: cannot read pool file {path}: {exc}") from exc


def _pool_image(task: Task) -> np.ndarray:
    return _read_pool(task, task.pool_image, "RGB").astype(np.float32) / 255.0


def _eligible_roi(task: Task, hair_fn) -> dict:
    img = _pool_image(task)
    mask = _read_pool(task, task.pool_mask, "L") > 0
    if mask.shape != img.shape[:2]:
        raise MaskGridError(f"{task.image_id}: mask {mask.shape} and pool image {img.shape[:2]} are not on the same grid")
    sha_ok = None if task.mask_sha256 is None else hashlib.sha256(Path(task.pool_mask).read_bytes()).hexdigest() == task.mask_sha256
    return _roi_record(e5_roi.ring_estimate(img, mask, hair_fn=hair_fn), estimator=PERILESIONAL,
                       state=task.mask_state, image=POOL, sha_ok=sha_ok)


def _roi(task: Task, canonical: np.ndarray, masks, hair_fn) -> dict:
    """The ring where an eligible mask exists, the border fallback where it does not, no_mask where nothing was declared."""
    if task.mask_pos is not None:
        lesion = _lesion(task, masks, canonical.shape[:2])
        return _roi_record(e5_roi.ring_estimate(canonical, lesion, hair_fn=hair_fn), estimator=PERILESIONAL,
                           state="c1_native", image=CANONICAL)
    if task.pool_mask is not None:
        return _eligible_roi(task, hair_fn)
    if task.mask_state is None:
        return _roi_record(e5_roi.ring_estimate(canonical, None), estimator=None, state=None, image=None)
    pooled = task.pool_image is not None
    img = _pool_image(task) if pooled else canonical
    return _roi_record(e5_roi.border_estimate(img, hair_fn=hair_fn), estimator=BORDER, state=task.mask_state,
                       image=POOL if pooled else CANONICAL)


def analyze(task: Task, *, masks, hair_fn: e5_roi.HairFn | None) -> dict:
    """The record of one image (see the module docstring)."""
    try:
        img = e5_ita.decode_canonical(Path(task.path))
    except e5_ita.ImageDecodeError as exc:
        return e5_counts.unreadable_record(task.image_id, task.split, task.label, str(exc))
    return {"image_id": task.image_id, "split": task.split, "label": task.label,
            **e5_ita.global_estimates(img), "roi": _roi(task, img, masks, hair_fn)}


_WORKER: dict = {}


def _init_worker(masks_path: str | None, use_hair: bool) -> None:
    _WORKER["masks"] = None if masks_path is None else np.load(masks_path, mmap_mode="r")
    _WORKER["hair_fn"] = load_hair_fn() if use_hair else None


def _work(task: Task) -> dict:
    return analyze(task, masks=_WORKER["masks"], hair_fn=_WORKER["hair_fn"])


def _stream(records, total: int, progress: Callable[[int, int], None] | None, every: int) -> list[dict]:
    out = []
    for record in records:
        out.append(record)
        if progress is not None and (len(out) % every == 0 or len(out) == total):
            progress(len(out), total)
    return out


def run_tasks(tasks: list[Task], *, workers: int, chunk_size: int = DEFAULT_CHUNK, masks_path: Path | None = None,
              use_hair: bool = False, progress: Callable[[int, int], None] | None = None,
              progress_every: int = DEFAULT_PROGRESS_EVERY) -> list[dict]:
    """Records in task order. ``workers`` = 1 runs in this process (same code path, no pool)."""
    path = None if masks_path is None else str(masks_path)
    if workers <= 1:
        _init_worker(path, use_hair)
        return _stream((_work(task) for task in tasks), len(tasks), progress, progress_every)
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(path, use_hair)) as pool:
        return _stream(pool.map(_work, tasks, chunksize=chunk_size), len(tasks), progress, progress_every)
