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

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dermssl import e5_counts, e5_ita, e5_roi

LESION = 1  # c1_mask_export.LESION_LABEL: 255 (uncertain) is NOT lesion
DEFAULT_CHUNK = 32


class MaskGridError(ValueError):
    """The mask is missing or is not on the grid of its image."""


class HairDetectorUnavailable(RuntimeError):
    """E5 excludes hair from the ring; without the detector the estimator would be a different one."""


@dataclass(frozen=True)
class Task:
    """One image to analyse; ``mask_pos`` indexes the mask array, ``None`` when the dataset has no masks."""

    image_id: str
    path: str
    split: str
    label: str
    mask_pos: int | None = None


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


def _roi_record(est: e5_roi.RoiEstimate) -> dict:
    return {"status": est.status, "bucket": None if est.bucket is None else e5_ita.bucket_label(est.bucket),
            "ita": est.ita, "L_median": est.L_median, "b_median": est.b_median, "n_px": est.n_px,
            "ring_px": est.ring_px, "valid_fraction": est.valid_fraction}


def _lesion(task: Task, masks, shape: tuple[int, int]) -> np.ndarray | None:
    if task.mask_pos is None:
        return None
    if masks is None:
        raise MaskGridError(f"{task.image_id}: has mask position {task.mask_pos} but no mask array was given")
    mask = np.asarray(masks[task.mask_pos]) == LESION
    if mask.shape != shape:
        raise MaskGridError(f"{task.image_id}: mask {mask.shape} and image {shape} are not on the same grid")
    return mask


def analyze(task: Task, *, masks, hair_fn: e5_roi.HairFn | None) -> dict:
    """The record of one image (see the module docstring)."""
    try:
        img = e5_ita.decode_canonical(Path(task.path))
    except e5_ita.ImageDecodeError as exc:
        return e5_counts.unreadable_record(task.image_id, task.split, task.label, str(exc))
    lesion = _lesion(task, masks, img.shape[:2])
    return {"image_id": task.image_id, "split": task.split, "label": task.label,
            **e5_ita.global_estimates(img), "roi": _roi_record(e5_roi.ring_estimate(img, lesion, hair_fn=hair_fn))}


_WORKER: dict = {}


def _init_worker(masks_path: str | None, use_hair: bool) -> None:
    _WORKER["masks"] = None if masks_path is None else np.load(masks_path, mmap_mode="r")
    _WORKER["hair_fn"] = load_hair_fn() if use_hair else None


def _work(task: Task) -> dict:
    return analyze(task, masks=_WORKER["masks"], hair_fn=_WORKER["hair_fn"])


def run_tasks(tasks: list[Task], *, workers: int, chunk_size: int = DEFAULT_CHUNK,
              masks_path: Path | None = None, use_hair: bool = False) -> list[dict]:
    """Records in task order. ``workers`` = 1 runs in this process (same code path, no pool)."""
    path = None if masks_path is None else str(masks_path)
    if workers <= 1:
        _init_worker(path, use_hair)
        return [_work(task) for task in tasks]
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(path, use_hair)) as pool:
        return list(pool.map(_work, tasks, chunksize=chunk_size))
