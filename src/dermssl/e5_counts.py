"""E5 (docs/33): descriptive counts of the per-image estimates of one dataset.

Class x bucket, bucket x split, class x L* tercile, tercile x split, status counts and, where the
dataset has lesion masks, the ROI-vs-Otsu bucket cross table. Every table lists ALL its columns
(zeros included) in a fixed order, abstentions in their own column. An image whose class or split
is not declared is a typed error: a new row is never created silently. Counts only, no inference.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from dermssl import e5_ita as ita

BUCKET_COLUMNS = (*ita.BUCKET_LABELS, ita.ABSTAIN)
TERCILE_COLUMNS = (*ita.TERCILES, ita.UNAVAILABLE)
UNREADABLE = "unreadable"
NO_ESTIMATE = {"bucket": None, "ita": None, "L_median": None, "b_median": None}


class UnknownKeyError(KeyError):
    """A record names a class or a split the dataset did not declare."""


def unreadable_record(image_id: str, split: str, label: str, error: str) -> dict:
    """A record for a file that could not be decoded: it abstains everywhere and says why."""
    est = {**NO_ESTIMATE, "status": UNREADABLE, "n_skin_px": 0}
    roi = {**NO_ESTIMATE, "status": UNREADABLE, "n_px": 0, "ring_px": 0, "valid_fraction": None}
    return {"image_id": image_id, "split": split, "label": label, "otsu": dict(est),
            "whole_image": dict(est), "roi": roi, "tercile": ita.UNAVAILABLE, "error": error}


def _bucket(estimate: dict) -> str:
    return ita.ABSTAIN if estimate["bucket"] is None else estimate["bucket"]


def _crosstab(pairs: Iterable[tuple[str, str]], rows: Sequence[str], columns: Sequence[str]) -> dict:
    counts = Counter(pairs)
    for row, col in counts:
        if row not in rows:
            raise UnknownKeyError(f"{row!r} is not one of {list(rows)}")
        if col not in columns:
            raise UnknownKeyError(f"{col!r} is not one of {list(columns)}")
    return {row: {col: counts[(row, col)] for col in columns} for row in rows}


def _check(records: Sequence[dict], classes: Sequence[str], splits: Sequence[str]) -> None:
    for rec in records:
        for key, allowed in (("label", classes), ("split", splits)):
            if rec[key] not in allowed:
                raise UnknownKeyError(f"{key} {rec[key]!r} of {rec['image_id']} is not one of {list(allowed)}")


def _estimator_tables(records, name, classes, splits) -> dict:
    est = [r[name] for r in records]
    buckets = [_bucket(e) for e in est]
    return {
        "status_counts": dict(sorted(Counter(e["status"] for e in est).items())),
        "bucket_counts": {b: buckets.count(b) for b in BUCKET_COLUMNS},
        "class_x_bucket": _crosstab(zip((r["label"] for r in records), buckets, strict=True), classes, BUCKET_COLUMNS),
        "bucket_x_split": _crosstab(zip((r["split"] for r in records), buckets, strict=True), splits, BUCKET_COLUMNS),
    }


def _agreement(records) -> dict:
    """Otsu bucket (rows) by ROI bucket (columns) on the same images; agreement is not accuracy."""
    pairs = [(_bucket(r["otsu"]), _bucket(r["roi"])) for r in records]
    both = [(a, b) for a, b in pairs if ita.ABSTAIN not in (a, b)]
    return {"agreement_with_otsu": _crosstab(pairs, BUCKET_COLUMNS, BUCKET_COLUMNS),
            "both_estimated": len(both), "same_bucket": sum(a == b for a, b in both)}


def _roi_report(records, classes, splits, applicable: bool, reason: str) -> dict:
    status_counts = dict(sorted(Counter(r["roi"]["status"] for r in records).items()))
    if not applicable:
        return {"applicable": False, "reason": reason, "status_counts": status_counts}
    return {"applicable": True, "reason": "", **_estimator_tables(records, "roi", classes, splits),
            **_agreement(records)}


def _tercile_report(records, classes, splits, cuts: dict) -> dict:
    labels = [r["tercile"] for r in records]
    return {
        "cuts": cuts,
        "counts": {t: labels.count(t) for t in TERCILE_COLUMNS},
        "class_x_tercile": _crosstab(zip((r["label"] for r in records), labels, strict=True), classes, TERCILE_COLUMNS),
        "tercile_x_split": _crosstab(zip((r["split"] for r in records), labels, strict=True), splits, TERCILE_COLUMNS),
    }


def build_report(records: Sequence[dict], *, classes: Sequence[str], splits: Sequence[str], tercile_cuts: dict,
                 roi_applicable: bool, roi_reason: str) -> dict:
    """All the descriptive tables of one dataset from its per-image records."""
    _check(records, classes, splits)
    estimators = {name: {"applicable": True, **_estimator_tables(records, name, classes, splits)}
                  for name in ita.ESTIMATORS}
    estimators["roi"] = _roi_report(records, classes, splits, roi_applicable, roi_reason)
    return {"n_images": len(records),
            "n_unreadable": sum(r["otsu"]["status"] == UNREADABLE for r in records),
            "estimators": estimators, "terciles": _tercile_report(records, classes, splits, tercile_cuts)}
