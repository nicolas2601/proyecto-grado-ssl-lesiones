"""Report helpers for the evaluation stage: percentiles, strata and overlays.

Split out of ``evaluate_segmentation.py`` so both files stay small. Nothing
here touches the model: it only turns per-image rows into summaries, tables
and PNG contact sheets. Masks arrive already decoded as boolean arrays.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

PERCENTILES = (5, 10, 25, 75, 90)
WORST_COUNT = 10
# Bootstrap of the mean: fixed seed so a summary is reproducible from its CSV.
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
UNAVAILABLE = "unavailable"
VALID_BUCKETS = tuple(str(value) for value in range(1, 7))
STRATA_METRICS = ("dice", "iou")
IN_POOL = "in_pool"
OUT_OF_POOL = "out_of_pool"


# ------------------------------------------------------------------ summary
def _bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    """95 % percentile bootstrap interval of the mean, seeded."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(values, size=(BOOTSTRAP_RESAMPLES, len(values)), replace=True)
    low, high = np.percentile(samples.mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def _describe(values: np.ndarray) -> dict:
    """Mean, std, median, percentiles, bootstrap CI and count of a NaN-free vector."""
    if not len(values):
        empty = {name: math.nan for name in ("mean", "std", "median", "ci95_low", "ci95_high")}
        empty.update({f"p{p}": math.nan for p in PERCENTILES})
        return {**empty, "n": 0}
    low, high = _bootstrap_ci(values)
    stats = {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "ci95_low": low,
        "ci95_high": high,
        "n": int(len(values)),
    }
    stats.update({f"p{p}": float(np.percentile(values, p)) for p in PERCENTILES})
    return stats


def worst_image_ids(rows: Sequence[Mapping], count: int = WORST_COUNT) -> list[str]:
    """Image ids of the ``count`` lowest Dice values, worst first."""
    usable = [r for r in rows if "image_id" in r and not math.isnan(float(r.get("dice", math.nan)))]
    usable.sort(key=lambda row: float(row["dice"]))
    return [str(row["image_id"]) for row in usable[:count]]


def summarize(rows: Sequence[Mapping], metric_names: Sequence[str]) -> dict:
    """Per-metric statistics ignoring NaN, plus the worst Dice image ids."""
    summary = {}
    for name in metric_names:
        values = np.array([row.get(name, math.nan) for row in rows], dtype=np.float64)
        summary[name] = _describe(values[~np.isnan(values)])
    worst = worst_image_ids(rows)
    if worst:
        summary["worst_ids"] = worst
    return summary


# ------------------------------------------------------------------- strata
def _manifest_rows(path: Path) -> list[Mapping]:
    """Rows of a CSV manifest, or the ``records`` list of the JSON one."""
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return list(payload["records"] if isinstance(payload, Mapping) else payload)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bucket_of(row: Mapping, image_id: str):
    """The bucket column is ``fst_bucket`` in the CSV and ``FST_bucket`` in the JSON."""
    for name in row:
        if str(name).lower() == "fst_bucket":
            return row[name]
    raise ValueError(f"the manifest row for {image_id!r} has no fst_bucket column")


def validate_fst_manifest(path: Path) -> None:
    """Fail before inference when the manifest cannot feed the stratification."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"el manifiesto FST no existe: {path}")
    rows = _manifest_rows(path)
    if not rows:
        raise ValueError(f"el manifiesto FST esta sin filas: {path}")
    names = {str(name).lower() for name in rows[0]}
    if "image_id" not in names:
        raise ValueError(f"al manifiesto FST le falta la columna image_id: {path}")
    if "fst_bucket" not in names:
        raise ValueError(f"al manifiesto FST le falta la columna fst_bucket: {path}")


def read_fst_manifest(path: Path) -> dict[str, str]:
    """``image_id`` to bucket, from either manifest written by the FST stage.

    Buckets are 1..6 or ``unavailable``; the JSON manifest stores them as
    integers, so every value is normalised to a string key.
    """
    buckets: dict[str, str] = {}
    for row in _manifest_rows(Path(path)):
        image_id = str(row["image_id"]).strip()
        bucket = str(_bucket_of(row, image_id)).strip()
        if bucket not in VALID_BUCKETS and bucket != UNAVAILABLE:
            raise ValueError(f"invalid fst_bucket {bucket!r} for {image_id!r}")
        buckets[image_id] = bucket
    return buckets


def _bucket_summary(bucket_rows: Sequence[Mapping], min_n: int) -> dict:
    entry = {"n": len(bucket_rows), "status": "OK" if len(bucket_rows) >= min_n else "SKIPPED"}
    for metric in STRATA_METRICS:
        values = np.array([row.get(metric, math.nan) for row in bucket_rows], dtype=np.float64)
        values = values[~np.isnan(values)]
        entry[metric] = {
            "mean": float(values.mean()) if len(values) else math.nan,
            "median": float(np.median(values)) if len(values) else math.nan,
            "n": int(len(values)),
        }
    return entry


def _pool_summaries(grouped: Mapping[str, Sequence[Mapping]], min_n: int) -> dict:
    """Everything the manifest gave a skin tone, against everything it did not.

    Two aggregates over the same rows the buckets split: they answer how the
    model does on the population the stratification can actually compare, and
    on the remainder, which no bucket ever reports.
    """
    inside = [row for key, group in grouped.items() if key != UNAVAILABLE for row in group]
    return {
        IN_POOL: _bucket_summary(inside, min_n),
        OUT_OF_POOL: _bucket_summary(grouped.get(UNAVAILABLE, []), min_n),
    }


def stratified_summary(
    rows: Sequence[Mapping], buckets: Mapping[str, str], *, min_n: int = 30
) -> dict:
    """Per-bucket Dice and IoU plus the gap between qualifying skin-tone buckets."""
    grouped: dict[str, list[Mapping]] = {}
    for row in rows:
        key = buckets.get(str(row.get("image_id", "")), UNAVAILABLE)
        grouped.setdefault(key, []).append(row)
    summaries = {key: _bucket_summary(grouped[key], min_n) for key in sorted(grouped)}
    qualifying = {
        key: entry["dice"]["mean"]
        for key, entry in summaries.items()
        if entry["status"] == "OK" and key != UNAVAILABLE and not math.isnan(entry["dice"]["mean"])
    }
    gap = max(qualifying.values()) - min(qualifying.values()) if qualifying else None
    worst = min(qualifying, key=qualifying.get) if qualifying else None
    return {
        "min_n": int(min_n),
        "buckets": summaries,
        "gap_dice": gap,
        "worst_bucket": worst,
        **_pool_summaries(grouped, min_n),
    }


# ------------------------------------------------------------------ writers
def write_per_image_csv(path: Path, rows: Sequence[Mapping], columns: Sequence[str]) -> None:
    fieldnames = ["image_id", *columns]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    """Recursive copy where every NaN becomes None, so the JSON stays valid."""
    if isinstance(value, Mapping):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_summary_json(path: Path, metrics: dict, meta: dict, strata: dict | None = None) -> None:
    """``allow_nan=False``: a bare NaN is not JSON and breaks strict readers."""
    payload = {"meta": meta, "metrics": metrics}
    if strata is not None:
        payload["strata"] = strata
    Path(path).write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


# ----------------------------------------------------------------- overlays
def select_overlay_groups(
    rows: Sequence[Mapping], count: int, seed: int = 0
) -> dict[str, list[str]]:
    """Worst, best and random image ids by Dice; groups may overlap on tiny sets."""
    ranked = sorted(
        (r for r in rows if not math.isnan(float(r.get("dice", math.nan)))),
        key=lambda row: float(row["dice"]),
    )
    identifiers = [str(row["image_id"]) for row in ranked]
    sample_size = min(count, len(identifiers))
    sampler = random.Random(seed)
    return {
        "worst": identifiers[:count],
        "best": identifiers[-count:][::-1] if count else [],
        "random": sorted(sampler.sample(identifiers, sample_size)),
    }


def _erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=bool), 1, constant_values=False)
    eroded = padded[1:-1, 1:-1].copy()
    for neighbour in (padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]):
        eroded &= neighbour
    return eroded


def contour(mask: np.ndarray) -> np.ndarray:
    """Four-connected inner boundary of a boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    return mask & ~_erode(mask)


def overlay_image(image: np.ndarray, reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Copy of ``image`` with the reference contour in green and the prediction in red."""
    painted = np.array(image, dtype=np.uint8, copy=True)
    painted[contour(reference)] = (0, 255, 0)
    painted[contour(prediction)] = (255, 0, 0)
    return painted


def _cell(array: np.ndarray, cell_size: int):
    """One grid cell: boolean masks become white on black, RGB passes through."""
    from PIL import Image

    array = np.asarray(array)
    if array.ndim == 2:
        array = np.repeat(array.astype(bool)[..., None], 3, axis=2).astype(np.uint8) * 255
    return Image.fromarray(array.astype(np.uint8)).resize((cell_size, cell_size), Image.NEAREST)


def write_overlay_grid(
    path: Path, records: Sequence[Mapping], *, cell_size: int = 224, title_height: int = 16
) -> Path:
    """Contact sheet with columns image, reference, prediction and overlay."""
    from PIL import Image, ImageDraw

    if not records:
        raise ValueError("an overlay grid needs at least one record")
    row_height = cell_size + title_height
    canvas = Image.new("RGB", (4 * cell_size, len(records) * row_height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, record in enumerate(records):
        top = index * row_height
        title = f"{record['image_id']}  dice={float(record['dice']):.3f}"
        draw.text((4, top + 3), title, fill=(255, 255, 255))
        painted = overlay_image(record["image"], record["reference"], record["prediction"])
        cells = (record["image"], record["reference"], record["prediction"], painted)
        for column, cell in enumerate(cells):
            canvas.paste(_cell(cell, cell_size), (column * cell_size, top + title_height))
    canvas.save(Path(path))
    return Path(path)


# --------------------------------------------------- helpers for the CLI
def strata_from_args(args, rows: Sequence[Mapping]) -> dict | None:
    """Stratification requested on the command line, or None."""
    if not args.fst_manifest:
        return None
    buckets = read_fst_manifest(Path(args.fst_manifest))
    return stratified_summary(rows, buckets, min_n=args.fst_min_n)


def reject_duplicate_ids(identifiers: Sequence[str]) -> None:
    """Two files with the same stem would collapse one row and one overlay."""
    repeated = sorted(name for name, count in Counter(identifiers).items() if count > 1)
    if repeated:
        raise ValueError(f"{len(repeated)} duplicate image ids (first: {repeated[:3]})")
