"""Dice and HD95 by luminance tercile, and the dark-minus-rest contrast (T2, docs/27).

The bins are the ``tone_tercile`` column of the published manifest: terciles
of the skin median L* over the pool. They are luminance, not skin tone.

The dark tercile and the rest hold different images, so the contrast is an
unpaired bootstrap stratified by group: each resample draws with replacement
inside the dark group and inside the rest separately, keeping both sizes.
The registered reading (docs/27, T2) lives in :func:`verdict`.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

DARK = "dark"
REST = ("medium", "light")
TERCILES = (DARK, *REST)
UNAVAILABLE = "unavailable"
KNOWN_LABELS = (*TERCILES, UNAVAILABLE)
METRICS = ("dice", "hd95")

#: "Claramente menor": the whole 95 % interval below this (twice the S4 margin).
CLEAR_MARGIN = -0.02
#: Under this many images in the dark group or the rest, no verdict.
MIN_N = 100
FULFILLED = "cumplida"
DISTINGUISHABLE = "distinguible pero no clara"
NOT_FULFILLED = "no cumplida"
NO_POWER = "sin potencia"


def verdict(ci_high: float, n_dark: int, n_rest: int) -> str:
    """The registered reading of the dark-minus-rest Dice interval."""
    if n_dark < MIN_N or n_rest < MIN_N:
        return NO_POWER
    if ci_high < CLEAR_MARGIN:
        return FULFILLED
    if ci_high < 0:
        return DISTINGUISHABLE
    return NOT_FULFILLED


def _percentile_ci(means: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def mean_ci(values: np.ndarray, n_resamples: int, seed: int) -> dict:
    """Mean with a percentile bootstrap 95 % interval; non-finite values are dropped and counted."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    out = {"n": int(finite.size), "n_dropped_non_finite": int(values.size - finite.size)}
    if not finite.size:
        return {**out, "mean": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.RandomState(seed)
    means = finite[rng.randint(0, finite.size, size=(n_resamples, finite.size))].mean(axis=1)
    low, high = _percentile_ci(means)
    return {**out, "mean": float(finite.mean()), "ci95_low": low, "ci95_high": high}


def unpaired_contrast(a: np.ndarray, b: np.ndarray, n_resamples: int, seed: int) -> dict:
    """mean(a) - mean(b), resampling inside each group separately (non-finite values dropped)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not a.size or not b.size:
        raise ValueError(f"un grupo del contraste esta vacio (n_a={a.size}, n_b={b.size})")
    rng = np.random.RandomState(seed)
    mean_a = a[rng.randint(0, a.size, size=(n_resamples, a.size))].mean(axis=1)
    mean_b = b[rng.randint(0, b.size, size=(n_resamples, b.size))].mean(axis=1)
    low, high = _percentile_ci(mean_a - mean_b)
    return {"delta": float(a.mean() - b.mean()), "ci95_low": low, "ci95_high": high,
            "n_a": int(a.size), "n_b": int(b.size)}


def coverage(rows: Mapping[str, Mapping[str, float]], terciles: Mapping[str, str]) -> dict:
    """How the evaluated ids fall into the bins; ids absent from the manifest are counted apart."""
    counts = dict.fromkeys((*KNOWN_LABELS, "not_in_manifest"), 0)
    for image_id in rows:
        label = terciles.get(image_id, "not_in_manifest")
        if label not in counts:
            raise ValueError(f"tercil desconocido {label!r} para {image_id}")
        counts[label] += 1
    return {**counts, "total": len(rows)}


def _values(rows: Mapping[str, Mapping[str, float]], ids: list[str], metric: str) -> np.ndarray:
    return np.array([rows[i][metric] for i in ids], dtype=float)


def _dark_minus(rows, dark_ids, other_ids, n_resamples, seed, with_verdict) -> dict:
    out = {}
    for metric in METRICS:
        contrast = unpaired_contrast(_values(rows, dark_ids, metric), _values(rows, other_ids, metric),
                                     n_resamples, seed)
        if with_verdict and metric == "dice":
            contrast["verdict"] = verdict(contrast["ci95_high"], len(dark_ids), len(other_ids))
        out[metric] = contrast
    return out


def analyze(rows: Mapping[str, Mapping[str, float]], terciles: Mapping[str, str],
            n_resamples: int, seed: int) -> dict:
    """Per-bin means with intervals, the registered contrast and the descriptive one against the global."""
    coverage(rows, terciles)  # refuses unknown labels before any number is computed
    ids = {t: sorted(i for i in rows if terciles.get(i) == t) for t in TERCILES}
    rest_ids = sorted(i for t in REST for i in ids[t])
    all_ids = sorted(ids[DARK] + rest_ids)
    bins = {t: {"n": len(ids[t]), **{m: mean_ci(_values(rows, ids[t], m), n_resamples, seed)
                                      for m in METRICS}} for t in TERCILES}
    return {
        "bins": bins, "rest_n": len(rest_ids), "global_n": len(all_ids),
        "global": {m: mean_ci(_values(rows, all_ids, m), n_resamples, seed) for m in METRICS},
        "dark_minus_rest": _dark_minus(rows, ids[DARK], rest_ids, n_resamples, seed, True),
        "dark_minus_global": _dark_minus(rows, ids[DARK], all_ids, n_resamples, seed, False),
    }
