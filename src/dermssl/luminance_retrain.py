"""T7 statistics (docs/27): the dark-minus-rest gap of several segmenters on the same resampled images.

Every segmenter is scored on the same images, so one stratified resample
(with replacement inside the dark group and inside the rest) is shared by all
of them. With two segmenters the draws are exactly those of
``luminance_threshold.gap_reduction``, so R(T7) against A2 is T6's paired
statistic bit for bit. The seed contrast S = R(T7) - R(A2-s43) is taken on
the same resamples. The registered readings (docs/27, T7) live here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from dermssl import luminance_threshold as lt

SEED_SEPARATED = "se separa del ruido de semilla"
SEED_NOT_SEPARATED = "no se separa del ruido de semilla"
SHRINKS = "T7 achica la brecha"
FULFILLED_WITHIN_NOISE = "cumplida, pero no se separa del ruido de semilla"


def _stat(point: float, samples: np.ndarray) -> dict:
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"delta": float(point), "ci95_low": float(low), "ci95_high": float(high)}


def shared_gaps(models: Mapping[str, tuple[Sequence[float], Sequence[float]]], n_resamples: int,
                seed: int) -> dict:
    """Point gap and resampled gaps per segmenter; every segmenter uses the same indices."""
    arrays = {name: (np.asarray(d, float), np.asarray(r, float)) for name, (d, r) in models.items()}
    sizes = {(d.size, r.size) for d, r in arrays.values()}
    if len(sizes) != 1:
        raise ValueError(f"los segmentadores no se evaluaron en las mismas imágenes: {sorted(sizes)}")
    (n_dark, n_rest), = sizes
    rng = np.random.RandomState(seed)
    dark_idx = rng.randint(0, n_dark, size=(n_resamples, n_dark))
    rest_idx = rng.randint(0, n_rest, size=(n_resamples, n_rest))
    return {"n_dark": n_dark, "n_rest": n_rest,
            "models": {name: {"point": float(d.mean() - r.mean()),
                              "samples": d[dark_idx].mean(axis=1) - r[rest_idx].mean(axis=1)}
                       for name, (d, r) in arrays.items()}}


def gap(gaps: dict, name: str) -> dict:
    model = gaps["models"][name]
    return _stat(model["point"], model["samples"])


def reduction(gaps: dict, reference: str, other: str) -> dict:
    """R(other) = |G(reference)| - |G(other)| with its paired interval."""
    ref, oth = gaps["models"][reference], gaps["models"][other]
    return _stat(abs(ref["point"]) - abs(oth["point"]), np.abs(ref["samples"]) - np.abs(oth["samples"]))


def reduction_difference(gaps: dict, first: str, second: str) -> dict:
    """R(first) - R(second) = |G(second)| - |G(first)| on the same resamples."""
    a, b = gaps["models"][first], gaps["models"][second]
    return _stat(abs(b["point"]) - abs(a["point"]), np.abs(b["samples"]) - np.abs(a["samples"]))


def seed_reading(ci_low: float) -> str:
    return SEED_SEPARATED if ci_low > 0 else SEED_NOT_SEPARATED


def conclusion(prediction: str, seed: str) -> str:
    """"T7 achica la brecha" only when Pau's prediction holds and it stands out of the seed noise."""
    if prediction != lt.FULFILLED:
        return prediction
    return SHRINKS if seed == SEED_SEPARATED else FULFILLED_WITHIN_NOISE
