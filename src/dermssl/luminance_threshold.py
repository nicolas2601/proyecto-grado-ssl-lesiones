"""One mask threshold per luminance tercile on A2's probabilities (T6, docs/27).

Cheap analogue of Fitzpatrick Thresholding (arXiv 2510.06655): the segmenter
is not retrained, only its probability is binarised with a threshold chosen
per bin on a fit part. Here the bins are the ``tone_tercile`` column of the
published manifest, terciles of the skin median L*: luminance, not skin tone.

Everything is computed from a per-image *Dice curve*, the Dice at each grid
threshold, so the choice and the contrasts can be rebuilt from a CSV. The
binarisation and the Dice are the evaluator's (strictly greater than the
threshold; two empty masks agree), and a test checks they match it bit for
bit. The registered reading (docs/27, T6) lives in the ``*_verdict`` helpers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

GRID = tuple(round(0.30 + 0.05 * k, 2) for k in range(9))
FIXED = 0.5
DARK = "dark"
REST = ("medium", "light")
TERCILES = (DARK, *REST)

#: Minimum images per group: under it no verdict (as in T2).
MIN_N = 100
#: Part 1: the lower end of the interval of the gap reduction must reach this.
GAP_MARGIN = 0.02
#: Part 2: the lower end of the interval of the global change must not fall under this.
GLOBAL_TOLERANCE = -0.005

GAP_FULFILLED = "cumplida"
GAP_PARTIAL = "se achica, pero menos de 0,02"
GAP_NOT_FULFILLED = "no cumplida"
GLOBAL_HOLDS = "se sostiene"
GLOBAL_FAILS = "no se sostiene"
FULFILLED = "cumplida"
NOT_FULFILLED = "no cumplida"
NO_POWER = "sin potencia"


def dice_at(probability: np.ndarray, truth: np.ndarray, threshold: float) -> float:
    """The evaluator's Dice of ``probability > threshold`` against ``truth``."""
    prediction = np.asarray(probability) > threshold
    truth = np.asarray(truth, dtype=bool)
    if prediction.shape != truth.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} vs {truth.shape}")
    true_positive = float(np.logical_and(prediction, truth).sum())
    false_positive = float(np.logical_and(prediction, ~truth).sum())
    false_negative = float(np.logical_and(~prediction, truth).sum())
    if true_positive + false_positive + false_negative == 0:
        return 1.0
    return 2 * true_positive / (2 * true_positive + false_positive + false_negative)


def dice_curve(probability: np.ndarray, truth: np.ndarray, grid: Sequence[float] = GRID) -> np.ndarray:
    return np.array([dice_at(probability, truth, t) for t in grid], dtype=float)


def _pick(mean_by_threshold: np.ndarray, grid: Sequence[float]) -> int:
    """Best mean Dice; ties go to the threshold closest to 0.5, then to the lower one."""
    best = mean_by_threshold.max()
    tied = [k for k, value in enumerate(mean_by_threshold) if value == best]
    return min(tied, key=lambda k: (abs(grid[k] - FIXED), grid[k]))


def select_thresholds(curves: Mapping[str, Sequence[float]], terciles: Mapping[str, str],
                      fit_ids: Iterable[str], grid: Sequence[float] = GRID, min_n: int = MIN_N,
                      terciles_used: Sequence[str] = TERCILES) -> dict:
    """Per tercile, the grid threshold with the highest mean Dice over its fit images."""
    fit_ids = list(fit_ids)
    fixed_index = list(grid).index(FIXED)
    chosen = {}
    for tercile in terciles_used:
        rows = np.array([curves[i] for i in fit_ids if terciles.get(i) == tercile], dtype=float)
        if len(rows) < min_n:
            raise ValueError(f"el tercil {tercile} tiene {len(rows)} imagenes de ajuste (< {min_n})")
        means = rows.mean(axis=0)
        best = _pick(means, grid)
        chosen[tercile] = {"threshold": grid[best], "fit_n": len(rows), "fit_dice": float(means[best]),
                           "fit_dice_at_fixed": float(means[fixed_index])}
    return chosen


def adapted_dice(curves: Mapping[str, Sequence[float]], terciles: Mapping[str, str],
                 thresholds: Mapping[str, float], ids: Iterable[str],
                 grid: Sequence[float] = GRID) -> dict[str, float]:
    """Dice of each image with its own tercile's threshold."""
    index = {t: list(grid).index(thresholds[t]) for t in thresholds}
    return {i: float(curves[i][index[terciles[i]]]) for i in ids}


def _ci(samples: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def _stat(point: float, samples: np.ndarray) -> dict:
    low, high = _ci(samples)
    return {"delta": float(point), "ci95_low": low, "ci95_high": high}


def gap_reduction(dark_fixed, dark_adapted, rest_fixed, rest_adapted, n_resamples: int, seed: int) -> dict:
    """R = |gap at 0.5| - |adapted gap|, resampled inside each group with the same indices for both rules."""
    dark_fixed, dark_adapted = np.asarray(dark_fixed, float), np.asarray(dark_adapted, float)
    rest_fixed, rest_adapted = np.asarray(rest_fixed, float), np.asarray(rest_adapted, float)
    rng = np.random.RandomState(seed)
    dark_idx = rng.randint(0, dark_fixed.size, size=(n_resamples, dark_fixed.size))
    rest_idx = rng.randint(0, rest_fixed.size, size=(n_resamples, rest_fixed.size))
    gap_fixed = dark_fixed[dark_idx].mean(axis=1) - rest_fixed[rest_idx].mean(axis=1)
    gap_adapted = dark_adapted[dark_idx].mean(axis=1) - rest_adapted[rest_idx].mean(axis=1)
    point_fixed = dark_fixed.mean() - rest_fixed.mean()
    point_adapted = dark_adapted.mean() - rest_adapted.mean()
    return {"n_dark": int(dark_fixed.size), "n_rest": int(rest_fixed.size),
            "gap_fixed": _stat(point_fixed, gap_fixed), "gap_adapted": _stat(point_adapted, gap_adapted),
            "reduction": _stat(abs(point_fixed) - abs(point_adapted),
                               np.abs(gap_fixed) - np.abs(gap_adapted))}


def global_change(fixed, adapted, n_resamples: int, seed: int) -> dict:
    """Mean per-image (adapted - fixed) Dice with a paired percentile interval."""
    diff = np.asarray(adapted, float) - np.asarray(fixed, float)
    rng = np.random.RandomState(seed)
    means = diff[rng.randint(0, diff.size, size=(n_resamples, diff.size))].mean(axis=1)
    return {**_stat(diff.mean(), means), "n": int(diff.size),
            "mean_fixed": float(np.mean(fixed)), "mean_adapted": float(np.mean(adapted))}


def gap_verdict(ci_low: float) -> str:
    if ci_low >= GAP_MARGIN:
        return GAP_FULFILLED
    return GAP_PARTIAL if ci_low > 0 else GAP_NOT_FULFILLED


def global_verdict(ci_low: float) -> str:
    return GLOBAL_HOLDS if ci_low >= GLOBAL_TOLERANCE else GLOBAL_FAILS


def prediction_verdict(gap: str, global_: str, n_dark: int, n_rest: int) -> str:
    if n_dark < MIN_N or n_rest < MIN_N:
        return NO_POWER
    return FULFILLED if gap == GAP_FULFILLED and global_ == GLOBAL_HOLDS else NOT_FULFILLED


def _per_tercile(fixed: Mapping[str, float], adapted: Mapping[str, float], ids_by_tercile) -> dict:
    return {t: {"n": len(ids), "dice_fixed": float(np.mean([fixed[i] for i in ids])) if ids else None,
                "dice_adapted": float(np.mean([adapted[i] for i in ids])) if ids else None}
            for t, ids in ids_by_tercile.items()}


def evaluate(curves, terciles, thresholds, eval_ids, n_resamples: int, seed: int,
             with_verdict: bool = True, grid: Sequence[float] = GRID) -> dict:
    """Both rules on the evaluation ids with a tercile: per-tercile means, the gap and the global change."""
    eval_ids = list(eval_ids)
    binned = sorted(i for i in eval_ids if terciles.get(i) in TERCILES)
    fixed_index = list(grid).index(FIXED)
    fixed = {i: float(curves[i][fixed_index]) for i in binned}
    adapted = adapted_dice(curves, terciles, {t: thresholds[t]["threshold"] for t in TERCILES}, binned, grid)
    by_tercile = {t: [i for i in binned if terciles[i] == t] for t in TERCILES}
    dark, rest = by_tercile[DARK], [i for t in REST for i in by_tercile[t]]
    gap = gap_reduction([fixed[i] for i in dark], [adapted[i] for i in dark],
                        [fixed[i] for i in rest], [adapted[i] for i in rest], n_resamples, seed)
    change = global_change([fixed[i] for i in binned], [adapted[i] for i in binned], n_resamples, seed)
    out = {"n_without_tercile": len(eval_ids) - len(binned), "terciles": _per_tercile(fixed, adapted, by_tercile),
           "gap": gap, "global": change}
    if with_verdict:
        parts = {"gap": gap_verdict(gap["reduction"]["ci95_low"]),
                 "global": global_verdict(change["ci95_low"])}
        out["verdict"] = {**parts, "prediction": prediction_verdict(parts["gap"], parts["global"],
                                                                    gap["n_dark"], gap["n_rest"])}
    return out


def analyze(curves, terciles, fit_ids, eval_ids, n_resamples: int, seed: int, min_fit_n: int = MIN_N,
            with_verdict: bool = True, grid: Sequence[float] = GRID) -> dict:
    """Choose the thresholds on ``fit_ids`` and score both rules on ``eval_ids``, which must not overlap."""
    shared = set(fit_ids) & set(eval_ids)
    if shared:
        raise ValueError(f"{len(shared)} imagenes estan en ajuste y evaluacion (p. ej. {sorted(shared)[:3]})")
    thresholds = select_thresholds(curves, terciles, fit_ids, grid, min_fit_n)
    return {"thresholds": thresholds,
            "evaluation": evaluate(curves, terciles, thresholds, eval_ids, n_resamples, seed, with_verdict, grid)}
