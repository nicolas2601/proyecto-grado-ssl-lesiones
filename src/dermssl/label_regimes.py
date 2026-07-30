"""Few-label regime sampling for frozen-backbone probes.

Why this module exists
----------------------
The whole point of self-supervised learning is that it should need *fewer*
labels. To measure that you evaluate the same frozen backbone while giving the
classifier 1 %, 5 %, 10 %, ... of the training labels. v4 started at 10 %, which
skips the regimes where SSL usually wins by the widest margin.

Two properties matter and neither is free:

**Every class must survive.** ``df`` (dermatofibroma) has roughly 80 training
images. One percent of the whole training set naively sampled can leave ``df``
with zero examples, and a classifier that never saw a class cannot predict it -
macro-F1 collapses for a reason that has nothing to do with the backbone.
``min_per_class`` prevents that.

**Regimes must nest.** The 1 % subset should be contained in the 5 % subset,
which is contained in 10 %, and so on. Nesting means the only difference between
two regimes is *added* labels, so the resulting curve measures label efficiency
instead of sampling noise. Achieved by shuffling each class once per seed and
taking prefixes.
"""

from __future__ import annotations

import numpy as np

DEFAULT_FRACTIONS = (0.01, 0.05, 0.10, 0.20, 0.50, 1.00)


class RegimeError(ValueError):
    """Raised when a requested regime cannot be satisfied."""


def _class_order(y: np.ndarray, seed: int) -> dict[int, np.ndarray]:
    """Per-class index list, shuffled deterministically from ``seed``.

    The same seed always yields the same order, which is what makes prefixes
    nest across fractions.
    """
    rng = np.random.RandomState(seed)
    order: dict[int, np.ndarray] = {}
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        order[int(cls)] = idx[rng.permutation(len(idx))]
    return order


def regime_size(n_class: int, fraction: float, min_per_class: int) -> int:
    """How many samples one class contributes to a regime.

    Never more than the class actually has, never fewer than
    ``min_per_class`` (unless the class is smaller than that).
    """
    if not 0.0 < fraction <= 1.0:
        raise RegimeError(f"fraction must be in (0, 1], got {fraction}")
    wanted = int(round(n_class * fraction))
    wanted = max(wanted, min(min_per_class, n_class))
    return min(wanted, n_class)


def stratified_regime_indices(
    y: np.ndarray,
    fraction: float,
    seed: int = 42,
    min_per_class: int = 2,
) -> np.ndarray:
    """Indices of a class-balanced-by-proportion subset of ``y``.

    Returns sorted, duplicate-free indices into ``y``. Guarantees that every
    class present in ``y`` is also present in the result.
    """
    y = np.asarray(y)
    if y.ndim != 1 or y.size == 0:
        raise RegimeError("y must be a non-empty 1-D array of class indices")

    order = _class_order(y, seed)
    picked: list[np.ndarray] = []
    for cls, idx in order.items():
        take = regime_size(len(idx), fraction, min_per_class)
        picked.append(idx[:take])
    return np.sort(np.concatenate(picked))


def build_regimes(
    y: np.ndarray,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    seed: int = 42,
    min_per_class: int = 2,
) -> dict[float, np.ndarray]:
    """All regimes for one seed, nested by construction.

    ``build_regimes(y)[0.01]`` is a subset of ``build_regimes(y)[0.05]`` because
    both are prefixes of the same per-class shuffle.
    """
    return {
        f: stratified_regime_indices(y, f, seed=seed, min_per_class=min_per_class)
        for f in sorted(fractions)
    }


def describe_regimes(y: np.ndarray, regimes: dict[float, np.ndarray]) -> list[dict]:
    """Human-readable audit of what each regime actually contains.

    Printed in the notebook so a reader can confirm no class was starved.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    rows = []
    for frac in sorted(regimes):
        idx = regimes[frac]
        counts = {int(c): int((y[idx] == c).sum()) for c in classes}
        rows.append(
            {
                "fraction": frac,
                "n_labels": int(len(idx)),
                "pct_of_train": round(100.0 * len(idx) / len(y), 2),
                "per_class": counts,
                "min_class_count": min(counts.values()),
            }
        )
    return rows
