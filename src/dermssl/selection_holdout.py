"""Internal holdout for choosing pilot variants (docs/28 section 4).

Selection happens on 20 % of the C1 TRAIN ids, grouped by lesion and
stratified by class; the validation set is scored once, at the end, and is
never an input here. The split is fold 0 of StratifiedGroupKFold(5, shuffle,
random_state=42), the same recipe that produced the C1 split itself, so it is
rebuilt identically from the ids alone.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

HOLDOUT_N_SPLITS = 5  # One fold of five: a 20 % holdout.
HOLDOUT_SEED = 42
TIE_MACRO_F1 = 0.005  # docs/28 §4.1: closer than this is a tie, broken by ECE.


def grouped_holdout(
    ids: Sequence[str],
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    forbidden: Sequence[str] = (),
    n_splits: int = HOLDOUT_N_SPLITS,
    seed: int = HOLDOUT_SEED,
) -> tuple[list[str], list[str]]:
    """``(fit_ids, holdout_ids)``: no lesion on both sides, no forbidden id."""
    from sklearn.model_selection import StratifiedGroupKFold  # noqa: PLC0415

    ids = list(ids)
    if not (len(ids) == len(labels) == len(groups)):
        raise ValueError(
            f"largos distintos: ids {len(ids)}, labels {len(labels)}, groups {len(groups)}"
        )
    leaked = set(ids) & set(forbidden)
    if leaked:
        raise ValueError(
            f"{len(leaked)} ids de validacion en la entrada: la seleccion no puede verlos"
        )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fit_idx, holdout_idx = next(splitter.split(ids, np.asarray(labels), np.asarray(groups)))
    fit = [ids[i] for i in fit_idx]
    holdout = [ids[i] for i in holdout_idx]
    shared = {groups[i] for i in fit_idx} & {groups[i] for i in holdout_idx}
    if shared:
        raise AssertionError(f"LEAKAGE: {len(shared)} lesiones en los dos lados")
    return fit, holdout


def select_winner(
    results: dict[str, dict],
    candidates: Sequence[str] | None = None,
    *,
    tie: float = TIE_MACRO_F1,
) -> str:
    """Highest macro-F1; among those within ``tie`` of it, the lowest ECE."""
    names = list(results) if candidates is None else list(candidates)
    if not names:
        raise ValueError("no hay variantes candidatas: elegir entre nada no es una seleccion")
    best = max(results[n]["macro_f1"] for n in names)
    close = [n for n in names if best - results[n]["macro_f1"] < tie]
    return min(close, key=lambda n: (results[n]["ece"], -results[n]["macro_f1"]))
