"""Paired bootstrap for arm-vs-control contrasts.

Why this module exists
----------------------
``resultados/clasificacion/bootstrap.json`` was committed as DATA: one commit,
alongside its consumer, with no generator anywhere in the repo. Its schema is
read by ``build_results_readme.py``, ``build_site_data.py`` and the demo
backend, so C1 has to produce the same shape -- and to produce it, someone has
to write the code that was never written.

The load-bearing property
-------------------------
The two arms are resampled with the SAME indices. Every resample asks "on
THESE images, how much better was the arm?", so the per-image noise the two
share cancels out. Drawing independently would add variance that is not in
the data, widen every interval, and turn real differences into
"indistinguishable" -- a conclusion that reads as admirably cautious and is
simply wrong.

The point estimate is NOT resampled. It is the metric on the full set, which
is why it can be used to check this implementation against the v6 numbers:
if it does not come out identical, the two are not measuring the same thing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from dermssl import metrics as _metrics

#: Metrics where a SMALLER value is better. The verdict must read this rather
#: than infer a side from the sign of the delta: inferring is exactly the
#: inverted-label bug that hit HD95, and it would flip every ECE row.
LOWER_IS_BETTER = frozenset({"ece", "mel_to_nv_rate"})  # melanomas sent to nevus: fewer is better

#: Metrics whose direction is known. An unlisted one is refused rather than
#: assumed higher-is-better, because the next lower-is-better metric to arrive
#: (HD95, error rate, latency) would otherwise have its verdict printed
#: backwards, and a backwards verdict still looks like a result.
KNOWN_METRICS = frozenset({
    "ece", "macro_f1", "balanced_accuracy", "auc",
    "accuracy", "mel_recall", "mel_auc",  # docs/29 2.3b secondaries, higher is better
    # docs/29 2.7 item 1 operating points at a fixed holdout threshold, higher is better
    "mel_sens_at_thr_spec80", "mel_sens_at_thr_spec90", "mel_spec_at_thr_sens95",
}) | LOWER_IS_BETTER


def metric_macro_f1(y_true: np.ndarray, proba: np.ndarray) -> float:
    n_classes = proba.shape[1]
    return float(_metrics.macro_f1(y_true, proba.argmax(axis=1), n_classes))


def metric_balanced_accuracy(y_true: np.ndarray, proba: np.ndarray) -> float:
    n_classes = proba.shape[1]
    return float(_metrics.balanced_accuracy(y_true, proba.argmax(axis=1), n_classes))


def metric_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    return float(_metrics.auc_ovr_macro(y_true, proba))


def metric_ece(y_true: np.ndarray, proba: np.ndarray) -> float:
    return float(_metrics.ece(y_true, proba))


METRIC_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "macro_f1": metric_macro_f1,
    "balanced_accuracy": metric_balanced_accuracy,
    "auc": metric_auc,
    "ece": metric_ece,
}


def direction_for(*, metric: str, point_delta: float, excludes_zero: bool) -> str:
    """``arm_better`` / ``control_better`` / ``indistinguishable``.

    Computed HERE, at write time, because the consumer deliberately refuses to
    compute it: ``build_results_readme._verdict`` reads the stored field
    rather than inferring from the sign, and says so in its docstring. That
    makes this function the single place the direction can be got wrong.
    """
    if metric not in KNOWN_METRICS:
        raise ValueError(
            f"no se conoce la direccion de {metric!r}. Asumir que mayor es mejor es como "
            "una metrica lower-is-better (HD95, error, latencia) termina con el veredicto "
            f"al reves; agregala a KNOWN_METRICS y, si corresponde, a LOWER_IS_BETTER"
        )

    # An interval spanning zero is not a win, whichever way the point leans.
    # Checked BEFORE the sign so the lower-is-better path cannot skip it.
    if not excludes_zero:
        return "indistinguishable"

    favours_arm = point_delta < 0 if metric in LOWER_IS_BETTER else point_delta > 0
    return "arm_better" if favours_arm else "control_better"


def index_sampler(n: int, groups: Sequence | None = None) -> Callable[[np.random.RandomState], np.ndarray]:
    """One bootstrap draw of row indices: rows, or whole groups when ``groups`` is given.

    Without groups the draw is ``rng.randint(0, n, n)``, the call every
    published interval used, so those numbers do not move. With groups
    (PAD-UFES patients, docs/30 2.4) whole groups are drawn with replacement
    and every row of a drawn group enters: rows of one patient are not
    independent, and resampling them one by one narrows the interval with
    dependence that is not new information.
    """
    if groups is None:
        return lambda rng: rng.randint(0, n, n)
    groups = np.asarray(groups)
    if len(groups) != n:
        raise ValueError(f"groups tiene {len(groups)} filas y los datos {n}: cada fila necesita su grupo")
    _, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    members = np.split(order, np.flatnonzero(np.diff(inverse[order])) + 1)
    k = len(members)
    return lambda rng: np.concatenate([members[g] for g in rng.randint(0, k, k)])


def paired_contrast(
    *,
    y_true: np.ndarray,
    arm_proba: np.ndarray,
    control_proba: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    groups: Sequence | None = None,
    control_metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> dict:
    """Point estimates plus a percentile interval on the paired delta.

    ``groups`` switches the resampling unit from the row to the group; the
    pairing does not change: both arms still see the same draw.
    ``control_metric_fn`` scores the control when it needs its own function
    (an operating point at the control's own threshold); default: ``metric_fn``.
    """
    control_fn = metric_fn if control_metric_fn is None else control_metric_fn
    y_true = np.asarray(y_true)
    arm_proba = np.asarray(arm_proba)
    control_proba = np.asarray(control_proba)

    if not (len(y_true) == len(arm_proba) == len(control_proba)):
        raise ValueError(
            f"largo distinto: y_true {len(y_true)}, brazo {len(arm_proba)}, control "
            f"{len(control_proba)}. Un contraste pareado exige las MISMAS imagenes en el "
            "mismo orden"
        )
    if n_resamples < 1:
        raise ValueError(
            f"n_resamples={n_resamples}: sin remuestras, np.percentile devuelve nan y "
            "`nan > 0` es False, asi que el intervalo incluiria el cero en silencio"
        )

    point_arm = float(metric_fn(y_true, arm_proba))
    point_control = float(control_fn(y_true, control_proba))
    point_delta = point_arm - point_control

    rng = np.random.RandomState(seed)
    draw = index_sampler(len(y_true), groups)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        # ONE index draw, used for both arms. This is the whole point.
        picks = draw(rng)
        deltas[index] = metric_fn(y_true[picks], arm_proba[picks]) - control_fn(
            y_true[picks], control_proba[picks]
        )

    if not np.isfinite(deltas).all():
        raise ValueError(
            f"{int((~np.isfinite(deltas)).sum())} de {n_resamples} remuestras dieron un "
            "valor no finito. np.percentile sobre NaN devuelve NaN y toda comparacion "
            "contra NaN es False, asi que el contraste diria 'indistinguible' para una "
            "metrica que en realidad no se pudo calcular"
        )

    low = float(np.percentile(deltas, 100 * alpha / 2))
    high = float(np.percentile(deltas, 100 * (1 - alpha / 2)))

    return {
        "point_arm": point_arm,
        "point_control": point_control,
        "point_delta": point_delta,
        "ci95_low": low,
        "ci95_high": high,
        # Strictly on one side: an interval touching zero is not a win.
        "excludes_zero": bool(low > 0 or high < 0),
    }


def contrast_record(
    *,
    arm: str,
    control: str,
    metric: str,
    probe: str,
    fraction: float,
    seed: int,
    y_true: np.ndarray,
    arm_proba: np.ndarray,
    control_proba: np.ndarray,
    n_resamples: int = 1000,
    rng_seed: int = 20260923,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    groups: Sequence | None = None,
    control_metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> dict:
    """One row of ``bootstrap.json``, in the published schema.

    The field set is fixed by the consumers already parsing that file, so it
    is asserted by a test rather than left to drift.
    """
    if metric_fn is None:
        if metric not in METRIC_FUNCTIONS:
            raise ValueError(
                f"no hay funcion para la metrica {metric!r}; pasala en metric_fn o "
                f"agregala a METRIC_FUNCTIONS ({sorted(METRIC_FUNCTIONS)})"
            )
        metric_fn = METRIC_FUNCTIONS[metric]

    stats = paired_contrast(
        y_true=y_true,
        arm_proba=arm_proba,
        control_proba=control_proba,
        metric_fn=metric_fn,
        n_resamples=n_resamples,
        seed=rng_seed,
        groups=groups,
        control_metric_fn=control_metric_fn,
    )

    return {
        "arm": arm,
        "control": control,
        "metric": metric,
        "probe": probe,
        "fraction": float(fraction),
        "seed": int(seed),
        **stats,
        "direction": direction_for(
            metric=metric,
            point_delta=stats["point_delta"],
            excludes_zero=stats["excludes_zero"],
        ),
    }


def bootstrap_document(
    records: Sequence[dict],
    *,
    n_resamples: int,
    rng_seed: int,
    probe: str,
    fraction: float,
    seed_used: int,
    n_val: int,
    val_order_sha256: str,
) -> dict:
    """The top-level shape of ``bootstrap.json``.

    ``note_ece`` travels WITH the data. A reader who sees a negative delta
    favouring the arm needs to be told why here, not in a file they may never
    open -- that note is what stops the next person from "fixing" the sign.
    """
    return {
        "n_resamples": int(n_resamples),
        "rng_seed": int(rng_seed),
        "stratified_by": "class",
        "paired": True,
        "probe": probe,
        "fraction": float(fraction),
        "seed_used": int(seed_used),
        "n_val": int(n_val),
        "val_order_sha256": val_order_sha256,
        "note_ece": "ECE is lower-is-better: a negative delta favours the arm.",
        # Taken verbatim from the published v6 document rather than reworded:
        # this note explains why a single seed is reported, and paraphrasing it
        # would quietly change what the file claims about its own uncertainty.
        "note_seed": (
            "At fraction=1.0 the seed only controls label subsampling, so the three bank "
            "seeds are one run repeated; seed 0 is used and the uncertainty reported here "
            "is resampling of the test set, not of seeds."
        ),
        "results": list(records),
    }
