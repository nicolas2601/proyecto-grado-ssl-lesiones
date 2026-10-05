"""Extra contrasts on the v7.1 val read (asked after the single read, not registered).

Three questions the main table does not answer:

* Fair melanoma operating points: each entry read at its OWN holdout
  threshold on the probe fitted on the 80 % fit split, exactly as the
  published per-entry numbers, and the paired delta between two entries.
  One draw serves both arms; each arm keeps its own threshold.
* Melanomas predicted as nevus (argmax of the full-train probe), the error
  with the highest clinical cost, and its paired delta.
* Macro-F1 and balanced accuracy against the v7 s43 primaries.

Every contrast is ``paired_bootstrap.contrast_record`` with the published
seed, so a row here and a row of ``v71_final_eval.json`` for the same pair
and metric are the same number.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import numpy as np

from dermssl import paired_bootstrap as pb
from dermssl import v71_metrics as vm

FAIR_OPERATING_METRICS = ("mel_sens_at_thr_spec80", "mel_sens_at_thr_spec90", "mel_spec_at_thr_sens95")
MEL_TO_NV = "mel_to_nv_rate"


def mel_to_nv(y_true, proba, *, mel: int, nv: int) -> dict:
    """True melanomas whose argmax is nevus: count, and rate over true melanomas."""
    is_mel = np.asarray(y_true) == mel
    n_mel = int(is_mel.sum())
    count = int((np.asarray(proba)[is_mel].argmax(axis=1) == nv).sum())
    return {"n_mel": n_mel, "count": count, "rate": count / n_mel if n_mel else float("nan")}


def mel_to_nv_rate_fn(mel: int, nv: int) -> Callable[[np.ndarray, np.ndarray], float]:
    return lambda y, p: mel_to_nv(y, p, mel=mel, nv=nv)["rate"]


def _record(arm, control, metric, y, probas, n_resamples, fn, control_fn=None) -> dict:
    return pb.contrast_record(arm=arm, control=control, metric=metric, probe="logreg", fraction=1.0, seed=0,
                              y_true=y, arm_proba=probas[arm], control_proba=probas[control],
                              n_resamples=n_resamples, rng_seed=vm.BOOTSTRAP_SEED, metric_fn=fn,
                              control_metric_fn=control_fn)


def _fixed_fn(metric: str, thresholds: dict, mel: int):
    return lambda y, p: vm.fixed_threshold_metrics(y, p, mel=mel, thresholds=thresholds)[metric]


def fair_operating_contrasts(y_true, probas_fit80: Mapping, thresholds: Mapping,
                             pairs: Iterable[tuple[str, str]], *, mel: int,
                             n_resamples: int = vm.BOOTSTRAP_RESAMPLES) -> list[dict]:
    """Paired operating-point deltas, each entry at its own holdout threshold."""
    return [_record(arm, control, metric, y_true, probas_fit80, n_resamples,
                    _fixed_fn(metric, thresholds[arm], mel), _fixed_fn(metric, thresholds[control], mel))
            for arm, control in pairs for metric in FAIR_OPERATING_METRICS]


def argmax_contrasts(y_true, probas: Mapping, pairs: Iterable[tuple[str, str]], metrics: Iterable[str], *,
                     n_resamples: int = vm.BOOTSTRAP_RESAMPLES, mel: int = 4, nv: int = 5) -> list[dict]:
    """Paired deltas of argmax metrics (published functions, or the mel->nv rate)."""
    functions = {**pb.METRIC_FUNCTIONS, MEL_TO_NV: mel_to_nv_rate_fn(mel, nv)}
    unknown = [m for m in metrics if m not in functions]
    if unknown:
        raise ValueError(f"no hay funcion para la metrica {unknown}; conocidas: {sorted(functions)}")
    return [_record(arm, control, metric, y_true, probas, n_resamples, functions[metric])
            for arm, control in pairs for metric in metrics]
