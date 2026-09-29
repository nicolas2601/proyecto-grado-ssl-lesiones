"""T3 (docs/28): C1-v7, every mask condition against ``none`` inside each luminance tercile.

The bins are the ``tone_tercile`` column of the published manifest: terciles
of the skin median L* over the pool. They are luminance, not skin tone.

Inside a tercile both sides score the SAME images, so the contrast is the
paired image bootstrap of :mod:`dermssl.paired_bootstrap`, reused rather than
rewritten. What this module adds is the per-bin metric definition (macro over
the classes the bin actually holds), the power rule and the registered
reading of the prediction. Pure numpy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from dermssl import metrics as _metrics
from dermssl import paired_bootstrap as pb
from dermssl.luminance_dice import MIN_N, TERCILES

CONTROL = "none"
CONDITIONS = ("bbox_crop", "neutral_bg", "mask_pool")
METRICS = ("macro_f1", "balanced_accuracy", "mel_recall")
PRIMARY_METRIC = "macro_f1"
SEGMENTERS = ("student_r3", "a2")
KEY_TERCILE = "dark"
KEY_CONDITIONS = ("bbox_crop", "mask_pool")
KEY_BACKBONES = ("v7_astar_s42", "v7_astar_s43", "dinov2_public_vitb14")
ASTAR_SEEDS = ("v7_astar_s42", "v7_astar_s43")
PUBLIC = "dinov2_public_vitb14"

IMPROVES = "mejora"
WORSENS = "empeora"
INDISTINGUISHABLE = "indistinguible"
NO_POWER = "sin potencia"
PREDICTION_HOLDS = "cumplida"
PREDICTION_FAILS = "no cumplida"

_VERDICT = {"arm_better": IMPROVES, "control_better": WORSENS, "indistinguishable": INDISTINGUISHABLE}

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def metric_functions(y_bin: np.ndarray, *, n_classes: int, mel: int) -> dict[str, MetricFn]:
    """The three metrics of one bin; classes absent from the whole bin never enter a mean.

    The class set is fixed on the full bin, before any resample, so a resample
    that happens to lose a rare class scores it 0 on BOTH sides of the pair.
    ``mel_recall`` is left out when the bin holds no melanoma: it is undefined.
    """
    present = np.unique(np.asarray(y_bin))

    def macro_f1(y: np.ndarray, proba: np.ndarray) -> float:
        return float(_metrics.per_class_f1(y, proba.argmax(axis=1), n_classes)[present].mean())

    def balanced_accuracy(y: np.ndarray, proba: np.ndarray) -> float:
        return float(_metrics.recall_per_class(y, proba.argmax(axis=1), n_classes)[present].mean())

    def mel_recall(y: np.ndarray, proba: np.ndarray) -> float:
        return float(_metrics.recall_per_class(y, proba.argmax(axis=1), n_classes)[mel])

    fns = {"macro_f1": macro_f1, "balanced_accuracy": balanced_accuracy}
    if mel in present:
        fns["mel_recall"] = mel_recall
    return fns


def _no_power(y, arm, control, fn) -> dict:
    point_arm, point_control = float(fn(y, arm)), float(fn(y, control))
    return {"point_arm": point_arm, "point_control": point_control,
            "point_delta": point_arm - point_control, "verdict": NO_POWER}


def _contrast(y, arm, control, metric, fn, n_resamples, seed) -> dict:
    stats = pb.paired_contrast(y_true=y, arm_proba=arm, control_proba=control, metric_fn=fn,
                               n_resamples=n_resamples, seed=seed)
    direction = pb.direction_for(metric=metric, point_delta=stats["point_delta"],
                                 excludes_zero=stats["excludes_zero"])
    return {**stats, "verdict": _VERDICT[direction]}


def _bin_entry(y, probas, n_resamples, seed, mel, n_classes) -> dict:
    fns = metric_functions(y, n_classes=n_classes, mel=mel)
    powered = len(y) >= MIN_N
    contrasts = {}
    for condition in (c for c in CONDITIONS if c in probas):
        arm, control = probas[condition], probas[CONTROL]
        contrasts[condition] = {
            m: (_contrast(y, arm, control, m, fns[m], n_resamples, seed) if powered
                else _no_power(y, arm, control, fns[m])) if m in fns else {"verdict": "no definida"}
            for m in METRICS
        }
    return {"n": int(len(y)), "n_mel": int((y == mel).sum()), "contrasts": contrasts}


def tercile_contrasts(y_true: np.ndarray, terciles: np.ndarray, probas: Mapping[str, np.ndarray], *,
                      n_resamples: int, seed: int, mel: int, n_classes: int) -> dict:
    """Per tercile: n, melanomas and every condition against ``none`` in every metric."""
    if CONTROL not in probas:
        raise ValueError(f"falta la condicion {CONTROL!r}: todo contraste se mide contra ella")
    y_true, terciles = np.asarray(y_true), np.asarray(terciles)
    out = {}
    for tercile in TERCILES:
        rows = terciles == tercile
        if not rows.any():
            out[tercile] = {"n": 0, "n_mel": 0, "contrasts": {}}
            continue
        sliced = {c: np.asarray(p)[rows] for c, p in probas.items()}
        out[tercile] = _bin_entry(y_true[rows], sliced, n_resamples, seed, mel, n_classes)
    return out


def _robust(cell: tuple[str, str, str], improving: set) -> bool:
    segmenter, backbone, condition = cell
    if backbone in ASTAR_SEEDS:
        return all((segmenter, seed, condition) in improving for seed in ASTAR_SEEDS)
    return all((other, backbone, condition) in improving for other in SEGMENTERS)


def prediction(key_verdicts: Mapping[tuple[str, str, str], str]) -> dict:
    """Paula's prediction over the 12 key cells: it holds unless one of them improves."""
    if any(v == NO_POWER for v in key_verdicts.values()):
        return {"verdict": NO_POWER, "n_key_cells": len(key_verdicts)}
    improving = {cell for cell, v in key_verdicts.items() if v == IMPROVES}
    return {
        "verdict": PREDICTION_FAILS if improving else PREDICTION_HOLDS,
        "n_key_cells": len(key_verdicts),
        "improves": sorted("/".join(c) for c in improving),
        "robust": sorted("/".join(c) for c in improving if _robust(c, improving)),
        "n_worsens": sum(v == WORSENS for v in key_verdicts.values()),
    }


def n_contrasts(*, n_backbones: int) -> int:
    """Every contrast the analysis computes, for the registered multiplicity."""
    return n_backbones * len(CONDITIONS) * len(TERCILES) * len(METRICS) * len(SEGMENTERS)
