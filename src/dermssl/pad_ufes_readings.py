"""The PAD-UFES-20 zero-shot readings and verdicts registered in docs/30 2.3-3.

Inputs are the seven-column HAM probabilities of the C1 ``logreg`` probe, one
row per PAD image. Each reading keeps its classes' columns and renormalizes
them (docs/30 2.3); metrics, pairing and verdict directions come from
``paired_bootstrap`` and ``final_eval``, with the resampling unit moved to
the patient (docs/30 2.4).
"""

from __future__ import annotations

import numpy as np

from dermssl import final_eval as fe
from dermssl import pad_ufes as pu
from dermssl import paired_bootstrap as pb

#: The probe's columns: sorted HAM ``dx``, as ``cluster_frozen_bank.build_split`` builds them.
HAM_CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
SEEDS = (42, 43)
PUBLIC = "public"
ARMS = ("plain", "proposed")
BOOTSTRAP = {"n_resamples": 1000, "rng_seed": 20260923}
MIN_BIN_N = 100
NO_POWER = "sin potencia"
NOT_CALCULABLE = pb.NOT_CALCULABLE


def role(arm: str, seed: int) -> str:
    return f"{arm}_s{seed}"


def roles(seeds=SEEDS) -> list[str]:
    return [PUBLIC] + [role(arm, seed) for seed in seeds for arm in ARMS]


def contrast_pairs(seeds=SEEDS) -> list[tuple[str, str]]:
    """Per seed, the three contrasts of docs/28: A*-public, B*-public, B*-A*."""
    pairs = []
    for seed in seeds:
        plain, proposed = role("plain", seed), role("proposed", seed)
        pairs += [(plain, PUBLIC), (proposed, PUBLIC), (proposed, plain)]
    return pairs


def _columns(reading: str) -> list[int]:
    return [HAM_CLASSES.index(c) for c in pu.READINGS[reading]]


def restrict(proba7: np.ndarray, reading: str) -> np.ndarray:
    """P(class | the image is one of the reading's classes)."""
    sub = np.asarray(proba7, dtype=float)[:, _columns(reading)]
    totals = sub.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError(f"{int((totals <= 0).sum())} filas sin masa en las clases de {reading}")
    return sub / totals


def targets(ham_labels, reading: str) -> tuple[np.ndarray, np.ndarray]:
    """``(mask of rows in the reading, y indexed into the reading's classes)``."""
    classes = pu.READINGS[reading]
    labels = np.asarray(ham_labels)
    mask = np.isin(labels, classes)
    return mask, np.array([classes.index(label) for label in labels[mask]], dtype=int)


def outside_rate(proba7: np.ndarray, reading: str) -> float:
    """Descriptive: share of images whose seven-way argmax is not a class of the reading."""
    return float((~np.isin(np.asarray(proba7).argmax(axis=1), _columns(reading))).mean())


def _resample_metrics(y: np.ndarray, proba: np.ndarray, groups: np.ndarray) -> dict:
    """Every metric on every patient-grouped resample (the draws ``paired_bootstrap`` makes)."""
    draw = pb.index_sampler(len(y), groups)
    rng = np.random.RandomState(BOOTSTRAP["rng_seed"])
    samples = {m: [] for m in fe.METRICS}
    for _ in range(BOOTSTRAP["n_resamples"]):
        picks = draw(rng)
        for metric in fe.METRICS:
            samples[metric].append(pb.METRIC_FUNCTIONS[metric](y[picks], proba[picks]))
    return {m: np.asarray(v, dtype=float) for m, v in samples.items()}


def _percentile_ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def arm_summary(y: np.ndarray, proba: np.ndarray, groups: np.ndarray) -> dict:
    """Point metrics plus a patient-grouped percentile interval for each; a non-finite resample aborts."""
    samples = _resample_metrics(y, proba, groups)
    bad = {m: int((~np.isfinite(v)).sum()) for m, v in samples.items() if not np.isfinite(v).all()}
    if bad:
        raise ValueError(f"remuestras con valor no finito por metrica, de {BOOTSTRAP['n_resamples']}: {bad}")
    out = {m: pb.METRIC_FUNCTIONS[m](y, proba) for m in fe.METRICS}
    out["ci95"] = {m: _percentile_ci(v) for m, v in samples.items()}
    return out


def stratum_arm_summary(y: np.ndarray, proba: np.ndarray, groups: np.ndarray) -> dict:
    """docs/30 6, rule (b): the point always; an interval only if every resample is finite, else k."""
    samples = _resample_metrics(y, proba, groups)
    out = {m: pb.METRIC_FUNCTIONS[m](y, proba) for m in fe.METRICS}
    out["n_nonfinite"] = {m: int((~np.isfinite(v)).sum()) for m, v in samples.items()}
    out["ci95"] = {m: NOT_CALCULABLE if out["n_nonfinite"][m] else _percentile_ci(v) for m, v in samples.items()}
    return out


def contrast_rows(y, probas: dict, groups, pairs, record=pb.contrast_record) -> list[dict]:
    return [
        record(arm=arm, control=control, metric=metric, probe="logreg", fraction=1.0,
               seed=0, y_true=y, arm_proba=probas[arm], control_proba=probas[control],
               groups=groups, **BOOTSTRAP)
        for arm, control in pairs
        for metric in fe.METRICS
    ]


def evaluate_subset(y, probas: dict, groups, pairs, stratum: bool = False) -> dict:
    """``stratum`` applies rule (b) of docs/30 6 (exploratory strata only); otherwise non-finite aborts."""
    groups = np.asarray(groups)
    summary = stratum_arm_summary if stratum else arm_summary
    record = pb.contrast_record_reporting_nonfinite if stratum else pb.contrast_record
    return {
        "n_images": int(len(y)),
        "n_patients": int(len(set(groups.tolist()))),
        "arms": {name: summary(y, proba, groups) for name, proba in probas.items()},
        "paired": contrast_rows(y, probas, groups, pairs, record),
    }


def evaluate_reading(reading: str, ham_labels, groups, probas7: dict, pairs) -> dict:
    mask, y = targets(ham_labels, reading)
    probas = {name: restrict(np.asarray(p)[mask], reading) for name, p in probas7.items()}
    result = evaluate_subset(y, probas, np.asarray(groups)[mask], pairs)
    result["classes"] = list(pu.READINGS[reading])
    result["outside_rate"] = {name: outside_rate(np.asarray(p)[mask], reading)
                              for name, p in probas7.items()}
    if reading in pu.READINGS_WITH_VERDICT:
        result["verdicts"] = verdicts(result["paired"])
    return result


def fst_strata(ham_labels, groups, fst_bins, probas7: dict, pairs) -> dict:
    """docs/30 2.5: the primary reading per phototype bin. Exploratory, no verdict.

    A resample with a non-finite metric does not abort here (docs/30 6, rule (b)):
    the point is kept, that interval is :data:`NOT_CALCULABLE` and ``n_nonfinite``
    carries the exact k of ``BOOTSTRAP["n_resamples"]``.
    """
    bins = np.asarray(fst_bins)
    out = {}
    for name in pu.FST_BINS:
        keep = bins == name
        mask, y = targets(np.asarray(ham_labels)[keep], "primary")
        if not mask.any():
            out[name] = {"n_images": 0, "power": NO_POWER}
            continue
        probas = {r: restrict(np.asarray(p)[keep][mask], "primary") for r, p in probas7.items()}
        entry = evaluate_subset(y, probas, np.asarray(groups)[keep][mask], pairs, stratum=True)
        entry["power"] = NO_POWER if entry["n_images"] < MIN_BIN_N else "ok"
        out[name] = entry
    return out


def _all_seeds(paired: dict, arm: str, control: str, test) -> str:
    ok = all(test(paired[(role(arm, s), control if control == PUBLIC else role(control, s))])
             for s in SEEDS)
    return fe.MET if ok else fe.NOT_MET


def verdicts(rows: list[dict]) -> dict:
    """PAD-P1..P4 of docs/30 3, plus the "mejora" reading of docs/30 2.4 per contrast."""
    paired = fe.paired_from_rows(rows)
    f1_not_up = lambda c: c["macro_f1"]["point_delta"] <= 0  # noqa: E731
    return {
        "PAD-P1": _all_seeds(paired, "proposed", PUBLIC, f1_not_up),
        "PAD-P2": _all_seeds(paired, "plain", PUBLIC, f1_not_up),
        "PAD-P3": _all_seeds(paired, "proposed", PUBLIC, lambda c: c["ece"]["point_delta"] < 0),
        "PAD-P4": _all_seeds(paired, "proposed", "plain", lambda c: not c["macro_f1"]["excludes_zero"]),
        "mejora_robusta": improvements(rows),
    }


def improvements(rows: list[dict]) -> dict:
    """``"<arm>-<control>/<metric>" -> True`` only if the arm is better with every seed."""
    wins: dict[str, list[bool]] = {}
    for row in rows:
        arm, control = row["arm"].split("_s")[0], row["control"].split("_s")[0]
        key = f"{arm}-{control}/{row['metric']}"
        wins.setdefault(key, []).append(row["direction"] == "arm_better")
    return {key: all(values) and len(values) == len(SEEDS) for key, values in wins.items()}
