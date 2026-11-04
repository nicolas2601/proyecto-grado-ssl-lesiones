"""Continuous-ITA stratified evaluation: the valid replacement for the G3 fairness gate.

PURE EVALUATION INSTRUMENT. This module measures; it transforms nothing. It
never calls ``tone_augment_lab.augment_skin_tone`` or
``tone_augment_lab_v2.augment_skin_tone_v2``, writes no pixel, and must not
be confused with the tonal augmentation arm.

Why this exists: ``dermssl.gates.G3`` (equity gate) and
``circle.should_run_stage2`` key CIRCLe stage 2 on Fitzpatrick skin-type
BUCKETS. ``docs/63-instrumento-ita-alcance-y-limites.md`` section 4 already
retired that construct with evidence: against PAD-UFES-20's real `fitspatrick`
field, optimized cuts give 53.3% accuracy against 55.8% for a constant
predictor -- the optimizer converges to the constant, and Spearman between
ITA and the real FST label is +0.027 (p=0.50). No JSON in this repo ever
carried G3's computed status; the gate that decides whether stage 2 runs was
never evaluated against real data, and is built on a construct the project's
own instrument has already declared invalid. This module is the replacement:
it stratifies by ITA in continuous degrees, never by a Fitzpatrick bucket or
``fitzpatrick.FST_THRESHOLDS``.

Three quantities ``gates.py`` does NOT separate are kept apart here on
purpose:

* Measurability itself. ``fitzpatrick.estimate_fst`` abstains
  (``status != "ok"``) on a declared fraction of any real pool --
  ~28% on ISIC 2018 with expert masks, 46.5% on the ISIC2019 + teacher-026
  pool (docs/63). Coverage is REPORTED, never silently backfilled: an
  unmeasurable row contributes to the coverage count and nowhere else.
* Mask provenance (``mask_source``). A pseudo-mask (e.g. a teacher model) and
  an Otsu fallback are not the same measurement regime -- docs/63 measures
  the masked estimator 2x-6x more accurate than the unmasked one, by lesion
  area stratum. Averaging the two regimes together hides exactly that gap.
  Stratify or at minimum report counts per ``mask_source``, never pool them
  silently.
* Cohort terciles vs any literature cut. When terciles are used for
  legibility, they are percentiles of THIS cohort's own measured ITA
  distribution, labeled as such -- never Groh's or Corbin's thresholds,
  which section 4 retired as a classification label and section 5 flags as
  internally inconsistent even as a magnitude cap.

A caveat that belongs here, not just in a test: ``dermssl.metrics.per_class_f1``
scores a class 0 when it has no true AND no predicted samples in a subset
(documented in that function). A stratum where only one or two of the seven
HAM10000 classes are actually present -- entirely plausible with ~58x class
imbalance and a thin mask_source or tercile split -- cannot reach macro-F1
near 1.0 even with every prediction correct, because the five or six absent
classes still score 0 each. Do not read a low per-stratum macro-F1 as model
failure without first checking which classes that stratum actually contains.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from dermssl import metrics
from dermssl.per_class_metrics import HAM10000_CLASSES

N_CLASSES = len(HAM10000_CLASSES)

# Reused from dermssl.metrics, not re-derived: the bootstrap resample count
# and seed already used for every other interval in this codebase's eval
# path (metrics._macro_f1_interval, metrics.bootstrap_interval).
BOOTSTRAP_RESAMPLES = metrics.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = metrics.BOOTSTRAP_SEED

MIN_STRATUM_N = 30


def load_npz_predictions(path: Path, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """``y_true`` and the probability matrix for one variant of a saved ``.npz``.

    Fails loudly if ``variant`` is absent or the column count does not match
    ``HAM10000_CLASSES`` -- reading a probability matrix with the wrong class
    order is a silent-corruption bug, not a recoverable one.
    """
    data = np.load(path, allow_pickle=False)
    if variant not in data.files:
        raise ValueError(f"variant {variant!r} not in {path}; available: {list(data.files)}")
    y_true = np.asarray(data["y_true"])
    proba = np.asarray(data[variant], dtype=np.float64)
    if proba.shape[1] != N_CLASSES:
        raise ValueError(
            f"{variant} has {proba.shape[1]} columns, expected {N_CLASSES} "
            f"(HAM10000_CLASSES={HAM10000_CLASSES})"
        )
    if proba.shape[0] != y_true.shape[0]:
        raise ValueError(f"{variant} has {proba.shape[0]} rows, y_true has {y_true.shape[0]}")
    return y_true, proba


def read_ita_manifest(path: Path) -> list[dict]:
    """One record per line: at minimum ``row`` (the index into the npz arrays).

    Optional per record: ``ita`` (degrees, NaN or absent when unmeasurable),
    ``ita_status`` (``"ok"`` or an abstention reason), ``b_median``,
    ``mask_source`` (arbitrary label, not validated against a fixed set --
    see ``metrics.stratified_by_label``), ``class`` (string class name, used
    ONLY as a cross-check against ``y_true``, never as the ground truth
    itself).
    """
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def join_predictions_with_ita(
    y_true: np.ndarray, ita_rows: list[dict]
) -> None:
    """Cross-checks alignment; raises on the first mismatch instead of computing on a bad join.

    ``row`` is the join key, not list position (the manifest need not be
    sorted, though it usually will be): a record's ``row`` indexes directly
    into ``y_true``/``proba``. When a record carries ``class``, it must name
    the SAME class ``y_true[row]`` encodes under ``HAM10000_CLASSES`` -- the
    same kind of independent cross-check that confirmed this join's row
    order for the v7 manifest in the first place (two independent checks,
    the sha256 of the id list and element-wise label agreement).
    """
    n = y_true.shape[0]
    for record in ita_rows:
        row = record["row"]
        if not (0 <= row < n):
            raise ValueError(f"row {row} out of range for {n} predictions")
        class_name = record.get("class")
        if class_name is not None:
            expected = HAM10000_CLASSES[int(y_true[row])]
            if class_name != expected:
                raise ValueError(
                    f"row {row}: manifest class {class_name!r} != y_true class {expected!r} "
                    "-- the ITA manifest and the npz are not aligned"
                )


def coverage_report(ita_rows: list[dict], n_total: int) -> dict:
    """Declares coverage; fills nothing in. Every count is of what IS present."""
    n_manifest = len(ita_rows)
    measurable = [r for r in ita_rows if r.get("ita_status") == "ok" and r.get("ita") is not None]
    by_status: dict[str, int] = {}
    for record in ita_rows:
        status = record.get("ita_status") or "sin_ita_status"
        by_status[status] = by_status.get(status, 0) + 1
    by_mask_source: dict[str, int] = {}
    for record in ita_rows:
        source = record.get("mask_source", "sin_declarar")
        by_mask_source[source] = by_mask_source.get(source, 0) + 1
    return {
        "n_total_predictions": n_total,
        "n_in_ita_manifest": n_manifest,
        "manifest_coverage_fraction": n_manifest / n_total if n_total else None,
        "n_ita_measurable": len(measurable),
        "ita_measurable_fraction_of_manifest": len(measurable) / n_manifest if n_manifest else None,
        "by_ita_status": by_status,
        "by_mask_source": by_mask_source,
    }


def _measurable_arrays(
    y_true: np.ndarray, proba: np.ndarray, ita_rows: list[dict]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Rows restricted to ``ita_status == "ok"``: indices, ITA values, and mask sources, in manifest order."""
    rows, ita_values, mask_sources = [], [], []
    for record in ita_rows:
        if record.get("ita_status") == "ok" and record.get("ita") is not None:
            rows.append(record["row"])
            ita_values.append(float(record["ita"]))
            mask_sources.append(record.get("mask_source", "sin_declarar"))
    idx = np.asarray(rows, dtype=np.int64)
    return y_true[idx], proba[idx], np.asarray(ita_values, dtype=np.float64), mask_sources


def _per_class_counts(y_true_subset: np.ndarray) -> dict[str, int]:
    return {
        class_name: int(np.sum(y_true_subset == class_index))
        for class_index, class_name in enumerate(HAM10000_CLASSES)
    }


def _stratum_mask(labels: np.ndarray, key: str) -> np.ndarray:
    if key == metrics.UNAVAILABLE_BUCKET:
        return np.array([metrics._is_unavailable_bucket(v) for v in labels])
    return np.array([not metrics._is_unavailable_bucket(v) and str(v) == key for v in labels])


def _augment_with_class_counts(
    stratum: dict, y_true: np.ndarray, labels: np.ndarray, min_n: int
) -> dict:
    """Attaches per-class counts to every OK entry: declares thin or absent classes, never hides them.

    A class with 0 true instances in a stratum is ABSENT (contributes 0 to
    that stratum's macro-F1 by ``per_class_f1``'s own documented
    convention); a class with ``0 < count < min_n`` is THIN (present, but
    not enough of it to trust). Different failure modes, reported
    separately -- maestro's requirement 3, ~38 melanoma cases across strata
    is exactly the regime where this distinction is the whole point.
    """
    augmented = {}
    for key, entry in stratum.items():
        if entry.get("status") != "OK":
            augmented[key] = entry
            continue
        mask = _stratum_mask(labels, key)
        counts = _per_class_counts(y_true[mask])
        augmented[key] = {
            **entry,
            "per_class_counts": counts,
            "classes_absent": [c for c, n in counts.items() if n == 0],
            "classes_below_min_n": [c for c, n in counts.items() if 0 < n < min_n],
        }
    return augmented


def macro_f1_by_mask_source(
    y_true: np.ndarray, proba: np.ndarray, ita_rows: list[dict], *, min_n: int = MIN_STRATUM_N
) -> dict:
    """``metrics.stratified_by_label`` keyed by ``mask_source`` -- never averaged silently."""
    y_sub, proba_sub, _, sources = _measurable_arrays(y_true, proba, ita_rows)
    if y_sub.size == 0:
        return {}
    labels = np.asarray(sources, dtype=object)
    stratum = metrics.stratified_by_label(y_sub, proba_sub, labels, N_CLASSES, min_n=min_n)
    return _augment_with_class_counts(stratum, y_sub, labels, min_n)


def cohort_terciles(ita_values: np.ndarray) -> np.ndarray:
    """Tercile labels of THIS cohort's own ITA distribution -- ``cohorte_t1/t2/t3``, nothing from the literature."""
    if ita_values.size == 0:
        return np.asarray([], dtype=object)
    cuts = np.percentile(ita_values, [100 / 3, 200 / 3])
    labels = np.where(
        ita_values <= cuts[0], "cohorte_t1_mas_oscuro",
        np.where(ita_values <= cuts[1], "cohorte_t2_medio", "cohorte_t3_mas_claro"),
    )
    return labels


def macro_f1_by_cohort_tercile(
    y_true: np.ndarray, proba: np.ndarray, ita_rows: list[dict], *, min_n: int = MIN_STRATUM_N
) -> dict:
    y_sub, proba_sub, ita_values, _ = _measurable_arrays(y_true, proba, ita_rows)
    if y_sub.size == 0:
        return {}
    labels = cohort_terciles(ita_values)
    stratum = metrics.stratified_by_label(y_sub, proba_sub, labels, N_CLASSES, min_n=min_n)
    return _augment_with_class_counts(stratum, y_sub, labels, min_n)


def correctness_ita_correlation(
    y_true: np.ndarray, proba: np.ndarray, ita_rows: list[dict]
) -> dict:
    """Spearman rho of per-image correctness against ITA in continuous degrees, with a bootstrap CI.

    Correctness is ``argmax(proba) == y_true`` per image -- a binary series,
    so Spearman reduces to the point-biserial rank correlation; reported
    anyway as Spearman because ITA itself is continuous and not assumed
    linear in anything.
    """
    y_sub, proba_sub, ita_values, _ = _measurable_arrays(y_true, proba, ita_rows)
    if y_sub.size < 2:
        return {"n": int(y_sub.size), "rho": None, "p_value": None, "ci95_low": None, "ci95_high": None}

    correct = (proba_sub.argmax(axis=1) == y_sub).astype(np.float64)
    rho, p_value = spearmanr(ita_values, correct)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = y_sub.size
    draws = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        if np.std(correct[idx]) == 0 or np.std(ita_values[idx]) == 0:
            continue
        r, _ = spearmanr(ita_values[idx], correct[idx])
        if r == r:  # skip NaN
            draws.append(r)
    if draws:
        lo, hi = np.percentile(draws, [2.5, 97.5])
    else:
        lo = hi = float("nan")

    return {
        "n": int(n),
        "rho": float(rho),
        "p_value": float(p_value),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def reachable_range_percentiles(ita_rows: list[dict]) -> dict | None:
    """Percentiles of the per-image reachable ITA span, when the manifest carries one.

    Optional: not every ITA manifest will have run tone_augment_lab_v2's
    solver (this module does not run it either -- it only reads
    ``reachable_ita_span`` if a record already has it, mirroring
    ``scripts/audit_pool_tone_eligibility.py``'s section 5).
    """
    spans = [r["reachable_ita_span"] for r in ita_rows if r.get("reachable_ita_span") is not None]
    if not spans:
        return None
    arr = np.asarray(spans, dtype=np.float64)
    return {f"p{p}": float(np.percentile(arr, p)) for p in (5, 25, 50, 75, 95)} | {"n": len(spans)}


def class_composition(y_true_full: np.ndarray, y_true_measurable: np.ndarray) -> dict:
    """Class mix of the full split next to the measurable subset, side by side.

    Required in the SAME output as any stratified metric, not a footnote:
    an evaluation that restricts to ITA-measurable images changes WHICH
    images it is measuring, and if that change concentrates or depletes the
    class the project exists to detect, no metric computed only on the
    subset can be read without this table next to it.
    """
    n_full = int(y_true_full.shape[0])
    n_sub = int(y_true_measurable.shape[0])
    full_counts = _per_class_counts(y_true_full)
    sub_counts = _per_class_counts(y_true_measurable)
    return {
        "n_full": n_full,
        "n_measurable": n_sub,
        "fraction_full": {c: full_counts[c] / n_full if n_full else None for c in HAM10000_CLASSES},
        "fraction_measurable": {c: sub_counts[c] / n_sub if n_sub else None for c in HAM10000_CLASSES},
        "counts_full": full_counts,
        "counts_measurable": sub_counts,
    }


NOT_COMPARABLE_CAVEAT = (
    "NO COMPARABLE con el macro-F1 por estrato de ITA: corre sobre el split completo "
    "(n y composicion de clases distintos al subconjunto medible). Ver class_composition."
)


def macro_f1_full_reference(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """The npz's own overall macro-F1, carrying its non-comparability warning IN the same dict.

    Never print this next to a per-stratum macro-F1 without the caveat
    attached -- which is why the caveat lives inside the return value
    itself, not in a comment the caller might drop.
    """
    pred = proba.argmax(axis=1)
    return {
        "macro_f1": metrics.macro_f1(y_true, pred, N_CLASSES),
        "n": int(y_true.shape[0]),
        "caveat": NOT_COMPARABLE_CAVEAT,
    }


def build_report(
    y_true: np.ndarray,
    proba: np.ndarray,
    ita_rows: list[dict],
    *,
    variant: str,
    min_n: int = MIN_STRATUM_N,
) -> dict:
    """Everything Fase 4 asked for, assembled: coverage, class composition, per-mask-source,
    per-cohort-tercile, correlation, the non-comparable full-split reference, verdict."""
    join_predictions_with_ita(y_true, ita_rows)
    coverage = coverage_report(ita_rows, n_total=y_true.shape[0])
    y_measurable, _, _, _ = _measurable_arrays(y_true, proba, ita_rows)
    composition = class_composition(y_true, y_measurable)
    by_mask_source = macro_f1_by_mask_source(y_true, proba, ita_rows, min_n=min_n)
    by_tercile = macro_f1_by_cohort_tercile(y_true, proba, ita_rows, min_n=min_n)
    correlation = correctness_ita_correlation(y_true, proba, ita_rows)
    reachable = reachable_range_percentiles(ita_rows)
    full_reference = macro_f1_full_reference(y_true, proba)

    thin_strata = [
        key
        for stratum in (by_mask_source, by_tercile)
        for key, entry in stratum.items()
        if entry.get("status") == "SKIPPED"
    ]
    thin_classes_by_stratum = {
        f"{group}/{key}": entry["classes_below_min_n"]
        for group, stratum in (("mask_source", by_mask_source), ("cohort_tercile", by_tercile))
        for key, entry in stratum.items()
        if entry.get("status") == "OK" and entry.get("classes_below_min_n")
    }
    return {
        "variant": variant,
        "classes": list(HAM10000_CLASSES),
        "coverage": coverage,
        "class_composition": composition,
        "macro_f1_by_mask_source": by_mask_source,
        "macro_f1_by_cohort_tercile": by_tercile,
        "correctness_vs_ita_spearman": correlation,
        "reachable_ita_span_percentiles": reachable,
        "macro_f1_full_split_reference": full_reference,
        "strata_with_insufficient_n": thin_strata,
        "strata_with_thin_classes": thin_classes_by_stratum,
    }
