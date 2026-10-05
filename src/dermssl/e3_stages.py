"""The `select` and `final` stages of E3 (docs/33 section 3), on already extracted features.

``select_stage`` scores every variant on the HAM10000 holdout (probe fitted on the fit split),
applies the registered adoption rule and returns the selection document. It receives a
:class:`FeatureReader` that refuses ``val`` and ``train`` and labels through lazy callables
that the tests replace with tripwires: it cannot open validation by construction.

``final_stage`` needs the selection JSON AND its pinned sha256 before it touches anything,
checks the feature files exist, takes the one-shot :class:`~dermssl.e3_store.ValLock`, and
only then reads validation labels: probe fitted on the full train, B71 and the planned
variant scored on val, paired bootstrap (seed 20260923, grouped by lesion) of each variant
against B71 and against the no-mask reference.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dermssl import c1_luminance as lum
from dermssl import e3_descriptors as desc
from dermssl import e3_seed as seed_map
from dermssl import e3_selection as sel
from dermssl import e3_store as st
from dermssl import e3_variants as var
from dermssl import paired_bootstrap as pb
from dermssl import v71_metrics as vm

BOOTSTRAP_SEED = 20260923
BOOTSTRAP_RESAMPLES = 1000
CONTRAST_METRICS = ("macro_f1", "balanced_accuracy", "mel_recall")
PROBE_NAME = "logreg_balanced"
INVARIANT_KEYS = ("protocol", "masks_sha256", "weights_sha256", "manifest_sha256", "holdout_sha256")
SEED_BOUND_KEYS = ("protocol", "weights_sha256")  # the only ones a declared B* seed may change (e3_seed)


@dataclass(frozen=True)
class Run:
    """What the stages need; labels, groups and terciles are lazy so a stage reads only its splits."""

    feats_dir: Path
    n_classes: int
    mel: int
    labels_of: Callable[[str], np.ndarray]
    groups_of: Callable[[str], np.ndarray]
    codes_of: Callable[[str], np.ndarray]
    fingerprint: dict = field(default_factory=dict)


def _load(reader: st.FeatureReader, split: str, name: str) -> dict:
    return {c: reader.load(split, c) for c in var.required_conditions(name)}


def _descriptors(reader, fit_split: str, eval_split: str, name: str):
    if name not in var.WITH_DESCRIPTORS:
        return None, None
    raw_fit, raw_eval = (np.asarray(reader.load(s, "descriptors")) for s in (fit_split, eval_split))
    z_fit, z_eval, _ = desc.standardize(raw_fit, raw_eval)
    return z_fit, z_eval


def _proba(run: Run, reader, fit_split: str, eval_split: str, name: str, y_fit) -> np.ndarray:
    d_fit, d_eval = _descriptors(reader, fit_split, eval_split, name)
    return var.variant_proba(name, _load(reader, fit_split, name), y_fit, _load(reader, eval_split, name),
                             run.n_classes, desc_fit=d_fit, desc_eval=d_eval)


def val_plan(selection: dict, *, report_candidate: str) -> list[str]:
    """B71, the best no-mask row, and the winner (or, with no adoption, the best candidate, report only)."""
    plan = _controls(selection)
    plan.append(selection["winner"] or report_candidate)
    return plan


def _controls(selection: dict) -> list[str]:
    """Reference and best no-mask row; a no-mask row that is not one of our variants (a v7.1 grid
    row added by the caller) is a bar for the rule only and is not re-evaluated on val."""
    names = dict.fromkeys([selection["reference"], selection["best_nomask"]])
    return [n for n in names if n in var.PARTS]


def select_stage(run: Run, cells: st.CellStore, *, variants, n_resamples: int, nomask_rows: dict | None = None) -> dict:
    """Holdout rows for every variant (one resumable cell each) and the registered decision."""
    reader = st.FeatureReader(run.feats_dir, forbidden=("val", "train"))
    y_fit, y_hold, codes = run.labels_of("fit"), run.labels_of("holdout"), run.codes_of("holdout")
    for name in variants:
        key = {"stage": "select", "variant": name, "n_resamples": n_resamples, **run.fingerprint}
        cells.run(f"select__{name}", key, lambda n=name: _variant_row(run, reader, n, y_fit, y_hold, codes, n_resamples))
    rows = {n: cells.read(f"select__{n}")["result"] for n in variants if cells.path(f"select__{n}").is_file()}
    failed = [n for n, _ in cells.failures]
    return _selection_doc(run, rows, failed, nomask_rows or {}, n_resamples)


def _variant_row(run, reader, name, y_fit, y_hold, codes, n_resamples) -> dict:
    proba = _proba(run, reader, "fit", "holdout", name, y_fit)
    return sel.score_variants(y_hold, {name: proba}, codes, mel=run.mel, n_resamples=n_resamples)[name]


def _selection_doc(run: Run, rows: dict, failed: list, extra_nomask: dict, n_resamples: int) -> dict:
    if var.REFERENCE not in rows:
        raise sel.SelectionError(f"no hay fila del holdout para la referencia {var.REFERENCE}: no se puede decidir")
    nomask = [var.REFERENCE, *extra_nomask]
    verdict = sel.adoption({**rows, **extra_nomask}, nomask=nomask)
    candidates = [n for n in rows if n != var.REFERENCE]
    report = sel.strict_winner(rows, candidates) if candidates else var.REFERENCE
    doc = {"stage": "select", "split": "holdout", "rule": sel.RULE, "manifest_sha256": sel.MANIFEST_SHA256,
           "n_resamples": n_resamples, "fingerprint": run.fingerprint, "rows": rows, "nomask_extra": extra_nomask,
           "adoption": verdict, "decision": verdict["decision"], "winner": verdict["winner"],
           "reference": verdict["reference"], "best_nomask": verdict["best_nomask"],
           "report_candidate": report, "failed": failed, "complete": not failed}
    doc["val_plan"] = val_plan(doc, report_candidate=report)
    return doc


def _grouped_ci(y, proba, groups, fns: dict, n_resamples: int) -> dict:
    rng, draw = np.random.RandomState(BOOTSTRAP_SEED), pb.index_sampler(len(y), groups)
    values = {m: [] for m in fns}
    for _ in range(n_resamples):
        picks = draw(rng)
        for metric, fn in fns.items():
            values[metric].append(fn(y[picks], proba[picks]))
    return {m: vm.percentile_interval(v) for m, v in values.items()}


def _metric_fns(y, n_classes: int, mel: int) -> dict:
    """The contrast metrics; ``mel_recall`` is left out when the rows hold no melanoma (undefined)."""
    fns = {m: pb.METRIC_FUNCTIONS[m] for m in ("macro_f1", "balanced_accuracy")}
    fns.update({m: f for m, f in lum.metric_functions(y, n_classes=n_classes, mel=mel).items() if m == "mel_recall"})
    return fns


def _contrasts(plan, controls, probas, y, groups, codes, *, mel, n_classes, n_resamples) -> list[dict]:
    dark = np.asarray(codes) == sel.DARK
    subsets = (("all", np.ones(len(y), bool)), ("dark_tercile", dark))
    return [record for arm in plan if arm not in controls for control in controls for subset, rows in subsets
            for record in _subset_records(arm, control, subset, rows, probas, y, groups, mel, n_classes, n_resamples)]


def _subset_records(arm, control, subset, rows, probas, y, groups, mel, n_classes, n_resamples) -> list[dict]:
    if not rows.any():
        return []
    ys = y[rows]
    return [{**pb.contrast_record(arm=arm, control=control, metric=m, probe=PROBE_NAME, fraction=1.0, seed=42,
                                  y_true=ys, arm_proba=probas[arm][rows], control_proba=probas[control][rows],
                                  n_resamples=n_resamples, rng_seed=BOOTSTRAP_SEED, metric_fn=fn,
                                  groups=np.asarray(groups)[rows]), "subset": subset}
            for m, fn in _metric_fns(ys, n_classes, mel).items()]


def final_stage(run: Run, *, selection_path, pinned_sha256: str, lock: st.ValLock, n_resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """The single validation read; see the module docstring for the order of the refusals."""
    selection, sha = st.require_selection(selection_path, pinned_sha256)
    if not selection.get("complete"):
        raise st.SelectionPinError("la seleccion esta incompleta (celdas fallidas): no se lee val")
    _check_fingerprint(selection, run)
    if lock.taken():
        raise st.ValAlreadyReadError(f"val ya se leyo ({lock.path})")
    plan = selection["val_plan"]
    reader = st.FeatureReader(run.feats_dir)
    for name in plan:  # preflight: every array exists before the lock is taken
        _load(reader, "train", name), _load(reader, "val", name)
        _descriptors(reader, "train", "val", name)
    lock.acquire(sha)
    return _evaluate(run, selection, sha, plan, reader, n_resamples)


def _check_fingerprint(selection: dict, run: Run) -> None:
    other_seed = run.fingerprint.get("b71_seed", seed_map.BASE_SEED) != seed_map.BASE_SEED
    for key in INVARIANT_KEYS:
        if other_seed and key in SEED_BOUND_KEYS:
            continue
        if selection.get("fingerprint", {}).get(key) != run.fingerprint.get(key):
            raise st.SelectionPinError(f"la corrida de final no coincide con la seleccion en {key!r}")


def _evaluate(run, selection, sha, plan, reader, n_resamples) -> dict:
    y_train, y_val = run.labels_of("train"), run.labels_of("val")
    groups, codes = run.groups_of("val"), run.codes_of("val")
    probas = {n: _proba(run, reader, "train", "val", n, y_train) for n in plan}
    fns = _metric_fns(y_val, run.n_classes, run.mel)
    controls = _controls(selection)
    rows = {n: {**vm.metric_bundle(y_val, p, mel=run.mel), "ci95_grouped": _grouped_ci(y_val, p, groups, fns, n_resamples)}
            for n, p in probas.items()}
    return {"stage": "final", "split": "val", "selection_sha256": sha, "decision": selection["decision"],
            "adopted": selection["decision"] == "adopt", "val_plan": plan, "fingerprint": run.fingerprint,
            "bootstrap": {"n_resamples": n_resamples, "seed": BOOTSTRAP_SEED, "grouped_by": "lesion_id"},
            "controls": controls, "rows": rows,
            "contrasts": _contrasts(plan, controls, probas, y_val, groups, codes, mel=run.mel,
                                    n_classes=run.n_classes, n_resamples=n_resamples)}
