"""MCC contrasts of an E3 final evaluation, from a probe refit on the cached features (CPU, no val read).

E3 never stored its validation probabilities. The final stage is deterministic given the cached
``train`` / ``val`` feature arrays, so the probabilities can be refitted and the contrasts recomputed
with the stage's own code (:func:`dermssl.e3_stages._contrasts`: same bootstrap, seed 20260923,
grouped by lesion, MCC last so the other metrics' draws do not move).

A refit only counts where it reproduces what the run stored: the macro-F1, balanced accuracy and
melanoma recall of each variant within :data:`TOLERANCE`, and every pre-existing contrast record of
a pair. Where it does not, the MCC record is written as unavailable with :data:`UNAVAILABLE_REASON`
and no number: an approximate MCC is never published. The stored records are kept as they are.

Validation labels are NOT read here: ``y_val`` is passed in (the caller takes it from the artifact
that already holds it, ``e3b_probas.npz``), train labels come from the training split, and the
``val_read.lock`` of the original run is neither read nor written (:func:`check_out_dir`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dermssl import e3_stages as sg
from dermssl import run_environment
from dermssl import e3_store as st
from dermssl import v71_metrics as vm

TOLERANCE = 1e-4
POINT_METRICS = ("macro_f1", "balanced_accuracy", "mel_recall")
UNAVAILABLE_REASON = "no disponible: el refit no reproduce los puntos registrados"
MCC = "mcc"


class RebuildInputError(st.E3Error):
    """An input of the rebuild is not what the original run used."""


def check_out_dir(out_dir: Path, final_path: Path) -> None:
    """The rebuild writes to its own directory: never next to the original final eval, never over a lock."""
    out_dir = Path(out_dir)
    if out_dir.resolve() == Path(final_path).resolve().parent:
        raise RebuildInputError(f"{out_dir} holds the original final eval: write to your own out dir")
    if (out_dir / st.LOCK_NAME).exists():
        raise RebuildInputError(f"{out_dir} holds a {st.LOCK_NAME}: that directory belongs to a run that read val")


def _no_labels(split: str):
    raise AssertionError(f"the rebuild never asks the dataset for the labels of {split!r}")


def refit_probas(feats_dir: Path, plan: list[str], y_train, *, n_classes: int, mel: int) -> dict:
    """Probabilities on val of every planned variant, fitted on the full train (the final stage's path)."""
    run = sg.Run(feats_dir=Path(feats_dir), n_classes=n_classes, mel=mel, labels_of=_no_labels,
                 groups_of=_no_labels, codes_of=_no_labels)
    reader = st.FeatureReader(Path(feats_dir))
    return {name: sg._proba(run, reader, "train", "val", name, y_train) for name in plan}


def reproduced_cells(final_doc: dict, probas: dict, y_val, mel: int) -> tuple[dict, dict]:
    """``({variant: bool}, {variant: max |refit - stored|})`` over the three registered point metrics."""
    ok, gap = {}, {}
    for name, proba in probas.items():
        bundle, stored = vm.metric_bundle(y_val, proba, mel=mel), final_doc["rows"][name]
        gap[name] = max(abs(bundle[m] - stored[m]) for m in POINT_METRICS)
        ok[name] = gap[name] <= TOLERANCE
    return ok, gap


def _same(stored: dict, fresh: dict) -> bool:
    for key, value in fresh.items():
        other = stored.get(key)
        same = abs(value - other) <= TOLERANCE if isinstance(value, float) and isinstance(other, float) else value == other
        if not same:
            return False
    return True


def _pair_key(record: dict) -> tuple:
    return record["arm"], record["control"], record["subset"]


def _unavailable(record: dict) -> dict:
    return {"arm": record["arm"], "control": record["control"], "metric": MCC, "subset": record["subset"],
            "available": False, "reason": UNAVAILABLE_REASON}


def rebuild(final_doc: dict, feats_dir: Path, *, y_train, y_val, groups, codes, n_classes: int, mel: int,
            n_resamples: int | None = None) -> tuple[list[dict], dict]:
    """``(contrasts, report)``: stored records unchanged, one MCC record per pair (a number or unavailable)."""
    plan, controls = list(final_doc["val_plan"]), list(final_doc["controls"])
    n_resamples = n_resamples or final_doc["bootstrap"]["n_resamples"]
    y_val = np.asarray(y_val)
    probas = refit_probas(feats_dir, plan, y_train, n_classes=n_classes, mel=mel)
    cells, gap = reproduced_cells(final_doc, probas, y_val, mel)
    fresh = sg._contrasts(plan, controls, probas, y_val, groups, codes, mel=mel, n_classes=n_classes,
                          n_resamples=n_resamples)
    stored = {(*_pair_key(r), r["metric"]): r for r in final_doc["contrasts"] if r["metric"] != MCC}
    records, pair_ok = [], {}
    for record in fresh:
        pair = _pair_key(record)
        if record["metric"] != MCC:
            old = stored.get((*pair, record["metric"]))
            if old is None:
                raise RebuildInputError(f"the stored final eval has no record for {pair} / {record['metric']}")
            pair_ok[pair] = pair_ok.get(pair, True) and _same(old, record)
            records.append(old)
            continue
        usable = pair_ok[pair] and cells[record["arm"]] and cells[record["control"]]
        records.append(record if usable else _unavailable(record))
    mcc = [r for r in records if r["metric"] == MCC]
    return records, {"tolerance": TOLERANCE, "cells": cells, "cell_max_abs_diff": gap, "n_resamples": n_resamples,
                     "records_compared": len(stored), "mcc_available": sum("available" not in r for r in mcc),
                     "mcc_unavailable": sum("available" in r for r in mcc)}


def verify_feature_hashes(final_doc: dict, feats_dir: Path) -> None:
    """The cached arrays are the ones the run fingerprinted (train/val x condition, plus the descriptors)."""
    expected = final_doc.get("fingerprint", {}).get("feature_sha256", {})
    for split, by_condition in expected.items():
        for condition, sha in by_condition.items():
            path = st.FeatureReader(feats_dir).path(split, condition)
            if not path.is_file():
                continue  # an array the plan does not use (e.g. fallbacks) is not needed by the refit
            if st.file_sha256(path) != sha:
                raise RebuildInputError(f"{split}/{condition}: {path.name} is not the array the run fingerprinted")


def write_output(out_dir: Path, final_doc: dict, records: list[dict], report: dict) -> Path:
    """``<out_dir>/e3_final_eval.json``: the stored document with its contrasts replaced, plus its sidecar."""
    out = Path(out_dir) / "e3_final_eval.json"
    st.write_json_atomic(out, {**final_doc, "contrasts": records, "contrasts_rebuild": report})
    run_environment.write_sidecar(out)
    return out

