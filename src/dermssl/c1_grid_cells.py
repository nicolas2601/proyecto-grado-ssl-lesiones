"""What the C1 grid does with a cell once its features exist.

Fit the probe, gate the cell, save it; reload the cells a previous launch
already finished; and contrast every condition against (0). None of it needs
a GPU, so all of it is tested without one -- the extraction that does need
one lives in :mod:`dermssl.c1_grid_inputs` and ``torch_c1_extraction``.
"""

from __future__ import annotations

import numpy as np

from dermssl import c1_cell_gates as cg
from dermssl import metrics as _metrics
from dermssl import paired_bootstrap as pb
from dermssl import probes

#: The question the professor actually asked, run before the other sixteen.
PRIORITY_BACKBONES = ("dinov2_public_vitb14", "ssl_proposed_vitb14")

#: (0) first: every contrast is measured against it, so a grid that dies early
#: still has the baseline every other cell needs.
CONDITION_ORDER = ("none", "bbox_crop", "neutral_bg", "mask_pool")

METRICS = ("macro_f1", "balanced_accuracy", "auc", "ece")

BASELINE_CONDITION = "none"

#: Fixed before the data: the contrast table is computed with these and only these.
CONTRAST_PROBE = "logreg"
N_RESAMPLES = 1000
RNG_SEED = 20260923


def non_finite_reasons(train_features, val_features, proba, metrics) -> list[str]:
    reasons = []
    for name, array in (("features train", train_features), ("features val", val_features),
                        ("proba", proba)):
        bad = int((~np.isfinite(np.asarray(array, dtype=np.float64))).sum())
        if bad:
            reasons.append(f"{bad} valores no finitos en {name}")
    bad_metrics = [k for k, v in metrics.items() if not np.isfinite(float(v))]
    if bad_metrics:
        reasons.append(f"metricas no finitas: {bad_metrics}")
    return reasons


def evaluate_cell(train_features, train_labels, val_features, val_labels, probe, n_classes):
    proba = probes.run_probe(probe, train_features, train_labels, val_features, n_classes)
    summary = _metrics.summarize(val_labels, np.asarray(proba), n_classes)
    return np.asarray(proba), summary


def _print_cell(reasons: list[str], cell_metrics: dict) -> None:
    for reason in reasons:
        print(f"  GATE: {reason}")
    print(f"  macro_f1 {cell_metrics['macro_f1']:.4f} | "
          f"bal_acc {cell_metrics['balanced_accuracy']:.4f} | "
          f"auc {cell_metrics['auc']:.4f} | ece {cell_metrics['ece']:.4f}")


def fit_and_save_cell(store, backbone, condition, *, train_features, train_labels, val_features,
                      val_labels, train_tally, val_tally, probe, n_classes, provenance) -> dict:
    """Fit the probe, gate the cell and save it; the train array hashed is the one fitted on."""
    # Review (both reviewers): the TRAIN side is gated too. The probe is fitted on
    # train, so a degraded train set contaminates every val number.
    val_passes, val_reasons = cg.evaluate_cell_gates(val_tally)
    train_passes, train_reasons = cg.evaluate_cell_gates(train_tally)
    reasons = [f"val: {r}" for r in val_reasons] + [f"train: {r}" for r in train_reasons]
    cell_summary = cg.summarise_cell(val_tally)

    proba, cell_metrics = evaluate_cell(
        train_features, train_labels, val_features, val_labels, probe, n_classes
    )
    # Review (opencode): a NaN in features, proba or a metric compares
    # False against every threshold and serialises into the JSON as if it
    # were a number. Non-finite anything fails the cell.
    finite_reasons = non_finite_reasons(train_features, val_features, proba, cell_metrics)
    reasons += finite_reasons
    passes = val_passes and train_passes and not finite_reasons
    cell_summary = {**cell_summary, "gate_passes": passes, "gate_reasons": reasons}
    _print_cell(reasons, cell_metrics)

    store.save(
        backbone, condition, val_features,
        proba=proba,
        train_features=train_features,
        provenance=provenance,
        extra={
            "gate": cell_summary,
            "train_gate": cg.summarise_cell(train_tally),
            "metrics": {k: float(v) for k, v in cell_metrics.items()},
            "n_classes": n_classes,
        },
    )
    return {"proba": proba, "metrics": cell_metrics, "gate": cell_summary, "passes": passes}


def load_done_cells(store, done, provenance_for) -> dict[tuple[str, str], dict]:
    """The cells a previous launch finished, in the shape a fresh cell has."""
    results: dict[tuple[str, str], dict] = {}
    for backbone, condition in done:
        cell = store.load(backbone, condition, provenance=provenance_for(backbone))
        gate = cell.manifest.get("gate", {})
        results[(backbone, condition)] = {
            "proba": cell.proba,
            "gate": gate,
            "metrics": cell.manifest.get("metrics", {}),
            "passes": bool(gate.get("gate_passes", False)),
        }
        print(f"  REUSADA {backbone} / {condition}")
    return results


def print_cell_table(results: dict) -> None:
    print("\n=== celdas ===")
    for (backbone, condition), payload in sorted(results.items()):
        print(f"  {backbone} / {condition}: {'PASA' if payload['passes'] else 'FRENA'}")


def _contrast_records(backbone, condition, payload, baseline, val_labels) -> list[dict]:
    return [
        pb.contrast_record(
            arm=f"{backbone}__{condition}",
            control=f"{backbone}__{BASELINE_CONDITION}",
            metric=metric,
            probe=CONTRAST_PROBE,
            fraction=1.0,
            seed=0,
            y_true=np.asarray(val_labels),
            arm_proba=payload["proba"],
            control_proba=baseline["proba"],
        )
        for metric in METRICS
    ]


def build_contrasts(results, val_labels, index) -> dict:
    """Every condition against (0), per backbone, paired.

    Only cells that PASSED their gate enter a contrast. A degraded cell still
    produces a number, and contrasting it would compare the baseline against
    something that is partly the baseline -- a tie that means nothing and
    reads like a finding.
    """
    records = []
    skipped = []
    for (backbone, condition), payload in sorted(results.items()):
        if condition == BASELINE_CONDITION:
            continue
        baseline = results.get((backbone, BASELINE_CONDITION))
        if baseline is None:
            skipped.append(f"{backbone}/{condition}: no hay celda (0) con la que contrastar")
            continue
        if not payload["passes"] or not baseline["passes"]:
            skipped.append(f"{backbone}/{condition}: una de las dos celdas no paso su gate")
            continue
        records += _contrast_records(backbone, condition, payload, baseline, val_labels)

    document = pb.bootstrap_document(
        records,
        n_resamples=N_RESAMPLES,
        rng_seed=RNG_SEED,
        probe=CONTRAST_PROBE,
        fraction=1.0,
        seed_used=0,
        n_val=int(len(val_labels)),
        val_order_sha256=index["c1_val_ids_sha256"],
    )
    # Recorded, not dropped: a contrast that could not be computed is a fact
    # about the grid, and an absent row would read as one that was never asked.
    document["not_contrasted"] = skipped
    return document
