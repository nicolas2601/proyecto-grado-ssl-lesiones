"""C1 contrasts BETWEEN backbones, and the crop-retention stratification.

The per-backbone table answers "does segmentation change this backbone". The
professor's question is a different one: "does the public DINOv2 WITH
segmentation reach the proposed backbone". That is a contrast across
backbones, and it is still paired: every cell was scored on the same 2004
validation images in the same order. Pairing is not assumed, it is checked on
the manifests (same val ids, same labels, same masks, same probe) before a
single resample is drawn.

The stratification exists for the confound declared in docs/27: condition (a)
``bbox_crop`` does not go through the bank centre crop, so for lesions near
the border it sees more lesion than (0), (b) and (c). Splitting by
``mask_area_retained_after_crop`` (>= 0.9 vs < 0.9) shows whether an effect
lives where the crop cut the lesion or where it did not.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from dermssl import paired_bootstrap as pb

PUBLIC = "dinov2_public_vitb14"
PROPOSED = "ssl_proposed_vitb14"

#: Fixed before looking at any cross-backbone number. The first row is the
#: professor's question; the last one is the reference that must reproduce
#: the sign already published in v6 (public above proposed in macro-F1).
CROSS_PAIRS: tuple[tuple[str, str], ...] = (
    (f"{PUBLIC}__mask_pool", f"{PROPOSED}__none"),
    (f"{PUBLIC}__bbox_crop", f"{PROPOSED}__none"),
    (f"{PUBLIC}__neutral_bg", f"{PROPOSED}__none"),
    (f"{PUBLIC}__none", f"{PROPOSED}__none"),
)

#: Both backbones under the SAME condition: segmentation on both sides, so
#: the gap is compared with the input held equal. (0) vs (0) is already the
#: reference row above and is not repeated.
SAME_CONDITION_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (f"{PUBLIC}__{condition}", f"{PROPOSED}__{condition}")
    for condition in ("bbox_crop", "neutral_bg", "mask_pool")
)

ALL_CROSS_PAIRS = CROSS_PAIRS + SAME_CONDITION_PAIRS

METRICS = ("macro_f1", "balanced_accuracy", "auc", "ece")
RETAINED_THRESHOLD = 0.9
#: Below this a stratum is reported by size only: a bootstrap over a handful
#: of images produces an interval that looks like a result and is not one.
MIN_STRATUM = 30

#: What must be identical for two cells to be compared row by row.
PAIRING_FIELDS = ("c1_val_ids_sha256", "labels_sha256", "masks_sha256", "probe")


def check_pairable(manifests: Mapping[str, dict]) -> None:
    """Refuse cells that were not scored on the same images and labels."""
    if not manifests:
        raise ValueError("no hay celdas: un contraste sin celdas es un no-op silencioso")
    for field in PAIRING_FIELDS:
        values = {name: manifest.get(field) for name, manifest in manifests.items()}
        if None in values.values() or len(set(values.values())) != 1:
            raise ValueError(
                f"las celdas no son pareables en {field}: {values}. Un bootstrap "
                "pareado sobre filas que no son las mismas imagenes mide ruido"
            )


def contrast_pairs(
    probas: Mapping[str, np.ndarray],
    passes: Mapping[str, bool],
    y_true: np.ndarray,
    pairs: Sequence[tuple[str, str]],
) -> tuple[list[dict], list[str]]:
    """Paired records for every pair whose two cells exist and passed."""
    records: list[dict] = []
    skipped: list[str] = []
    for arm, control in pairs:
        missing = [name for name in (arm, control) if name not in probas]
        if missing:
            skipped.append(f"{arm} vs {control}: falta {missing}")
            continue
        if not (passes.get(arm) and passes.get(control)):
            skipped.append(f"{arm} vs {control}: una de las dos celdas no paso su gate")
            continue
        for metric in METRICS:
            try:
                records.append(
                    pb.contrast_record(
                        arm=arm, control=control, metric=metric, probe="logreg",
                        fraction=1.0, seed=0, y_true=np.asarray(y_true),
                        arm_proba=probas[arm], control_proba=probas[control],
                    )
                )
            except ValueError as error:
                # Small strata can miss a class; that is reported, never dropped.
                skipped.append(f"{arm} vs {control} [{metric}]: no calculable ({error})")
    return records, skipped


def retention_strata(retained: np.ndarray, threshold: float = RETAINED_THRESHOLD) -> dict:
    """Row indices per stratum. An empty mask (NaN) is its own stratum."""
    retained = np.asarray(retained, dtype=np.float64)
    empty = np.isnan(retained)
    return {
        f"retained_ge_{threshold}": np.flatnonzero(~empty & (retained >= threshold)),
        f"retained_lt_{threshold}": np.flatnonzero(~empty & (retained < threshold)),
        "empty_mask": np.flatnonzero(empty),
    }


def within_backbone_pairs(cells: Sequence[str]) -> list[tuple[str, str]]:
    """Every condition against the (0) of its own backbone."""
    pairs = []
    for name in sorted(cells):
        backbone, condition = name.rsplit("__", 1)
        if condition != "none":
            pairs.append((name, f"{backbone}__none"))
    return pairs


def cell_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    out = {}
    for metric in METRICS:
        try:
            out[metric] = float(pb.METRIC_FUNCTIONS[metric](y_true, proba))
        except ValueError:
            out[metric] = None
    return out


def stratified_summary(
    probas: Mapping[str, np.ndarray],
    passes: Mapping[str, bool],
    y_true: np.ndarray,
    retained: np.ndarray,
) -> list[dict]:
    """Per stratum: its size, every cell's metrics, and the paired contrasts."""
    y_true = np.asarray(y_true)
    if len(retained) != len(y_true):
        raise ValueError(
            f"retained tiene {len(retained)} filas y y_true {len(y_true)}: no es el "
            "mismo orden de validacion"
        )
    pairs = within_backbone_pairs(list(probas)) + list(ALL_CROSS_PAIRS)
    out = []
    for stratum, rows in retention_strata(retained).items():
        entry = {"stratum": stratum, "n": int(len(rows))}
        if len(rows) < MIN_STRATUM:
            reason = f"{stratum}: n={len(rows)} < {MIN_STRATUM}, un IC ahi no dice nada"
            entry.update(cells={}, results=[], not_contrasted=[reason])
            out.append(entry)
            continue
        sub_y = y_true[rows]
        sub = {name: np.asarray(proba)[rows] for name, proba in probas.items()}
        records, skipped = contrast_pairs(sub, passes, sub_y, pairs)
        entry.update(
            cells={name: cell_metrics(sub_y, proba) for name, proba in sorted(sub.items())},
            results=records,
            not_contrasted=skipped,
        )
        out.append(entry)
    return out
