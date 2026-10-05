"""Paired contrasts of E3b on the shared validation set (docs/33 E3b).

Arms are named ``<model>`` (full image), ``<model>+F4``, and the proposed arm of the bank,
B71, in both of its protocols: ``B71`` / ``B71+F4`` with the baselines' single view and
probe (apples to apples) and ``B71_tta`` / ``B71+F4_tta`` with its registered cls4 + D4 TTA.
The bootstrap is the repo's paired one: seed 20260923, resampled by ``lesion_id``.
"""

from __future__ import annotations

import numpy as np

from dermssl import e3_stages as stages
from dermssl import paired_bootstrap as pb

PROBE_NAME = "logreg_balanced"


def pairs_for(model: str) -> list[tuple[str, str, str]]:
    """``(arm, control, family)`` for one baseline: the F4 effect and each side against B71."""
    seg = f"{model}+F4"
    return [
        (seg, model, "seg_effect"),
        (model, "B71", "vs_proposed_notta"),
        (seg, "B71+F4", "vs_proposed_seg_notta"),
        (model, "B71_tta", "vs_proposed_registered"),
        (seg, "B71+F4_tta", "vs_proposed_seg_registered"),
    ]


def paired(y, probas: dict, pairs, groups, *, n_classes: int, mel: int, n_resamples: int) -> list[dict]:
    """One record per pair and metric (macro-F1, balanced accuracy, melanoma recall)."""
    y, fns = np.asarray(y), stages._metric_fns(np.asarray(y), n_classes, mel)
    records = []
    for arm, control, family in pairs:
        for metric, fn in fns.items():
            record = pb.contrast_record(
                arm=arm, control=control, metric=metric, probe=PROBE_NAME, fraction=1.0, seed=42, y_true=y,
                arm_proba=probas[arm], control_proba=probas[control], n_resamples=n_resamples,
                rng_seed=stages.BOOTSTRAP_SEED, metric_fn=fn, groups=np.asarray(groups))
            records.append({**record, "family": family, "n_resamples": n_resamples,
                           "rng_seed": stages.BOOTSTRAP_SEED})
    return records
