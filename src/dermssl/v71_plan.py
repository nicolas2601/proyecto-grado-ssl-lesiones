"""The v7.1 run plan (docs/29 §1, §2.1, §2.4, §2.7): checks, selection, val read, cost.

Pure decisions, kept apart from the GPU script so each one is tested:

* the holdout is the docs/28 §4.1 one (digests match AND it is what
  ``selection_holdout.grouped_holdout`` rebuilds from the C1 train), with no
  validation id in it;
* the §2.1 pre-test: alpha = 0 must be public and alpha = 1 the final, in the
  weights and in every holdout metric, within 1e-6;
* per arm, the docs/28 §4.1 rule over that arm's configurations; public gets
  its own winner too, reported as a secondary;
* the single val read: each arm's winner, public under the SAME protocol as
  that winner, the v7 primary of every arm, and public's own winner.
"""

from __future__ import annotations

from dermssl import c1_split_identity as si
from dermssl import selection_holdout as sh
from dermssl import v71_protocol as vp

PRETEST_TOLERANCE = 1e-6
VAL_LABELS = ("A71", "B71", "public_A71", "public_B71", "A_v7", "B_v7", "public_v7", "public71")
VAL_CONTRASTS = (
    ("A71", "public_A71"), ("B71", "public_B71"),  # the weights, protocol held fixed
    ("A71", "A_v7"), ("B71", "B_v7"),  # what v7.1 adds to the v7 primary
    ("A71", "public_v7"), ("B71", "public_v7"),  # against the v7 reference
    ("B71", "A71"), ("public71", "public_v7"),
)
# Cost model. Throughputs are assumptions for ViT-B/14 under autocast on one
# RTX PRO 4000 and are re-measured by the --limit smoke (it prints img/s);
# the bootstrap figure was measured on 2026-09-28 (1602 rows, 1000 resamples).
FORWARD_IMG_PER_S = {224: 700.0, 448: 130.0}
DECODE_IMG_PER_S_PER_CORE = 60.0
LOGREG_SECONDS_PER_1K_DIMS = 15.0  # per fit on LOGREG_REFERENCE_ROWS rows, linear in rows
LOGREG_REFERENCE_ROWS = 6409
BOOTSTRAP_SECONDS_PER_CELL = 2.5
BOOTSTRAP_REFERENCE_ROWS = 1602
FEATURE_DIMS = {"cls": 768, "cls_mean": 1536, "cls4": 3072, "cls_gem": 1536, "cls_attn": 1536}
PRETEST_MODELS = 7  # public, A, B directly; A and B at alpha 0 and 1 through WiSE
VAL_UNITS_UPPER_BOUND = 5  # A71, B71, A_v7, B_v7, public: at most five weight sets


def check_holdout(doc: dict, train_ids, labels, groups, *, val_ids) -> str:
    """The docs/28 holdout or an exception; returns its ``holdout_ids_sha256``."""
    for key in ("fit_ids", "holdout_ids"):
        if si.ids_sha256(doc[key]) != doc[f"{key}_sha256"]:
            raise ValueError(f"{key}: el digest guardado no coincide con los ids")
    leaked = set(val_ids) & (set(doc["fit_ids"]) | set(doc["holdout_ids"]))
    if leaked:
        raise ValueError(f"{len(leaked)} ids de validacion en el holdout")
    fit, holdout = sh.grouped_holdout(list(train_ids), list(labels), list(groups), forbidden=val_ids)
    if fit != list(doc["fit_ids"]) or holdout != list(doc["holdout_ids"]):
        raise ValueError("el holdout no es el de docs/28 §4.1 reconstruido desde el train de C1")
    return doc["holdout_ids_sha256"]


def limited_ids(ids, labels, n: int | None) -> list:
    """First ``n`` ids taken round-robin over the classes (smoke runs), else all."""
    ids = list(ids)
    if n is None or n >= len(ids):
        return ids
    queues = {c: [i for i, lab in zip(ids, labels) if lab == c] for c in sorted(set(labels))}
    picked: list = []
    while len(picked) < n:
        for queue in queues.values():
            if queue and len(picked) < n:
                picked.append(queue.pop(0))
    order = {image_id: k for k, image_id in enumerate(ids)}
    return sorted(picked, key=order.__getitem__)


def pretest_check(arm: str, alpha: float, direct: dict, wise: dict, *, weights_diff: float,
                  tol: float = PRETEST_TOLERANCE) -> dict:
    deltas = {k: abs(float(direct[k]) - float(wise[k])) for k in direct}
    passed = weights_diff == 0.0 and all(d <= tol for d in deltas.values())
    return {"arm": arm, "alpha": alpha, "weights_max_abs_diff": weights_diff,
            "metric_abs_deltas": deltas, "tolerance": tol, "passed": bool(passed)}


def build_selection(grid: dict) -> dict:
    """Winners per arm (and public's own), with the §2.7 item 7 phase-2 condition."""
    winners = {arm: vp.select_for_arm(grid, arm) for arm in (*vp.ARMS, vp.PUBLIC)}
    mel = {arm: grid[winners[arm]]["mel_sens_at_spec80"] for arm in vp.ARMS}
    return {
        "rule": f"max macro_f1; within {sh.TIE_MACRO_F1} of the best, lowest ece (docs/28 4.1), per arm",
        "winners": winners,
        "winner_rows": {arm: grid[cid] for arm, cid in winners.items()},
        "n_candidates": {arm: sum(vp.arm_of(vp.parse_config_id(c)["weights"]) == arm for c in grid)
                         for arm in winners},
        "phase2": {"mel_sens_at_spec80_holdout": mel, "threshold": vp.PHASE2_MEL_SENS_AT_SPEC80,
                   "enabled": vp.phase2_enabled(mel),
                   "note": "habilita PROPONER la fase 2 (docs/29 §3); no se lanza sin el OK de Paula"},
    }


def val_entries(selection: dict) -> dict:
    winners = selection["winners"]
    return {
        "A71": winners["A"], "B71": winners["B"],
        "public_A71": vp.with_weights(winners["A"], vp.PUBLIC),
        "public_B71": vp.with_weights(winners["B"], vp.PUBLIC),
        "A_v7": vp.primary_config("A"), "B_v7": vp.primary_config("B"),
        "public_v7": vp.primary_config(vp.PUBLIC), "public71": winners[vp.PUBLIC],
    }


def val_units(entries: dict) -> dict:
    """``(weights, resolution) -> {features, tta}`` to extract for the val read."""
    units: dict = {}
    for cid in entries.values():
        cfg = vp.parse_config_id(cid)
        unit = units.setdefault((cfg["weights"], cfg["resolution"]), {"features": set(), "tta": False})
        unit["features"].add(cfg["feature"])
        unit["tta"] = unit["tta"] or cfg["tta"]
    return units


def cost_projection(*, n_fit: int, n_hold: int, n_train: int, n_val: int, resolutions,
                    n_weights: int, val_units: int = VAL_UNITS_UPPER_BOUND) -> dict:
    """GPU-h of extraction and CPU-h of decoding, probes and bootstrap (upper bounds for val)."""
    views = 8
    forwards = {
        "pretest": {224: PRETEST_MODELS * (n_fit + n_hold)},
        "grid": {r: n_weights * (n_fit + views * n_hold) for r in resolutions},
        "val": {r: val_units * (n_train + views * n_val) for r in resolutions},
    }
    gpu_s = sum(n / FORWARD_IMG_PER_S[r] for stage in forwards.values() for r, n in stage.items())
    n_units = n_weights * len(resolutions)
    n_configs = n_units * len(vp.FEATURES) * len(vp.WEIGHTINGS) * len(vp.TTA_OPTIONS)
    dims_k = sum(FEATURE_DIMS.values()) / 1000 * len(vp.FITTED_WEIGHTINGS)
    fitted_rows = n_units * n_fit + val_units * len(resolutions) * n_train
    probe_s = dims_k * LOGREG_SECONDS_PER_1K_DIMS * fitted_rows / LOGREG_REFERENCE_ROWS
    boot_s = (n_configs * n_hold + len(VAL_LABELS) * n_val) / BOOTSTRAP_REFERENCE_ROWS \
        * BOOTSTRAP_SECONDS_PER_CELL
    decode_s = (n_train + n_val) * len(resolutions) / DECODE_IMG_PER_S_PER_CORE
    return {
        "forwards": forwards, "n_units": n_units, "n_configs": n_configs,
        "gpu_hours": gpu_s / 3600,
        "cpu_hours": (probe_s + boot_s + decode_s) / 3600,
        "cpu_hours_breakdown": {"probes": probe_s / 3600, "bootstrap": boot_s / 3600,
                                "decode": decode_s / 3600},
        "assumptions": {"forward_img_per_s": FORWARD_IMG_PER_S,
                        "logreg_s_per_1k_dims": LOGREG_SECONDS_PER_1K_DIMS,
                        "bootstrap_s_per_cell": BOOTSTRAP_SECONDS_PER_CELL,
                        "decode_img_per_s_per_core": DECODE_IMG_PER_S_PER_CORE},
    }
