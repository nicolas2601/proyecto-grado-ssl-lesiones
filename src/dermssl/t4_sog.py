"""T4 (docs/31): Shades of Gray p = 6 before DINOv2, as data and pure decisions.

The input is one more coordinate of every cached artefact. ``identity`` keeps
the v7.1 names and run fingerprint unchanged; ``sog_p6`` adds itself to both,
so a unit or an array computed with one input is refused under the other.
"""

from __future__ import annotations

import math

import numpy as np

from dermssl import feature_cache as fc
from dermssl import metrics
from dermssl import v71_metrics as vm
from dermssl import v71_plan as plan
from dermssl import v71_protocol as vp
from dermssl.pretrain import color_constancy as cc

IDENTITY = "identity"
SOG = "sog_p6"
SOG_P = 6  # docs/31 §2: fixed a priori, never tuned
INPUTS = (IDENTITY, SOG)


def _check_input(name: str) -> None:
    if name not in INPUTS:
        raise ValueError(f"entrada {name!r} no registrada en docs/31; se esperaba una de {INPUTS}")


def preprocess(image: np.ndarray, name: str) -> np.ndarray:
    """HWC uint8 after the bank geometry -> the array the encoder sees."""
    _check_input(name)
    if name == IDENTITY:
        return image
    return cc.shades_of_gray_uint8(image, p=SOG_P)


def input_suffix(name: str) -> str:
    """File-name infix of a decoded array: empty for identity (v7.1 names)."""
    _check_input(name)
    return "" if name == IDENTITY else f"_{name}"


def run_fingerprint(fit_ids, holdout_ids, name: str) -> str:
    """The v7.1 run fingerprint, extended with the input unless it is identity."""
    _check_input(name)
    tokens = [*fit_ids, "|", *holdout_ids]
    if name != IDENTITY:
        tokens += ["|input", name]
    return fc.fingerprint_ids(tokens)


# --- arms, protocols and the reproduction guard (docs/31 §3, §6.1) ---

WEIGHTS = {"public": "public", "A": "A_soup_a1.0", "B": "B_s42_a1.0"}
FIXED_PROTOCOLS = {  # the committed v7.1 winners (resultados/clasificacion/v71/selection.json)
    "A": "A_soup_a1.0@224|cls_attn|la0.5|tta",
    "B": "B_s42_a1.0@224|cls4|balanced|tta",
    "public": "public@224|cls4|la0.5|tta",
}
REPRODUCTION_TOLERANCE = 1e-6
REPRODUCED_METRICS = vm.GRID_METRICS


def _delta(a, b) -> float:
    """|a - b|, two NaNs counting as equal (an undefined metric on both sides)."""
    a, b = float(a), float(b)
    if math.isnan(a) and math.isnan(b):
        return 0.0
    return abs(a - b)


def reproduction_check(recomputed: dict, committed: dict, tol: float = REPRODUCTION_TOLERANCE) -> dict:
    """The identity cells of the fixed protocols against the committed v7.1 holdout grid."""
    checks = {}
    for cid in FIXED_PROTOCOLS.values():
        if cid not in recomputed or cid not in committed:
            checks[cid] = {"missing": True, "passed": False}
            continue
        deltas = {m: _delta(recomputed[cid][m], committed[cid][m]) for m in REPRODUCED_METRICS}
        checks[cid] = {"deltas": deltas, "passed": all(d <= tol for d in deltas.values())}
    return {"tolerance": tol, "checks": checks, "passed": all(c["passed"] for c in checks.values())}


# --- selection and the single val read (docs/31 §6.2, §6.3) ---

PRIMARY_CONTRASTS = tuple((f"{arm}_sog", f"{arm}_id") for arm in WEIGHTS)
SECONDARY_CONTRASTS = tuple((f"{arm}_sogwin", f"{arm}_id") for arm in WEIGHTS)


def sog_winners(sog_rows: dict) -> dict:
    """docs/28 §4.1 per arm over that arm's SoG cells at 224."""
    return {arm: vp.select_for_arm(sog_rows, arm) for arm in WEIGHTS}


def val_entries(winners: dict) -> dict:
    """``label -> (input, config_id)``: identity and SoG at the fixed protocol, and the SoG winner."""
    entries = {}
    for arm, fixed in FIXED_PROTOCOLS.items():
        entries[f"{arm}_id"] = (IDENTITY, fixed)
        entries[f"{arm}_sog"] = (SOG, fixed)
        entries[f"{arm}_sogwin"] = (SOG, winners[arm])
    return entries


def val_units(entries: dict) -> dict:
    """``(input, weights, resolution) -> {features, tta}``: one extraction per key."""
    units: dict = {}
    for input_name, cid in entries.values():
        cfg = vp.parse_config_id(cid)
        unit = units.setdefault((input_name, cfg["weights"], cfg["resolution"]), {"features": set(), "tta": False})
        unit["features"].add(cfg["feature"])
        unit["tta"] = unit["tta"] or cfg["tta"]
    return units


# --- the luminance-tercile gap (docs/31 §4) ---

TERCILES = {"dark": 0, "medium": 1, "light": 2}
UNAVAILABLE_CODE = -1
GAP_METRIC = "luminance_gap_dark_light"
RANGE_METRIC = "luminance_gap_range"


def tercile_codes(labels) -> np.ndarray:
    """Manifest ``tone_tercile`` labels -> 0/1/2; unavailable or empty -> -1."""
    codes = []
    for label in labels:
        text = "" if label is None else str(label).strip()
        if text in ("", "unavailable"):
            codes.append(UNAVAILABLE_CODE)
        elif text in TERCILES:
            codes.append(TERCILES[text])
        else:
            raise ValueError(f"tercil de luminancia desconocido {label!r}; se esperaba {sorted(TERCILES)}")
    return np.array(codes, dtype=int)


def pack_strata(y: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """``(n, 2)`` labels: a paired bootstrap then draws each image with its tercile."""
    return np.stack([np.asarray(y, dtype=int), np.asarray(codes, dtype=int)], axis=1)


def _tercile_f1(packed: np.ndarray, proba: np.ndarray) -> list[float]:
    y, codes, pred = packed[:, 0], packed[:, 1], np.asarray(proba).argmax(axis=1)
    n_classes = proba.shape[1]
    return [metrics.macro_f1(y[codes == c], pred[codes == c], n_classes) for c in TERCILES.values()]


def gap_dark_light(packed: np.ndarray, proba: np.ndarray) -> float:
    """|macro-F1(dark) - macro-F1(light)|; lower is better."""
    dark, _, light = _tercile_f1(packed, proba)
    return abs(dark - light)


def gap_range(packed: np.ndarray, proba: np.ndarray) -> float:
    """max - min of the three terciles' macro-F1; lower is better."""
    f1s = _tercile_f1(packed, proba)
    return max(f1s) - min(f1s)


# --- cost (docs/31 §8) ---

FORWARD_IMG_PER_S = 720.0  # measured on the pod at 224
PROBE_SECONDS_PER_UNIT = 0.4 * 3600 / 21  # v7.1: ~0.4 CPU-h per 21 units, all features, fit split
ALL_FEATURE_DIMS = sum(plan.FEATURE_DIMS.values())
WINNER_DIMS_BOUND = max(plan.FEATURE_DIMS.values())  # the SoG winner is unknown before select
N_VIEWS = 8
N_CELLS_PER_UNIT = len(vp.FEATURES) * len(vp.WEIGHTINGS) * len(vp.TTA_OPTIONS)
N_VAL_METRICS = 8  # the six v7.1 contrast metrics and the two gaps


def cost_projection(*, n_fit: int, n_hold: int, n_train: int, n_val: int) -> dict:
    n_units = len(INPUTS) * len(WEIGHTS)
    forwards = {"grid": n_units * (n_fit + N_VIEWS * n_hold), "val": n_units * (n_train + N_VIEWS * n_val)}
    gpu_s = sum(forwards.values()) / FORWARD_IMG_PER_S
    val_share = sum(min(1.0, (plan.FEATURE_DIMS[vp.parse_config_id(c)["feature"]] + WINNER_DIMS_BOUND)
                        / ALL_FEATURE_DIMS) for c in FIXED_PROTOCOLS.values()) * len(INPUTS)
    cpu = {
        "probes_grid": n_units * PROBE_SECONDS_PER_UNIT,
        "probes_val": val_share * PROBE_SECONDS_PER_UNIT * (n_train + n_fit) / n_fit,
        "bootstrap": (n_units * N_CELLS_PER_UNIT * n_hold
                      + (len(WEIGHTS) * 3 + N_VAL_METRICS * (len(PRIMARY_CONTRASTS) + len(SECONDARY_CONTRASTS)))
                      * n_val) / plan.BOOTSTRAP_REFERENCE_ROWS * plan.BOOTSTRAP_SECONDS_PER_CELL,
        "decode": len(INPUTS) * (n_fit + n_hold + n_train + n_val) / plan.DECODE_IMG_PER_S_PER_CORE,
    }
    return {"forwards": forwards, "n_units": n_units, "gpu_hours": gpu_s / 3600,
            "cpu_hours": sum(cpu.values()) / 3600,
            "cpu_hours_breakdown": {k: v / 3600 for k, v in cpu.items()},
            "assumptions": {"forward_img_per_s": FORWARD_IMG_PER_S, "probe_s_per_unit": PROBE_SECONDS_PER_UNIT,
                            "bootstrap_s_per_cell": plan.BOOTSTRAP_SECONDS_PER_CELL,
                            "decode_img_per_s_per_core": plan.DECODE_IMG_PER_S_PER_CORE}}
