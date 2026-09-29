"""docs/30 8: the WiSE-FT alpha curve of A* on PAD-UFES-20.

The grid is v7.1's (``v71_protocol.ALPHAS``) plus alpha = 0, which is exactly
public and is one role shared by the three curves: it is extracted and fitted
once. Each point is named as v7.1 names its weights (``A_s42_a0.8``), so
``v71_io.WeightBank.state`` builds it.
"""

from __future__ import annotations

import math

import numpy as np

from dermssl import final_eval as fe
from dermssl import pad_ufes as pu
from dermssl import pad_ufes_readings as pr
from dermssl import v71_protocol as vp

ARM = "A"
ALPHAS = (0.0, *vp.ALPHAS)
#: model -> the v7.1 weight source it interpolates towards (the soup is s42+s43, the base of A71).
SOURCES = {"A_s42": "s42", "A_s43": "s43", "A_soup": "soup"}
MODELS = tuple(SOURCES)
ADAPTED, W1_ALPHA = 1.0, 0.8
W1_METRIC = "macro_f1"
#: PAD-W2: the most a step down in alpha, or public below the curve's max, may lose in macro-F1.
W2_TOLERANCE = 0.01
#: A registered tolerance is compared on decimals; binary floats put 0.60 - 0.59 a hair above 0.01.
FLOAT_SLACK = 1e-12
MET, NOT_MET = fe.MET, fe.NOT_MET
#: The committed HAM holdout curve (v7.1): the primary protocol of every weight set, 7 classes.
HAM_PROTOCOL = {"resolution": 224, "feature": "cls", "weighting": "none", "tta": False}
HAM_METRICS = fe.METRICS
SCOPE = ("docs/30 8: PAD-W1 and PAD-W2 decide, on the primary reading's macro-F1 only. ECE, balanced "
         "accuracy, AUC, the secondary and exploratory readings, the phototype strata and the HAM holdout "
         "curve are descriptive.")
HAM_SCOPE = ("v7.1 holdout of the C1 train (7 classes, cls|none|id at 224), already committed and known "
             "before docs/30 8: descriptive, not a prediction. Not the same classes as the PAD readings.")


def alpha_key(alpha: float) -> str:
    """The one text key of an alpha in the document (JSON keys are text; never rely on a float's repr)."""
    return f"{alpha:.1f}"


def weights_of(model: str, alpha: float) -> str:
    """The v7.1 weights name of one point; alpha = 0 is ``public`` for every model."""
    if model not in SOURCES:
        raise ValueError(f"modelo {model!r} fuera del registro de docs/30 8: {MODELS}")
    if alpha not in ALPHAS:
        raise ValueError(f"alpha {alpha} fuera de la grilla registrada {ALPHAS}")
    return vp.PUBLIC if alpha == 0 else vp.weights_name(ARM, SOURCES[model], alpha)


def curve(model: str) -> dict:
    """``{alpha: role}`` from public (alpha = 0) to the adapted final (alpha = 1)."""
    return {alpha: weights_of(model, alpha) for alpha in ALPHAS}


def roles() -> list[str]:
    """Public once, then every interpolated point of every model."""
    return [vp.PUBLIC] + [weights_of(m, a) for m in MODELS for a in ALPHAS if a]


def w1_pair(model: str) -> tuple[str, str]:
    return weights_of(model, W1_ALPHA), weights_of(model, ADAPTED)


def contrast_pairs() -> list[tuple[str, str]]:
    """Per model, every lower alpha against the adapted end; PAD-W1 reads the alpha = 0.8 one."""
    return [(weights_of(m, a), weights_of(m, ADAPTED)) for m in MODELS for a in ALPHAS if a != ADAPTED]


def _within(loss: float) -> bool:
    return loss <= W2_TOLERANCE + FLOAT_SLACK


def _checked_points(points: dict) -> dict:
    if set(points) != set(ALPHAS):
        raise ValueError(f"la curva tiene alpha {sorted(points)}, la grilla registrada es {ALPHAS}")
    if not all(math.isfinite(points[a]) for a in ALPHAS):
        raise ValueError(f"macro-F1 no finito en la curva: {points}")
    return {a: float(points[a]) for a in ALPHAS}


def w2_model(points: dict) -> dict:
    """PAD-W2 on one curve ``{alpha: primary macro-F1}``: the steps that break it and public's gap."""
    f1 = _checked_points(points)
    steps = [(high, low, f1[high] - f1[low]) for low, high in zip(ALPHAS, ALPHAS[1:])]
    violations = [{"from_alpha": high, "to_alpha": low, "drop": drop}
                  for high, low, drop in reversed(steps) if not _within(drop)]
    gap = max(f1.values()) - f1[0.0]
    return {"macro_f1": {alpha_key(a): v for a, v in f1.items()}, "violations": violations,
            "public_gap_to_max": gap,
            "holds": not violations and _within(gap)}


def _all_models(by_model: dict) -> None:
    if set(by_model) != set(MODELS):
        raise ValueError(f"faltan o sobran modelos: {sorted(by_model)} != {sorted(MODELS)}")


def w2_verdict(curves: dict) -> dict:
    """``{"PAD-W2": cumplida/no cumplida, "by_model": ...}``; it holds only if it holds in all three."""
    _all_models(curves)
    by_model = {m: w2_model(curves[m]) for m in MODELS}
    return {"PAD-W2": MET if all(r["holds"] for r in by_model.values()) else NOT_MET,
            "tolerance": W2_TOLERANCE, "by_model": by_model}


def _w1_row(rows: list[dict], model: str) -> dict:
    arm, control = w1_pair(model)
    found = [r for r in rows if (r["arm"], r["control"], r["metric"]) == (arm, control, W1_METRIC)]
    if len(found) != 1:
        raise ValueError(f"{model}: se esperaba una fila {arm} - {control} de {W1_METRIC}, hay {len(found)}")
    return found[0]


def w1_model(row: dict) -> dict:
    """Holds if the point delta is > 0; "mejora" only if the paired interval also lies above 0."""
    stats = {k: row[k] for k in ("point_delta", "ci95_low", "ci95_high", "excludes_zero")}
    return {**stats, "holds": row["point_delta"] > 0,
            "mejora": bool(row["excludes_zero"]) and row["ci95_low"] > 0}


def w1_verdict(rows: list[dict]) -> dict:
    """PAD-W1 from the primary reading's paired rows: it holds only if it holds in all three models."""
    by_model = {m: w1_model(_w1_row(rows, m)) for m in MODELS}
    return {"PAD-W1": MET if all(r["holds"] for r in by_model.values()) else NOT_MET,
            "contrast": f"alpha {W1_ALPHA} - alpha {ADAPTED}", "metric": W1_METRIC, "by_model": by_model}


def _ham_point(rows: dict, model: str, alpha: float) -> dict:
    cid = vp.config_id(weights_of(model, alpha), **HAM_PROTOCOL)
    if cid not in rows:
        raise ValueError(f"la curva de HAM de {model} no tiene {cid}")
    row = rows[cid]
    return {"config": cid, **{m: row[m] for m in HAM_METRICS}, "ci95": {m: row["ci95"][m] for m in HAM_METRICS}}


def _has_curve(rows: dict, model: str) -> bool:
    return any(vp.config_id(weights_of(model, a), **HAM_PROTOCOL) in rows for a in ALPHAS if a)


def ham_curves(grid: dict) -> dict:
    """Each model's committed HAM holdout curve (descriptive); a model v7.1 never ran has none."""
    rows = grid["rows"]
    out = {}
    for model in MODELS:
        if not _has_curve(rows, model):
            out[model] = {"available": False,
                          "note": f"v7.1 no corrio {SOURCES[model]} en el holdout de HAM: sin contraparte"}
            continue
        out[model] = {"available": True, "points": {alpha_key(a): _ham_point(rows, model, a) for a in ALPHAS}}
    return out


def check_roles(probas7: dict) -> None:
    missing = [r for r in roles() if r not in probas7]
    if missing or len(probas7) != len(roles()):
        raise ValueError(f"probabilidades de PAD sin {missing} o con roles de mas: {sorted(probas7)}")


def evaluate_reading(reading: str, ham_labels, groups, probas7: dict, pairs) -> dict:
    """The first pass's reading (:mod:`pad_ufes_readings`) without its PAD-P verdicts."""
    mask, y = pr.targets(ham_labels, reading)
    probas = {name: pr.restrict(np.asarray(p)[mask], reading) for name, p in probas7.items()}
    result = pr.evaluate_subset(y, probas, np.asarray(groups)[mask], pairs)
    result["classes"] = list(pu.READINGS[reading])
    result["outside_rate"] = {name: pr.outside_rate(np.asarray(p)[mask], reading) for name, p in probas7.items()}
    return result


def verdicts(primary: dict) -> dict:
    curves = {m: {a: primary["arms"][role]["macro_f1"] for a, role in curve(m).items()} for m in MODELS}
    return {"PAD-W1": w1_verdict(primary["paired"]), "PAD-W2": w2_verdict(curves)}


def build_document(probas7: dict, pad: dict, provenance: dict, ham_grid: dict) -> dict:
    check_roles(probas7)
    ordered = {role: probas7[role] for role in roles()}
    pairs = contrast_pairs()
    args = (pad["ham_labels"], pad["groups"])
    readings = {name: evaluate_reading(name, *args, ordered, pairs) for name in pu.READINGS}
    return {
        **provenance,
        "scope": SCOPE,
        "alphas": list(ALPHAS),
        "curves": {m: {alpha_key(a): role for a, role in curve(m).items()} for m in MODELS},
        "contrasts": [list(pair) for pair in pairs],
        "bootstrap": {**pr.BOOTSTRAP, "paired": True, "resampling_unit": "patient_id"},
        "verdicts": verdicts(readings["primary"]),
        "readings": readings,
        "fst_strata": {"reading": "primary", "exploratory": True, "min_n": pr.MIN_BIN_N,
                       "note": pu.fst_note(pu.fst_composition(pad["fst"])),
                       "bins": pr.fst_strata(*args, pad["fst_bins"], ordered, pairs)},
        "ham_holdout": {"scope": HAM_SCOPE, "curves": ham_curves(ham_grid)},
    }


#: Points this curve shares with the first pass (docs/30 2): same weights, same protocol.
FIRST_PASS_ROLES = {vp.PUBLIC: pr.PUBLIC, weights_of("A_s42", ADAPTED): pr.role("plain", 42),
                    weights_of("A_s43", ADAPTED): pr.role("plain", 43)}


def first_pass_agreement(probas7: dict, first_pass: dict) -> dict:
    """Descriptive: largest probability gap to the first pass on the three points both computed."""
    return {role: {"first_pass_role": old,
                   "max_abs_diff": float(np.abs(np.asarray(probas7[role]) - np.asarray(first_pass[old])).max())}
            for role, old in FIRST_PASS_ROLES.items()}
