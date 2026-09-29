"""docs/30 8: the WiSE-FT alpha curve of A* on PAD-UFES-20.

The grid is v7.1's (``v71_protocol.ALPHAS``) plus alpha = 0, which is exactly
public and is one role shared by the three curves: it is extracted and fitted
once. Each point is named as v7.1 names its weights (``A_s42_a0.8``), so
``v71_io.WeightBank.state`` builds it.
"""

from __future__ import annotations

import math

from dermssl import final_eval as fe
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
    return {"macro_f1": f1, "violations": violations, "public_gap_to_max": gap,
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
