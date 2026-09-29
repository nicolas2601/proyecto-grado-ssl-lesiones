"""docs/30 8: the WiSE-FT alpha curve of A* on PAD-UFES-20.

The grid is v7.1's (``v71_protocol.ALPHAS``) plus alpha = 0, which is exactly
public and is one role shared by the three curves: it is extracted and fitted
once. Each point is named as v7.1 names its weights (``A_s42_a0.8``), so
``v71_io.WeightBank.state`` builds it.
"""

from __future__ import annotations

from dermssl import v71_protocol as vp

ARM = "A"
ALPHAS = (0.0, *vp.ALPHAS)
#: model -> the v7.1 weight source it interpolates towards (the soup is s42+s43, the base of A71).
SOURCES = {"A_s42": "s42", "A_s43": "s43", "A_soup": "soup"}
MODELS = tuple(SOURCES)
ADAPTED, W1_ALPHA = 1.0, 0.8


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
