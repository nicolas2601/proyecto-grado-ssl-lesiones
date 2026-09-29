"""T4 (docs/31): Shades of Gray p = 6 before DINOv2, as data and pure decisions.

The input is one more coordinate of every cached artefact. ``identity`` keeps
the v7.1 names and run fingerprint unchanged; ``sog_p6`` adds itself to both,
so a unit or an array computed with one input is refused under the other.
"""

from __future__ import annotations

import numpy as np

from dermssl import feature_cache as fc
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
