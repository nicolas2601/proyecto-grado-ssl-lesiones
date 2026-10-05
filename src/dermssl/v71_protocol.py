"""The v7.1 grid registered in docs/29, as data.

A configuration is ``weights @ resolution | feature | weighting | tta``:

* weights (§2.1): ``public`` (the reference, never a candidate), and per arm
  ``<arm>_s42_a<alpha>`` (WiSE-FT between public and the seed-42 final) and
  ``<arm>_soup_a<alpha>`` (WiSE-FT between public and the 42/43 seed soup);
* resolution (§2.3): 224 or 448, same weights;
* feature (§2.2 and §2.7 item 3): CLS, CLS + patch mean, the CLS of the last
  four blocks, CLS + GeM(p=3) of the patches, CLS + CLS->patch attention pool;
* weighting (§2.2 and §2.7 item 2): none, ``balanced``, or post-hoc logit
  adjustment ``la<tau>`` on the flat probe (tau = 0 is ``none``);
* tta (§2.7 item 4): the probe's softmax averaged over the eight D4 views.

Selection (§2.4) is the docs/28 §4.1 rule from :mod:`selection_holdout`,
applied per arm over that arm's configurations only.
"""

from __future__ import annotations

import numpy as np

from dermssl import selection_holdout as sh

ARMS = ("A", "B")
PUBLIC = "public"
ALPHAS = (0.2, 0.4, 0.6, 0.8, 1.0)
SOURCES = ("s42", "soup")
RESOLUTIONS = (224, 448)
FEATURES = ("cls", "cls_mean", "cls4", "cls_gem", "cls_attn")
TAUS = (0.5, 1.0, 1.5, 2.0)
FITTED_WEIGHTINGS = ("none", "balanced")  # each one logistic regression fit
WEIGHTINGS = FITTED_WEIGHTINGS + tuple(f"la{t}" for t in TAUS)
TTA_OPTIONS = (False, True)
PRIMARY = {"resolution": 224, "feature": "cls", "weighting": "none", "tta": False}
# §2.3: 448 is a secondary diagnostic at fixed weights, never a selection axis.
SELECTION_RESOLUTION = PRIMARY["resolution"]
PHASE2_MEL_SENS_AT_SPEC80 = 0.70  # §2.7 item 7: phase 2 only if no arm passes this


def weights_name(arm: str, source: str, alpha: float) -> str:
    return f"{arm}_{source}_a{alpha:.1f}"


def weight_configs(arm: str) -> list[str]:
    return [weights_name(arm, s, a) for s in SOURCES for a in ALPHAS]


def parse_weights(name: str) -> dict:
    """``A_soup_a0.4`` -> arm, source, alpha; ``public`` is alpha 0 of every arm."""
    if name == PUBLIC:
        return {"arm": PUBLIC, "source": PUBLIC, "alpha": 0.0}
    arm, source, alpha = name.split("_")
    return {"arm": arm, "source": source, "alpha": float(alpha[1:])}


def arm_of(weights: str) -> str:
    return parse_weights(weights)["arm"]


def config_id(weights: str, resolution: int, feature: str, weighting: str, tta: bool) -> str:
    return f"{weights}@{resolution}|{feature}|{weighting}|{'tta' if tta else 'id'}"


def parse_config_id(cid: str) -> dict:
    unit, feature, weighting, tta = cid.split("|")
    weights, resolution = unit.split("@")
    return {"weights": weights, "resolution": int(resolution), "feature": feature,
            "weighting": weighting, "tta": tta == "tta"}


def with_weights(cid: str, weights: str) -> str:
    """The same protocol on other weights: how public is matched to a chosen config."""
    parts = parse_config_id(cid)
    parts["weights"] = weights
    return config_id(**parts)


def primary_config(arm: str) -> str:
    """The v7 primary protocol on the seed-42 final (alpha = 1), or on public."""
    weights = PUBLIC if arm == PUBLIC else weights_name(arm, "s42", 1.0)
    return config_id(weights, **PRIMARY)


def tau_of(weighting: str) -> float:
    return float(weighting[2:]) if weighting.startswith("la") else 0.0


def fit_of(weighting: str) -> str:
    """Which fitted probe a weighting reads: logit adjustment is post-hoc on the flat one."""
    return "balanced" if weighting == "balanced" else "none"


def class_prior(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(np.asarray(y), minlength=n_classes).astype(float)
    if (counts == 0).any():
        raise ValueError(f"clases sin ejemplos en el train del probe: {np.flatnonzero(counts == 0)}")
    return counts / counts.sum()


def logit_adjust(proba: np.ndarray, prior: np.ndarray, tau: float) -> np.ndarray:
    """softmax(log p - tau * log pi) (Menon et al. 2021); tau = 0 returns ``proba``."""
    proba = np.asarray(proba, dtype=float)
    if tau == 0:
        return proba
    logits = np.log(np.clip(proba, 1e-300, None)) - tau * np.log(prior)[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def select_for_arm(grid: dict, arm: str, candidates=None) -> str:
    """docs/28 §4.1 on this arm's configurations at the selection resolution."""
    if candidates is None:
        candidates = [c for c in grid if arm_of(parse_config_id(c)["weights"]) == arm
                      and parse_config_id(c)["resolution"] == SELECTION_RESOLUTION]
    return sh.select_winner(grid, candidates=candidates)


def phase2_enabled(mel_sens_at_spec80_by_arm: dict) -> bool:
    return all(v <= PHASE2_MEL_SENS_AT_SPEC80 for v in mel_sens_at_spec80_by_arm.values())
