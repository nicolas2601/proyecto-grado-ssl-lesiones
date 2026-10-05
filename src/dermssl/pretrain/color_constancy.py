"""Color constancy as a stochastic SSL augmentation (thesis novelty 2).

Why this module exists
----------------------
Dermoscopy literature applies color constancy (shades-of-gray, gray-world) as
a DETERMINISTIC preprocess: normalize every image once, then train. v6 instead
applies it STOCHASTICALLY and INDEPENDENTLY to each view of an SSL pair. The
encoder then cannot rely on illuminant color to match views, so it is pushed
to become invariant to the dermatoscope's lighting - which is precisely the
device-to-device variability that hurts cross-center generalization.

All functions take float arrays of shape (H, W, 3) with values in [0, 1] and
return the same. Pure numpy so the exact transform the cluster applies is the
one tested here.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def gray_world(img: np.ndarray) -> np.ndarray:
    """Scale each channel so all channel means match the global mean."""
    means = img.reshape(-1, 3).mean(axis=0)
    target = means.mean()
    if target < _EPS:  # black image: nothing to correct
        return img.copy()
    gains = target / np.maximum(means, _EPS)
    return np.clip(img * gains, 0.0, 1.0)


def _minkowski_norms(img: np.ndarray, p: float) -> np.ndarray:
    """Per-channel illuminant estimate; ``p = inf`` is the channel maximum (max-RGB)."""
    if p < 1:
        raise ValueError(f"p = {p}: la norma de Minkowski de shades of gray exige p >= 1")
    flat = img.reshape(-1, 3)
    if np.isinf(p):
        return flat.max(axis=0)
    return (flat**p).mean(axis=0) ** (1.0 / p)


def illuminant_gains(img: np.ndarray, p: float = 6) -> np.ndarray:
    """``g_c = mean_k(n_k) / n_c``: the corrected channel norms all equal their old mean.

    A black image has no illuminant to remove and gets unit gains.
    """
    return _gains_from_norms(_minkowski_norms(img, p))


def _gains_from_norms(norms: np.ndarray) -> np.ndarray:
    target = norms.mean()
    if target < _EPS:
        return np.ones(3)
    return target / np.maximum(norms, _EPS)


def shades_of_gray(img: np.ndarray, p: float = 6) -> np.ndarray:
    """Minkowski-norm illuminant estimate; p=1 is gray-world, p->inf is max-RGB.

    p=6 is the standard choice in the dermoscopy color-constancy literature
    (Finlayson & Trezzi 2004; Barata et al. 2015; docs/31).
    """
    if p == 1:
        return gray_world(img)
    norms = _minkowski_norms(img, p)
    if norms.mean() < _EPS:
        return img.copy()
    return np.clip(img * _gains_from_norms(norms), 0.0, 1.0)


def max_rgb(img: np.ndarray) -> np.ndarray:
    """Shades of gray at p = inf: each channel scaled by its maximum."""
    return shades_of_gray(img, p=np.inf)


def shades_of_gray_uint8(img: np.ndarray, p: float = 6) -> np.ndarray:
    """HWC uint8 in and out, the pretraining composition (``torch_data._finish_view``):
    float in [0, 1], the correction, then truncation back to uint8."""
    if img.dtype != np.uint8:
        raise ValueError(f"se esperaba uint8 HWC, llego {img.dtype}")
    arr = shades_of_gray(img.astype(np.float64) / 255.0, p=p)
    return (arr * 255.0).astype(np.uint8)


OPS = {
    "identity": lambda img: img,
    "gray_world": gray_world,
    "shades_of_gray": shades_of_gray,
}


def sample_op(rng: np.random.Generator, probs: tuple[tuple[str, float], ...]) -> str:
    """Draw an op name; probabilities are normalized before the draw.

    numpy's ``choice`` raises unless probabilities sum to exactly 1.0, and
    hand-written configs routinely sum to 0.99 (v5 gotcha #5). Normalizing
    here makes the config robust without changing relative weights.
    """
    names = [name for name, _ in probs]
    unknown = [n for n in names if n not in OPS]
    if unknown:
        raise ValueError(f"operación de constancia de color desconocida {unknown}; se esperaba {sorted(OPS)}")
    weights = np.array([w for _, w in probs], dtype=float)
    weights = weights / weights.sum()
    return str(rng.choice(names, p=weights))


def apply_stochastic(
    img: np.ndarray,
    rng: np.random.Generator,
    probs: tuple[tuple[str, float], ...],
) -> np.ndarray:
    """Apply one op drawn from ``probs``. Call once PER VIEW: the independent
    draws are what turn a preprocess into an invariance signal."""
    return OPS[sample_op(rng, probs)](img)
