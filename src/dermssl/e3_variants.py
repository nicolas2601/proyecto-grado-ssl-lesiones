"""The E3 variants of docs/33 section 3, built from per-condition features.

Every feature array is ``(n_views, n_images, dim)`` produced by the B71 protocol
(``B_s42_a1.0@224|cls4|balanced|tta``: DINOv2 cls4, balanced logistic probe fitted on
the identity view, probabilities averaged over the 8 D4 views). A crop condition is
extracted with the SAME extractor, TTA and pooling as the full image, so view ``k`` of
a concatenation joins view ``k`` of each part.

``B71``     the no-mask reference, full features only;
``F1``      concat(full, crop with 10 % margin cut from the 224 view);
``F2``      concat(full, crop cut from the original and resized to 224);
``F2_448``  the same crop resized to 448 (full stays at 224: the registered config);
``F3``      mask-guided TTA: mean of the class probabilities of the full image and of K = 4
            shifted crops, all under ONE probe fitted on the full-image features;
``F4``      F1 plus the standardised mask descriptors (``e3_descriptors``);
``F5``      F1 plus the perilesional ring features.

Descriptors do not depend on the D4 view, so they are repeated on every view. The
probe's own ``StandardScaler`` re-standardises every column; the descriptors are
standardised on the fit split anyway so the stored array is already comparable.
"""

from __future__ import annotations

import numpy as np

from dermssl import e3_render as render
from dermssl import probe_threads
from dermssl import v71_grid as vg

REFERENCE = "B71"
PROBE_WEIGHTING = "balanced"
PARTS = {
    "B71": ("full",),
    "F1": ("full", "crop_view"),
    "F2": ("full", "crop_orig224"),
    "F2_448": ("full", "crop_orig448"),
    "F3": ("full", *render.SHIFT_CONDITIONS),
    "F4": ("full", "crop_view"),
    "F5": ("full", "crop_view", "ring"),
}
VARIANTS = tuple(PARTS)
ALL_CONDITIONS = render.CONDITIONS
WITH_DESCRIPTORS = frozenset({"F4"})


class UnknownVariantError(ValueError):
    """The name is not one of :data:`VARIANTS`."""


class MissingInputError(ValueError):
    """A condition, the descriptors or an image count needed by a variant is absent or inconsistent."""


def required_conditions(name: str) -> tuple[str, ...]:
    if name not in PARTS:
        raise UnknownVariantError(f"variante {name!r} no registrada; se esperaba una de {VARIANTS}")
    return PARTS[name]


def _checked_parts(name: str, feats: dict) -> list[np.ndarray]:
    parts = []
    for condition in required_conditions(name):
        if condition not in feats:
            raise MissingInputError(f"{name}: falta la condicion {condition!r}")
        parts.append(np.asarray(feats[condition]))
    if len({p.shape[:2] for p in parts}) != 1:
        raise MissingInputError(f"{name}: las condiciones no tienen las mismas vistas e imagenes: "
                                f"{[p.shape[:2] for p in parts]}")
    return parts


def concat_features(name: str, feats: dict, descriptors: np.ndarray | None = None) -> np.ndarray:
    """``(n_views, n, D)``: the parts of the variant joined on the feature axis."""
    parts = _checked_parts(name, feats)
    if name in WITH_DESCRIPTORS:
        n_views, n = parts[0].shape[:2]
        if descriptors is None or len(descriptors) != n:
            raise MissingInputError(f"{name}: necesita un descriptor por imagen ({n})")
        parts.append(np.broadcast_to(np.asarray(descriptors)[None], (n_views, n, descriptors.shape[1])))
    return np.concatenate(parts, axis=-1)


def _score(x_fit: np.ndarray, y_fit, x_views: np.ndarray, n_classes: int) -> np.ndarray:
    return probe_threads.call_pinned(vg.fit_predict_views, x_fit, y_fit, x_views, n_classes, PROBE_WEIGHTING)


def f3_members(fit_feats: dict, y_fit, eval_feats: dict, n_classes: int) -> np.ndarray:
    """``(5, n, classes)``: TTA probabilities of the full image, then of each shifted crop."""
    names = required_conditions("F3")
    _checked_parts("F3", eval_feats)
    stacked = np.concatenate([np.asarray(eval_feats[c]) for c in names], axis=0)
    out = _score(np.asarray(fit_feats["full"])[0], y_fit, stacked, n_classes)
    return out.reshape(len(names), -1, *out.shape[1:]).mean(axis=1)


def variant_proba(name: str, fit_feats: dict, y_fit, eval_feats: dict, n_classes: int, *,
                  desc_fit: np.ndarray | None = None, desc_eval: np.ndarray | None = None) -> np.ndarray:
    """Class probabilities ``(n, classes)`` of one variant on the evaluated split."""
    if name == "F3":
        return f3_members(fit_feats, y_fit, eval_feats, n_classes).mean(axis=0)
    x_fit = concat_features(name, fit_feats, desc_fit)[0]
    x_eval = concat_features(name, eval_feats, desc_eval)
    return _score(x_fit, y_fit, x_eval, n_classes).mean(axis=0)
