"""Pure helpers for the student: which images have usable pseudo-labels."""

from __future__ import annotations

import numpy as np

from cache_io import IGNORE

SELECTION_CHOICES = ("best-val", "last")


def indices_with_confident_pixels(
    pseudo_masks: np.ndarray, candidates: np.ndarray, chunk_size: int = 512
) -> np.ndarray:
    """Keep the candidate rows that have at least one non-ignored pixel.

    Samples excluded by the uncertainty module (all pixels ignored) would add
    no loss, so they are dropped up front instead of wasting GPU time.
    """
    candidates = np.asarray(candidates)
    kept = []
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        has_confident = (pseudo_masks[chunk] != IGNORE).reshape(len(chunk), -1).any(axis=1)
        kept.append(chunk[has_confident])
    return np.concatenate(kept) if kept else candidates[:0]


def require_training_images(train_indices: np.ndarray) -> np.ndarray:
    """Fail early and clearly when no image has usable pseudo-labels."""
    if len(train_indices) == 0:
        raise ValueError(
            "ninguna imagen de entrenamiento tiene pixeles confiables: el modulo de "
            "incertidumbre excluyo todas las muestras (revise los umbrales o entrene mas "
            "la ronda anterior)"
        )
    return train_indices
