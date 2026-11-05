"""Literal discrete Eq.2 of Li et al., arXiv2209.15007; not classification AUROC."""
import numpy as np


def singular_auc(singular_values):
    values = np.asarray(singular_values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Expected finite nonnegative singular values')
    total = values.sum()
    if total == 0:
        return None
    return float(np.cumsum(np.sort(values)[::-1]).mean() / total)
