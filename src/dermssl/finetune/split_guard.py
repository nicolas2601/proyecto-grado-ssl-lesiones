"""Refuse a fine-tuning split that is not the committed one (docs/33 G1.3).

The ISIC 2020 fine-tuning table is only comparable with the frozen bank if both
validate on the same patients. ``StratifiedGroupKFold(shuffle=True)`` changed in
scikit-learn 1.8, so the rebuilt split is checked against the committed val
supports and val id digest (``isic_eval.splits``) before any GPU time is spent.
No torch here: it can run on a login node.
"""

from __future__ import annotations

import numpy as np

from dermssl.isic_eval import splits as sp

REQUIRED_SKLEARN = "1.7.2"


class SklearnVersionError(RuntimeError):
    """scikit-learn cannot reproduce the committed StratifiedGroupKFold split."""


def expectation_for(dataset: str) -> sp.SplitExpectation | None:
    """The committed split a dataset must reproduce (None = nothing committed to check)."""
    return sp.ISIC2020_EXPECTED if dataset == "isic2020" else None


def require_sklearn() -> None:
    import sklearn  # noqa: PLC0415

    if sklearn.__version__ != REQUIRED_SKLEARN:
        raise SklearnVersionError(
            f"scikit-learn {sklearn.__version__} installed; the committed split only "
            f"reproduces with {REQUIRED_SKLEARN} (pip install scikit-learn=={REQUIRED_SKLEARN})")


def verify_split(expected: sp.SplitExpectation, classes, ids, y, groups, paths,
                 train_idx, val_idx) -> dict:
    """Raise ``SplitMismatchError`` unless val supports and val ids equal the committed ones."""
    ds = sp.EvalDataset(expected.name, tuple(classes), tuple(ids),
                        np.asarray(y, dtype=np.int64), groups, paths,
                        np.asarray(train_idx), np.asarray(val_idx))
    return sp.verify_split(ds, expected)
