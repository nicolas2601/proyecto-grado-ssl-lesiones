"""dermssl - reusable core of the Objective 2 SSL pipeline.

Every module here is dependency-light (numpy + standard library) so the logic can
be unit tested on any machine, with no GPU, no Kaggle session and no network.
The heavy, torch-dependent code lives in the notebook, which is generated from
these files by ``scripts/build_notebook_v5.py`` - so the tested code and the
executed code are the same code.
"""

from . import (
    data_resolver,
    feature_cache,
    gates,
    label_regimes,
    metrics,
    patch_guard,
    probes,
)

__all__ = [
    "data_resolver",
    "feature_cache",
    "gates",
    "label_regimes",
    "metrics",
    "patch_guard",
    "probes",
]

MODULE_ORDER = (
    "patch_guard",
    "data_resolver",
    "label_regimes",
    "metrics",
    "probes",
    "feature_cache",
    "fitzpatrick",
    "gates",
    "circle",
)
