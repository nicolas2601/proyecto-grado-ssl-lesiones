"""dermssl - reusable core of the Objective 2 self-supervised pipeline.

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
    # gates.py is stdlib-only by design (design.md D1) and metrics.py's
    # stratified_summary reuses its DEFAULT_FAIRNESS_MIN_BUCKET_N as the
    # single source of truth for the "min samples per fairness bucket"
    # threshold, instead of each module hardcoding its own literal that
    # could silently drift out of sync. gates must be inlined here, before
    # metrics, so the constant is already a plain name in the notebook
    # core cell's namespace when metrics.stratified_summary's default
    # argument is evaluated - see build_notebook_v5.module_source()'s
    # "from dermssl" strip.
    "gates",
    "metrics",
    "probes",
    "feature_cache",
    # fitzpatrick.py's Otsu fallback reuses pretrain/lesion_guided.lesion_mask
    # (design.md D3: "one contract", not a duplicate). It must be inlined
    # here, before fitzpatrick, so lesion_mask is already a plain name in
    # the notebook core cell's namespace by the time fitzpatrick's code
    # runs - see build_notebook_v5.module_source()'s "from dermssl" strip.
    "pretrain/lesion_guided",
    "fitzpatrick",
)
