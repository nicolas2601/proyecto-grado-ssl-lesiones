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
    # gates.py no tiene dependencias externas, y metrics.py (stratified_summary)
    # reusa su DEFAULT_FAIRNESS_MIN_BUCKET_N como unica fuente de verdad para
    # el minimo de muestras por bucket de fairness, en vez de que cada modulo
    # tenga su propio literal que pueda desincronizarse en silencio. gates
    # tiene que inlinearse aca, antes de metrics, para que la constante ya
    # exista como nombre suelto en el namespace de la celda del notebook
    # cuando se evalua el default de stratified_summary - ver el "from
    # dermssl" que build_notebook_v5.module_source() elimina al inlinear.
    "gates",
    "metrics",
    "probes",
    "feature_cache",
    # El fallback Otsu de fitzpatrick.py reusa pretrain/lesion_guided.lesion_mask
    # (un solo contrato, no una copia duplicada). Tiene que inlinearse aca,
    # antes de fitzpatrick, para que lesion_mask ya exista como nombre suelto
    # en el namespace de la celda del notebook cuando corre el codigo de
    # fitzpatrick - ver el "from dermssl" que build_notebook_v5.module_source()
    # elimina al inlinear.
    "pretrain/lesion_guided",
    "fitzpatrick",
)
