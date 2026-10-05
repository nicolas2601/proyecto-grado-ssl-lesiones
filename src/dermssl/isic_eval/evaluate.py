"""One cell: fit the fixed probe on the dataset TRAIN split, score the val split (docs/33 E1).

The config is fixed, so nothing is selected on val: the probe (``probes.fit_logreg``, C = 1, standardised,
via ``v71_grid.fit_predict_views``) is fitted on the identity view of the train features, scored on the
eight D4 views of val, and combined as the config says (class weight ``balanced`` or post-hoc logit
adjustment on the flat fit with the train prior, then the mean over views for TTA). Each fit runs under
``probe_threads.PROBE_BLAS_THREADS`` like every published v7.1 number.

``extract(weights, split, ids, n_views, parts)`` returns ``part -> (n_views, n_ids, dim)``; the real one
decodes images and runs the backbone, a test passes a fake.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from dermssl import probe_threads
from dermssl import v71_grid as vg
from dermssl import v71_protocol as vp
from dermssl.isic_eval import protocol as pr
from dermssl.isic_eval import scoring
from dermssl.isic_eval.splits import EvalDataset

N_VIEWS = 8  # the D4 group (torch_v71.N_D4; no torch import here)
Extractor = Callable[[str, str, Sequence[str], int, tuple[str, ...]], dict]


def probe_proba(train_parts: dict, y_train, eval_parts: dict, cfg: dict, n_classes: int) -> np.ndarray:
    """Probabilities of the fixed config on the eval views: ``(n_eval, n_classes)``."""
    x_fit = vg.compose(train_parts, cfg["feature"])[0]
    x_eval = vg.compose(eval_parts, cfg["feature"])
    class_weight = "balanced" if vp.fit_of(cfg["weighting"]) == "balanced" else None
    views = probe_threads.call_pinned(vg.fit_predict_views, x_fit, y_train, x_eval, n_classes, class_weight)
    tau = vp.tau_of(cfg["weighting"])
    if tau:
        prior = vp.class_prior(y_train, n_classes)
        views = np.stack([vp.logit_adjust(v, prior, tau) for v in views])
    return views.mean(axis=0) if cfg["tta"] else views[0]


def load_or_extract(extract: Extractor, ds: EvalDataset, config: str, cache: Path | None) -> tuple[dict, dict]:
    """``(train_parts, val_parts)``; features are kept in ``cache`` so a retry never re-extracts."""
    if cache is not None and cache.is_file():
        saved = np.load(cache)
        return ({p: saved[f"train_{p}"] for p in pr.parts_needed(config)},
                {p: saved[f"val_{p}"] for p in pr.parts_needed(config)})
    weights, parts = pr.parse(config)["weights"], pr.parts_needed(config)
    train = extract(weights, "train", ds.train_ids, 1, parts)
    val = extract(weights, "val", ds.val_ids, N_VIEWS, parts)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.stem + ".tmp.npz")
        np.savez(tmp, **{f"train_{p}": train[p] for p in parts}, **{f"val_{p}": val[p] for p in parts})
        tmp.replace(cache)
    return train, val


def evaluate_config(ds: EvalDataset, config: str, extract: Extractor, *, cache: Path | None = None,
                    n_resamples: int = scoring.N_RESAMPLES) -> tuple[dict, np.ndarray]:
    """``(metrics block, val probabilities)`` of ``config`` fitted on train and scored on val."""
    cfg = pr.parse(config)
    train, val = load_or_extract(extract, ds, config, cache)
    proba = probe_proba(train, ds.y_train, val, cfg, len(ds.classes))
    block = scoring.score(ds.y_val, proba, ds.classes, groups=ds.val_groups, n_resamples=n_resamples)
    return block, proba
