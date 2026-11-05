"""Image-only supervised cache reader inspired by Tip-Adapter, without text features."""
import time

import numpy as np

from evaluate_overlay_readout import group_sha, validate_support
from frozen_reader_heads import unit
from probe_preprocessing_readout import C_GRID, INNER_FOLDS, OUTER_FOLDS, SEEDS, _fit, grouped_folds, metrics

ALPHAS = (0., 1., 5.)
BETAS = (5., 20.)


def cache_scores(reference_x, reference_y, query_x, reference_groups, query_groups,
                 classes, beta, device="cuda:0"):
    import torch
    import torch.nn.functional as functional
    reference_x, query_x = unit(reference_x), unit(query_x)
    reference_y = np.asarray(reference_y)
    reference_groups, query_groups = np.asarray(reference_groups, dtype=str), np.asarray(query_groups, dtype=str)
    if (reference_y.shape != (len(reference_x),) or reference_groups.shape != reference_y.shape
            or query_groups.shape != (len(query_x),) or beta <= 0
            or np.any(reference_groups == "") or np.any(query_groups == "")):
        raise ValueError("Expected aligned cache labels and patient identities")
    labels = np.asarray([classes.index(value) for value in reference_y])
    _, codes = np.unique(np.concatenate((reference_groups, query_groups)), return_inverse=True)
    with torch.inference_mode():
        reference = torch.as_tensor(reference_x, dtype=torch.float32, device=device)
        query = torch.as_tensor(query_x, dtype=torch.float32, device=device)
        r_groups = torch.as_tensor(codes[:len(reference_x)], device=device)
        q_groups = torch.as_tensor(codes[len(reference_x):], device=device)
        allowed = (q_groups[:, None] != r_groups[None, :]).float()
        indicator = functional.one_hot(torch.as_tensor(labels, device=device), num_classes=len(classes)).float()
        support = allowed @ indicator
        if torch.any(support == 0):
            raise ValueError("Reference class missing after whole-patient exclusion")
        affinity = torch.exp(-beta * (1 - (query @ reference.T).clamp(-1, 1)))
        return (((affinity * allowed) @ indicator) / support).cpu().numpy()


def blend(probabilities, cache, alpha):
    p, cache = np.asarray(probabilities), np.asarray(cache)
    if p.shape != cache.shape or not np.isfinite(p).all() or not np.isfinite(cache).all() or alpha < 0:
        raise ValueError("Invalid baseline/cache scores")
    if alpha == 0:
        return p.copy()
    logits = np.log(np.maximum(p, np.finfo(float).tiny)) + alpha * cache
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def evaluate(features, truth, groups, classes, device="cuda:0"):
    from sklearn.metrics import f1_score
    validate_support(truth, groups, classes)
    x = unit(features)
    class_array = np.asarray(classes)
    runs = []
    for seed in SEEDS:
        started = time.monotonic()
        oof, seen, traces = np.zeros((len(truth), len(classes))), np.zeros(len(truth), int), []
        for fold, (train, test) in enumerate(grouped_folds(truth, groups, OUTER_FOLDS, seed)):
            grid = {(c, alpha, beta): [] for c in C_GRID for alpha in ALPHAS for beta in BETAS}
            for inner, valid in grouped_folds(truth[train], groups[train], INNER_FOLDS, seed + fold):
                ref, query = train[inner], train[valid]
                caches = {beta: cache_scores(x[ref], truth[ref], x[query], groups[ref], groups[query], classes, beta, device) for beta in BETAS}
                for c in C_GRID:
                    model = _fit(x[ref], truth[ref], c)
                    if tuple(model.classes_) != tuple(classes):
                        raise ValueError("Cache base reader class order mismatch")
                    p = model.predict_proba(x[query])
                    for alpha in ALPHAS:
                        for beta in BETAS:
                            predicted = class_array[blend(p, caches[beta], alpha).argmax(1)]
                            grid[c, alpha, beta].append(f1_score(truth[query], predicted, labels=classes,
                                                                average="macro", zero_division=0))
            c, alpha, beta = max(grid, key=lambda key: (np.mean(grid[key]), -key[1], -key[0], -key[2]))
            model = _fit(x[train], truth[train], c)
            cache = cache_scores(x[train], truth[train], x[test], groups[train], groups[test], classes, beta, device)
            oof[test] = blend(model.predict_proba(x[test]), cache, alpha)
            seen[test] += 1
            traces.append({"fold": fold, "C": c, "alpha": alpha, "beta": beta,
                           "train_groups_sha256": group_sha(train, groups),
                           "test_groups_sha256": group_sha(test, groups),
                           "cache_scope": "Outer-training patients only; entire query patient excluded"})
        if not np.all(seen == 1):
            raise ValueError("Expected one OOF prediction per allowed image")
        runs.append({"seed": seed, "metrics": metrics(truth, oof, classes), "folds": traces,
                     "oof_probabilities": oof.tolist(), "seconds": time.monotonic() - started})
        print(f"prototype-cache-reader CV_seed={seed} macro_f1={runs[-1]['metrics']['macro_f1']:.6f}", flush=True)
    return runs
