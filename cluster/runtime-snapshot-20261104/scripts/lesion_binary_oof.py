"""Lesion-level aggregation of fixed image OOF probabilities, never new training."""
import numpy as np


def binary_metrics(truth, probabilities, classes):
    from sklearn.metrics import roc_auc_score, roc_curve
    p = np.asarray(probabilities, dtype=float)
    truth = np.asarray(truth)
    cancer = {'BCC', 'MEL', 'SCC'}
    if (set(classes) != {'ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK'}
            or len(classes) != 6 or set(truth) - set(classes)
            or p.ndim != 3 or p.shape[1:] != (len(truth), len(classes))
            or not np.isfinite(p).all() or np.any(p < 0)
            or not np.allclose(p.sum(axis=2), 1)):
        raise ValueError('Expected normalized six-class PAD OOF probabilities')
    y = np.isin(truth, list(cancer))
    keys = ('auc', 'sensitivity_at_specificity_0.80', 'sensitivity_at_specificity_0.90')
    if len(np.unique(y)) != 2:
        return dict.fromkeys(keys)
    values = []
    for run in p:
        score = run[:, [i for i, name in enumerate(classes) if name in cancer]].sum(axis=1)
        fpr, tpr, _ = roc_curve(y, score, drop_intermediate=False)
        values.append([roc_auc_score(y, score)] +
                      [float(tpr[fpr <= 1 - specificity + 1e-12].max()) for specificity in (.8, .9)])
    return dict(zip(keys, np.mean(values, axis=0).tolist()))


def aggregate_lesions(truth, groups, lesion_ids, probabilities):
    truth, groups, lesion_ids = map(np.asarray, (truth, groups, lesion_ids))
    p = np.asarray(probabilities, dtype=float)
    if (truth.ndim != 1 or truth.shape != groups.shape or truth.shape != lesion_ids.shape
            or not len(truth) or p.ndim != 3 or p.shape[1] != len(truth)
            or not np.isfinite(p).all() or np.any(p < 0)
            or not np.allclose(p.sum(axis=2), 1) or np.any(lesion_ids == '')):
        raise ValueError('Expected aligned nonempty identities and normalized OOF probabilities')
    buckets = {}
    for index, lesion in enumerate(lesion_ids):
        buckets.setdefault(str(lesion), []).append(index)
    labels, patients, averaged = [], [], []
    for indices in buckets.values():
        if len(set(truth[indices])) != 1 or len(set(groups[indices])) != 1:
            raise ValueError('Conflicting lesion diagnosis or patient identity')
        labels.append(truth[indices[0]])
        patients.append(groups[indices[0]])
        averaged.append(p[:, indices].mean(axis=1))
    return np.asarray(labels), np.asarray(patients), np.stack(averaged, axis=1)
