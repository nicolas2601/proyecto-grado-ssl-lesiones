"""Native PAD reference voting with patient-excluded diagnostic priors."""
from __future__ import annotations

import numpy as np

NATIVE_CLASSES = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")
TEMPERATURE = 0.07


def feature_views(cls, mean_patch):
    cls, mean_patch = np.asarray(cls), np.asarray(mean_patch)
    if (cls.ndim != 2 or not cls.shape[0] or cls.shape != mean_patch.shape
            or not np.isfinite(cls).all() or not np.isfinite(mean_patch).all()):
        raise ValueError("Feature blocks must be aligned finite matrices")
    cls_norm, patch_norm = np.linalg.norm(cls, axis=1, keepdims=True), np.linalg.norm(mean_patch, axis=1, keepdims=True)
    if np.any(cls_norm == 0) or np.any(patch_norm == 0):
        raise ValueError("Feature blocks must have nonzero row norms")
    return {"cls": cls, "mean_patch": mean_patch,
            "unit_concat": np.concatenate([cls / cls_norm, mean_patch / patch_norm], axis=1)}


def patient_excluded_vote(indices, similarities, labels, reference_groups, query_groups, balanced=True):
    indices = np.asarray(indices)
    similarities = np.asarray(similarities, dtype=np.float64)
    labels = np.asarray(labels, dtype=str)
    reference_groups, query_groups = np.asarray(reference_groups, dtype=str), np.asarray(query_groups, dtype=str)
    if (indices.ndim != 2 or not indices.shape[0] or not indices.shape[1]
            or similarities.shape != indices.shape or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= len(labels))):
        raise ValueError("Neighbor indices must be nonempty aligned unique integer rows")
    if any(len(set(row)) != len(row) for row in indices):
        raise ValueError("Neighbor indices cannot repeat within a query")
    if labels.ndim != 1 or set(labels) - set(NATIVE_CLASSES):
        raise ValueError("Reference diagnoses must use the six native PAD classes")
    if (reference_groups.shape != labels.shape or query_groups.shape != (len(indices),)
            or np.isin(reference_groups, ["", "None", "nan"]).any()
            or np.isin(query_groups, ["", "None", "nan"]).any()):
        raise ValueError("Every reference and query requires an aligned patient group")
    if not np.isfinite(similarities).all():
        raise ValueError("Neighbor similarities must be finite")
    class_index = {name: index for index, name in enumerate(NATIVE_CLASSES)}
    coded = np.asarray([class_index[label] for label in labels], dtype=np.int64)
    scores = np.zeros((len(indices), len(NATIVE_CLASSES)))
    for query, chosen in enumerate(indices):
        allowed = reference_groups != query_groups[query]
        if not allowed[chosen].all():
            raise ValueError("A neighbor from the query patient is not permitted")
        # The query patient's labels must affect neither neighbors nor class priors.
        counts = np.bincount(coded[allowed], minlength=len(NATIVE_CLASSES)) if balanced else np.ones(len(NATIVE_CLASSES))
        if np.any(counts[coded[chosen]] <= 0):
            raise ValueError("A voted class has no allowed patient-excluded support")
        weights = np.exp((similarities[query] - similarities[query].max()) / TEMPERATURE)
        weights /= counts[coded[chosen]]
        scores[query] = np.bincount(coded[chosen], weights=weights, minlength=len(NATIVE_CLASSES))
    return scores / scores.sum(axis=1, keepdims=True)
