"""Paired patient bootstrap conditional on already frozen OOF predictions."""
import numpy as np

from probe_preprocessing_readout import metrics


def patient_indices(groups, rng):
    groups = np.asarray(groups, dtype=str)
    if groups.ndim != 1 or not len(groups) or np.any(groups == ""):
        raise ValueError("Expected a nonempty patient-group vector")
    unique = np.unique(groups)
    selected = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(groups == patient) for patient in selected])


def flat_metrics(truth, probabilities, classes):
    records = [metrics(truth, run, classes) for run in probabilities]
    output = {key: float(np.mean([r[key] for r in records]))
              for key in ("macro_f1", "balanced_accuracy", "ece")}
    for name in ("recall", "auc_ovr"):
        for label in classes:
            values = [r[name][label] for r in records if r[name][label] is not None]
            output[f"{name}/{label}"] = float(np.mean(values)) if values else None
    return output


def interval(values, point):
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return {"point": point, "ci95": np.quantile(finite, [.025, .975]).tolist() if finite else None,
            "valid_resamples": len(finite)}


def bootstrap(truth, groups, probabilities_by_arm, classes, resamples=500, seed=20261025):
    truth, groups = np.asarray(truth), np.asarray(groups)
    arms = {name: np.asarray(value, dtype=float) for name, value in probabilities_by_arm.items()}
    baseline = "balanced-linear-reader"
    if baseline not in arms or truth.shape != groups.shape or resamples < 1:
        raise ValueError("Expected aligned patient labels, baseline and positive resample count")
    for probabilities in arms.values():
        if (probabilities.ndim != 3 or probabilities.shape[1:] != (len(truth), len(classes))
                or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(axis=2), 1)):
            raise ValueError("Expected finite normalized CV x image x class OOF probabilities")
    points = {name: flat_metrics(truth, p, classes) for name, p in arms.items()}
    samples = {name: {key: [] for key in point} for name, point in points.items()}
    differences = {name: {key: [] for key in point} for name, point in points.items() if name != baseline}
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        indices = patient_indices(groups, rng)
        current = {name: flat_metrics(truth[indices], p[:, indices], classes) for name, p in arms.items()}
        for name, record in current.items():
            for key, value in record.items():
                samples[name][key].append(value)
                if name != baseline:
                    reference = current[baseline][key]
                    differences[name][key].append(value - reference if value is not None and reference is not None else None)
    results = {name: {key: interval(values, points[name][key]) for key, values in record.items()}
               for name, record in samples.items()}
    paired = {}
    for name, record in differences.items():
        paired[name] = {}
        for key, values in record.items():
            a, b = points[name][key], points[baseline][key]
            paired[name][key] = interval(values, a - b if a is not None and b is not None else None)
    return {"conditional_on_fixed_OOF_predictions": True, "resamples": resamples, "seed": seed,
            "unit": "identity_component_id; sample whole patient image vectors with replacement",
            "arms": results, "paired_differences": paired,
            "limits": "No model refitting, hyperparameter-selection uncertainty or independent training-seed replication. Undefined AUC resamples omitted per metric with valid counts reported."}
