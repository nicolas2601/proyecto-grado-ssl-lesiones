"""Paired patient uncertainty with several explicit frozen-reader reference models."""
import numpy as np

from grouped_readout_bootstrap import flat_metrics, interval, patient_indices


def bootstrap_references(truth, groups, probabilities_by_arm, classes, references,
                         resamples=500, seed=20261025):
    truth, groups = np.asarray(truth), np.asarray(groups)
    arms = {name: np.asarray(value, dtype=float) for name, value in probabilities_by_arm.items()}
    references = tuple(references)
    if (truth.shape != groups.shape or resamples < 1 or not references
            or len(set(references)) != len(references) or set(references) - set(arms)):
        raise ValueError("Expected aligned patients and distinct existing reference readers")
    cv_counts = set()
    for p in arms.values():
        if (p.ndim != 3 or not p.shape[0] or p.shape[1:] != (len(truth), len(classes))
                or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=2), 1)):
            raise ValueError("Expected finite normalized CV x image x class probabilities")
        cv_counts.add(p.shape[0])
    if len(cv_counts) != 1:
        raise ValueError("Reference readers must use paired CV repetitions")
    points = {name: flat_metrics(truth, p, classes) for name, p in arms.items()}
    samples = {name: {key: [] for key in point} for name, point in points.items()}
    differences = {reference: {name: {key: [] for key in point} for name, point in points.items() if name != reference}
                   for reference in references}
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        indices = patient_indices(groups, rng)
        current = {name: flat_metrics(truth[indices], p[:, indices], classes) for name, p in arms.items()}
        for name, record in current.items():
            for key, value in record.items():
                samples[name][key].append(value)
        for reference in references:
            for name, record in differences[reference].items():
                for key, values in record.items():
                    a, b = current[name][key], current[reference][key]
                    values.append(a - b if a is not None and b is not None else None)
    arm_intervals = {name: {key: interval(values, points[name][key]) for key, values in record.items()}
                     for name, record in samples.items()}
    paired = {}
    for reference, comparisons in differences.items():
        paired[reference] = {}
        for name, record in comparisons.items():
            paired[reference][name] = {}
            for key, values in record.items():
                a, b = points[name][key], points[reference][key]
                paired[reference][name][key] = interval(values, a - b if a is not None and b is not None else None)
    return {"conditional_on_fixed_OOF_predictions": True, "references": references,
            "resamples": resamples, "seed": seed, "unit": "Whole identity_component_id image vectors",
            "arms": arm_intervals, "paired_differences": paired,
            "limits": "No model refitting or hyperparameter/method-selection uncertainty. Reused CV seeds are not independent training replicates. Many reported comparisons are exploratory,not multiplicity-corrected clinical evidence."}
