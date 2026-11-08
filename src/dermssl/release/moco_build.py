"""Small validation helpers shared by the MoCo release builder and tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dermssl.release.moco_inputs import CLASSES, ENCODER_SHA256

PROBABILITY_TOLERANCE = 1e-4
MANIFEST_RECIPE = {"feature": "cls4", "layers": [8, 9, 10, 11], "layer_norm": True,
                   "l2_normalize": True, "n_views": 1, "tta": "none", "tau": 0.0,
                   "color_constancy": False, "hue_jitter": 0.0}


def load_reader(path, task):
    artifact = json.loads(Path(path).read_text())
    if task not in CLASSES or artifact.get("classes") != list(CLASSES[task]):
        raise ValueError("Reader classes differ from the registered ontology")
    provenance = artifact.get("provenance", {})
    if provenance.get("encoder_sha256") != ENCODER_SHA256 or provenance.get("task") != task:
        raise ValueError("Reader source encoder/task mismatch")
    weight, bias = np.asarray(artifact["weight"]), np.asarray(artifact["bias"])
    if (artifact.get("feature_dim") != 3072 or weight.shape != (len(CLASSES[task]), 3072)
            or bias.shape != (len(CLASSES[task]),)):
        raise ValueError("Reader shape differs from CLS4")
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("Reader parameters must be finite")
    return artifact


def _check_public(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("image_ids", "image_path", "oof_probabilities", "patient_ids", "group_ids"):
                raise ValueError("Report contains private per-case data")
            _check_public(item)
    elif isinstance(value, list):
        for item in value:
            _check_public(item)


def api_report(metrics):
    _check_public(metrics)
    if metrics.get("release_version") != "10.0.0" or set(metrics.get("cohorts", {})) != {"HAM", "PAD"}:
        raise ValueError("Generated V10 native-cohort metrics are required")
    return {"release": "v10.0.0", "method": {"family": "MoCo-v3-style", "backbone": "DINOv2 ViT-B/14",
            "batch_quota": {"source_fit": 12, "pad_adapt": 4}, "feature": "CLS4 + per-image L2",
            "feature_dim": 3072, "deployment_seed": 42, "inference_views": 1,
            "reader": "Separate development-fitted balanced logistic heads; C is recorded per head"},
            "classification": metrics["cohorts"], "aggregation": metrics.get("aggregation"),
            "source_plan_sha256": metrics.get("source_plan_sha256"),
            "scope": "Exploratory development OOF evidence; not performance of the final refitted head",
            "caveats": ["PAD images were seen unlabeled by SSL: transductive evaluation.",
                        "HAM development has repeatedly informed model selection.",
                        "Three encoder seeds are not an ensemble or nine independent trainings.",
                        "No sealed-test or tonal T0/M1 evaluation is included.",
                        "Research only; not a clinical diagnostic device."]}


def probability_gate(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.ndim != 2:
        raise ValueError("Probability shape mismatch")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Probabilities must be finite")
    normalized = bool(np.all(actual >= 0) and np.all(expected >= 0)
                      and np.allclose(actual.sum(1), 1, atol=1e-5)
                      and np.allclose(expected.sum(1), 1, atol=1e-5))
    maximum = float(np.abs(actual - expected).max(initial=0))
    argmax_equal = bool(np.array_equal(actual.argmax(1), expected.argmax(1)))
    return {"passed": normalized and argmax_equal and maximum <= PROBABILITY_TOLERANCE,
            "normalized": normalized, "max_abs_diff": maximum, "argmax_equal": argmax_equal,
            "tolerance": PROBABILITY_TOLERANCE}
