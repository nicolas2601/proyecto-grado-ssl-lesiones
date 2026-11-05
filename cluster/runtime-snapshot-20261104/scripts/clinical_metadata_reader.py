"""Patient metadata readers with explicit leakage exclusions and fold-local fitting."""
import csv
import hashlib
from pathlib import Path
import time

import numpy as np

from evaluate_overlay_readout import group_sha
from probe_preprocessing_readout import C_GRID, INNER_FOLDS, OUTER_FOLDS, SEEDS, grouped_folds, metrics

CATEGORICAL = ("region", "gender", "smoke", "drink", "pesticide", "itch", "grew", "hurt", "changed", "bleed", "elevation")
FEATURE_COLUMNS = ("age",) + CATEGORICAL
FORBIDDEN_COLUMNS = {"diagnostic", "biopsed", "lesion_id", "img_id", "patient_id", "skin_cancer_history", "cancer_history"}
MISSING = {"", "unk", "unknown", "nan", "none", "n/a", "na"}


def load_metadata(rows, path, expected_sha):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_sha:
        raise ValueError("PAD metadata changed")
    wanted = {row["image_id"] for row in rows}
    if len(wanted) != len(rows):
        raise ValueError("Duplicate allowed identities")
    found = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(FEATURE_COLUMNS) | {"img_id"} <= set(reader.fieldnames or ()):
            raise ValueError("Missing allowlisted metadata columns")
        for row in reader:
            identity = Path(row["img_id"]).stem
            if identity not in wanted:
                continue
            if identity in found:
                raise ValueError("Duplicate allowed metadata identity")
            values = []
            for name in FEATURE_COLUMNS:
                value = str(row[name]).strip().lower() if row[name] is not None else ""
                if value in MISSING:
                    values.append(np.nan)
                elif name == "age":
                    number = float(value)
                    if not np.isfinite(number) or not 0 <= number <= 120:
                        raise ValueError("Age outside declared range")
                    values.append(number)
                else:
                    values.append(value)
            found[identity] = values
    if set(found) != wanted:
        raise ValueError("Missing allowed metadata identity")
    return np.asarray([found[row["image_id"]] for row in rows], dtype=object)


def preprocessor(image_dimensions):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import MissingIndicator, SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    offset = image_dimensions
    transformers = []
    if image_dimensions:
        transformers.append(("image", "passthrough", list(range(image_dimensions))))
    transformers.extend([
        ("age", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                           ("scale", StandardScaler())]), [offset]),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="MISSING", keep_empty_features=True)),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         list(range(offset + 1, offset + len(FEATURE_COLUMNS)))),
        ("missing", MissingIndicator(features="all", sparse=False),
         list(range(offset, offset + len(FEATURE_COLUMNS)))),
    ])
    return ColumnTransformer(transformers, sparse_threshold=0)


def estimator(c, image_dimensions):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    return Pipeline([("preprocess", preprocessor(image_dimensions)),
                     ("reader", LogisticRegression(C=c, class_weight="balanced", max_iter=2000))])


def evaluate(table, image_features, labels, groups, classes, fold_truth=None):
    from sklearn.metrics import f1_score
    image_dimensions = 0 if image_features is None else image_features.shape[1]
    features = table if image_features is None else np.concatenate((image_features, table), axis=1)
    stratification = labels if fold_truth is None else fold_truth
    runs = []
    for seed in SEEDS:
        started = time.monotonic()
        oof, seen, traces = np.zeros((len(labels), len(classes))), np.zeros(len(labels), int), []
        for number, (train, test) in enumerate(grouped_folds(stratification, groups, OUTER_FOLDS, seed)):
            scores = {c: [] for c in C_GRID}
            inner_folds = grouped_folds(stratification[train], groups[train], INNER_FOLDS, seed + number)
            for inner_train, valid in inner_folds:
                for c in C_GRID:
                    model = estimator(c, image_dimensions).fit(features[train[inner_train]], labels[train[inner_train]])
                    predicted = model.predict(features[train[valid]])
                    scores[c].append(f1_score(labels[train[valid]], predicted, labels=classes,
                                              average="macro", zero_division=0))
            c = max(C_GRID, key=lambda value: np.mean(scores[value]))
            model = estimator(c, image_dimensions).fit(features[train], labels[train])
            probabilities = model.predict_proba(features[test])
            for column, label in enumerate(model.classes_):
                oof[test, classes.index(label)] = probabilities[:, column]
            seen[test] += 1
            traces.append({"fold": number, "C": c, "train_groups_sha256": group_sha(train, groups),
                           "test_groups_sha256": group_sha(test, groups),
                           "preprocessing_fit": "Outer train only; inner preprocessors fitted on inner train only"})
        if not np.all(seen == 1):
            raise ValueError("Expected exactly one OOF prediction per allowed image")
        runs.append({"seed": seed, "folds": traces, "metrics": metrics(labels, oof, classes),
                     "oof_probabilities": oof.tolist(), "seconds": time.monotonic() - started})
        print(f"metadata image_dimensions={image_dimensions} CV_seed={seed} macro_f1={runs[-1]['metrics']['macro_f1']:.6f}", flush=True)
    return runs
