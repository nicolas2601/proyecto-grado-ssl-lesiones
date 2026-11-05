"""Línea 3 supervised readers on frozen SimSiam features (declared in ledger before running; never pooled with pure SSL)."""
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from evaluate_overlay_readout import INNER_FOLDS, OUTER_FOLDS, SEEDS, grouped_folds

MLP_ALPHAS = (1e-4, 1e-3, 1e-2)


def balanced_oversample(features, labels, rng):
    classes, counts = np.unique(labels, return_counts=True)
    target = counts.max()
    picked = [np.arange(len(labels))]
    for label, count in zip(classes, counts):
        pool = np.flatnonzero(labels == label)
        picked.append(rng.choice(pool, size=target - count, replace=True) if count < target else np.array([], int))
    index = np.concatenate(picked)
    return features[index], labels[index]


def fit_mlp(features, labels, alpha, seed):
    rng = np.random.default_rng(seed)
    scaler = StandardScaler().fit(features)
    x, y = balanced_oversample(scaler.transform(features), labels, rng)
    model = MLPClassifier(hidden_layer_sizes=(64,), alpha=alpha, max_iter=500, random_state=seed).fit(x, y)
    return scaler, model


def mlp_probabilities(model_pair, features):
    scaler, model = model_pair
    return model.predict_proba(scaler.transform(features))


def evaluate_readers(features, labels, groups, classes, seeds=SEEDS):
    classes = list(classes)
    runs = []
    for seed in seeds:
        oof_mlp = np.zeros((len(labels), len(classes)))
        seen = np.zeros(len(labels), int)
        for fold, (train, test) in enumerate(grouped_folds(labels, groups, OUTER_FOLDS, seed)):
            scores = {a: [] for a in MLP_ALPHAS}
            for inner_train, valid in grouped_folds(labels[train], groups[train], INNER_FOLDS, seed + fold):
                for a in MLP_ALPHAS:
                    pair = fit_mlp(features[train[inner_train]], labels[train[inner_train]], a, seed)
                    predicted = np.array(classes)[np.argmax(mlp_probabilities(pair, features[train[valid]]), 1)]
                    scores[a].append(f1_score(labels[train[valid]], predicted, labels=classes, average='macro', zero_division=0))
            selected = max(MLP_ALPHAS, key=lambda a: np.mean(scores[a]))
            pair = fit_mlp(features[train], labels[train], selected, seed)
            oof_mlp[test] = mlp_probabilities(pair, features[test])
            seen[test] += 1
        if not np.all(seen == 1):
            raise ValueError('OOF coverage must be exactly one prediction per allowed image')
        runs.append({'seed': seed, 'oof_mlp': oof_mlp.tolist()})
        print(json.dumps({'CV_seed': seed, 'mlp_macro_f1': f1_score(labels, np.array(classes)[np.argmax(oof_mlp, 1)], labels=classes, average='macro', zero_division=0)}), flush=True)
    return runs


if __name__ == '__main__':
    print('library module; invoked by the declared run scripts only', file=sys.stderr)
