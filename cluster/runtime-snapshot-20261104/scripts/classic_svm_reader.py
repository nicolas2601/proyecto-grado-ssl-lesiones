"""Nested-CV RBF-SVM on handcrafted features (PAD target_adapt cohort only; no test roles)."""
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from evaluate_overlay_readout import (C_GRID, CLASSES, INNER_FOLDS, OUTER_FOLDS, SEEDS,
                                      grouped_folds, load_labels, metrics, validate_support)

ROOT = Path(__file__).resolve().parents[1]


def model(c, seed):
    return make_pipeline(StandardScaler(), SVC(C=c, kernel='rbf', gamma='scale', class_weight='balanced',
                                               probability=True, random_state=seed))


def evaluate(features, labels, groups, classes):
    validate_support(labels, groups, classes)
    runs = []
    for seed in SEEDS:
        oof, seen = np.zeros((len(labels), len(classes))), np.zeros(len(labels), int)
        for fold, (train, test) in enumerate(grouped_folds(labels, groups, OUTER_FOLDS, seed)):
            scores = {c: [] for c in C_GRID}
            for inner_train, valid in grouped_folds(labels[train], groups[train], INNER_FOLDS, seed + fold):
                for c in C_GRID:
                    fitted = model(c, seed).fit(features[train[inner_train]], labels[train[inner_train]])
                    predicted = fitted.predict(features[train[valid]])
                    scores[c].append(f1_score(labels[train[valid]], predicted, labels=list(classes),
                                              average='macro', zero_division=0))
            selected = max(C_GRID, key=lambda c: np.mean(scores[c]))
            final = model(selected, seed).fit(features[train], labels[train])
            if tuple(final.classes_) != tuple(classes):
                raise ValueError('Training fold class order differs from fixed ontology')
            oof[test] = final.predict_proba(features[test])
            seen[test] += 1
        if not np.all(seen == 1):
            raise ValueError('OOF coverage must be exactly one prediction per allowed image')
        runs.append({'seed': seed, 'metrics': metrics(labels, oof, classes), 'oof_probabilities': oof.tolist()})
        print(json.dumps({'CV_seed': seed, 'macro_f1': runs[-1]['metrics']['macro_f1']}), flush=True)
    return runs


def main(protocol_path):
    protocol = json.loads(Path(protocol_path).read_text())
    rows = [json.loads(line) for line in Path(protocol['manifest']).read_text().splitlines()]
    if Path(protocol['manifest']).name != 'pad_adapt.jsonl':
        raise ValueError('Only the approved PAD adaptation cohort is allowed')
    data = np.load(protocol['features'], allow_pickle=False)
    if data['image_ids'].tolist() != [r['image_id'] for r in rows]:
        raise ValueError('Feature rows do not match the approved manifest order')
    labels = load_labels('PAD', rows, Path(protocol['metadata']), protocol['metadata_sha256'])
    groups = np.asarray([r['identity_component_id'] for r in rows])
    runs = evaluate(data['features'], labels, groups, CLASSES['PAD'])
    out = Path(protocol['output'])
    if out.exists():
        raise FileExistsError('Preserve reader evidence')
    out.write_text(json.dumps({'reader': 'rbf-svm-handcrafted', 'classes': CLASSES['PAD'], 'runs': runs,
                               'scope': 'PAD target_adapt development only; not clinical validation'}))
    print(json.dumps({'mean_macro_f1': float(np.mean([r['metrics']['macro_f1'] for r in runs]))}), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
