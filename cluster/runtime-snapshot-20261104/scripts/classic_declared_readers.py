"""Declared classical readers on the 60 handcrafted features (rules fixed in the v3 ledger before running).

PRIMARY   : shared balanced logistic reader (evaluate_encoder), features standardised without labels.
SECONDARY : RBF-SVM class_weight=balanced, argmax of one-vs-rest decision scores, no further tuning.
"""
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluate_overlay_readout import (C_GRID, CLASSES, INNER_FOLDS, OUTER_FOLDS, SEEDS, evaluate_encoder,
                                      grouped_folds, load_labels, validate_support)

PRIMARY, SECONDARY = 'primary', 'secondary'


def standardise(features):
    return StandardScaler().fit_transform(features)


def svm_scores(features, labels, groups, classes):
    validate_support(labels, groups, classes)
    runs = []
    for seed in SEEDS:
        oof, seen = np.zeros((len(labels), len(classes))), np.zeros(len(labels), int)
        for fold, (train, test) in enumerate(grouped_folds(labels, groups, OUTER_FOLDS, seed)):
            scores = {c: [] for c in C_GRID}
            for inner_train, valid in grouped_folds(labels[train], groups[train], INNER_FOLDS, seed + fold):
                for c in C_GRID:
                    fitted = SVC(C=c, kernel='rbf', gamma='scale', class_weight='balanced').fit(
                        features[train[inner_train]], labels[train[inner_train]])
                    predicted = fitted.predict(features[train[valid]])
                    scores[c].append(f1_score(labels[train[valid]], predicted, labels=list(classes),
                                              average='macro', zero_division=0))
            selected = max(C_GRID, key=lambda c: np.mean(scores[c]))
            final = SVC(C=selected, kernel='rbf', gamma='scale', class_weight='balanced').fit(
                features[train], labels[train])
            if tuple(final.classes_) != tuple(classes):
                raise ValueError('Training fold class order differs from fixed ontology')
            oof[test] = final.decision_function(features[test])
            seen[test] += 1
        if not np.all(seen == 1):
            raise ValueError('OOF coverage must be exactly one prediction per allowed image')
        predicted = np.array(classes)[np.argmax(oof, 1)]
        runs.append({'seed': seed, 'macro_f1': float(f1_score(labels, predicted, labels=list(classes),
                                                               average='macro', zero_division=0)),
                     'balanced_accuracy': float(balanced_accuracy_score(labels, predicted)),
                     'oof_scores': oof.tolist()})
        print(json.dumps({'CV_seed': seed, 'macro_f1': runs[-1]['macro_f1']}), flush=True)
    return runs


def main(protocol_path, mode):
    protocol = json.loads(Path(protocol_path).read_text())
    rows = [json.loads(line) for line in Path(protocol['manifest']).read_text().splitlines()]
    if Path(protocol['manifest']).name != 'pad_adapt.jsonl':
        raise ValueError('Only the approved PAD adaptation cohort is allowed')
    data = np.load(protocol['features'], allow_pickle=False)
    if data['image_ids'].tolist() != [r['image_id'] for r in rows]:
        raise ValueError('Feature rows do not match the approved manifest order')
    labels = load_labels('PAD', rows, Path(protocol['metadata']), protocol['metadata_sha256'])
    groups = np.asarray([r['identity_component_id'] for r in rows])
    classes = CLASSES['PAD']
    features = standardise(data['features'])
    out = Path(protocol['output_directory']) / f'{mode}.json'
    if out.exists():
        raise FileExistsError('Preserve reader evidence')
    out.parent.mkdir(parents=True, exist_ok=True)
    if mode == PRIMARY:
        runs = evaluate_encoder(features, labels, groups, classes)
        payload = {'reader': 'balanced-logistic (shared)', 'runs': runs}
    elif mode == SECONDARY:
        payload = {'reader': 'rbf-svm balanced argmax-decision', 'runs': svm_scores(features, labels, groups, classes)}
    else:
        raise ValueError('mode must be primary or secondary')
    payload.update({'classes': list(classes), 'image_ids': [r['image_id'] for r in rows],
                    'groups': groups.tolist(), 'labels': labels.tolist(),
                    'scope': 'PAD target_adapt development only; not clinical validation'})
    out.write_text(json.dumps(payload))
    print(json.dumps({'mode': mode, 'mean_macro_f1': float(np.mean([r['metrics']['macro_f1'] if 'metrics' in r
                                                                  else r['macro_f1'] for r in payload['runs']]))}),
          flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
