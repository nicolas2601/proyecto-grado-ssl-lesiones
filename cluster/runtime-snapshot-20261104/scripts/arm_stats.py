"""Per-arm statistics against the baseline readout (fast; no build_union import).

Δ media 3 CV [IC95]: mean over CV runs of the arm minus mean over CV runs of the baseline,
paired patient bootstrap (500 resamples, seed 20261025), same statistic as reactivation-20261026-v2.
Usage: python arm_stats.py <arm_eval_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from evaluate_overlay_readout import load_labels

BASE = Path('results/baseline-current-recipe-seed42-evaluation-v1')
BASE_FILES = {'PAD': BASE / 'balanced-linear-reader-pad-adapt.json', 'HAM': BASE / 'balanced-linear-reader-ham-dev.json'}
META = {
    'PAD': ('/home/jovyan/work/pad_ufes_20/metadata.csv', '14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527'),
    'HAM': ('/home/jovyan/work/datos/ham10000/HAM10000_metadata.csv', '06aee5e603d313ba69de38263053c792a1dff030eb9c44a96ceb477ceefb99a5'),
}
CANCER = {'MEL', 'BCC', 'SCC'}
SEED = 20261025


def labels(kind, base):
    meta, sha = META[kind]
    return base['image_ids'], load_labels(kind, [{'image_id': i} for i in base['image_ids']], Path(meta), sha)


def runs_scores(obj):
    key = 'oof_probabilities' if 'oof_probabilities' in obj['runs'][0] else 'oof_scores'
    return [np.array(r[key]) for r in obj['runs']]


def arm_vs_base(kind, arm_file):
    base = json.loads(BASE_FILES[kind].read_text())
    arm = json.loads(Path(arm_file).read_text())
    if base['image_ids'] != arm['image_ids']:
        raise ValueError(f'id order mismatch for {arm_file}')
    classes = np.array(base['classes'])
    ids, y = labels(kind, base)
    y = np.asarray(y)
    groups = np.asarray(base['groups'])
    nr, br = runs_scores(arm), runs_scores(base)
    mf = lambda yt, p: f1_score(yt, classes[np.argmax(p, 1)], labels=list(classes), average='macro', zero_division=0)
    ug = np.unique(groups)
    idx = {k: np.where(groups == k)[0] for k in ug}
    rng = np.random.default_rng(SEED)
    samples = [np.concatenate([idx[k] for k in rng.choice(ug, len(ug))]) for _ in range(500)]
    nm = [mf(y, p) for p in nr]
    bm = [mf(y, p) for p in br]
    ds = [np.mean([mf(y[s], p[s]) for p in nr]) - np.mean([mf(y[s], p[s]) for p in br]) for s in samples]
    return {
        'mean_f1': float(np.mean(nm)), 'base_mean_f1': float(np.mean(bm)),
        'dmean': float(np.mean(nm) - np.mean(bm)), 'dseed': float(nm[0] - bm[0]),
        'ic': [round(float(np.percentile(ds, 2.5)), 4), round(float(np.percentile(ds, 97.5)), 4)],
    }


def auc_recall(arm_file, kind='PAD'):
    arm = json.loads(Path(arm_file).read_text())
    classes = arm['classes']
    ids, y = labels(kind, arm)
    y = np.asarray([classes.index(v) if isinstance(v, str) else int(v) for v in y])
    cancer_idx = [classes.index(c) for c in CANCER]
    aucs, rec = [], {c: [] for c in ('MEL', 'SCC', 'BCC')}
    for p in runs_scores(arm):
        aucs.append(roc_auc_score(np.isin(y, cancer_idx).astype(int), p[:, cancer_idx].sum(1)))
        pred = p.argmax(1)
        for c in rec:
            k = classes.index(c)
            rec[c].append(float((pred[y == k] == k).mean()))
    return {'auc_cancer': float(np.mean(aucs)), 'recall': {c: float(np.mean(v)) for c, v in rec.items()}}


if __name__ == '__main__':
    eval_dir = Path(sys.argv[1])
    out = {}
    for kind, name in (('PAD', 'PAD-readout.json'), ('HAM', 'HAM-readout-06aee5.json')):
        out[kind] = arm_vs_base(kind, eval_dir / name)
    out.update(auc_recall(eval_dir / 'PAD-readout.json'))
    print(json.dumps(out))
