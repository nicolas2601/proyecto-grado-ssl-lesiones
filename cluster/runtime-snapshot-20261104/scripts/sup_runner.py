import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
sys.path.insert(0, 'scripts')
from evaluate_overlay_readout import CLASSES, load_labels
from supervised_readers import evaluate_readers
kind = sys.argv[1]
R = Path('results/content-exclusion-overlay-v1')
if kind == 'PAD':
    feat = 'results/baseline-current-recipe-seed42-evaluation-v1/adapted-simsiam-pad-adapt-features.npz'
    manifest, meta, sha = R / 'pad_adapt.jsonl', '/home/jovyan/work/pad_ufes_20/metadata.csv', '14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527'
    base = 'results/baseline-current-recipe-seed42-evaluation-v1/balanced-linear-reader-pad-adapt.json'
else:
    feat = 'results/baseline-current-recipe-seed42-evaluation-v1/adapted-simsiam-ham-dev-features.npz'
    manifest, meta, sha = R / 'ham_dev.jsonl', '/home/jovyan/work/datos/ham10000/HAM10000_metadata.csv', '06aee5e603d313ba69de38263053c792a1dff030eb9c44a96ceb477ceefb99a5'
    base = 'results/baseline-current-recipe-seed42-evaluation-v1/balanced-linear-reader-ham-dev.json'
rows = [json.loads(l) for l in open(manifest)]
data = np.load(feat, allow_pickle=False)
assert data['image_ids'].tolist() == [r['image_id'] for r in rows]
labels = load_labels(kind, rows, Path(meta), sha)
groups = np.asarray([r['identity_component_id'] for r in rows])
classes = list(CLASSES[kind])
runs = evaluate_readers(data['features'], labels, groups, classes)
b = json.load(open(base))
out = {'kind': kind, 'classes': classes, 'image_ids': [r['image_id'] for r in rows], 'groups': groups.tolist(),
       'labels': labels.tolist(), 'base_runs': [r['oof_probabilities'] for r in b['runs']],
       'mlp_runs': [r['oof_mlp'] for r in runs], 'label_metadata_sha256': sha,
       'scope': 'Línea 3 SUPERVISED, development only; not pooled with pure SSL'}
Path('results/supervised-readers-v1').mkdir(parents=True, exist_ok=True)
Path(f'results/supervised-readers-v1/{kind}-readers.json').write_text(json.dumps(out))
print('done', kind, flush=True)
