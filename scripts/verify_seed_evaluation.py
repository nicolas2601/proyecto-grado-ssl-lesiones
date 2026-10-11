"""Verify persisted expert-pilot labels, metrics and paired bootstrap."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'plan-segmentacion/comun/evaluacion'))
import evaluate_segmentation as ev

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    records = [json.loads(line) for line in (root / 'records.jsonl').read_text().splitlines()]
    selection = json.loads((root / 'selection.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    assert len(records) == len(selection) == summary['n'] == 100
    assert [r['image_id'] for r in records] == [r['image_id'] for r in selection]
    pixel_accuracy = {arm: [] for arm in ['S0', 'S1', 'S2', 'S3', 'S4']}
    for selected, row in zip(selection, records):
        for key in ['image', 'mask']:
            path = Path(selected[key + '_path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == selected[key + '_sha256']
        folder = root / row['image_id']
        truth = np.asarray(Image.open(folder / 'reference.png')) > 0
        for arm, stored in row['arms'].items():
            path = folder / (arm + '_labels.png')
            assert hashlib.sha256(path.read_bytes()).hexdigest() == stored['mask_sha256']
            labels = np.asarray(Image.open(path))
            assert set(np.unique(labels)) <= {0, 1, 255}
            pixel_accuracy[arm].append(float(np.mean((labels == 1) == truth)))
            for metric, value in ev.compute_binary_metrics(labels == 1, truth).items():
                if stored[metric] is None:
                    assert not np.isfinite(value)
                else:
                    assert np.isclose(value, stored[metric], rtol=0, atol=1e-12)
    draws = np.random.default_rng(45).integers(0, 100, size=(4000, 100))
    for arm in ['S0', 'S1', 'S2', 'S3', 'S4']:
        values = np.array([row['arms'][arm]['dice'] for row in records])
        assert np.isclose(values.mean(), summary['arms'][arm]['mean_dice'], rtol=0, atol=1e-12)
        if arm != 'S0':
            delta = values - np.array([row['arms']['S0']['dice'] for row in records])
            interval = np.quantile(delta[draws].mean(axis=1), [.025, .975])
            assert np.allclose(interval, summary['paired_dice'][arm + '_minus_S0']['ci95_percentile'], rtol=0, atol=1e-12)
    result = {'status': 'passed', 'pairs': 100, 'label_files': 500,
              'source_file_hashes': True, 'binary_metrics_recomputed': True,
              'dice_bootstrap_recomputed': True,
              'mean_pixel_accuracy': {arm: float(np.mean(values))
                                      for arm, values in pixel_accuracy.items()},
              'hd95_recomputed': False}
    (root / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
