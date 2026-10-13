"""Validate the fixed W3 image sample and emit readiness evidence without training."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dermssl.w3_preflight import original_manifest, readiness, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--data-preflight', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Use a new output directory; existing evidence is never overwritten')
    registry = json.loads(args.registry.read_text())
    data = json.loads(args.data_preflight.read_text())
    gates = readiness(registry)
    rows = original_manifest(data, registry['sample_count'])
    versions = {}
    for name in ('numpy', 'scipy', 'scikit-learn', 'scikit-image', 'pyod', 'jpype1', 'numba', 'torch', 'wandb'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    report = {'status': 'inputs_verified_methods_gated', 'code_base_revision': revision,
              'source_hashes': {str(p.relative_to(Path(__file__).resolve().parents[1])): sha256(p)
                                for p in (Path(__file__).resolve(),
                                          Path(__file__).resolve().parents[1] / 'src/dermssl/w3_preflight.py')},
              'registry_sha256': sha256(args.registry), 'data_preflight_sha256': sha256(args.data_preflight),
              'data_validation_revision': data['source_revision'], 'count': len(rows),
              'selection': 'unchanged W0 development IDs; not an independent expert evaluation',
              'candidate_readiness': gates, 'python': platform.python_version(), 'versions': versions,
              'methods_executed': [], 'gpu_training_started': False,
              'baseline_note': 'W0 student-r2 pseudomasks are not W2 S0 CCAM seeds',
              'wandb_note': 'No run created: this is preparation, not an experiment result'}
    args.out.mkdir(parents=True)
    (args.out / 'original_inputs.json').write_text(json.dumps(rows, indent=2) + '\n')
    (args.out / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'count': len(rows),
                      'candidate_readiness': gates, 'methods_executed': []}, indent=2))


if __name__ == '__main__':
    main()
