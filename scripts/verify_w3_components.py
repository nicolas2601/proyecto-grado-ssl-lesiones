"""Synthetic W3 component verification with optional aggregate-only W&B tracking."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'plan-segmentacion' / 'comun'))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wandb-project', default='')
    parser.add_argument('--wandb-run-name', default='w3-components-synthetic-v8')
    parser.add_argument('--require-online', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refuse to overwrite prior verification evidence')
    if args.require_online and (not args.wandb_project or os.environ.get('WANDB_MODE', '').lower() in {'offline', 'disabled'}):
        raise ValueError('Explicit online W&B project required')
    files = ['src/dermssl/w3_components.py', 'src/dermssl/w3_sled_adapter.py',
             'src/dermssl/w3_usl_models.py', 'tests/test_w3_components.py',
             'tests/test_w3_usl_models.py', 'scripts/verify_w3_components.py']
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}
    suite = unittest.TestSuite()
    for pattern in ('test_w3_components.py', 'test_w3_usl_models.py'):
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern=pattern))
    start = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    elapsed = time.monotonic() - start
    report = {'status': 'component_tests_passed_not_method_smoke' if result.wasSuccessful() else 'tests_failed',
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'tests_run': result.testsRun, 'tests_failed': len(result.failures),
              'tests_errors': len(result.errors), 'tests_skipped': len(result.skipped),
              'elapsed_seconds': elapsed, 'files_sha256': hashes,
              'real_images_read': 0, 'ground_truth_read': False,
              'third_party_sled_executed': False, 'training_started': False,
              'methods_executed': [], 'gpu_used': False, 'wandb_run_created': False}
    if args.wandb_project:
        log_dir = args.output.parent.resolve() / 'wandb-components'
        log_dir.mkdir(parents=True, exist_ok=True)
        prior_dir = os.environ.get('WANDB_DIR')
        os.environ['WANDB_DIR'] = str(log_dir)
        try:
            from nucleo.wandb_logging import WandbSink
            sink = WandbSink(project=args.wandb_project, run_name=args.wandb_run_name,
                             group='w3-reproducibility-v8', tags=['synthetic', 'component-verification'],
                             config={'stage': 'synthetic_components', 'seed': 42,
                                     'scientific_efficacy_evaluation': False, 'files_sha256': hashes})
            report['wandb_run_created'] = sink.enabled
            if sink.enabled:
                report['wandb_url'] = sink._run.url
                report['wandb_run_id'] = sink._run.id
                sink.log({'tests_run': result.testsRun, 'tests_failed': len(result.failures),
                          'tests_errors': len(result.errors), 'tests_skipped': len(result.skipped),
                          'elapsed_seconds': elapsed, 'real_images_read': 0, 'training_started': 0})
                sink.finish()
        finally:
            if prior_dir is None:
                os.environ.pop('WANDB_DIR', None)
            else:
                os.environ['WANDB_DIR'] = prior_dir
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'files_sha256'}, indent=2))
    return 0 if result.wasSuccessful() and (not args.require_online or report['wandb_run_created']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
