"""Detached single-GPU launch for one-factor SimSiam variants; no campaign GPU-hour cap (v3)."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from one_factor_gate import validate_one_factor
from supervise_phone_source_job import free_memory

ROOT = Path(__file__).resolve().parents[1]
MIN_FREE_MIB = 12288
BASELINE = ROOT / 'config/ssl-overlay-v1/A-s42.json'


def gate(smoke, free_mib):
    if smoke.get('status') != 'COMPLETE' or smoke.get('steps') != 40:
        raise ValueError('Exact 40-step smoke replay required before full launch')
    if free_mib <= MIN_FREE_MIB:
        raise ValueError('Selected GPU needs more than 12GiB free memory')


def launch(run_config, smoke_record_path, gpu):
    config = json.loads(Path(run_config).read_text())
    baseline = json.loads(BASELINE.read_text())['base_config']
    validate_one_factor(config['base_config'], baseline)
    gate(json.loads(Path(smoke_record_path).read_text()), free_memory()[gpu])
    out = Path(config['base_config']['output_dir'])
    if out.exists():
        raise FileExistsError('No duplicate launch or implicit resume')
    logs = ROOT / 'results/one-factor-v1' / out.name
    logs.mkdir(parents=True)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONPATH': str(ROOT / 'scripts')}
    seconds = int(config['training_gpu_hours_limit'] * 3600)
    with (logs / 'training.log').open('xb') as log:
        child = subprocess.Popen(['/usr/bin/timeout', '--signal=TERM', '--kill-after=5s', str(seconds),
                                  sys.executable, str(ROOT / 'scripts/one_factor_ssl_runtime.py'), run_config],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        (logs / 'child.json').write_text(json.dumps({'pid': child.pid, 'gpu': gpu, 'seconds': seconds}))
        reason = 'process_exit'
        while True:
            try:
                code = child.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                if free_memory()[gpu] < 2048:
                    reason = 'own_job_stopped_to_preserve_GPU_memory'
                    os.killpg(child.pid, signal.SIGTERM)
                    code = child.wait()
                    break
    record = {'exit_code': code, 'reason': reason, 'gpu': gpu}
    (logs / 'supervisor-receipt.json').write_text(json.dumps(record, indent=2))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_config', type=Path)
    parser.add_argument('--smoke-record', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    print(json.dumps(launch(args.run_config, args.smoke_record, args.gpu)), flush=True)
