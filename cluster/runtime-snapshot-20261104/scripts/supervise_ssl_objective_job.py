"""Detached, budgeted single-GPU launches for predeclared SSL-only objective arms."""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from supervise_phone_source_job import free_memory

ROOT = Path(__file__).resolve().parents[1]
RUNTIMES = {'simsiam-explora-style': 'explora_ssl_runtime.py',
            'dino-ibot-ema-continued': 'dino_ibot_ssl_runtime.py'}
DEADLINE = datetime.datetime(2026, 10, 26, 10, tzinfo=datetime.timezone.utc)


def validate_budget(debit, request):
    if (not math.isfinite(debit) or not math.isfinite(request)
            or debit < 0 or request <= 0 or debit + request > 20):
        raise ValueError('SSL launch exceeds the20GPU-hour campaign budget')
    return debit + request


def reserve(arm, config, protocol_sha, gpu, memory):
    control = ROOT / 'results' / (arm + '-seed42-launch-v1')
    ledger = json.loads((ROOT / 'results/gpu-budget-ledger-20261026-v1.json').read_text())
    with (ROOT / 'results/ssl-night-budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        debit = ledger['committed_budget_debit_including_active_full_cap_and_reserves']
        for name in RUNTIMES:
            receipt = ROOT / 'results' / (name + '-seed42-launch-v1/receipt.json')
            if receipt.exists():
                debit += json.loads(receipt.read_text())['reserved_gpu_hours']
            smoke = ROOT / 'config' / (name + '-v1/smoke.json')
            if smoke.exists():
                path = Path(json.loads(smoke.read_text())['base_config']['output_dir']) / 'replay-smoke.json'
                if path.exists():
                    record = json.loads(path.read_text())
                    debit += record.get('training_gpu_hours', record.get('allocated_gpu_hours', 0))
        total = validate_budget(debit, config['training_gpu_hours_limit'])
        control.mkdir(exist_ok=False)
        record = {'arm': arm, 'protocol_sha256': protocol_sha, 'gpu': gpu,
                  'reserved_gpu_hours': config['training_gpu_hours_limit'],
                  'committed_budget_gpu_hours': total, 'free_memory_at_launch_mib': memory,
                  'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'scope': 'SSL-only,no diagnostic encoder gradients,no W&B;conservative cap reservations'}
        (control / 'receipt.json').write_text(json.dumps(record, indent=2))
    return control


def main(arm, gpu, protocol_sha):
    folder = ROOT / 'config' / (arm + '-v1')
    path = folder / 'protocol.json'
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if digest(path) != protocol_sha:
        raise ValueError('Protocol digest changed')
    protocol = json.loads(path.read_text())
    for source, expected in protocol['sources_sha256'].items():
        if digest(source) != expected:
            raise ValueError('Sealed config or source changed')
    config_path = folder / 'run.json'
    config = json.loads(config_path.read_text())
    if Path(config['base_config']['output_dir']).exists():
        raise FileExistsError('No duplicate launch or implicit resume')
    smoke_config = json.loads((folder / 'smoke.json').read_text())
    smoke = json.loads((Path(smoke_config['base_config']['output_dir']) / 'replay-smoke.json').read_text())
    if (smoke.get('status') != 'PASS' or smoke.get('model_and_optimizer_bitwise_equal') is not True
            or smoke.get('actual_ID_counts_equal') is not True):
        raise ValueError('Exact GPU replay smoke required')
    memory = free_memory()[gpu]
    if memory <= 12288:
        raise ValueError('Selected GPU needs more than12GiB free')
    remaining = (DEADLINE - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    seconds = int(min(config['training_gpu_hours_limit'] * 3600 - 10, remaining - 10))
    if seconds < 60:
        raise ValueError('Overnight execution window closed')
    control = reserve(arm, config, protocol_sha, gpu, memory)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OPENBLAS_NUM_THREADS': '2',
           'OMP_NUM_THREADS': '2', 'PYTHONPATH': str(ROOT / 'scripts')}
    started = time.monotonic()
    reason = 'process_exit'
    with (control / 'training.log').open('xb') as log:
        child = subprocess.Popen(['/usr/bin/timeout', '--signal=TERM', '--kill-after=5s', str(seconds),
                                  sys.executable, str(ROOT / 'scripts' / RUNTIMES[arm]), str(config_path)],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        (control / 'child.json').write_text(json.dumps({'pid': child.pid, 'gpu': gpu, 'seconds': seconds}))
        (control / 'state').write_text(f'RUNNING gpu={gpu} pid={child.pid}\n')
        while True:
            try:
                code = child.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                if free_memory()[gpu] < 2048:
                    reason = 'own_job_stopped_to_preserve_GPU_memory'
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        code = child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        code = child.wait()
                    break
    record = {'arm': arm, 'gpu': gpu, 'exit_code': code, 'reason': reason,
              'allocated_gpu_hours_upper_bound': (time.monotonic() - started) / 3600}
    (control / 'execution-receipt.json').write_text(json.dumps(record, indent=2))
    (control / 'state').write_text(f'FINISHED exit={code}\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', choices=RUNTIMES)
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args()
    main(args.arm, args.gpu, args.protocol_sha)
