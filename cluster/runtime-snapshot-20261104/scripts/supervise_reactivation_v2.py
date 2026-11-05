"""Detached, deterministic A/B launches sharing the original20GPU-hour campaign cap."""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from resume_predictor_distill_job import telemetry, stop_own

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    'simsiam-distill-with-predictor-memoryfix': ('simsiam-distill-with-predictor-memoryfix-v2', 'predictor_distill_ssl_runtime.py'),
    'crop-scale-0.4': ('crop-scale-0.4-v1', 'crop_scale_ssl_runtime.py'),
}
PROBE_REPLAY_EVALUATION_ALLOWANCE = .4


def checked_total(base, reserved, requested):
    values = [base, *reserved, requested]
    if any(not math.isfinite(x) or x < 0 for x in values) or requested <= 0 or sum(values) > 20:
        raise ValueError('Reactivated jobs exceed the shared20GPU-hour cap')
    return sum(values)


def main(arm, gpu, protocol_sha):
    folder_name, runtime_name = ARMS[arm]
    folder = ROOT / 'config' / folder_name
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if digest(folder / 'protocol.json') != protocol_sha:
        raise ValueError('Protocol changed')
    protocol = json.loads((folder / 'protocol.json').read_text())
    for source, expected in protocol['sources_sha256'].items():
        if digest(source) != expected:
            raise ValueError('Sealed source/config changed')
    config_path = folder / 'run.json'
    config = json.loads(config_path.read_text())
    run = Path(config['base_config']['output_dir'])
    if not run.resolve().is_relative_to((ROOT / 'runs').resolve()) or run.exists() or config.get('resume'):
        raise ValueError('Only fresh isolated output is allowed;no duplicate/implicit resume')
    smoke_config = json.loads((folder / 'smoke.json').read_text())
    smoke_dir = Path(smoke_config['base_config']['output_dir'])
    smoke = json.loads((smoke_dir / 'replay-smoke.json').read_text())
    if (smoke.get('status') != 'PASS' or smoke.get('model_and_optimizer_bitwise_equal') is not True
            or smoke.get('actual_ID_counts_equal') is not True):
        raise ValueError('Exact GPU replay must pass first')
    receipt_name = 'predictor-distillation-receipt.json' if arm.startswith('simsiam') else 'crop-scale-receipt.json'
    hook = json.loads((smoke_dir / receipt_name).read_text())
    if hook['hook_sha256'] != config['runtime_hook_sha256']:
        raise ValueError('GPU smoke used different hook code')
    if arm.startswith('simsiam'):
        proof = json.loads((ROOT / 'runs/distill-memory-regression-green300-v2/memory-longevity-report.json').read_text())
        if proof['status'] != 'PASS' or proof['steps'] != 300 or proof['runtime_sha256'] != config['runtime_hook_sha256']:
            raise ValueError('The real300-step memory regression must be GREEN on this code')
    else:
        prior = ROOT / 'results/simsiam-distill-with-predictor-memoryfix-launch-v2/child.json'
        if not prior.exists():
            raise ValueError('Launch the fixed distillation arm before crop branch B')
    sample = telemetry()
    if sample['devices'][gpu]['free_mib'] <= 12288:
        raise ValueError('Selected GPU needs more than12GiB free')
    base_ledger = json.loads((ROOT / 'results/ssl-night-comparison-20261026-v1/final-gpu-budget-ledger.json').read_text())
    base = base_ledger['itemized_plus_allowances'] + PROBE_REPLAY_EVALUATION_ALLOWANCE
    with (ROOT / 'results/reactivation-v2-budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reserved = []
        for name in ARMS:
            control = ROOT / 'results' / (name + '-launch-v2')
            if (control / 'receipt.json').exists():
                record = json.loads((control / 'receipt.json').read_text())
                reserved.append(record['reserved_gpu_hours'])
                if record['gpu'] == gpu and (control / 'state').read_text().startswith('RUNNING'):
                    raise ValueError('Another own reactivated run occupies this GPU')
        debit = checked_total(base, reserved, config['training_gpu_hours_limit'])
        control = ROOT / 'results' / (arm + '-launch-v2')
        control.mkdir(exist_ok=False)
        receipt = {'arm': arm, 'gpu': gpu, 'protocol_sha256': protocol_sha,
                   'run_config_sha256': digest(config_path), 'reserved_gpu_hours': config['training_gpu_hours_limit'],
                   'old_campaign_with_allowances': base_ledger['itemized_plus_allowances'],
                   'new_probe_replay_evaluation_allowance': PROBE_REPLAY_EVALUATION_ALLOWANCE,
                   'total_committed_gpu_hours': debit, 'initial_telemetry': sample,
                   'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        (control / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    seconds = int(config['training_gpu_hours_limit'] * 3600) - 10
    assert seconds > 0 and seconds + 5 < config['training_gpu_hours_limit'] * 3600
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
           'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2', 'PYTHONPATH': str(ROOT / 'scripts')}
    started, reason, code = time.monotonic(), 'process_exit', None
    with (control / 'training.log').open('xb') as log, (control / 'gpu-telemetry.jsonl').open('x') as samples:
        child = subprocess.Popen(['/usr/bin/timeout', '--signal=TERM', '--kill-after=5s', str(seconds),
                                  sys.executable, str(ROOT / 'scripts' / runtime_name), str(config_path)],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        (control / 'child.json').write_text(json.dumps({'pid': child.pid, 'gpu': gpu, 'seconds': seconds}))
        (control / 'state').write_text(f'RUNNING gpu={gpu} pid={child.pid}\n')
        try:
            while True:
                try:
                    code = child.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    sample = telemetry()
                    sample['own_timeout_pid'] = child.pid
                    samples.write(json.dumps(sample) + '\n')
                    samples.flush()
                    if sample['devices'][gpu]['free_mib'] < 2048:
                        reason = 'own_job_stopped_low_GPU_memory'
                        code = stop_own(child)
                        break
        except Exception as error:
            reason = 'own_job_stopped_monitor_error_' + type(error).__name__
            code = stop_own(child)
        finally:
            if child.poll() is None:
                code = stop_own(child)
    record = {'arm': arm, 'gpu': gpu, 'exit_code': code, 'reason': reason,
              'allocated_gpu_hours_upper_bound': (time.monotonic() - started) / 3600}
    (control / 'execution-receipt.json').write_text(json.dumps(record, indent=2))
    (control / 'state').write_text(f'FINISHED exit={code}\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', choices=ARMS)
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args()
    main(args.arm, args.gpu, args.protocol_sha)
