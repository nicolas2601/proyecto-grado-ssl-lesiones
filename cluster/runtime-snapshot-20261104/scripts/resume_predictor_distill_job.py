"""One explicit checkpoint resume; preserve prior failure and monitor only our child."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG_SHA = '76ad960ce32eb245523eb61338e122d8facb20d4e0c626a0696b6f64db27149a'
CHECKPOINT_SHA = '173ff50935e7ded190a2114682b6d75f9d446c586fbc91a7f040da2528e7ddda'
DEADLINE = datetime.datetime(2026, 10, 26, 8, 30, tzinfo=datetime.timezone.utc)


def remaining_seconds(prior, limit):
    if not math.isfinite(prior) or not math.isfinite(limit) or prior < 0 or prior >= limit:
        raise ValueError('Invalid or exhausted combined GPU budget')
    seconds = int((limit - prior) * 3600) - 10
    if seconds < 60:
        raise ValueError('Insufficient remaining combined GPU budget')
    return seconds


def telemetry():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True, timeout=5)
    devices = {}
    for line in text.strip().splitlines():
        index, uuid, free, used, utilization = [x.strip() for x in line.split(',')]
        devices[int(index)] = {'uuid': uuid, 'free_mib': int(free), 'used_mib': int(used), 'utilization': utilization}
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                    '--format=csv,noheader,nounits'], text=True, timeout=5)
    processes = []
    for line in text.strip().splitlines():
        if line.strip():
            uuid, pid, used = [x.strip() for x in line.split(',')]
            processes.append({'uuid': uuid, 'pid': int(pid), 'used_mib': used})
    return {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'devices': devices, 'processes': processes}


def stop_own(child):
    if child.poll() is not None:
        return child.returncode
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return child.wait()
    try:
        return child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        return child.wait()


def main():
    from overlay_ssl_runtime import config_seal
    cfg_path = ROOT / 'config/simsiam-distill-with-predictor-v1/resume-run-attempt2.json'
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if digest(cfg_path) != CONFIG_SHA:
        raise ValueError('Explicit resume config changed')
    config = json.loads(cfg_path.read_text())
    run = Path(config['base_config']['output_dir'])
    if digest(config['base_config']['resume_checkpoint']) != CHECKPOINT_SHA:
        raise ValueError('Preserved step500 checkpoint changed')
    seal = json.loads((run / 'attempt1-launch-seal.json').read_text())
    if config_seal(config) != config_seal(seal['config']):
        raise ValueError('Resume changes sealed training parameters')
    for path, expected in seal['source_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Training source changed')
    if digest(ROOT / 'scripts/predictor_distill_ssl_runtime.py') != config['runtime_hook_sha256']:
        raise ValueError('Distillation hook changed')
    prior_record = json.loads((ROOT / 'results/simsiam-distill-with-predictor-seed42-launch-v1/execution-receipt.json').read_text())
    prior = prior_record['allocated_gpu_hours_upper_bound']
    seconds = remaining_seconds(prior, config['training_gpu_hours_limit'])
    remaining = (DEADLINE - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    if remaining < seconds + 10:
        raise ValueError('Full remaining cap cannot finish before03:30Colombia')
    sample = telemetry()
    gpu = max(sample['devices'], key=lambda k: sample['devices'][k]['free_mib'])
    if sample['devices'][gpu]['free_mib'] <= 12288:
        raise ValueError('No GPU has more than12GiB free;resume not started')
    control = ROOT / 'results/simsiam-distill-with-predictor-seed42-resume-attempt2-v1'
    control.mkdir(exist_ok=False)
    receipt = {'gpu': gpu, 'prior_gpu_hours': prior, 'combined_cap_gpu_hours': config['training_gpu_hours_limit'],
               'maximum_child_seconds': seconds, 'resume_step': 500, 'discarded_unsaved_steps': 180,
               'resume_config_sha256': CONFIG_SHA, 'checkpoint_sha256': CHECKPOINT_SHA,
               'supervisor_sha256': digest(__file__), 'reuses_original_budget_reservation': True,
               'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'initial_telemetry': sample}
    (control / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OPENBLAS_NUM_THREADS': '2',
           'OMP_NUM_THREADS': '2', 'PYTHONPATH': str(ROOT / 'scripts')}
    started, reason = time.monotonic(), 'process_exit'
    with (control / 'training.log').open('xb') as log, (control / 'gpu-telemetry.jsonl').open('x') as samples:
        child = subprocess.Popen(['/usr/bin/timeout', '--signal=TERM', '--kill-after=5s', str(seconds),
                                  sys.executable, str(ROOT / 'scripts/predictor_distill_ssl_runtime.py'), str(cfg_path)],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        (control / 'child.json').write_text(json.dumps({'pid': child.pid, 'gpu': gpu}))
        (control / 'state').write_text(f'RUNNING gpu={gpu} pid={child.pid}\n')
        code = None
        try:
            while True:
                try:
                    code = child.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    try:
                        sample = telemetry()
                    except (OSError, subprocess.SubprocessError, ValueError):
                        reason = 'own_job_stopped_telemetry_unavailable'
                        code = stop_own(child)
                        break
                    sample['own_timeout_pid'] = child.pid
                    samples.write(json.dumps(sample) + '\n')
                    samples.flush()
                    if sample['devices'][gpu]['free_mib'] < 2048:
                        reason = 'own_job_stopped_low_GPU_memory_again_no_further_retry'
                        code = stop_own(child)
                        break
        except Exception as error:
            reason = 'own_job_stopped_monitor_error_' + type(error).__name__
            code = stop_own(child)
        finally:
            if child.poll() is None:
                code = stop_own(child)
    elapsed = (time.monotonic() - started) / 3600
    record = {'exit_code': code, 'reason': reason, 'gpu': gpu,
              'allocated_gpu_hours_this_attempt_upper_bound': elapsed,
              'prior_gpu_hours': prior, 'combined_gpu_hours_upper_bound': prior + elapsed}
    (control / 'execution-receipt.json').write_text(json.dumps(record, indent=2))
    (control / 'state').write_text(f'FINISHED exit={code}\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
