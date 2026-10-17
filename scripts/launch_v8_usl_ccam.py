"""Launch the reviewed CCAM stage once, detached, with immutable input receipts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

ROOT = Path('/home/jovyan/work/proyectonp/repo_v8_canonical')
PYTHON = Path('/home/jovyan/work/proyectonp/.venv/bin/python')
BASE = ROOT / 'plan-segmentacion/metodos-v8/launch-preparation/configs'
DRAFT = BASE / 'usl_ccam.draft.json'
CONFIG = BASE / 'usl_ccam.run-20261017.json'
MANIFEST = Path('/home/jovyan/work/validation_v8/accepted-manifests-20261016-002/learning.manifest.json')
REVIEW = Path('/home/jovyan/work/validation_v8/usl-ccam-resource-review-20261017-001')
RECEIPT_DIR = Path('/home/jovyan/work/validation_v8/usl-ccam-launch-20261017-001')
OUT = Path('/home/jovyan/work/proyectonp/runs_v8/usl_ccam-20261017-001')
TRACKING = OUT.parent / (OUT.name + '-tracking')
GATE = RECEIPT_DIR / 'ccam.gate.json'
STOP = False
OWNS_RECEIPTS = False


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def status(value):
    target = RECEIPT_DIR / 'status.json'
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temp, target)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path, expected):
    require(digest(path) == expected, 'Input hash mismatch: ' + str(path))


def process_conflicts():
    conflicts = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (b'--stage' in args and b'ccam' in args
                and any(b'run_segmentation_v8.py' in item for item in args)):
            conflicts.append(int(path.parent.name))
    return conflicts


def prepare(args):
    global OWNS_RECEIPTS
    require(Path.cwd().resolve() == ROOT, 'Run from the canonical checkout')
    require(Path(sys.executable).absolute() == PYTHON, 'Use the reviewed project virtualenv')
    require(not RECEIPT_DIR.exists(), 'New launch receipt directory required')
    require(not OUT.exists() and not TRACKING.exists(), 'New output and tracking directories required')
    require(not process_conflicts(), 'An equivalent CCAM process is already active')
    pin(DRAFT, 'f8a3589613b4608ea68d7efe5a08dfaf70d8467d2805c143c0ea18064082e792')
    pin(REVIEW / 'resource-review.json', '6e3d896e59b473d075e1e81cfeda08755230fa7068df609c72f2a15c601eb2c9')
    pin(REVIEW / 'ccam.gate.json', '2f8e644680aacf5eaeca90561a6d0f7a16655632e50020e022e4b8f55aa6b062')
    pin(MANIFEST, '7749471410aa98a76dd978cca2ef58f9642d14404530c98e534484ba40a7a6ff')
    draft = json.loads(DRAFT.read_text())
    config = json.loads(CONFIG.read_text())
    expected = dict(draft, execution_enabled=True)
    require(config == expected, 'Enabled config must differ from reviewed draft only by execution_enabled')
    review = json.loads((REVIEW / 'resource-review.json').read_text())
    sources = dict(review['source_hashes'])
    sources.update(review['reviewed_metadata_consumer_sources'])
    for relative, expected_hash in sources.items():
        pin(ROOT / relative, expected_hash)
    cleanup = json.loads(args.cleanup_receipt.read_text())
    require(cleanup.get('cleanup_complete') is True and cleanup.get('evidence_preserved') is True,
            'Completed synthetic cleanup with preserved evidence is required')
    require(cleanup.get('wandb_cleanup_complete') is True, 'W&B synthetic cleanup evidence required')
    for synthetic in (Path('/home/jovyan/work/public_ssl_v8_preparation/usl-net-synthetic-forward-20261014.json'),
                      Path('/home/jovyan/work/validation_v8/f1dc0d2-20261014/controller_dryrun')):
        require(not synthetic.exists(), 'Operational synthetic artifact remains: ' + str(synthetic))
    pin(config['source_bank'], config['preparation_review']['source_bank_manifest_sha256'])
    require(bool(args.wandb_entity.strip()), 'Explicit authenticated W&B entity required')
    require(args.wandb_entity == 'nmoreno534-universidad-autonoma-de-bucaramanga',
            'Only the reviewed authenticated W&B entity is allowed')
    sys.path.insert(0, str(ROOT / 'plan-segmentacion/comun'))
    from nucleo.wandb_logging import run_id
    import wandb
    expected_run_id = run_id('w3-four-methods-v8', OUT.name)
    wandb_path = f'{args.wandb_entity}/dermssl-segmentacion/{expected_run_id}'
    try:
        wandb.Api().run(wandb_path)
    except (wandb.errors.CommError, ValueError) as error:
        message = str(error).lower()
        require('not found' in message or 'could not find' in message or 'does not exist' in message,
                'Could not confirm a fresh W&B run identity')
    else:
        raise ValueError('Deterministic W&B run identity already exists; CCAM cannot resume')
    from dermssl.segmentation_methods.common.training import gate_report, _json_hash
    gate = json.loads((REVIEW / 'ccam.gate.json').read_text())
    gate.update(config_sha256=_json_hash(config), config_file_sha256=digest(CONFIG),
                preparation_only=False, training_authorized=True, execution_enabled=True,
                authorization_note='User explicitly authorized the CCAM launch on 2026-10-17',
                parent_gate_sha256=digest(REVIEW / 'ccam.gate.json'),
                synthetic_cleanup_receipt=str(args.cleanup_receipt),
                synthetic_cleanup_receipt_sha256=digest(args.cleanup_receipt))
    RECEIPT_DIR.mkdir(parents=True)
    OWNS_RECEIPTS = True
    write_new(GATE, gate)
    gate_report(GATE, 'usl_ccam', MANIFEST, config=config, stage='ccam')
    command = [str(PYTHON), '-u', str(ROOT / 'scripts/run_segmentation_v8.py'),
               '--method', 'usl_net', '--stage', 'ccam', '--config', str(CONFIG),
               '--manifest', str(MANIFEST), '--gate', str(GATE), '--out', str(OUT),
               '--wandb-project', 'dermssl-segmentacion']
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    record = {'created_at': now(), 'status': 'prepared_not_started', 'training_started': False,
              'method': 'usl_net', 'stage': 'ccam', 'seed': 42, 'command': command,
              'checkout': str(ROOT), 'checkout_revision': revision,
              'config': {'path': str(CONFIG), 'sha256': digest(CONFIG)},
              'manifest': {'path': str(MANIFEST), 'sha256': digest(MANIFEST)},
              'gate': {'path': str(GATE), 'sha256': digest(GATE)},
              'source_hashes': sources, 'launcher_sha256': digest(__file__),
              'cleanup_receipt_sha256': digest(args.cleanup_receipt),
              'cleanup_receipt': str(args.cleanup_receipt),
              'wandb_project': 'dermssl-segmentacion', 'wandb_entity': args.wandb_entity,
              'expected_wandb_run_id': expected_run_id, 'wandb_identity_verified_absent': True,
              'expected_wandb_url': 'https://wandb.ai/' + wandb_path.replace('/' + expected_run_id, '/runs/' + expected_run_id),
              'output': str(OUT), 'tracking': str(TRACKING),
              'ccam_resume_supported': False, 'other_methods_automatic': False}
    write_new(RECEIPT_DIR / 'launch.json', record)
    return record


def request_stop(signum, frame):
    global STOP
    STOP = True


def descendants(pid):
    result = [pid]
    for parent in result:
        try:
            children = Path(f'/proc/{parent}/task/{parent}/children').read_text().split()
        except (FileNotFoundError, PermissionError):
            continue
        result.extend(int(child) for child in children)
    return result


def terminate(child, grace):
    # Only the launched runner and its descendants; no broad process-name killing.
    def birth(pid):
        try:
            return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        except (FileNotFoundError, PermissionError):
            return None
    owned = {pid: birth(pid) for pid in descendants(child.pid)}
    for pid in reversed(owned):
        try:
            if owned[pid] is not None and birth(pid) == owned[pid]:
                os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and any(birth(pid) == stamp for pid, stamp in owned.items() if stamp is not None):
        time.sleep(1)
    for pid in reversed(owned):
        try:
            if owned[pid] is not None and birth(pid) == owned[pid]:
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    child.wait()


def supervise():
    global OWNS_RECEIPTS
    with (RECEIPT_DIR.parent / '.usl-ccam-real-launch.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (RECEIPT_DIR / 'status.json').exists() and not (RECEIPT_DIR / 'exit.json').exists(),
                'A supervisor already owns this receipt directory')
        OWNS_RECEIPTS = True
        require(not process_conflicts(), 'Equivalent CCAM process appeared before launch')
        record = json.loads((RECEIPT_DIR / 'launch.json').read_text())
        for key in ('config', 'manifest', 'gate'):
            pin(record[key]['path'], record[key]['sha256'])
        pin(record['cleanup_receipt'], record['cleanup_receipt_sha256'])
        for relative, expected_hash in record['source_hashes'].items():
            pin(ROOT / relative, expected_hash)
        config = json.loads(CONFIG.read_text())
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', PYTHONPATH=str(ROOT / 'src'),
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', WANDB_MODE='online',
                   WANDB_ENTITY=record['wandb_entity'], WANDB_PROJECT='dermssl-segmentacion')
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        started = time.monotonic()
        reason = None
        with (RECEIPT_DIR / 'runner.log').open('xb') as log:
            child = subprocess.Popen(record['command'], cwd=ROOT, env=env,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            info = {'status': 'runner_process_started_training_unconfirmed',
                    'started_at': now(), 'supervisor_pid': os.getpid(),
                    'runner_pid': child.pid, 'training_started': None, 'output': str(OUT)}
            status(info)
            try:
                while child.poll() is None:
                    size = sum(path.stat().st_size for directory in (OUT, TRACKING, RECEIPT_DIR)
                               if directory.exists() for path in directory.rglob('*') if path.is_file())
                    if STOP:
                        reason = 'supervisor_stop_requested'
                    elif time.monotonic() - started > config['resources']['timeout_seconds']:
                        reason = 'outer_walltime_budget_exceeded'
                    elif size > config['resources']['max_output_mb'] * 1024 ** 2:
                        reason = 'output_tracking_log_budget_exceeded'
                    if reason:
                        terminate(child, config['resources']['shutdown_grace_seconds'])
                        break
                    time.sleep(5)
            finally:
                if child.poll() is None:
                    terminate(child, config['resources']['shutdown_grace_seconds'])
            code = child.wait()
        checkpoint = OUT / 'ccam.pt'
        provenance_path = OUT / 'ccam.pt.json'
        rows = []
        try:
            rows = [json.loads(line) for line in (OUT / 'metrics.jsonl').read_text().splitlines() if line.strip()]
        except (FileNotFoundError, ValueError):
            pass
        valid_rows = [row for row in rows if isinstance(row, dict)
                      and type(row.get('epoch')) is int
                      and isinstance(row.get('mean_ccam_loss'), (int, float))
                      and math.isfinite(row['mean_ccam_loss']) and row.get('sample_count') == 34784]
        provenance_ok = False
        if checkpoint.is_file() and provenance_path.is_file():
            try:
                provenance = json.loads(provenance_path.read_text())
                provenance_ok = (provenance.get('method') == 'usl_ccam'
                                 and provenance.get('checkpoint_sha256') == digest(checkpoint)
                                 and provenance.get('input_manifest_sha256') == record['manifest']['sha256']
                                 and provenance.get('lineage', {}).get('training_gate_sha256') == record['gate']['sha256'])
            except (ValueError, AttributeError):
                pass
        complete = (code == 0 and provenance_ok and len(valid_rows) == config['epochs']
                    and [row['epoch'] for row in valid_rows] == list(range(1, config['epochs'] + 1)))
        info.update(status='completed' if complete else 'failed', ended_at=now(), exit_code=code,
                    elapsed_seconds=time.monotonic() - started, stop_reason=reason,
                    training_started=True if valid_rows else None, completed_epoch_records=len(valid_rows),
                    checkpoint_sha256=digest(checkpoint) if checkpoint.is_file() else None,
                    provenance_sha256=digest(provenance_path) if provenance_path.is_file() else None)
        status(info)
        write_new(RECEIPT_DIR / 'exit.json', info)
        return 0 if complete else 1


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / 'src'))
    if sys.argv[1:] == ['--_supervise']:
        return supervise()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cleanup-receipt', type=Path, required=True)
    parser.add_argument('--wandb-entity', required=True)
    args = parser.parse_args()
    prepare(args)
    with (RECEIPT_DIR / 'supervisor.log').open('xb') as log:
        process = subprocess.Popen([str(PYTHON), '-u', str(Path(__file__).resolve()), '--_supervise'],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    write_new(RECEIPT_DIR / 'submission.json', {'submitted_at': now(),
              'supervisor_pid': process.pid, 'status': 'submitted_training_unconfirmed'})
    print(json.dumps({'status': 'submitted_training_unconfirmed', 'supervisor_pid': process.pid,
                      'receipt_dir': str(RECEIPT_DIR)}), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        if OWNS_RECEIPTS and RECEIPT_DIR.is_dir() and not (RECEIPT_DIR / 'exit.json').exists():
            failure = {'status': 'launch_failed', 'ended_at': now(),
                       'error_type': type(error).__name__, 'training_started': None}
            status(failure)
            write_new(RECEIPT_DIR / 'exit.json', failure)
        raise
