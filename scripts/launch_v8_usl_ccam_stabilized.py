"""Submit only the user-authorized stabilized CCAM run; no pretraining smoke."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path('/home/jovyan/work/proyectonp/repo_v8_ccam_stabilized_004')
PYTHON = Path('/home/jovyan/work/proyectonp/.venv/bin/python')
RECEIPTS = Path('/home/jovyan/work/validation_v8/usl-ccam-stabilized-launch-20261018-005')
OUT = Path('/home/jovyan/work/proyectonp/runs_v8/usl_ccam-20261018-005')
CONFIG = RECEIPTS / 'ccam.run.approved.json'
GATE = RECEIPTS / 'ccam.gate.json'
DRAFT = ROOT / 'plan-segmentacion/metodos-v8/launch-preparation/configs/usl_ccam.stabilized.draft.json'
PARENT_GATE = Path('/home/jovyan/work/validation_v8/usl-ccam-launch-20261017-003/ccam.gate.json')
CLEANUP = Path('/home/jovyan/work/validation_v8/usl-ccam-launch-preparation-4a6832c-20261017-001/review/approved-submission/cleanup.receipt.json')
MANIFEST = Path('/home/jovyan/work/validation_v8/accepted-manifests-20261016-002/learning.manifest.json')
ENTITY = 'nmoreno534-universidad-autonoma-de-bucaramanga'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def supervisor_module():
    spec = importlib.util.spec_from_file_location('bounded_ccam_supervisor', ROOT / 'scripts/launch_v8_usl_ccam.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.CONFIG, module.GATE = ROOT, CONFIG, GATE
    module.RECEIPT_DIR, module.OUT = RECEIPTS, OUT
    module.TRACKING = Path(str(OUT) + '-tracking')
    return module


def prepare(module):
    require(not any(path.exists() for path in (RECEIPTS, OUT, module.TRACKING)), 'Fresh run destinations required')
    require(not module.process_conflicts(), 'Another CCAM process is active')
    require(digest(PARENT_GATE) == 'd98c73c022d7809ae0b3d4fe94b16aea0ef3f92d6bdf1ef9b96b7ac953189bd0', 'Historical gate changed')
    require(digest(CLEANUP) == '0ca336b90aecf3fd999c9b03f7a83b5cc95d961c55650a78cdac091a470922ab', 'Cleanup evidence changed')
    cleanup = read(CLEANUP)
    require(all(cleanup.get(key) is True for key in ('cleanup_complete', 'evidence_preserved', 'wandb_cleanup_complete')), 'Scoped cleanup evidence required')
    require(digest(MANIFEST) == '7749471410aa98a76dd978cca2ef58f9642d14404530c98e534484ba40a7a6ff', 'Frozen manifest changed')
    config = read(DRAFT)
    require(config.get('execution_enabled') is False, 'Committed proposal must remain disabled')
    require(config.get('ccam_variant') == 'ccam_stabilized_v1' and config.get('epochs') == 30
            and config.get('effective_batch_size') == 64 and config.get('encoder_micro_batch_size') == 2
            and config.get('seed') == 42 and config.get('resolution') == 224, 'Unexpected authorized CCAM scope')
    require(digest(config['source_bank']) == config['source_bank_sha256'], 'Reviewed source bank changed')
    compute = subprocess.run(['nvidia-smi', '--id=0', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, check=True, timeout=15)
    require(not compute.stdout.strip(), 'GPU 0 already has a compute process')
    memory = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=15)
    gpu_required = config['resources']['max_gpu_memory_mb'] + config['resources']['gpu_reserve_mb']
    require(int(memory.stdout.strip()) >= gpu_required, 'GPU 0 has insufficient free planning capacity')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    config = dict(config, execution_enabled=True)
    from dermssl.segmentation_methods.common.training import gate_report, _json_hash
    parent = read(PARENT_GATE)
    paths = set(parent['source_hashes'])
    paths.update(str(path.relative_to(ROOT)) for path in (ROOT / 'src/dermssl/segmentation_methods').rglob('*.py'))
    paths.update((str(Path(__file__).relative_to(ROOT)), str(DRAFT.relative_to(ROOT)),
                  'scripts/launch_v8_usl_ccam.py'))
    sources = {name: digest(ROOT / name) for name in sorted(paths)}
    gate = dict(parent)
    gate.update(config_sha256=_json_hash(config), preparation_only=False,
                training_authorized=True, execution_enabled=True, verification_revision=revision,
                resource_budget_measured=False, smoke_passed=False,
                optimizer_memory_measured=False, full_dataset_runtime_measured=False,
                admission_waiver=config['admission_waiver'], source_hashes=sources,
                source_review_status='static_review_completed_no_tests',
                authorization_note=config['admission_waiver']['authorization_note'],
                parent_gate_sha256=digest(PARENT_GATE),
                resource_measurement_scope='Prior-run evidence only; this variant has no GPU smoke or measured resource budget',
                synthetic_cleanup_receipt=str(CLEANUP), synthetic_cleanup_receipt_sha256=digest(CLEANUP))
    from nucleo.wandb_logging import run_id
    import wandb
    identifier = run_id('w3-four-methods-v8', OUT.name)
    remote = ENTITY + '/dermssl-segmentacion/' + identifier
    try:
        wandb.Api().run(remote)
    except (wandb.errors.CommError, ValueError) as error:
        require(any(marker in str(error).lower() for marker in ('not found', 'could not find', 'does not exist')),
                'Could not establish a fresh W&B identity')
    else:
        raise ValueError('W&B identity already exists')
    RECEIPTS.mkdir()
    write_new(CONFIG, config)
    gate['config_file_sha256'] = digest(CONFIG)
    write_new(GATE, gate)
    gate_report(GATE, 'usl_ccam', MANIFEST, config=config, stage='ccam')
    command = [str(PYTHON), '-u', str(ROOT / 'scripts/run_segmentation_v8.py'), '--method', 'usl_net',
               '--stage', 'ccam', '--config', str(CONFIG), '--manifest', str(MANIFEST),
               '--gate', str(GATE), '--out', str(OUT), '--wandb-project', 'dermssl-segmentacion']
    record = dict(created_at=module.now(), status='prepared_not_started', training_started=False,
                  method='usl_net', stage='ccam', variant='ccam_stabilized_v1', seed=42,
                  command=command, checkout=str(ROOT), checkout_revision=revision,
                  config={'path': str(CONFIG), 'sha256': digest(CONFIG)},
                  manifest={'path': str(MANIFEST), 'sha256': digest(MANIFEST)},
                  gate={'path': str(GATE), 'sha256': digest(GATE)}, source_hashes=sources,
                  cleanup_receipt=str(CLEANUP), cleanup_receipt_sha256=digest(CLEANUP),
                  cleanup_basis='Prior diagnostics archived; this preparation ran no tests and created no synthetic training artifacts',
                  wandb_entity=ENTITY, expected_wandb_run_id=identifier,
                  expected_wandb_url='https://wandb.ai/' + ENTITY + '/dermssl-segmentacion/runs/' + identifier,
                  ccam_resume_supported=True, resume_requested=False, other_methods_automatic=False,
                  admission_waiver=config['admission_waiver'], gpu_pretraining_validation_performed=False,
                  launcher_sha256=digest(__file__), output=str(OUT), tracking=str(module.TRACKING))
    write_new(RECEIPTS / 'launch.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-reviewed-ccam-30', action='store_true')
    parser.add_argument('--_supervise', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(ROOT == EXPECTED_ROOT and Path(sys.executable).absolute() == PYTHON, 'Isolated cluster checkout and project venv required')
    require(args.execute_reviewed_ccam_30 or args._supervise, 'Explicit user-authorized execution flag required')
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'plan-segmentacion/comun')]
    os.environ.pop('WANDB_SERVICE', None)
    module = supervisor_module()
    if args._supervise:
        return module.supervise()
    prepare(module)
    env = os.environ.copy()
    with (RECEIPTS / 'supervisor.log').open('xb') as log:
        child = subprocess.Popen([str(PYTHON), '-u', str(Path(__file__).resolve()), '--_supervise'],
                                 cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_new(RECEIPTS / 'submission.json', {'supervisor_pid': child.pid, 'training_started': None})
    print(json.dumps({'supervisor_pid': child.pid, 'receipt_directory': str(RECEIPTS), 'training_started': None}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
