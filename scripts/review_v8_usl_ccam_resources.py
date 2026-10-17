"""Review existing CCAM measurements and modeled allowances; never run a model."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


RECEIPT_SHA = 'ed7275b299a2a8cf237c31d25375148de60b41ac5b9f4e7c42c6726e0cf30d84'
GATE_SHA = 'bcb9f329dbc0ff9acde28fe9e06f1853fc6a2f67022bd3d1fd74ce411cd16264'
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(path, expected):
    require(sha(path) == expected, 'SHA-256 mismatch: ' + str(path))
    return json.loads(Path(path).read_text())


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--smoke-config', required=True, type=Path)
    parser.add_argument('--smoke-receipt', required=True, type=Path)
    parser.add_argument('--smoke-gate', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), 'Output directory must not exist')
    require(not Path('/home/jovyan/work/proyectonp/salidas/segmentation_v8/usl_ccam_20261017_001').exists(),
            'Planned final training output already exists; do not reuse it')
    from dermssl.segmentation_methods.common.training import _json_hash, gate_report
    from dermssl.segmentation_methods.common.protocol import validate_binding

    receipt = pinned(args.smoke_receipt, RECEIPT_SHA)
    old_gate = pinned(args.smoke_gate, GATE_SHA)
    old_config = pinned(args.smoke_config, receipt['inputs']['config']['sha256'])
    config = json.loads(args.config.read_text())
    require(config.get('execution_enabled') is False, 'Configuration must remain disabled')
    baseline = copy.deepcopy(old_config)
    baseline['weight_decay'] = config.get('weight_decay')
    baseline['preparation_review'] = config.get('preparation_review')
    require(baseline == config, 'Only weight_decay and preparation_review may change')
    require(config.get('weight_decay') == 0.01, 'Declare actual existing AdamW default')
    require(receipt.get('status') == 'smoke_passed_preparation_only', 'Smoke did not pass')
    require(not receipt.get('errors') and not receipt.get('oom'), 'Smoke has errors or OOM')
    require(receipt.get('optimizer_created') is False and receipt.get('optimizer_steps') == 0,
            'Expected no optimizer updates in historical smoke')
    require(receipt.get('model_state_unchanged') is True and
            receipt['state_before_sha256'] == receipt['state_after_sha256'], 'Model state changed')
    require(receipt.get('finite_gradient_tensors') == 2, 'Expected two finite head gradients')
    require(old_gate.get('smoke_passed') is True, 'Historical gate smoke flag missing')
    source_hashes = receipt['source_hashes']
    for relative, expected in source_hashes.items():
        require(sha(ROOT / relative) == expected, 'Measured implementation changed: ' + relative)
    require(sha(ROOT / 'scripts/probe_v8_usl_ccam_cuda.py') == receipt['script_sha256'],
            'Historical probe script changed')
    review = config['preparation_review']
    manifest_ref = review['learning_manifest']
    manifest = pinned(manifest_ref['path'], manifest_ref['sha256'])
    freeze_ref = review['frozen_receipt']
    freeze = pinned(freeze_ref['path'], freeze_ref['sha256'])
    pinned(config['evaluation_protocol']['path'], config['evaluation_protocol']['sha256'])
    bank = pinned(config['source_bank'], review['source_bank_manifest_sha256'])
    require(bank is not None, 'Bank manifest missing')
    require(freeze.get('manifest_frozen') is True, 'Data manifest not frozen')
    require(manifest.get('partition') == 'train' and len(manifest['records']) == 34784,
            'Unexpected learning cohort')
    validate_binding(config, manifest)
    for key in ('manifest', 'freeze_receipt', 'protocol', 'source_bank'):
        actual = receipt['inputs'][key]
        expected = {'manifest': manifest_ref, 'freeze_receipt': freeze_ref,
                    'protocol': config['evaluation_protocol'],
                    'source_bank': {'path': config['source_bank'],
                                    'sha256': review['source_bank_manifest_sha256']}}[key]
        require(actual['path'] == expected['path'] and actual['sha256'] == expected['sha256'],
                'Historical smoke input differs: ' + key)
    caps = config['resources']
    moments_mib = 9217 * 4 * 2 / 1024**2
    temporary_mib = 16
    gpu_envelope = receipt['cuda_peak_reserved_mib'] + moments_mib + temporary_mib
    batches = 34784 // config['batch_size']
    linear_seconds = receipt['batch_seconds'] * batches * config['epochs']
    require(gpu_envelope < caps['max_gpu_memory_mb'], 'GPU operational allowance exceeds cap')
    require(receipt['rss_peak_mib'] < caps['max_rss_mb'], 'Observed RSS exceeds cap')
    require(linear_seconds < caps['timeout_seconds'], 'Representative time estimate exceeds cap')
    require(caps['max_output_mb'] >= 1024, 'Output precheck requires 1024 MiB')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    report = {
        'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'operational_budget_reviewed_pending_user_approval',
        'training_started': False, 'execution_enabled': False,
        'new_model_execution': False, 'new_optimizer_execution': False,
        'source_hashes': source_hashes, 'review_script_sha256': sha(__file__),
        'reviewed_metadata_consumer_sources': {
            relative: sha(ROOT / relative) for relative in (
                'src/dermssl/segmentation_methods/common/training.py',
                'src/dermssl/segmentation_methods/usl_net/seeds.py',
                'scripts/run_segmentation_v8.py')},
        'config': {'path': str(args.config.resolve()), 'sha256': sha(args.config),
                   'canonical_sha256': _json_hash(config)},
        'historical_smoke': {'receipt': str(args.smoke_receipt), 'receipt_sha256': RECEIPT_SHA,
                             'gate': str(args.smoke_gate), 'gate_sha256': GATE_SHA},
        'observed': {key: receipt[key] for key in ('batch_seconds', 'elapsed_seconds',
                     'model_load_seconds', 'rss_peak_mib', 'cuda_peak_allocated_mib',
                     'cuda_peak_reserved_mib', 'finite_gradient_tensors')},
        'modeled_not_measured': {
            'head_fp32_parameters': 9217, 'head_parameter_bytes': 36868,
            'adamw_two_moment_bytes': 73736, 'adamw_two_moment_mib': moments_mib,
            'optimizer_temporary_allowance_mib': temporary_mib,
            'temporary_allowance_basis': 'conservative assumption, over 450 times FP32 head bytes',
            'gpu_reserved_plus_modeled_allowance_mib': gpu_envelope,
            'batches_per_epoch': batches, 'epochs': config['epochs'],
            'linear_batch_seconds': linear_seconds,
            'timeout_to_linear_estimate_ratio': caps['timeout_seconds'] / linear_seconds,
            'time_estimate_excludes': ['model_loading', 'image_loading', 'optimizer_step',
                                     'filesystem_io', 'full_dataset_variability'],
            'output_cap_mib': caps['max_output_mb'], 'output_size_measured': False,
            'output_basis': 'existing_runner_ccam_storage_precheck_1024_MiB'},
        'resource_review_definition': 'observed representative forward/loss/backward plus explicit modeled optimizer allowance; not full-run measurement',
        'optimizer_memory_measured': False, 'full_dataset_runtime_measured': False,
        'historical_training_consumable_false_is_runtime_blocker': False,
        'historical_freeze_preserved': True, 'readiness_modified': False,
        'patient_independence_accredited': False,
        'remaining_before_launch': ['user_review', 'synthetic_cleanup_with_preserved_receipts',
                                    'enabled_config_and_regenerated_hash_bound_gate'],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / 'resource-review.json'
    write(report_path, report)
    gate = {
        'method': 'usl_ccam', 'input_manifest_sha256': manifest_ref['sha256'],
        'config_sha256': _json_hash(config), 'config_file_sha256': sha(args.config),
        'evaluation_protocol': config['evaluation_protocol'], 'verification_revision': revision,
        'data_verified': True, 'weights_provenance_verified': True, 'smoke_passed': True,
        'resource_budget_measured': True,
        'resource_measurement_scope': report['resource_review_definition'],
        'optimizer_memory_measured': False, 'full_dataset_runtime_measured': False,
        'patient_independence_accredited': False, 'retrospective_protocol_reviewed': True,
        'preparation_only': True, 'training_authorized': False, 'execution_enabled': False,
        'authorization_note': 'gate_report does not enforce these informational authorization flags; disabled configuration is the effective execution guard',
        'resource_review': {'path': str(report_path.resolve()), 'sha256': sha(report_path)},
        'historical_smoke_receipt': {'path': str(args.smoke_receipt), 'sha256': RECEIPT_SHA},
    }
    gate_path = args.output_dir / 'ccam.gate.json'
    write(gate_path, gate)
    gate_report(gate_path, 'usl_ccam', manifest_ref['path'], config=config, stage='ccam')
    launch = {
        'status': 'pending_user_approval_not_executed', 'execution_enabled': False,
        'cwd': str(ROOT), 'requires_gate_regeneration_after_any_config_change': True,
        'effective_guard': 'current config execution_enabled=false',
        'environment': {'PYTHONPATH': 'src', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                        'WANDB_ENTITY': 'nmoreno534-universidad-autonoma-de-bucaramanga',
                        'WANDB_MODE': 'online'},
        'argv': ['/home/jovyan/work/proyectonp/.venv/bin/python',
                 'scripts/run_segmentation_v8.py', '--method', 'usl_net', '--stage', 'ccam',
                 '--config', str(args.config.resolve()), '--manifest', manifest_ref['path'],
                 '--gate', str(gate_path.resolve()), '--out',
                 '/home/jovyan/work/proyectonp/salidas/segmentation_v8/usl_ccam_20261017_001',
                 '--wandb-project', 'dermssl-segmentacion'],
        'remaining_before_launch': report['remaining_before_launch'],
        'downstream_usl_student_ready': False,
        'downstream_pending': ['CCAM_checkpoint_and_lineage', 'pseudo_mask_manifest',
                               'student_and_serial_cycle_execution_plan'],
    }
    write(args.output_dir / 'launch-review.json', launch)
    envelope = {'report': report, 'report_sha256': sha(report_path), 'gate': gate,
                'gate_sha256': sha(gate_path), 'gate_consumer_metadata_validation_passed': True,
                'launch': launch, 'launch_sha256': sha(args.output_dir / 'launch-review.json')}
    with (args.output_dir / 'resource-review.log').open('x') as stream:
        stream.write(json.dumps(envelope) + '\n')
    print(json.dumps({'status': report['status'], 'resource_review': str(report_path),
                      'sha256': sha(report_path), 'gate_sha256': sha(gate_path)}))


if __name__ == '__main__':
    main()
