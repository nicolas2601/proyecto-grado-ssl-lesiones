"""One bounded CCAM CUDA diagnostic without optimization or learned artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_pin(reference):
    path = Path(reference['path'])
    if sha(path) != reference['sha256']:
        raise ValueError('Pinned artifact changed: ' + str(path))
    return json.loads(path.read_text())


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--timeout-seconds', type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 1800:
        parser.error('Diagnostic timeout must be between 1 and 1800 seconds')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        'schema_version': 1, 'method': 'usl_ccam',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'blocked', 'training_started': False,
        'optimizer_created': False, 'optimizer_steps': 0,
        'learned_checkpoint_written': False, 'tracking_started': False,
        'diagnostic_scope': 'one_real_learning_batch_forward_loss_backward_no_update',
        'diagnostic_timeout_seconds': args.timeout_seconds,
        'script_sha256': sha(__file__), 'errors': [],
    }
    torch = None
    device = None

    def deadline(_signum, _frame):
        raise TimeoutError('Bounded CUDA diagnostic deadline exceeded')

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.timeout_seconds)
    gate = None
    try:
        import torch as torch_module
        torch = torch_module
        from dermssl.segmentation_methods.common.bank import source_bank
        from dermssl.segmentation_methods.common.data import rgb_tensor_array
        from dermssl.segmentation_methods.common.protocol import validate_binding
        from dermssl.segmentation_methods.common.training import _json_hash, seed_all
        from dermssl.segmentation_methods.usl_net.ccam import existing_ccam_loss
        from dermssl.segmentation_methods.usl_net.model import ContrastiveFusion, USLSeedGenerator

        config_path = Path(args.config)
        config = json.loads(config_path.read_text())
        if config.get('execution_enabled') is not False or config.get('method') != 'usl_net':
            raise ValueError('Only explicitly disabled USL config permitted')
        if config.get('batch_size') != 2 or config.get('resolution') != 224:
            raise ValueError('Probe contract requires batch two at resolution 224')
        review = config['preparation_review']
        if Path(args.manifest).resolve() != Path(review['learning_manifest']['path']).resolve():
            raise ValueError('CLI manifest differs from pinned learning manifest')
        document = read_pin(review['learning_manifest'])
        freeze = read_pin(review['frozen_receipt'])
        read_pin(config['evaluation_protocol'])
        if freeze.get('manifest_frozen') is not True:
            raise ValueError('Data freeze receipt is not frozen')
        if document.get('partition') != 'train' or len(document['records']) != 34784:
            raise ValueError('Expected frozen learning partition and count')
        validate_binding(config, document)
        if sha(config['source_bank']) != review['source_bank_manifest_sha256']:
            raise ValueError('Source bank pin mismatch')
        report['inputs'] = {
            'config': {'path': str(config_path), 'sha256': sha(config_path),
                       'canonical_sha256': _json_hash(config)},
            'manifest': review['learning_manifest'], 'freeze_receipt': review['frozen_receipt'],
            'protocol': config['evaluation_protocol'],
            'source_bank': {'path': config['source_bank'],
                            'sha256': review['source_bank_manifest_sha256']},
        }
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        report['checkout'] = {'cwd': os.getcwd(), 'git_head': revision,
                              'note': 'Working source hashes, not git HEAD alone, identify deployed code'}
        root = Path(__file__).resolve().parents[1]
        source_paths = [
            'src/dermssl/segmentation_methods/usl_net/model.py',
            'src/dermssl/segmentation_methods/usl_net/ccam.py',
            'src/dermssl/segmentation_methods/common/bank.py',
            'src/dermssl/segmentation_methods/common/networks.py',
            'src/dermssl/segmentation_methods/common/checkpoints.py',
            'src/dermssl/segmentation_methods/common/protocol.py',
            'src/dermssl/segmentation_methods/common/data.py',
            'src/dermssl/w3_usl_models.py',
            'plan-segmentacion/camino-b-cnn/pseudo-mascaras/ccam_engine.py',
        ]
        report['source_hashes'] = {p: sha(root / p) for p in source_paths}
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable')
        device = torch.device(config['device'])
        if device.type != 'cuda':
            raise ValueError('Real CUDA device required')
        torch.cuda.set_device(device)
        free, total = torch.cuda.mem_get_info(device)
        caps = config['resources']
        if caps['max_gpu_memory_mb'] + caps['gpu_reserve_mb'] > free / 1024**2:
            raise ValueError('Registered GPU budget and reserve exceed current free memory')
        torch.cuda.set_per_process_memory_fraction(caps['max_gpu_memory_mb'] * 1024**2 / total, device)
        report['runtime'] = {'python': sys.version, 'torch': torch.__version__,
                             'cuda': torch.version.cuda, 'device': str(device),
                             'gpu_name': torch.cuda.get_device_name(device),
                             'capability': list(torch.cuda.get_device_capability(device)),
                             'gpu_total_mib': total / 1024**2, 'gpu_free_before_mib': free / 1024**2}
        seed_all(config.get('seed', 42))
        rows = document['records'][:2]
        images = torch.stack([torch.from_numpy(rgb_tensor_array(row, 224)) for row in rows])
        report['sampled_originals'] = rows
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        load_start = time.monotonic()
        encoders = source_bank(config['source_bank'], ('simclr', 'mocov1', 'mocov2'))
        model = USLSeedGenerator(fusion=ContrastiveFusion(encoders))
        before = state_digest(model)
        model = model.to(device)
        model.train()
        torch.cuda.synchronize(device)
        report['model_load_seconds'] = time.monotonic() - load_start
        batch_start = time.monotonic()
        images = images.to(device)
        features = model.fusion(images)
        foreground = 1 - torch.sigmoid(model.ccam.background(features))
        loss = existing_ccam_loss(features, foreground, config.get('ccam_alpha', .25))
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite CCAM loss')
        loss.backward()
        gradients = [p.grad for p in model.ccam.parameters() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise ValueError('Missing or nonfinite CCAM gradients')
        torch.cuda.synchronize(device)
        report['batch_seconds'] = time.monotonic() - batch_start
        report['loss'] = float(loss.detach().cpu())
        report['features_shape'] = list(features.shape)
        report['foreground_shape'] = list(foreground.shape)
        report['finite_gradient_tensors'] = len(gradients)
        after = state_digest(model)
        report['state_before_sha256'] = before
        report['state_after_sha256'] = after
        if before != after:
            raise ValueError('Parameters or buffers changed during no-update diagnostic')
        report['model_state_unchanged'] = True
        report['planning'] = {
            'epochs': config['epochs'], 'batches_per_epoch': math.ceil(len(document['records']) / 2),
            'cold_batch_linear_seconds': report['batch_seconds'] * math.ceil(len(document['records']) / 2) * config['epochs'],
            'estimate_is_not_validated_runtime': True,
            'optimizer_memory_measured': False, 'full_dataset_runtime_measured': False,
            'output_budget_measured': False,
            'scope_limit': 'Peak includes model and one forward/backward, excludes AdamW states and optimizer step; first-batch extrapolation includes cold kernels',
        }
        report['status'] = 'smoke_passed_preparation_only'
        gate = {
            'method': 'usl_ccam', 'input_manifest_sha256': sha(args.manifest),
            'config_sha256': _json_hash(config),
            'config_file_sha256': sha(config_path),
            'evaluation_protocol': config['evaluation_protocol'],
            'verification_revision': revision,
            'data_verified': True,
            'data_verification_scope': 'reuse_pinned_frozen_manifest_and_receipt; only_two_original_bytes_rechecked_here',
            'weights_provenance_verified': True, 'smoke_passed': True,
            'resource_budget_measured': False,
            'resource_measurement_scope': 'one_forward_backward_without_optimizer; incomplete_training_envelope',
            'patient_independence_accredited': False,
            'retrospective_protocol_reviewed': True,
            'training_authorized': False, 'preparation_only': True,
            'execution_enabled': False,
            'blocked_reasons': ['optimizer_and_full_run_resource_envelope_not_measured',
                                'frozen_receipt_training_consumable_false',
                                'user_review_and_pre_real_run_cleanup_pending'],
        }
    except BaseException as exc:
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc),
                                  'traceback': traceback.format_exc()})
        report['oom'] = isinstance(exc, torch.cuda.OutOfMemoryError) if torch is not None else False
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic() - started
        report['rss_peak_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if torch is not None and device is not None:
            try:
                report['cuda_peak_allocated_mib'] = torch.cuda.max_memory_allocated(device) / 1024**2
                report['cuda_peak_reserved_mib'] = torch.cuda.max_memory_reserved(device) / 1024**2
            except Exception as exc:
                report['measurement_error'] = str(exc)
        report_path = out / 'receipt.json'
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        if gate is None:
            gate = {'method': 'usl_ccam', 'smoke_passed': False, 'resource_budget_measured': False,
                    'preparation_only': True, 'training_authorized': False, 'execution_enabled': False,
                    'blocked_reasons': ['diagnostic_failed']}
        gate['diagnostic_receipt'] = {'path': str(report_path.resolve()), 'sha256': sha(report_path)}
        gate_path = out / 'ccam.gate.json'
        gate_path.write_text(json.dumps(gate, indent=2) + '\n')
        envelope = {'report_sha256': sha(report_path), 'report': report,
                    'gate_path': str(gate_path.resolve()), 'gate_sha256': sha(gate_path), 'gate': gate}
        (out / 'receipt.log').write_text(json.dumps(envelope) + '\n')
        print(json.dumps({'status': report['status'], 'receipt': str(report_path),
                          'receipt_sha256': sha(report_path), 'gate': str(gate_path)}), flush=True)
    return 0 if report['status'] == 'smoke_passed_preparation_only' else 1


if __name__ == '__main__':
    raise SystemExit(main())
