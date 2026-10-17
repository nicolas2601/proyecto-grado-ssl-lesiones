"""Finish the recorded synthetic cleanup after a system-managed W&B artifact refusal."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import wandb

FIRST = Path('/home/jovyan/work/validation_v8/pre-real-run-cleanup-20261017-001')
OUTPUT = Path('/home/jovyan/work/validation_v8/pre-real-run-cleanup-20261017-002')
ARCHIVE = FIRST / 'preserved-provenance'
ENTITY = 'nmoreno534-universidad-autonoma-de-bucaramanga'
PROJECT = 'dermssl-segmentacion'
RUN_ID = 'g8us8pz9'
RUN_NAME = 'usl_net_real_bank_synthetic_forward_20261014'
ARTIFACT = 'usl-net-synthetic-forward-20261014:v0'
HISTORY = 'run-g8us8pz9-history:v0'
RECEIPT = Path('/home/jovyan/work/public_ssl_v8_preparation/usl-net-synthetic-forward-20261014.json')
RECEIPT_SHA = 'b392e7d32eba94fe663777afa4754f47f554df50c715baaa0807ae535b7f6a83'
DRYRUN = Path('/home/jovyan/work/validation_v8/f1dc0d2-20261014/controller_dryrun')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def no_links(path):
    for part in (path, *path.parents):
        require(not part.is_symlink(), 'Symlink prohibited: ' + str(part))


def remote_object(kind, identifier):
    try:
        return getattr(wandb.Api(), kind)(identifier)
    except (wandb.errors.CommError, ValueError) as error:
        message = str(error).lower()
        require(any(word in message for word in ('not found', 'could not find', 'does not exist')),
                'Cannot establish remote object absence')
        return None


def absent(kind, identifier):
    return remote_object(kind, identifier) is None


def main():
    require(not OUTPUT.exists(), 'New recovery output directory required')
    for path in (FIRST, ARCHIVE, OUTPUT, RECEIPT, DRYRUN):
        no_links(path)
    require(sha(FIRST / 'cleanup.receipt.json') == '1d68eb8741895403d6a95fe8c9a550bfbdea26fab2b716394c3a857527844f24',
            'Original partial cleanup receipt changed')
    require(sha(ARCHIVE / 'archive.files.json') == '24b7ffd2feabea2e6d04efcf3102defd461c4aa266ae7a63a1f73f5f79628306',
            'Original provenance manifest changed')
    files = json.loads((ARCHIVE / 'archive.files.json').read_text())
    for name, digest in files.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe preserved file path')
        path = ARCHIVE / relative
        no_links(path)
        require(sha(path) == digest, 'Preserved provenance file changed')
    dryrun_hashes = json.loads((ARCHIVE / 'controller_dryrun.files.json').read_text())
    with zipfile.ZipFile(ARCHIVE / 'controller_dryrun.zip') as package:
        require(set(package.namelist()) == set(dryrun_hashes), 'Preserved dry-run ZIP members differ')
        for name, digest in dryrun_hashes.items():
            require(hashlib.sha256(package.read(name)).hexdigest() == digest, 'Preserved dry-run ZIP content changed')
    require(sha(RECEIPT) == RECEIPT_SHA, 'Operational synthetic receipt changed')
    require(sha(ARCHIVE / RECEIPT.name) == RECEIPT_SHA, 'Archived synthetic receipt differs')
    require(DRYRUN.is_dir(), 'Original dry-run directory missing')
    for path in DRYRUN.rglob('*'):
        no_links(path)
    require({path.relative_to(DRYRUN).as_posix(): sha(path) for path in DRYRUN.rglob('*') if path.is_file()} == dryrun_hashes,
            'Operational dry-run contents differ from the preserved ZIP')
    run_path = f'{ENTITY}/{PROJECT}/{RUN_ID}'
    custom_path = f'{ENTITY}/{PROJECT}/{ARTIFACT}'
    history_path = f'{ENTITY}/{PROJECT}/{HISTORY}'
    require(absent('artifact', custom_path), 'Previously removed custom artifact is unexpectedly present')
    run = wandb.Api().run(run_path)
    require(run.id == RUN_ID and run.name == RUN_NAME and run.state == 'finished',
            'Owning synthetic run identity or state changed')
    OUTPUT.mkdir(parents=True)
    report = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
              'cleanup_complete': False, 'evidence_preserved': True,
              'wandb_cleanup_complete': False, 'wandb_cleanup_confirmed': False,
              'operational_cleanup_complete': False, 'all_remote_objects_removed': False,
              'wandb_cleanup_scope': 'synthetic_run_and_user_validation_artifact; service_managed_history_explicitly_observed',
              'residual_remote_objects': [],
              'parent_receipt': str(FIRST / 'cleanup.receipt.json'),
              'parent_receipt_sha256': sha(FIRST / 'cleanup.receipt.json'),
              'archive': str(ARCHIVE), 'archive_manifest_sha256': sha(ARCHIVE / 'archive.files.json'),
              'removed_local_paths': [], 'removed_wandb_runs': [],
              'custom_artifact': {'path': custom_path, 'verified_absent_before_recovery': True},
              'system_managed_history': {'path': history_path, 'direct_delete_retried': False},
              'protected': ['real datasets', 'public weights', 'historical audit receipts'],
              'recovery_script_sha256': sha(__file__)}
    try:
        run.delete(delete_artifacts=False)
        report['removed_wandb_runs'].append(run_path)
        require(absent('run', run_path), 'Owning run removal not confirmed')
        require(absent('artifact', custom_path), 'Custom synthetic artifact removal not confirmed')
        history = remote_object('artifact', history_path)
        history_absent = history is None
        if history is not None:
            require(history.name == HISTORY and history.type == 'wandb-history',
                    'Residual service-managed artifact identity changed')
            report['residual_remote_objects'].append({'id': history.id, 'name': history.name,
                                                       'type': history.type, 'size': history.size,
                                                       'digest': history.digest, 'path': history_path})
        report['system_managed_history'].update(
            observed_status='removed_with_run' if history_absent else 'retained_system_managed_not_directly_deletable',
            owning_run_verified_absent=True,
            limitation=None if history_absent else 'W&B retains the service-managed history artifact; the owning synthetic run is removed. No direct-delete retry was attempted.')
        report.update(wandb_cleanup_complete=True, wandb_cleanup_confirmed=True,
                      all_remote_objects_removed=history_absent,
                      service_managed_retention_limitation=not history_absent)
        no_links(RECEIPT)
        require(sha(RECEIPT) == RECEIPT_SHA, 'Operational receipt changed before removal')
        RECEIPT.unlink()
        report['removed_local_paths'].append(str(RECEIPT))
        no_links(DRYRUN)
        for path in DRYRUN.rglob('*'):
            no_links(path)
        require({path.relative_to(DRYRUN).as_posix(): sha(path) for path in DRYRUN.rglob('*') if path.is_file()} == dryrun_hashes,
                'Dry-run source changed before removal')
        shutil.rmtree(DRYRUN)
        report['removed_local_paths'].append(str(DRYRUN))
        require(not RECEIPT.exists() and not DRYRUN.exists(), 'Operational synthetic paths remain')
        report.update(cleanup_complete=True, operational_cleanup_complete=True,
                      status='completed', finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        report.update(status='incomplete', error_type=type(error).__name__,
                      finished_at=datetime.now(timezone.utc).isoformat())
        with (OUTPUT / 'cleanup.receipt.json').open('x') as stream:
            stream.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
        raise
    with (OUTPUT / 'cleanup.receipt.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'cleanup_complete': True, 'receipt': str(OUTPUT / 'cleanup.receipt.json'),
                      'service_managed_retention_limitation': not history_absent}))


if __name__ == '__main__':
    main()
