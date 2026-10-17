"""Archive and remove only the explicitly approved pre-run synthetic artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

RECEIPT = Path('/home/jovyan/work/public_ssl_v8_preparation/usl-net-synthetic-forward-20261014.json')
RECEIPT_SHA = 'b392e7d32eba94fe663777afa4754f47f554df50c715baaa0807ae535b7f6a83'
DRYRUN = Path('/home/jovyan/work/validation_v8/f1dc0d2-20261014/controller_dryrun')
RUN_ID = 'g8us8pz9'
PROJECT = 'dermssl-segmentacion'
ENTITY = 'nmoreno534-universidad-autonoma-de-bucaramanga'
RUN_NAME = 'usl_net_real_bank_synthetic_forward_20261014'
ARTIFACTS = ('usl-net-synthetic-forward-20261014:v0', 'run-g8us8pz9-history:v0')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def no_links(path):
    for part in (path, *path.parents):
        require(not part.is_symlink(), 'Symlink prohibited: ' + str(part))


def safe_name(name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts and '\\' not in name,
            'Unsafe archive member name')
    return path


def sanitize(value):
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()
                if not re.search(r'api.?key|token|secret|password|cookie|authorization|credential', str(key), re.I)}
    if isinstance(value, (tuple, list)):
        return [sanitize(item) for item in value]
    return value


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--wandb-entity', required=True)
    args = parser.parse_args()
    require(args.wandb_entity == ENTITY, 'Only the reviewed W&B entity is authorized')
    output = args.output_dir.absolute()
    expected = Path('/home/jovyan/work/validation_v8/pre-real-run-cleanup-20261017-001')
    require(output == expected and output.is_dir(), 'Use the existing targeted inventory directory')
    no_links(output)
    archive = output / 'preserved-provenance'
    require(not archive.exists(), 'New provenance archive directory required')
    require(not (output / 'cleanup.receipt.json').exists(), 'Cleanup receipt already exists')
    for source in (RECEIPT, DRYRUN):
        no_links(source)
        require(source.exists(), 'Inventoried synthetic source is missing: ' + str(source))
    require(sha(RECEIPT) == RECEIPT_SHA, 'Synthetic receipt has changed since approval')
    require(DRYRUN.is_dir(), 'Expected the exact controller dry-run directory')
    archive.mkdir()
    report = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
              'cleanup_complete': False, 'evidence_preserved': False,
              'wandb_cleanup_complete': False, 'wandb_cleanup_confirmed': False,
              'removed_local_paths': [], 'removed_wandb_artifacts': [],
              'removed_wandb_runs': [], 'protected': ['real datasets', 'public source weights',
              'all historical audit receipts including usl-ccam-preparation-20261017-003'],
              'unreviewed_local_wandb_directories': 'not removed',
              'cleanup_script_sha256': sha(__file__)}
    try:
        import wandb
        api = wandb.Api()
        run_path = f'{args.wandb_entity}/{PROJECT}/{RUN_ID}'
        run = api.run(run_path)
        require(run.id == RUN_ID and run.name == RUN_NAME and run.state == 'finished',
                'Only the exact finished synthetic run may be removed')
        logged = list(run.logged_artifacts())
        require({artifact.name for artifact in logged} == set(ARTIFACTS),
                'Synthetic run logged artifacts differ from the reviewed allowlist')
        artifacts = [api.artifact(f'{args.wandb_entity}/{PROJECT}/{name}') for name in ARTIFACTS]
        require([(artifact.type, artifact.size) for artifact in artifacts]
                == [('validation-receipt', 732), ('wandb-history', 1054)],
                'Artifact types or sizes differ from the reviewed inventory')
        require({artifact.id for artifact in logged} == {artifact.id for artifact in artifacts},
                'Artifact identities differ between the run and allowlist')
        run_meta = {'id': run.id, 'name': run.name, 'state': run.state, 'path': run_path,
                    'url': run.url, 'config': sanitize(dict(run.config)),
                    'summary': sanitize(dict(run.summary)), 'tags': list(run.tags)}
        write(archive / 'wandb-run.json', run_meta)
        with (archive / 'wandb-history.jsonl').open('x', encoding='utf-8') as stream:
            for row in run.scan_history():
                stream.write(json.dumps(sanitize(row), default=str, sort_keys=True) + '\n')
        run_dir = archive / 'wandb-run-files'
        run_dir.mkdir()
        for file in run.files():
            safe_name(file.name)
            file.download(root=str(run_dir), replace=False)
        artifact_metadata = []
        for index, artifact in enumerate(artifacts):
            entries = artifact.manifest.entries
            for name, entry in entries.items():
                safe_name(name)
                require(not entry.ref, 'External artifact references are outside this cleanup scope')
            destination = archive / f'wandb-artifact-{index}'
            artifact.download(root=str(destination))
            artifact_metadata.append({'id': artifact.id, 'name': artifact.name,
                                      'version': artifact.version, 'type': artifact.type,
                                      'aliases': list(artifact.aliases), 'size': artifact.size,
                                      'digest': artifact.digest,
                                      'manifest': {name: {'digest': entry.digest, 'size': entry.size}
                                                   for name, entry in entries.items()}})
        write(archive / 'wandb-artifacts.json', artifact_metadata)
        shutil.copyfile(RECEIPT, archive / RECEIPT.name)
        require(sha(archive / RECEIPT.name) == RECEIPT_SHA, 'Archived synthetic receipt mismatch')
        dryrun_hashes = {}
        with zipfile.ZipFile(archive / 'controller_dryrun.zip', 'x', zipfile.ZIP_DEFLATED) as package:
            for source in sorted(DRYRUN.rglob('*')):
                no_links(source)
                if source.is_dir():
                    continue
                require(source.is_file(), 'Only regular dry-run files may be archived')
                relative = source.relative_to(DRYRUN).as_posix()
                dryrun_hashes[relative] = sha(source)
                package.write(source, relative)
        with zipfile.ZipFile(archive / 'controller_dryrun.zip') as package:
            require(set(package.namelist()) == set(dryrun_hashes), 'Dry-run ZIP members differ')
            for name, digest in dryrun_hashes.items():
                require(hashlib.sha256(package.read(name)).hexdigest() == digest, 'Dry-run ZIP checksum mismatch')
        write(archive / 'controller_dryrun.files.json', dryrun_hashes)
        hashes = {}
        for source in sorted(archive.rglob('*')):
            no_links(source)
            if source.is_file():
                hashes[source.relative_to(archive).as_posix()] = sha(source)
        write(archive / 'archive.files.json', hashes)
        # Reopen every preserved payload before any destructive operation.
        for name, digest in hashes.items():
            require(sha(archive / name) == digest, 'Preserved payload checksum mismatch')
        report.update(evidence_preserved=True, archive=str(archive),
                      archive_manifest_sha256=sha(archive / 'archive.files.json'),
                      original_run=run_meta, original_artifacts=artifact_metadata)
        write(output / 'archive-verified.json', report)
        for artifact in artifacts:
            current = api.artifact(f'{args.wandb_entity}/{PROJECT}/{artifact.name}')
            require(current.id == artifact.id, 'Artifact identity changed before deletion')
            artifact.delete(delete_aliases=True)
            report['removed_wandb_artifacts'].append({'id': artifact.id, 'name': artifact.name})
        run.delete(delete_artifacts=False)
        report['removed_wandb_runs'].append(run_path)
        # Use fresh API objects so a cached object cannot count as an absence check.
        for kind, identifier in [('run', run_path), *[('artifact', f'{args.wandb_entity}/{PROJECT}/{name}') for name in ARTIFACTS]]:
            try:
                getattr(wandb.Api(), kind)(identifier)
            except (wandb.errors.CommError, ValueError) as error:
                message = str(error).lower()
                require('not found' in message or 'could not find' in message or 'does not exist' in message,
                        'Remote deletion absence was not confirmed')
            else:
                raise ValueError('Deleted W&B object is still retrievable')
        report.update(wandb_cleanup_complete=True, wandb_cleanup_confirmed=True)
        require(sha(RECEIPT) == RECEIPT_SHA, 'Synthetic receipt changed before deletion')
        no_links(RECEIPT)
        RECEIPT.unlink()
        report['removed_local_paths'].append(str(RECEIPT))
        require({path.relative_to(DRYRUN).as_posix(): sha(path)
                 for path in DRYRUN.rglob('*') if path.is_file()} == dryrun_hashes,
                'Dry-run contents changed after archival')
        for path in DRYRUN.rglob('*'):
            no_links(path)
        no_links(DRYRUN)
        shutil.rmtree(DRYRUN)
        report['removed_local_paths'].append(str(DRYRUN))
        require(not RECEIPT.exists() and not DRYRUN.exists(), 'Synthetic operational paths remain')
        report.update(cleanup_complete=True, status='completed',
                      finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        report.update(status='incomplete', error_type=type(error).__name__,
                      finished_at=datetime.now(timezone.utc).isoformat())
        write(output / 'cleanup.receipt.json', report)
        raise
    write(output / 'cleanup.receipt.json', report)
    print(json.dumps({'cleanup_complete': True, 'receipt': str(output / 'cleanup.receipt.json')}))


if __name__ == '__main__':
    main()
