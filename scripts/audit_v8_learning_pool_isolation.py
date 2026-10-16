#!/usr/bin/env python3
"""Audit saved V8 pool eligibility and declared inventory isolation only.

No image bytes, ZIPs, readiness documents, or learned artifacts are inspected.
This report is not a protocol acceptance receipt or historical exposure review.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(items):
    result = {}
    for key, value in items:
        require(key not in result, f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def parse(data):
    return json.loads(data, object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError(f'Nonfinite JSON constant: {value}')))


def jsonl(data):
    rows = [parse(line) for line in data.decode('utf-8').splitlines() if line.strip()]
    require(all(isinstance(row, dict) for row in rows), 'JSONL records must be objects')
    return rows


def path_key(value):
    require(isinstance(value, str) and value.startswith('/') and '\\' not in value
            and '\x00' not in value, 'Expected absolute POSIX inventory path')
    require(not value.startswith('//'), 'Ambiguous double-slash inventory path')
    return posixpath.normpath(value)


def base_id(value):
    require(isinstance(value, str), 'Expected image ID string')
    return value.removesuffix('_downsampled')


def sha(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value),
            'Expected lowercase SHA-256')
    return value


def audit(args, report):
    checked = {}

    def pinned(reference, label, size_required=True):
        require(isinstance(reference, dict), f'Missing reference: {label}')
        path = Path(reference['path'])
        data = path.read_bytes()
        actual = digest(data)
        require(actual == sha(reference['sha256']), f'SHA-256 mismatch: {label}')
        size = reference.get('bytes', reference.get('size_bytes'))
        require(not size_required or type(size) is int, f'Missing byte count: {label}')
        require(size is None or size == len(data), f'Byte count mismatch: {label}')
        checked[label] = {'path': str(path.absolute()), 'sha256': actual, 'bytes': len(data)}
        report['inputs'] = checked.copy()
        return data

    app = parse(pinned({'path': str(args.application_receipt),
                        'sha256': args.application_receipt_sha256}, 'application_receipt', False))
    geometry = parse(pinned({'path': str(args.geometry_report),
                             'sha256': args.geometry_report_sha256}, 'geometry_report', False))
    require(app.get('schema') == 'v8_provisional_exclusion_application.v2',
            'Unexpected application schema')
    for flag in ('execution_enabled', 'launch_ready', 'training_started',
                 'manifest_frozen', 'training_consumable', 'similarity_adjudication_complete'):
        require(app.get(flag) is False, f'Application audit-only flag differs: {flag}')
    sources = app['sources']
    loaded = {name: pinned(sources[name], name) for name in (
        'register', 'pool_records', 'phash_pair_decisions', 'phash_path_decisions',
        'phash_adjudication_receipt')}
    filtered_data = pinned(app['outputs']['filtered_pool'], 'filtered_pool')
    ledger_data = pinned(app['outputs']['phash_ledger'], 'phash_ledger')
    report['not_verified'] = {
        name: {'reference': ref, 'reason': 'Outside inventory audit validation inputs; historical pins preserved.'}
        for name, ref in sources.items() if name not in loaded}
    report['not_verified']['evaluation_zip'] = {
        'reference': sources['evaluation_zip'],
        'reason': 'ZIP bytes not reread; only cross-bound to pinned completed geometry report.'}
    require(geometry.get('stage') == 'evaluation_geometry_audit_only'
            and geometry.get('status') == 'passed' and geometry.get('errors') == []
            and geometry.get('review_findings') == [], 'Geometry report is incomplete or has findings')
    require(sha(geometry['inputs']['images']['sha256']) == sha(sources['evaluation_zip']['sha256']),
            'Geometry image ZIP differs from application evaluation ZIP')
    records = jsonl(pinned(geometry['records'], 'geometry_records', False))
    require(len(records) == geometry['expected_count'] == geometry['checked_count']
            == geometry['passed_count'] and len(records) > 0, 'Geometry record counts differ')
    evaluation = {}
    reserved_hashes, mask_hashes, reserved_members, mask_members = set(), set(), set(), set()
    for row in records:
        identifier = row['image_id']
        require(isinstance(identifier, str) and re.fullmatch(r'ISIC_\d+', identifier)
                and identifier not in evaluation and row['status'] == 'passed',
                'Invalid or duplicate geometry identity')
        evaluation[identifier] = row['image']['member']
        reserved_hashes.add(sha(row['image']['sha256']))
        mask_hashes.add(sha(row['mask']['sha256']))
        reserved_members.add(row['image']['member'])
        mask_members.add(row['mask']['member'])
    register, phash = parse(loaded['register']), parse(loaded['phash_adjudication_receipt'])
    require(register.get('schema') == 'v8_exclusion_review_register.v1'
            and register.get('consistency', {}).get('all_checks_passed') is True,
            'Unverified register')
    require(phash.get('schema') == 'v8_phash_conservative_adjudication.v1'
            and phash.get('policy') == 'exclude_training_path_conservatively'
            and phash.get('duplicate_identity_status') == 'unverified', 'Unexpected pHash policy')
    for name in ('register', 'pool_records'):
        require(phash['sources'][name]['sha256'] == sources[name]['sha256'],
                f'pHash source binding mismatch: {name}')
    for label, name in [('pairs', 'phash_pair_decisions'), ('paths', 'phash_path_decisions')]:
        require(phash['outputs'][label]['sha256'] == sources[name]['sha256'],
                f'pHash decision binding mismatch: {label}')
    require(register['sources']['pool_records']['sha256'] == sources['pool_records']['sha256']
            and register['sources']['evaluation_zip']['sha256'] == sources['evaluation_zip']['sha256'],
            'Register source binding mismatch')
    require({row['eval_id']: row['eval_member'] for row in register['evaluation_images']}
            == evaluation and len(register['evaluation_images']) == len(evaluation),
            'Register reserved identities differ from geometry')

    def inventory(data):
        indexed = {}
        fields = {'bytes', 'dataset', 'height', 'image_id', 'image_path', 'image_sha256', 'width'}
        for row in jsonl(data):
            require(set(row) == fields, 'Unexpected learning inventory fields; review schema explicitly')
            key = path_key(row['image_path'])
            require(key not in indexed, 'Duplicate normalized inventory path')
            sha(row['image_sha256'])
            base_id(row['image_id'])
            indexed[key] = row
        return indexed

    original, kept = inventory(loaded['pool_records']), inventory(filtered_data)
    base_paths = set()
    for proposal in register['base_id_exclusion_proposals']:
        row = proposal['pool_record']
        key = path_key(row['image_path'])
        require(key in original and row == original[key] and key not in base_paths,
                'Invalid or duplicate base-ID proposal')
        require(proposal['eval_id'] == base_id(row['image_id'])
                and proposal['eval_member'] == evaluation[proposal['eval_id']], 'Invalid proposal identity')
        base_paths.add(key)
    derived_base = {key for key, row in original.items() if base_id(row['image_id']) in evaluation}
    require(base_paths == derived_base, 'Base-ID proposal coverage differs from independent derivation')
    decisions, pairs = jsonl(loaded['phash_path_decisions']), jsonl(loaded['phash_pair_decisions'])
    pair_fields = ('eval_id', 'eval_member', 'train_id', 'train_path',
                   'hamming_distance_63bit', 'exact_file_sha256_match')

    def immutable_pair(row):
        require(isinstance(row, dict) and set(pair_fields) <= row.keys(),
                'Missing immutable pair evidence')
        key = path_key(row['train_path'])
        require(key in original and row['train_path'] == original[key]['image_path'],
                'Pair training path must match exact original path')
        require(row['eval_id'] in evaluation
                and row['eval_member'] == evaluation[row['eval_id']]
                and row['train_id'] == base_id(original[key]['image_id']),
                'Immutable pair identity differs from source inventories')
        require(type(row['hamming_distance_63bit']) is int
                and 0 <= row['hamming_distance_63bit'] <= 8
                and row['exact_file_sha256_match'] is False,
                'Invalid immutable pHash distance or exact-match evidence')
        return tuple(row[field] for field in pair_fields)

    register_pairs = register.get('phash_candidate_pairs')
    require(isinstance(register_pairs, list), 'Missing pinned register candidate pairs')
    expected_pairs = Counter(immutable_pair(row) for row in register_pairs)
    require(all(count == 1 for count in expected_pairs.values()),
            'Duplicate candidate pair in pinned register')
    require(Counter(immutable_pair(row) for row in pairs) == expected_pairs,
            'Decision pairs omit, add, repeat, or alter pinned register evidence')
    expected_candidate_paths = {path_key(row['train_path']) for row in register_pairs}
    suspicious = set()
    for row in decisions:
        key = path_key(row['train_path'])
        require(key in original and key not in suspicious, 'Invalid or duplicate path decision')
        require(row['train_id'] == base_id(original[key]['image_id'])
                and row['dataset'] == original[key]['dataset'], 'Path decision inventory mismatch')
        require(row['disposition'] == phash['policy'] and row['duplicate_identity_status'] == 'unverified'
                and row['exclusion_applied'] is False, 'Unexpected path decision policy')
        suspicious.add(key)
    seen = set()
    for row in pairs:
        key = path_key(row['train_path'])
        identity = (row['eval_id'], key)
        require(identity not in seen and key in suspicious and key in original,
                'Duplicate or unmapped pair decision')
        require(row['eval_member'] == evaluation[row['eval_id']]
                and row['train_id'] == base_id(original[key]['image_id'])
                and row['disposition'] == phash['policy']
                and row['duplicate_identity_status'] == 'unverified'
                and row['exclusion_applied'] is False, 'Pair decision identity/policy mismatch')
        seen.add(identity)
    require({key for _, key in seen} == suspicious, 'Pair/path decision coverage differs')
    require(suspicious == expected_candidate_paths,
            'Candidate paths differ from independently derived pinned register coverage')
    excluded = base_paths | suspicious
    counts = {'original_records': len(original), 'filtered_records': len(kept),
              'candidate_pairs': len(pairs), 'candidate_paths': len(suspicious),
              'base_exclusion_paths': len(base_paths), 'union_removed_paths': len(excluded),
              'remaining_candidate_pairs': sum(key in kept for _, key in seen),
              'remaining_candidate_paths': len(suspicious & kept.keys()),
              'reserved_base_id_overlaps': sum(base_id(row['image_id']) in evaluation for row in kept.values()),
              'reserved_image_sha256_overlaps': sum(row['image_sha256'] in reserved_hashes for row in kept.values()),
              'expert_mask_sha256_overlaps': sum(row['image_sha256'] in mask_hashes for row in kept.values()),
              'expert_mask_filename_overlaps': sum(PurePosixPath(key).name.endswith('_segmentation.png') for key in kept)}
    report['counts'] = counts
    require(kept.keys() == original.keys() - excluded, 'Filtered inventory is not exact original minus union')
    require(all(row == original[key] for key, row in kept.items()), 'Filtered record content changed')
    require(all(counts[key] == 0 for key in counts if key.startswith('remaining_') or key.endswith('_overlaps')),
            'Reserved/suspicious identity, hash, or mask filename retained')
    # ZIP members are relative, not physical extracted paths. Check suffix aliases
    # without asserting any filesystem/symlink identity or extraction location.
    aliases = {member.lstrip('/') for member in reserved_members | mask_members}
    overlaps = [key for key in kept if any(key.endswith('/' + member) for member in aliases)]
    counts['reserved_member_path_suffix_overlaps'] = len(overlaps)
    require(not overlaps, 'Reserved ZIP member path suffix retained')
    expected_counts = {'pool_records': len(original), 'candidate_pair_decisions': len(pairs),
                       'phash_excluded_training_paths': len(suspicious),
                       'base_id_exclusion_proposals': len(base_paths),
                       'union_excluded_training_paths': len(excluded), 'remaining_candidate_count': len(kept)}
    require(all(phash['counts'].get(key) == value for key, value in expected_counts.items()),
            'pHash receipt counts differ from independent audit')
    ledger = list(csv.DictReader(io.StringIO(ledger_data.decode('utf-8'))))
    counts['ledger_rows'] = len(ledger)
    # Check every inspected input again before issuing the completion report.
    for label, reference in checked.items():
        current = Path(reference['path']).read_bytes()
        require(len(current) == reference['bytes'] and digest(current) == reference['sha256'],
                f'Input changed during audit: {label}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--application-receipt', type=Path, required=True)
    parser.add_argument('--application-receipt-sha256', required=True)
    parser.add_argument('--geometry-report', type=Path, required=True)
    parser.add_argument('--geometry-report-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=False, exist_ok=False)
    script_hash = digest(Path(__file__).read_bytes())
    report = {'schema': 'v8_learning_pool_inventory_isolation_audit.v1', 'status': 'failed',
              'started_at': datetime.now(timezone.utc).isoformat(), 'script_sha256': script_hash,
              'command': [sys.executable, str(Path(__file__).absolute()),
                          *(sys.argv[1:] if argv is None else argv)],
              'inputs': {}, 'errors': [], 'counts': {},
              'validation_domain': 'saved_pair_and_path_decisions_and_declared_learning_inventory_only',
              'similarity_adjudication_complete': False, 'duplicates_identity_verified': False,
              'patient_lesion_independence': 'unresolved', 'exposure_review_complete': False,
              'isolation_global_verified': False, 'execution_enabled': False, 'execution_ready': False,
              'launch_ready': False, 'training_started': False, 'manifest_frozen': False,
              'training_consumable': False,
              'limitations': ['No visual/clinical duplicate identity adjudication.',
                             'No historical checkpoint or pseudo-mask exposure review.',
                             'Declared inventory hashes are not rehashed image bytes.',
                             'Paths are lexical POSIX identities; symlinks are not resolved.',
                             'ZIP geometry is consumed as pinned evidence, not rerun.',
                             'ZIP member suffix checks do not establish global physical isolation.',
                             'CSV ledger is hash-verified and counted only; its row semantics are not audited.',
                             'No final protocol receipt or cohort acceptance is emitted.']}
    try:
        audit(args, report)
        require(digest(Path(__file__).read_bytes()) == script_hash, 'Audit script changed during run')
        report['status'] = 'eligibility_and_inventory_isolation_verified'
    except Exception as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    temporary, final = args.output_dir / '.receipt.pending', args.output_dir / 'receipt.json'
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, final)
    print(json.dumps({'status': report['status'], 'receipt': str(final.absolute()),
                      'sha256': digest(final.read_bytes()), 'errors': len(report['errors'])}, sort_keys=True))
    return 0 if not report['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
