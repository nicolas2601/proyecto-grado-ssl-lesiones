#!/usr/bin/env python3
"""Freeze accepted retrospective data artifacts offline; never enable execution."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import zipfile

from dermssl.segmentation_methods.common.protocol import (
    ACCEPTED_SOURCE_PINS, cohort_digest, file_digest, object_digest,
    validate_accepted_policy,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(reference, tracked):
    path = Path(reference['path']).resolve(strict=True)
    require(file_digest(path) == reference['sha256'], 'Input SHA-256 mismatch: ' + str(path))
    size = reference.get('bytes', reference.get('size_bytes'))
    require(size is None or size == path.stat().st_size, 'Input byte count mismatch: ' + str(path))
    tracked[str(path)] = reference['sha256']
    return path


def read_json(reference, tracked):
    value = json.loads(load(reference, tracked).read_text(encoding='utf-8'))
    require(isinstance(value, dict), 'JSON object required')
    return value


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return file_digest(path)


def canonical(identifier):
    require(isinstance(identifier, str), 'Image ID must be a string')
    result = identifier.removesuffix('_downsampled')
    require(re.fullmatch(r'ISIC_\d+', result), 'Full ISIC canonical ID required')
    return result


def archive_identity(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev


def extract_member(archive, evidence, target):
    member = evidence['member']
    parts = PurePosixPath(member)
    require(not parts.is_absolute() and '..' not in parts.parts and '\\' not in member,
            'Unsafe declared ZIP member')
    matches = [item for item in archive.infolist() if item.filename == member]
    require(len(matches) == 1 and not matches[0].is_dir(), 'Missing or duplicate declared ZIP member')
    info = matches[0]
    require(info.file_size == evidence['size_bytes'], 'ZIP member byte count mismatch')
    with archive.open(info) as source, target.open('xb') as destination:
        shutil.copyfileobj(source, destination, 1024 * 1024)
        destination.flush()
        os.fsync(destination.fileno())
    require(file_digest(target) == evidence['sha256'], 'Extracted member SHA-256 mismatch')


def materialize(args, staging, final, tracked):
    references = {}
    for label, expected in ACCEPTED_SOURCE_PINS.items():
        supplied = getattr(args, label + '_sha256')
        require(supplied == expected, 'Unexpected accepted source pin: ' + label)
        references[label] = {'path': str(getattr(args, label).resolve()), 'sha256': supplied}
    sources = {key: read_json(ref, tracked) for key, ref in references.items()}
    policy_ref = {'path': str(args.acceptance.resolve()), 'sha256': args.acceptance_sha256}
    policy = validate_accepted_policy(read_json(policy_ref, tracked))
    app, isolation, metadata, geometry = (sources[key] for key in
                                        ('application', 'isolation', 'metadata', 'geometry'))
    require(app.get('schema') == 'v8_provisional_exclusion_application.v2', 'Application schema mismatch')
    require(isolation.get('schema') == 'v8_learning_pool_inventory_isolation_audit.v1'
            and isolation.get('status') == 'eligibility_and_inventory_isolation_verified'
            and isolation.get('errors') == [], 'Inventory isolation is incomplete')
    require(metadata.get('schema') == 'v8_cached_current_pool_group_metadata.v1'
            and metadata.get('status') == 'cached_metadata_current_pool_join_completed', 'Metadata schema/status mismatch')
    coverage = metadata['evaluation_image_linkage_by_collection']['63']['group_coverage_among_joins']
    require(coverage['patient_id']['non_null_rows'] == 390
            and coverage['lesion_id']['non_null_rows'] == 1098,
            'Accepted incomplete official metadata coverage differs')
    require(geometry.get('stage') == 'evaluation_geometry_audit_only'
            and geometry.get('status') == 'passed' and geometry.get('errors') == []
            and geometry.get('review_findings') == []
            and all(geometry.get(key) == 2594 for key in ('expected_count', 'checked_count', 'passed_count')),
            'Geometry evidence is incomplete or has findings')
    pool_ref = app['outputs']['filtered_pool']
    require(pool_ref['sha256'] == isolation['inputs']['filtered_pool']['sha256']
            == metadata['pool_inventory']['sha256'], 'Sources bind different learning inventories')
    require(isolation['inputs']['application_receipt']['sha256'] == references['application']['sha256']
            and isolation['inputs']['geometry_report']['sha256'] == references['geometry']['sha256']
            and isolation['inputs']['geometry_records']['sha256'] == geometry['records']['sha256'],
            'Isolation evidence source binding mismatch')
    counts = isolation['counts']
    require(counts['filtered_records'] == 34784 and counts['union_removed_paths'] == 6112
            and counts['candidate_pairs'] == 15702
            and app['counts']['pool_records_after'] == 34784
            and app['counts']['union_exclusions_applied'] == 6112
            and app['counts']['phash_candidate_pairs'] == 15702, 'Accepted conservative counts mismatch')
    require(all(value == 0 for key, value in counts.items()
                if key.startswith('remaining_') or key.endswith('_overlaps')), 'Isolation still contains overlaps')
    for field in ('duplicates_identity_verified', 'exposure_review_complete', 'isolation_global_verified'):
        require(isolation.get(field) is False, 'Unexpected clinical/exposure evidence claim')
    pool = [json.loads(line) for line in load(pool_ref, tracked).read_text().splitlines() if line.strip()]
    require(len(pool) == 34784, 'Learning row count mismatch')
    fields = {'image_id', 'dataset', 'image_path', 'image_sha256', 'bytes', 'height', 'width'}
    require(all(isinstance(row, dict) and set(row) == fields for row in pool), 'Learning inventory schema mismatch')
    require(len({canonical(row['image_id']) for row in pool}) == len(pool)
            and len({row['image_path'] for row in pool}) == len(pool), 'Duplicate learning identity/path')
    hash_multiplicity = Counter(row['image_sha256'] for row in pool)
    repeated_counts = [count for count in hash_multiplicity.values() if count > 1]
    learning_content_multiplicity = {
        'unique_image_sha256s': len(hash_multiplicity),
        'duplicate_hash_groups': len(repeated_counts),
        'affected_rows': sum(repeated_counts),
        'excess_rows': sum(count - 1 for count in repeated_counts),
        'maximum_multiplicity': max(hash_multiplicity.values()),
    }
    records = [json.loads(line) for line in load(geometry['records'], tracked).read_text().splitlines() if line.strip()]
    require(len(records) == 2594 and len({row['image_id'] for row in records}) == 2594,
            'Geometry requires exact unique evaluation IDs')
    require(all(row.get('status') == 'passed' and canonical(row['image_id']) == row['image_id']
                for row in records), 'Invalid evaluation geometry record')
    # Compare full canonical IDs to cached official collection 63, without clinical inference.
    official_ref = metadata['sources']['63']['cached_provenance']
    official_path = load(official_ref, tracked)
    field = metadata['sources']['63']['current_parse']['image_id_field']
    with official_path.open(encoding='utf-8-sig', newline='') as stream:
        official_ids = [canonical(row[field]) for row in csv.DictReader(stream)]
    require(len(official_ids) == len(set(official_ids)) == 2594
            and set(official_ids) == {row['image_id'] for row in records}, 'Official/ZIP evaluation canonical IDs differ')
    evaluation_ids = {row['image_id'] for row in records}
    evaluation_hashes = {row['image']['sha256'] for row in records}
    mask_hashes = {row['mask']['sha256'] for row in records}
    require(len(evaluation_hashes) == 2594, 'Duplicate evaluation image bytes')
    require(not any(canonical(row['image_id']) in evaluation_ids
                    or row['image_sha256'] in evaluation_hashes | mask_hashes for row in pool),
            'Reserved image/mask identity or bytes in learning')
    learning = []
    for row in pool:
        load({'path': row['image_path'], 'sha256': row['image_sha256'], 'bytes': row['bytes']}, tracked)
        learning.append({key: row[key] for key in ('image_id', 'dataset', 'image_path', 'image_sha256')})
    learning.sort(key=lambda row: row['image_id'])
    evaluation, reference_rows = [], []
    (staging / 'evaluation-images').mkdir()
    (staging / 'evaluation-masks').mkdir()
    archive_paths = {key: load(ref, tracked) for key, ref in geometry['inputs'].items()
                     if key in ('images', 'masks')}
    archive_stats = {str(path): archive_identity(path) for path in archive_paths.values()}
    with ExitStack() as stack:
        archives = {key: stack.enter_context(zipfile.ZipFile(path)) for key, path in archive_paths.items()}
        for row in sorted(records, key=lambda item: item['image_id']):
            identifier = row['image_id']
            image_rel = Path('evaluation-images') / (identifier + '.jpg')
            mask_rel = Path('evaluation-masks') / (identifier + '_segmentation.png')
            extract_member(archives['images'], row['image'], staging / image_rel)
            extract_member(archives['masks'], row['mask'], staging / mask_rel)
            evaluation.append({'image_id': identifier, 'dataset': 'isic2018_task1',
                'image_path': str(final / image_rel), 'image_sha256': row['image']['sha256']})
            reference_rows.append({'image_id': identifier, 'image_sha256': row['image']['sha256'],
                'mask_path': str(final / mask_rel), 'mask_sha256': row['mask']['sha256']})
    cohort = cohort_digest(evaluation)
    learning_hashes = [cohort_digest(learning)]
    reference_hash = object_digest(sorted((row['image_id'], row['image_sha256'], row['mask_sha256'])
                                          for row in reference_rows))
    revision = file_digest(__file__)
    bindings = {'evaluation_cohort_sha256': cohort, 'reference_cohort_sha256': reference_hash,
                'learning_cohort_sha256s': learning_hashes}
    evidence = {}
    for check in ('exclusions_verified', 'mask_content_validated', 'evaluation_masks_isolated'):
        name = check + '.receipt.json'
        checksum = write_json(staging / name, {'check': check, 'verified': True,
            'review_revision': revision, 'scope': 'accepted_retrospective_benchmark',
            'source_evidence': references, 'duplicates_identity_verified': False,
            'exposure_review_complete': False, 'patient_independence_accredited': False, **bindings})
        evidence[check] = {'path': str(final / name), 'sha256': checksum}
    protocol = {'schema_version': 1, 'protocol_id': policy['policy_id'],
        'regime': 'retrospective_public_benchmark', 'scope': policy['scope'],
        'review_revision': revision, 'accepted_by': 'user', 'accepted_at': '2026-10-16',
        'accepted_data_policy': policy_ref, 'source_evidence': references,
        'patient_independence_accredited': False, 'expert_masks_used_for_learning': False,
        'diagnoses_used_for_learning': False, 'duplicates_identity_verified': False,
        'exposure_review_complete': False, 'global_clinical_identity_complete': False,
        'reserved_images': [{key: row[key] for key in ('image_id', 'image_sha256')} for row in evaluation],
        'prediction_threshold': .5, 'threshold_inclusive': True,
        'bootstrap_unit': 'image', 'bootstrap_seed': 42, 'bootstrap_replicates': 2000,
        'evidence': evidence, **bindings}
    protocol_sha = write_json(staging / 'protocol.json', protocol)
    protocol_ref = {'path': str(final / 'protocol.json'), 'sha256': protocol_sha, 'protocol_id': policy['policy_id']}
    eligibility = {'verified': True, 'excludes_target_test': True,
        'patient_independence_accredited': False, 'patient_lesion_groups_preserved': False,
        'deduplication_preserved': False, 'duplicates_identity_verified': False,
        'exposure_review_complete': False, 'global_clinical_identity_complete': False,
        'conservative_exclusion_status': 'accepted_conservative_eligibility',
        'source_manifest_sha256': pool_ref['sha256'], 'exclusions_sha256': references['application']['sha256'],
        'verification_revision': revision}
    outputs = {}
    for name, partition, rows in [('learning', 'train', learning), ('evaluation', 'evaluation', evaluation)]:
        filename = name + '.manifest.json'
        outputs[name] = {'path': str(final / filename), 'sha256': write_json(staging / filename,
            {'schema_version': 1, 'partition': partition, 'count': len(rows), 'records': rows,
             'eligibility': eligibility, 'evaluation_protocol': protocol_ref})}
    outputs['references'] = {'path': str(final / 'references.manifest.json'),
        'sha256': write_json(staging / 'references.manifest.json', {'schema_version': 1,
            'partition': 'evaluation', 'count': len(reference_rows), 'records': reference_rows,
            'evaluation_protocol': protocol_ref, 'evaluation_cohort_sha256': cohort})}
    # Recheck small provenance inputs. Image bytes were checked once above; readers rehash at use.
    image_paths = {str(Path(row['image_path']).resolve()) for row in pool}
    for path, checksum in tracked.items():
        if path in archive_stats:
            require(archive_identity(path) == archive_stats[path], 'Archive changed during extraction: ' + path)
        elif path not in image_paths:
            require(file_digest(path) == checksum, 'Provenance input changed before publication: ' + path)
    write_json(staging / 'freeze.receipt.json', {'schema': 'v8_accepted_manifest_freeze.v1',
        'status': 'accepted_data_artifacts_frozen', 'manifest_frozen': True,
        'scope': policy['scope'], 'outputs': outputs, 'evaluation_protocol': protocol_ref,
        'source_evidence': references, 'acceptance': policy_ref, **bindings,
        'execution_enabled': False, 'execution_ready': False, 'launch_ready': False,
        'training_started': False, 'training_consumable': False,
        'duplicates_identity_verified': False, 'exposure_review_complete': False,
        'patient_independence_accredited': False, 'global_clinical_identity_complete': False,
        'learning_content_multiplicity': learning_content_multiplicity,
        'limitations': ['Learning inventory preserves accepted image-level sampling multiplicity.',
                        'Equal byte hashes do not certify clinical identity or deduplication.'],
        'learned_artifact_lineage': 'pending_before_runtime_execution',
        'created_at': datetime.now(timezone.utc).isoformat()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*ACCEPTED_SOURCE_PINS, 'acceptance'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    final = args.output_dir.resolve()
    staging = None
    tracked = {}
    lock = final.with_name(final.name + '.freeze.lock')
    locked = False
    try:
        require(not final.exists(), 'Output must be a new absent directory')
        with lock.open('x'):
            locked = True
        require(not final.exists(), 'Output appeared before publication')
        staging = Path(tempfile.mkdtemp(prefix='.' + final.name + '.pending-', dir=final.parent))
        materialize(args, staging, final, tracked)
        require(not final.exists(), 'Output appeared before publication')
        os.rename(staging, final)
        staging = None
        print(json.dumps({'status': 'accepted_data_artifacts_frozen', 'output_dir': str(final),
                          'receipt_sha256': file_digest(final / 'freeze.receipt.json')}))
        return 0
    except Exception as error:
        failure = {'schema': 'v8_accepted_manifest_freeze_failure.v1', 'status': 'failed_closed',
            'error_type': type(error).__name__, 'diagnostic': str(error),
            'manifest_frozen': False, 'execution_enabled': False, 'training_started': False,
            'output_dir': str(final), 'verified_inputs': tracked}
        if locked:
            failure_path = final.with_name(final.name + '.failure.json')
            if not failure_path.exists():
                write_json(failure_path, failure)
        print(json.dumps(failure))
        return 1
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        if locked:
            lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
