"""Explicit, hash-bound retrospective benchmark acceptance; never patient accreditation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

REGIME = 'retrospective_public_benchmark'
CHECKS = ('exclusions_verified', 'duplicates_adjudicated', 'mask_content_validated',
          'exposure_review_complete', 'evaluation_masks_isolated')
ACCEPTED_POLICY = 'v8_conservative_retrospective_20261016'
ACCEPTED_COUNTS = {'excluded_paths': 6112, 'excluded_pairs': 15702,
                   'learning_images': 34784, 'evaluation_images': 2594,
                   'official_patient_ids_present': 390, 'official_lesion_ids_present': 1098}
ACCEPTED_SOURCE_PINS = {
    'application': '45d113fc0271543bb11d7d3b415951415beaf0b29798097fa2fcc35906baed1f',
    'isolation': '8fe770c15ec22cd067d005f49be7777009d983a0563f3edd4d33955d63b07ac7',
    'metadata': '0d1ecfbb523d37d7e6db495382f5c1d185296a7b43e7556e0cbe8cf6cbf8d54c',
    'geometry': 'e871a95705e2dfef32466e1ddb17099603fc4bdd8c518d91fcaa8073a7ce8366',
}


def validate_accepted_policy(policy):
    """Validate explicit acceptance without manufacturing clinical identity evidence."""
    if not isinstance(policy, dict) or policy.get('policy_id') != ACCEPTED_POLICY:
        raise ValueError('Unknown accepted data policy')
    if policy.get('schema_version') != 1 or policy.get('accepted_by') != 'user':
        raise ValueError('Explicit user acceptance required')
    if policy.get('acceptance_source') != 'user_chat_2026-10-16':
        raise ValueError('Explicit user acceptance source required')
    for key in ('duplicates_identity_verified', 'exposure_review_complete',
                'patient_independence_accredited', 'global_clinical_identity_complete'):
        if policy.get(key) is not False:
            raise ValueError('Unverified identity/exposure claims must remain false: ' + key)
    if policy.get('accepted_counts') != ACCEPTED_COUNTS:
        raise ValueError('Accepted conservative cohort/count policy mismatch')
    if policy.get('scope') != 'accepted_retrospective_benchmark':
        raise ValueError('Accepted retrospective benchmark scope required')
    return policy


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def cohort_digest(rows):
    """Bind exact original IDs and image bytes, independently of absolute paths."""
    return object_digest(sorted((r['image_id'], r['image_sha256']) for r in rows))


def checked_json(reference):
    if not isinstance(reference, dict) or not isinstance(reference.get('path'), str):
        raise ValueError('Explicit artifact path and SHA-256 required')
    if not re.fullmatch('[0-9a-f]{64}', reference.get('sha256', '')):
        raise ValueError('Full lowercase SHA-256 required')
    if file_digest(reference['path']) != reference['sha256']:
        raise ValueError('Reviewed artifact hash mismatch')
    value = json.loads(Path(reference['path']).read_text())
    if not isinstance(value, dict):
        raise ValueError('Reviewed JSON object required')
    return value


def validate_accepted_source_cohorts(protocol, sources, policy):
    """Bind cohort declarations to the exact inventories in accepted audit receipts."""
    application, isolation, geometry, metadata = (sources[key] for key in
                                                 ('application', 'isolation', 'geometry', 'metadata'))
    if (application.get('schema') != 'v8_provisional_exclusion_application.v2'
            or isolation.get('status') != 'eligibility_and_inventory_isolation_verified'
            or geometry.get('status') != 'passed' or geometry.get('errors') != []
            or geometry.get('review_findings') != []
            or metadata.get('schema') != 'v8_cached_current_pool_group_metadata.v1'):
        raise ValueError('Accepted source audit schema/status mismatch')
    pool_ref = application['outputs']['filtered_pool']
    if (pool_ref['sha256'] != isolation['inputs']['filtered_pool']['sha256']
            or pool_ref['sha256'] != metadata['pool_inventory']['sha256']):
        raise ValueError('Accepted source learning inventory binding mismatch')

    def jsonl(reference):
        if file_digest(reference['path']) != reference['sha256']:
            raise ValueError('Accepted inventory hash mismatch')
        rows = [json.loads(line) for line in Path(reference['path']).read_text().splitlines() if line.strip()]
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError('Accepted inventory object rows required')
        return rows

    pool, geometry_rows = jsonl(pool_ref), jsonl(geometry['records'])
    counts = policy['accepted_counts']
    if (len(pool) != counts['learning_images'] or len(geometry_rows) != counts['evaluation_images']
            or len({row['image_id'] for row in pool}) != len(pool)
            or len({row['image_id'] for row in geometry_rows}) != len(geometry_rows)
            or any(row.get('status') != 'passed' for row in geometry_rows)):
        raise ValueError('Accepted inventory counts/identities differ')
    evaluation = [{'image_id': row['image_id'], 'image_sha256': row['image']['sha256']}
                  for row in geometry_rows]
    references = sorted((row['image_id'], row['image']['sha256'], row['mask']['sha256'])
                        for row in geometry_rows)
    if (protocol['learning_cohort_sha256s'] != [cohort_digest(pool)]
            or protocol['evaluation_cohort_sha256'] != cohort_digest(evaluation)
            or protocol['reference_cohort_sha256'] != object_digest(references)):
        raise ValueError('Accepted protocol cohorts differ from pinned source inventories')


def validate_binding(*documents):
    """Return the reviewed protocol or None for the unchanged historical regime.

    Every supplied document must bind the same protocol whenever one opts in.
    Evidence files are review receipts, not substitutes for the underlying audits.
    """
    references = [d.get('evaluation_protocol') for d in documents]
    if not any(r is not None for r in references):
        return None
    if not references or any(r is None for r in references):
        raise ValueError('Config, gate and manifest must explicitly bind the same evaluation protocol')
    first = references[0]
    if any(r != first for r in references[1:]):
        raise ValueError('Evaluation protocol identity/hash/path mismatch')
    if any('patient_independence_accredited' in d and d['patient_independence_accredited'] is not False for d in documents):
        raise ValueError('Contradictory patient independence claim')
    p = checked_json(first)
    if p.get('schema_version') != 1 or p.get('regime') != REGIME:
        raise ValueError('Reviewed retrospective public benchmark protocol required')
    if not p.get('protocol_id') or p['protocol_id'] != first.get('protocol_id'):
        raise ValueError('Protocol identity mismatch')
    for key in ('review_revision', 'accepted_by', 'accepted_at'):
        if not isinstance(p.get(key), str) or not p[key].strip():
            raise ValueError('Explicit protocol review and user acceptance required')
    for key in ('patient_independence_accredited', 'expert_masks_used_for_learning',
                'diagnoses_used_for_learning'):
        if p.get(key) is not False:
            raise ValueError('Patient accreditation and expert/diagnostic learning prohibited in this regime')
    if not re.fullmatch('[0-9a-f]{64}', p.get('evaluation_cohort_sha256', '')):
        raise ValueError('Frozen evaluation cohort SHA-256 required')
    if not re.fullmatch('[0-9a-f]{64}', p.get('reference_cohort_sha256', '')):
        raise ValueError('Frozen reference-mask cohort SHA-256 required')
    reserved=p.get('reserved_images')
    if (not isinstance(reserved,list) or not reserved
            or any(not isinstance(r,dict) or set(r)!={'image_id','image_sha256'} for r in reserved)
            or any(not isinstance(r['image_id'],str) or not r['image_id'] for r in reserved)
            or any(not isinstance(r['image_sha256'],str) or not re.fullmatch('[0-9a-f]{64}',r['image_sha256']) for r in reserved)
            or len({r['image_id'] for r in reserved})!=len(reserved)
            or cohort_digest(reserved)!=p['evaluation_cohort_sha256']):
        raise ValueError('Exact unique reserved image identities required')
    learning=p.get('learning_cohort_sha256s')
    if (not isinstance(learning,list) or not learning
            or any(not isinstance(s,str) or not re.fullmatch('[0-9a-f]{64}',s) for s in learning)):
        raise ValueError('Frozen eligible learning cohort hashes required')
    if p.get('prediction_threshold') != .5 or p.get('threshold_inclusive') is not True:
        raise ValueError('Frozen inclusive >=0.5 prediction threshold required')
    if p.get('bootstrap_unit') != 'image' or type(p.get('bootstrap_seed')) is not int:
        raise ValueError('Explicit image bootstrap and seed required')
    if type(p.get('bootstrap_replicates')) is not int or p['bootstrap_replicates'] < 100:
        raise ValueError('At least 100 prespecified bootstrap replicates required')
    accepted = p.get('accepted_data_policy')
    if accepted is not None:
        policy = validate_accepted_policy(checked_json(accepted))
        evidence = p.get('source_evidence', {})
        if set(evidence) != set(ACCEPTED_SOURCE_PINS):
            raise ValueError('Accepted policy requires all four pinned source receipts')
        source_documents = {}
        for label, pin in ACCEPTED_SOURCE_PINS.items():
            if evidence[label].get('sha256') != pin:
                raise ValueError('Accepted source receipt pin mismatch: ' + label)
            source_documents[label] = checked_json(evidence[label])
        validate_accepted_source_cohorts(p, source_documents, policy)
        for key in ('duplicates_identity_verified', 'exposure_review_complete',
                    'global_clinical_identity_complete'):
            if p.get(key) is not False:
                raise ValueError('Accepted protocol must preserve unresolved claims: ' + key)
        if p.get('scope') != 'accepted_retrospective_benchmark':
            raise ValueError('Accepted protocol scope required')
    checks = ('exclusions_verified', 'mask_content_validated',
              'evaluation_masks_isolated') if accepted is not None else CHECKS
    for key in checks:
        receipt = checked_json(p.get('evidence', {}).get(key))
        if (receipt.get('check') != key or receipt.get('verified') is not True
                or receipt.get('review_revision') != p['review_revision']):
            raise ValueError('Incomplete or contradictory protocol evidence: ' + key)
        if any(receipt.get(field)!=p[field] for field in (
                'evaluation_cohort_sha256','reference_cohort_sha256','learning_cohort_sha256s')):
            raise ValueError('Protocol evidence belongs to different cohorts: ' + key)
    return p


def protocol_metadata(document):
    p = validate_binding(document)
    if p is None:
        return {}
    return {'evaluation_protocol': document['evaluation_protocol'], 'evaluation_regime': REGIME,
            'patient_independence_accredited': False, 'bootstrap_unit': 'image'}
