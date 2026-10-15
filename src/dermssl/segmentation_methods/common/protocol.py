"""Explicit, hash-bound retrospective benchmark acceptance; never patient accreditation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

REGIME = 'retrospective_public_benchmark'
CHECKS = ('exclusions_verified', 'duplicates_adjudicated', 'mask_content_validated',
          'exposure_review_complete', 'evaluation_masks_isolated')


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
    for key in CHECKS:
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
