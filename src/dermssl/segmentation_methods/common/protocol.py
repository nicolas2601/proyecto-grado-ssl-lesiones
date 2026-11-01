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
FRESH_POOL_CONTRACT = {
    'pool_id': 'isic2019_isic2020_candidate_v1',
    'source_pool_manifest_sha256': '60775b56e0f636ebc2d861a20b429c2f01bc55e74232582f56c3ebdff5e480ee',
    'source_receipt_sha256': '178809f412552d367b1bcfea475f213b40da4d99dbec758225ff8b0d8894f2cf',
    'exclusions_sha256': '002af9db83ce16e567a2b339f65a4ff06eea71e91dde155d492a4b9ba37e5cf4',
    'row_count': 53756,
    'dataset_counts': {'ISIC2019': 22574, 'ISIC2020': 31182},
    'evaluation_manifest_sha256': 'ca1b49b9c3d878ba7d2b94413e224604782aa3416e4523f93df42708e23a267f',
    'evaluation_count': 2594,
    'isic2020_patient_id_preserved': True,
    'group_eligibility_evidence_in_receipt': True,
    'patient_independence_accredited': False,
    'historical_exposure_resolved': False,
}
FRESH_POOL_ROW_FIELDS = {'image_id', 'dataset', 'image_path', 'image_sha256'}
V2_SHARED_POOL_CONTRACT = {
    'contract_id': 'classification_split_master_v2',
    'source_receipt_sha256': '412b71ef4a46e87a2fe877567980bced5e7eb778142328e59eb9943daea2636d',
    'source_pool_manifest_sha256': '60775b56e0f636ebc2d861a20b429c2f01bc55e74232582f56c3ebdff5e480ee',
    'source_exclusions_sha256': '9cd28d5f0fa099afe14e58088d4cbe8776147bed60add6bbe1dfb422a54acccc',
    'evaluation_manifest_sha256': '896c67e13a21f7eb069254a032e82b8435fbd48e93eb1d3ceab281805a56019d',
    'evaluation_exclusion_receipt_sha256': 'c0c15e6395edf338057695d33f69c2ac23546f97d6c056cbe6cc6955d6c9c5a7',
    'evaluation_count': 1822,
    'ph2_manifest_sha256': 'd2a159bb1f846e0ee2195406268d457d3c1b5a26a650c201b31e9f815371b9bf',
    'ph2_exclusions_sha256': 'b3253b3c1748f4057b497b8e2ceea6bfb4aa91344bba932b4c63d22253e9c09e',
    'ph2_eligible_count': 195,
    'selection_count': 34330,
    'selection_dataset_counts': {'ISIC2019': 14411, 'ISIC2020': 19919},
    'selection_source_manifest_sha256': '331457804f6bde1b65c4f184a1a529d9a6e4802b0349f6aebc9f2ab6d1945dc8',
    'selection_cohort_sha256': '469c275f2a22bd21a2cf496f4a85701f53776adbaea21f07aac1827de69810c2',
    'refit_count': 42977,
    'refit_dataset_counts': {'ISIC2019': 18053, 'ISIC2020': 24924},
    'refit_source_manifest_sha256': '21903961c680cf8cc3bdffeb8be394e2e32aec2ca93b295eea47eeb859d0bea0',
    'refit_cohort_sha256': 'ab65cb7273ed5402ee9e8a7509fd3fe7c1f417b0f0f14af2ebd994577bb75cba',
    'holdout_novelty_claim': False,
    'patient_independence_accredited': False,
    'execution_enabled': False,
    'execution_ready': False,
    'launch_ready': False,
    'training_started': False,
}
V2_SHARED_POOL_FIELDS = FRESH_POOL_ROW_FIELDS
DRAFT_EXECUTION_FIELDS = {
    'execution_enabled': False,
    'execution_ready': False,
    'launch_ready': False,
    'training_started': False,
    'resource_caps_measured': False,
    'smoke_passed': False,
}


def draft_execution_state():
    """Return the execution/readiness fields allowed in preparation-only artifacts."""
    return DRAFT_EXECUTION_FIELDS.copy()


def validate_fresh_pool_contract(contract, *, row_count=None, dataset_counts=None,
                                 row_fields=None, evaluation_manifest_sha256=None,
                                 evaluation_count=None):
    """Validate the exact 2019+2020 candidate pool while preserving unresolved claims."""
    if not isinstance(contract, dict) or contract != FRESH_POOL_CONTRACT:
        raise ValueError('Fresh ISIC2019+ISIC2020 source and evaluation pins mismatch')
    if row_count is not None and row_count != FRESH_POOL_CONTRACT['row_count']:
        raise ValueError('Fresh learning cohort count mismatch')
    if dataset_counts is not None and dataset_counts != FRESH_POOL_CONTRACT['dataset_counts']:
        raise ValueError('Fresh learning cohort dataset counts mismatch')
    if row_fields is not None and set(row_fields) != FRESH_POOL_ROW_FIELDS:
        raise ValueError('Fresh learning rows must use the exact image-only schema')
    if (evaluation_manifest_sha256 is not None
            and evaluation_manifest_sha256 != FRESH_POOL_CONTRACT['evaluation_manifest_sha256']):
        raise ValueError('Fresh evaluation manifest does not match the frozen evaluation cohort')
    if evaluation_count is not None and evaluation_count != FRESH_POOL_CONTRACT['evaluation_count']:
        raise ValueError('Fresh evaluation cohort count mismatch')
    return contract


# --- v3: source-cohort-disjoint pool contract (additive; v2 above unchanged) ---
# Derived from preflight/source-disjoint-pools-v3-20261101-003/selection.manifest.json
# Generated, not hand-written: the comparison is full dict equality.
V3_SHARED_POOL_CONTRACT = {
    "contract_id": "classification_split_master_v2",
    "evaluation_count": 1822,
    "evaluation_exclusion_receipt_sha256": "c0c15e6395edf338057695d33f69c2ac23546f97d6c056cbe6cc6955d6c9c5a7",
    "evaluation_manifest_sha256": "896c67e13a21f7eb069254a032e82b8435fbd48e93eb1d3ceab281805a56019d",
    "evaluation_reference_caveat": {
        "contract_evaluation_count": 1822,
        "contract_evaluation_manifest_sha256": "896c67e13a21f7eb069254a032e82b8435fbd48e93eb1d3ceab281805a56019d",
        "exclusion_computed_against_evaluation_count": 2594,
        "note": "The source-cohort exclusion in this manifest was computed against the 2594-image ISIC 2018 Task 1 split actually used for evaluation. The inherited contract pins a different, 1822-image evaluation manifest. The contract's disjointness statement therefore does not cover the evaluation split in use, and this manifest does not claim that it does."
    },
    "execution_enabled": False,
    "execution_ready": False,
    "holdout_novelty_claim": False,
    "launch_ready": False,
    "patient_independence_accredited": False,
    "ph2_eligible_count": 195,
    "ph2_exclusions_sha256": "b3253b3c1748f4057b497b8e2ceea6bfb4aa91344bba932b4c63d22253e9c09e",
    "ph2_manifest_sha256": "d2a159bb1f846e0ee2195406268d457d3c1b5a26a650c201b31e9f815371b9bf",
    "refit_cohort_sha256": "baff6ca6a6c6ab784f426f8dcaba2bf638f6387644161af659fad6c6e02c5c53",
    "refit_count": 42728,
    "refit_dataset_counts": {
        "ISIC2019": 17804,
        "ISIC2020": 24924
    },
    "refit_source_manifest_sha256": "21903961c680cf8cc3bdffeb8be394e2e32aec2ca93b295eea47eeb859d0bea0",
    "selection_cohort_sha256": "326f7316fcf2095a39122e3a198c6b0189a2bdafc43d7f137f6f5872f2aa2ae8",
    "selection_count": 34115,
    "selection_dataset_counts": {
        "ISIC2019": 14196,
        "ISIC2020": 19919
    },
    "selection_source_manifest_sha256": "331457804f6bde1b65c4f184a1a529d9a6e4802b0349f6aebc9f2ab6d1945dc8",
    "source_exclusions_sha256": "9cd28d5f0fa099afe14e58088d4cbe8776147bed60add6bbe1dfb422a54acccc",
    "source_pool_manifest_sha256": "60775b56e0f636ebc2d861a20b429c2f01bc55e74232582f56c3ebdff5e480ee",
    "source_receipt_sha256": "412b71ef4a46e87a2fe877567980bced5e7eb778142328e59eb9943daea2636d",
    "training_started": False
}


def validate_v2_shared_pool_contract(contract, records=None):
    """Validate the separate v2 segmentation-pool bridge without rebinding v1 pins."""
    if not isinstance(contract, dict):
        raise ValueError('Versioned v2 shared-pool contract required')
    stage = contract.get('stage')
    if stage not in {'selection', 'refit'}:
        raise ValueError('V2 shared-pool stage must be selection or refit')
    # Two pinned contracts are accepted: the original v2 pool and the v3
    # source-cohort-disjoint pool. Both are full-equality comparisons against a
    # constant, so a third shape still raises. v2 is unchanged, so every run that
    # validated before validates identically now.
    expected = None
    for _constant in (V2_SHARED_POOL_CONTRACT, V3_SHARED_POOL_CONTRACT):
        _candidate = dict(_constant)
        _candidate.update({
            'stage': stage,
            'stage_count': _constant[f'{stage}_count'],
            'stage_dataset_counts': _constant[f'{stage}_dataset_counts'],
            'stage_source_manifest_sha256': _constant[f'{stage}_source_manifest_sha256'],
            'stage_cohort_sha256': _constant[f'{stage}_cohort_sha256'],
        })
        if contract == _candidate:
            expected = _candidate
            break
    if expected is None:
        raise ValueError('V2 shared-pool source, split, evaluation, or PH2 pins mismatch')
    if records is not None:
        if (len(records) != expected['stage_count']
                or any(not isinstance(row, dict) or set(row) != V2_SHARED_POOL_FIELDS
                       for row in records)):
            raise ValueError('V2 shared-pool rows must have the exact image-only schema and count')
        counts = {dataset: sum(row['dataset'] == dataset for row in records)
                  for dataset in expected['stage_dataset_counts']}
        if counts != expected['stage_dataset_counts']:
            raise ValueError('V2 shared-pool dataset counts mismatch')
        if (len({row['image_id'] for row in records}) != len(records)
                or len({row['image_sha256'] for row in records}) != len(records)
                or cohort_digest(records) != expected['stage_cohort_sha256']):
            raise ValueError('V2 shared-pool identities or cohort hash mismatch')
    return contract


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
    fresh_pool = p.get('learning_pool_contract')
    if fresh_pool is not None:
        validate_fresh_pool_contract(fresh_pool)
        if (p.get('evaluation_manifest_sha256') != fresh_pool['evaluation_manifest_sha256']
                or p.get('evaluation_count') != fresh_pool['evaluation_count']):
            raise ValueError('Fresh-pool protocol evaluation binding mismatch')
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
    checks = (('exclusions_verified', 'mask_content_validated', 'evaluation_masks_isolated')
              if accepted is not None or fresh_pool is not None else CHECKS)
    for key in checks:
        receipt = checked_json(p.get('evidence', {}).get(key))
        if (receipt.get('check') != key or receipt.get('verified') is not True
                or receipt.get('review_revision') != p['review_revision']):
            raise ValueError('Incomplete or contradictory protocol evidence: ' + key)
        if any(receipt.get(field)!=p[field] for field in (
                'evaluation_cohort_sha256','reference_cohort_sha256','learning_cohort_sha256s')):
            raise ValueError('Protocol evidence belongs to different cohorts: ' + key)
    return p


def is_accepted_abstention_subset(rows, protocol, *, max_abstention_fraction=0.10, documented_abstentions=None):
    """Return True when rows are a strict non-empty subset of the pinned learning pool within the abstention cap,
    and all omitted IDs strictly correspond to documented abstentions."""
    if not isinstance(protocol, dict):
        return False
    if protocol.get('learning_pool_contract') is not None:
        contract = protocol['learning_pool_contract']
        try:
            validate_fresh_pool_contract(contract)
            reference = protocol.get('learning_pool_manifest')
            if reference is None:
                reference = protocol.get('evidence', {}).get('learning_pool_manifest')
            if (not isinstance(reference, dict)
                    or reference.get('sha256') != contract['source_pool_manifest_sha256']):
                return False
            pool_document = checked_json(reference)
            pool_rows = pool_document.get('records')
            if (not isinstance(pool_rows, list)
                    or len(pool_rows) != contract['row_count']
                    or pool_document.get('count') != contract['row_count']):
                return False
            pool_ids = [row.get('image_id') for row in pool_rows if isinstance(row, dict)]
            pool_hashes = [row.get('image_sha256') for row in pool_rows if isinstance(row, dict)]
            if (len(pool_ids) != contract['row_count']
                    or len(set(pool_ids)) != contract['row_count']
                    or len(set(pool_hashes)) != contract['row_count']):
                return False
        except (KeyError, TypeError, ValueError, OSError):
            return False
        doc_source = documented_abstentions
        if doc_source is None:
            for key in ('documented_abstentions', 'abstentions', 'abstention_ledger', 'abstention_ids'):
                if key in protocol:
                    doc_source = protocol[key]
                    break
                if key in protocol.get('evidence', {}):
                    doc_source = protocol['evidence'][key]
                    break
                if key in protocol.get('source_evidence', {}):
                    doc_source = protocol['source_evidence'][key]
                    break
        return _valid_pool_subset(rows, pool_rows, max_abstention_fraction, doc_source)
    if protocol.get('accepted_data_policy') is None:
        return False
    evidence = protocol.get('source_evidence', {})
    app_ref = evidence.get('application')
    if not isinstance(app_ref, dict) or app_ref.get('sha256') != ACCEPTED_SOURCE_PINS.get('application'):
        return False
    try:
        application = checked_json(app_ref)
        pool_ref = application['outputs']['filtered_pool']
        if file_digest(pool_ref['path']) != pool_ref['sha256']:
            return False
        pool_rows = [json.loads(line) for line in Path(pool_ref['path']).read_text().splitlines() if line.strip()]
    except Exception:
        return False
    doc_source = documented_abstentions
    if doc_source is None:
        for key in ('documented_abstentions', 'abstentions', 'abstention_ledger', 'abstention_ids'):
            if key in protocol:
                doc_source = protocol[key]
                break
            if key in protocol.get('evidence', {}):
                doc_source = protocol['evidence'][key]
                break
            if key in protocol.get('source_evidence', {}):
                doc_source = protocol['source_evidence'][key]
                break
    return _valid_pool_subset(
        rows, pool_rows, max_abstention_fraction, doc_source,
        pool_count=ACCEPTED_COUNTS['learning_images'])


def _valid_pool_subset(rows, pool_rows, max_abstention_fraction, documented_abstentions,
                       *, pool_count=None):
    import math
    import re
    if not isinstance(rows, list) or not rows or not isinstance(pool_rows, list):
        return False
    if (not isinstance(max_abstention_fraction, (int, float))
            or not 0 <= max_abstention_fraction < 1):
        return False
    pool_pairs = set()
    pool_ids = set()
    for row in pool_rows:
        if (not isinstance(row, dict) or not isinstance(row.get('image_id'), str)
                or not row['image_id'] or not re.fullmatch('[0-9a-f]{64}', row.get('image_sha256', ''))):
            return False
        pair = (row['image_id'], row['image_sha256'])
        if pair in pool_pairs or row['image_id'] in pool_ids:
            return False
        pool_pairs.add(pair)
        pool_ids.add(row['image_id'])
    if pool_count is not None and len(pool_pairs) != pool_count:
        return False

    row_pairs = set()
    row_ids = set()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get('image_id'), str)
                or not row['image_id'] or not re.fullmatch('[0-9a-f]{64}', row.get('image_sha256', ''))):
            return False
        pair = (row['image_id'], row['image_sha256'])
        if pair in row_pairs or row['image_id'] in row_ids:
            return False
        row_pairs.add(pair)
        row_ids.add(row['image_id'])
    min_required = math.ceil(len(pool_pairs) * (1.0 - float(max_abstention_fraction)))
    if len(row_pairs) < min_required or len(row_pairs) > len(pool_pairs):
        return False
    if not row_pairs.issubset(pool_pairs):
        return False

    omitted_ids = pool_ids - row_ids
    if isinstance(documented_abstentions, (list, tuple, set)):
        doc_ids = {item.get('image_id', '') if isinstance(item, dict) else str(item)
                   for item in documented_abstentions}
    elif isinstance(documented_abstentions, dict):
        doc_ids = set(documented_abstentions)
    elif documented_abstentions is None:
        doc_ids = set()
    else:
        return False
    return omitted_ids == doc_ids


def protocol_metadata(document):
    p = validate_binding(document)
    if p is None:
        return {}
    return {'evaluation_protocol': document['evaluation_protocol'], 'evaluation_regime': REGIME,
            'patient_independence_accredited': False, 'bootstrap_unit': 'image'}
