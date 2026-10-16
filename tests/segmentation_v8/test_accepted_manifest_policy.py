"""Accepted benchmark policy contracts; execution is reserved for UNABIA."""
import copy
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from dermssl.segmentation_methods.common.data import validate_manifest_document
from dermssl.segmentation_methods.common.protocol import (
    CHECKS, cohort_digest, file_digest, object_digest, validate_accepted_policy, validate_binding,
)


@pytest.fixture
def accepted(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding='utf-8')
        return {'path': str(path), 'sha256': file_digest(path)}

    policy = json.loads((Path(__file__).resolve().parents[2] /
        'plan-segmentacion/metodos-v8/launch-preparation/accepted-data-policy.json').read_text())
    policy['accepted_counts']['learning_images'] = 1
    policy['accepted_counts']['evaluation_images'] = 1
    image = tmp_path / 'image.jpg'
    image.write_bytes(b'original reserved bytes')
    rows = [{'image_id': 'ISIC_0000001', 'image_path': str(image),
             'image_sha256': file_digest(image), 'dataset': 'isic2018_task1'}]
    learning = [{'image_id': 'ISIC_0000002', 'image_sha256': 'e' * 64}]
    pool_ref = write('pool.jsonl', learning[0])
    geometry_ref = write('geometry.jsonl', {'image_id': rows[0]['image_id'], 'status': 'passed',
        'image': {'sha256': rows[0]['image_sha256']}, 'mask': {'sha256': 'c' * 64}})
    sources = {
        'application': write('application.json', {'schema': 'v8_provisional_exclusion_application.v2',
            'outputs': {'filtered_pool': pool_ref}}),
        'isolation': write('isolation.json', {'status': 'eligibility_and_inventory_isolation_verified',
            'inputs': {'filtered_pool': pool_ref}}),
        'geometry': write('geometry.json', {'status': 'passed', 'errors': [], 'review_findings': [],
            'records': geometry_ref}),
        'metadata': write('metadata.json', {'schema': 'v8_cached_current_pool_group_metadata.v1',
            'pool_inventory': pool_ref}),
    }
    protocol = {
        'schema_version': 1, 'regime': 'retrospective_public_benchmark',
        'protocol_id': 'accepted-fixture', 'review_revision': 'fixture',
        'accepted_by': 'user', 'accepted_at': '2026-10-16',
        'scope': 'accepted_retrospective_benchmark',
        'accepted_data_policy': write('acceptance.json', policy), 'source_evidence': sources,
        'patient_independence_accredited': False, 'expert_masks_used_for_learning': False,
        'diagnoses_used_for_learning': False, 'duplicates_identity_verified': False,
        'exposure_review_complete': False, 'global_clinical_identity_complete': False,
        'evaluation_cohort_sha256': cohort_digest(rows),
        'reference_cohort_sha256': object_digest([(rows[0]['image_id'], rows[0]['image_sha256'], 'c' * 64)]),
        'learning_cohort_sha256s': [cohort_digest(learning)],
        'reserved_images': [{k: rows[0][k] for k in ('image_id', 'image_sha256')}],
        'prediction_threshold': .5, 'threshold_inclusive': True,
        'bootstrap_unit': 'image', 'bootstrap_seed': 42, 'bootstrap_replicates': 100,
        'evidence': {},
    }
    for key in ('exclusions_verified', 'mask_content_validated', 'evaluation_masks_isolated'):
        protocol['evidence'][key] = write(key + '.json', {
            'check': key, 'verified': True, 'review_revision': 'fixture',
            **{name: protocol[name] for name in ('evaluation_cohort_sha256',
               'reference_cohort_sha256', 'learning_cohort_sha256s')}})

    def bind():
        return dict(write('protocol.json', protocol), protocol_id='accepted-fixture')

    proof = {'verified': True, 'excludes_target_test': True,
        'patient_lesion_groups_preserved': False, 'patient_independence_accredited': False,
        'deduplication_preserved': False, 'duplicates_identity_verified': False,
        'exposure_review_complete': False, 'global_clinical_identity_complete': False,
        'conservative_exclusion_status': 'accepted_conservative_eligibility',
        'source_manifest_sha256': 'a' * 64, 'exclusions_sha256': 'b' * 64,
        'verification_revision': 'fixture'}
    with patch('dermssl.segmentation_methods.common.protocol.ACCEPTED_SOURCE_PINS',
               {key: value['sha256'] for key, value in sources.items()}), patch(
               'dermssl.segmentation_methods.common.protocol.ACCEPTED_COUNTS', policy['accepted_counts']):
        yield policy, protocol, bind, rows, proof


def test_accepted_policy_allows_honest_conservative_manifest(accepted):
    _, protocol, bind, rows, proof = accepted
    document = {'schema_version': 1, 'partition': 'evaluation', 'count': 1,
                'records': rows, 'eligibility': proof, 'evaluation_protocol': bind()}
    assert validate_manifest_document(document, purpose='predict') is document
    assert protocol['duplicates_identity_verified'] is False


@pytest.mark.parametrize('field', ['duplicates_identity_verified', 'exposure_review_complete',
                                 'patient_independence_accredited', 'global_clinical_identity_complete'])
def test_false_clinical_claims_cannot_be_promoted(accepted, field):
    policy, protocol, bind, _, _ = accepted
    changed = copy.deepcopy(policy)
    changed[field] = True
    with pytest.raises(ValueError):
        validate_accepted_policy(changed)
    protocol[field] = True
    with pytest.raises(ValueError):
        validate_binding({'evaluation_protocol': bind()})


def test_unknown_policy_and_missing_acceptance_rejected(accepted):
    policy, protocol, bind, _, _ = accepted
    policy['policy_id'] = 'unreviewed'
    with pytest.raises(ValueError):
        validate_accepted_policy(policy)
    del protocol['accepted_data_policy']
    with pytest.raises(ValueError):
        validate_binding({'evaluation_protocol': bind()})


def test_tampered_source_pin_rejected(accepted):
    _, protocol, bind, _, _ = accepted
    protocol['source_evidence']['application']['sha256'] = 'f' * 64
    with pytest.raises(ValueError):
        validate_binding({'evaluation_protocol': bind()})


@pytest.mark.parametrize('field', ['learning_cohort_sha256s', 'evaluation_cohort_sha256',
                                 'reference_cohort_sha256'])
def test_substituted_cohort_rejected_despite_matching_source_receipt_pins(accepted, field):
    _, protocol, bind, _, _ = accepted
    protocol[field] = ['f' * 64] if field == 'learning_cohort_sha256s' else 'f' * 64
    if field == 'evaluation_cohort_sha256':
        protocol['reserved_images'][0]['image_id'] = 'ISIC_9999999'
        protocol[field] = cohort_digest(protocol['reserved_images'])
    # Forge a self-consistent protocol and all generic wrappers. Only the
    # unchanged source inventories can expose this cohort substitution.
    for reference in protocol['evidence'].values():
        path = Path(reference['path'])
        receipt = json.loads(path.read_text())
        for key in ('learning_cohort_sha256s', 'evaluation_cohort_sha256', 'reference_cohort_sha256'):
            receipt[key] = protocol[key]
        path.write_text(json.dumps(receipt))
        reference['sha256'] = file_digest(path)
    with pytest.raises(ValueError, match='cohorts differ from pinned source inventories'):
        validate_binding({'evaluation_protocol': bind()})


def test_production_policy_counts_cannot_shrink():
    from dermssl.segmentation_methods.common.protocol import ACCEPTED_COUNTS
    assert ACCEPTED_COUNTS['learning_images'] == 34784
    assert ACCEPTED_COUNTS['evaluation_images'] == 2594


def test_explicit_acceptance_source_required(accepted):
    policy, _, _, _, _ = accepted
    del policy['acceptance_source']
    with pytest.raises(ValueError):
        validate_accepted_policy(policy)


def test_materializer_wrong_pin_has_only_failure_receipt(tmp_path):
    script = Path(__file__).resolve().parents[2] / 'scripts/materialize_v8_accepted_manifests.py'
    spec = importlib.util.spec_from_file_location('accepted_materializer_contract', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / 'new-output'
    argv = ['--output-dir', str(output)]
    for label in ('application', 'isolation', 'metadata', 'geometry', 'acceptance'):
        argv.extend(['--' + label, str(tmp_path / (label + '.json')),
                     '--' + label + '-sha256', 'f' * 64])
    assert module.main(argv) == 1
    assert not output.exists()
    receipt = json.loads((tmp_path / 'new-output.failure.json').read_text())
    assert receipt['status'] == 'failed_closed'
    assert receipt['manifest_frozen'] is False
    assert not list(tmp_path.glob('*.pending-*'))


def test_materializer_never_overwrites_existing_output(tmp_path):
    script = Path(__file__).resolve().parents[2] / 'scripts/materialize_v8_accepted_manifests.py'
    spec = importlib.util.spec_from_file_location('accepted_materializer_existing', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / 'existing'
    output.mkdir()
    marker = output / 'preserve'
    marker.write_bytes(b'unchanged')
    argv = ['--output-dir', str(output)]
    for label in ('application', 'isolation', 'metadata', 'geometry', 'acceptance'):
        argv.extend(['--' + label, str(tmp_path / (label + '.json')),
                     '--' + label + '-sha256', 'f' * 64])
    assert module.main(argv) == 1
    assert marker.read_bytes() == b'unchanged'
    assert not (tmp_path / 'existing.failure.json').exists()


def test_legacy_protocol_still_requires_all_five_checks(accepted):
    _, protocol, bind, _, _ = accepted
    del protocol['accepted_data_policy']
    assert len(CHECKS) == 5
    with pytest.raises(ValueError):
        validate_binding({'evaluation_protocol': bind()})
    assert validate_binding({'schema_version': 1}) is None


def test_mask_field_cannot_enter_accepted_original_manifest(accepted):
    _, _, bind, rows, proof = accepted
    rows[0]['mask_path'] = '/evaluation/mask.png'
    with pytest.raises(ValueError):
        validate_manifest_document({'schema_version': 1, 'partition': 'evaluation', 'count': 1,
            'records': rows, 'eligibility': proof, 'evaluation_protocol': bind()}, purpose='predict')
