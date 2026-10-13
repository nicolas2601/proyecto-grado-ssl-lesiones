from pathlib import Path
import pytest
from dermssl.w3_preflight import original_manifest, readiness, sha256


def test_modified_input_is_rejected(tmp_path):
    image = tmp_path / 'image.png'
    image.write_bytes(b'original')
    row = {'image_id': 'i', 'dataset': 'd', 'original': str(image),
           'source_sha256': sha256(image), 'size': [224, 224]}
    image.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        original_manifest({'status': 'preflight_passed', 'verified_cases': [row]}, 1)


def test_manifest_does_not_expose_annotations_or_other_views(tmp_path):
    image = tmp_path / 'image.png'
    image.write_bytes(b'original')
    row = {'image_id': 'i', 'dataset': 'd', 'original': str(image),
           'source_sha256': sha256(image), 'size': [224, 224],
           'pseudo_mask': 'student_r2.png', 'transformed': 'tone.png',
           'diagnosis': 'never export', 'expert_mask': 'never export'}
    out = original_manifest({'status': 'preflight_passed', 'verified_cases': [row]}, 1)
    assert set(out[0]) == {'image_id', 'dataset', 'image_path', 'image_sha256', 'source_size'}


def test_duplicate_ids_are_rejected_before_replacing_fixed_sample():
    with pytest.raises(ValueError, match='unique'):
        original_manifest({'status': 'preflight_passed', 'verified_cases': [{'image_id':'i'}, {'image_id':'i'}]}, 2)


def test_review_does_not_make_unimplemented_candidates_ready():
    from dermssl.w3_preflight import REQUIRED_GATES
    registry = {'candidates': [{'id': name, 'status': 'reviewed',
                 'gates': {g: g != 'smoke_passed' for g in REQUIRED_GATES}}
                 for name in ('sled', 'usl_net', 'rpi_net')]}
    assert all(not row['ready_for_pilot'] for row in readiness(registry))
    assert all(row['missing_gates'] == ['smoke_passed'] for row in readiness(registry))


def test_missing_candidate_cannot_be_silently_replaced():
    with pytest.raises(ValueError, match='three'):
        readiness({'candidates': [{'id':'sled'}, {'id':'usl_net'}, {'id':'tokencut'}]})
