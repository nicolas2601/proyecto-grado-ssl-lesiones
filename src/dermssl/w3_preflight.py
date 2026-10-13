"""Prepare label-free W3 inputs and report readiness; never execute a method."""
from __future__ import annotations

import hashlib
from pathlib import Path

REQUIRED_GATES = (
    'implementation_verified', 'source_use_resolved', 'label_free_inputs_verified',
    'weights_provenance_resolved', 'isolated_environment_verified', 'smoke_passed',
)


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def readiness(registry: dict) -> list[dict]:
    candidates = registry['candidates']
    if {c['id'] for c in candidates} != {'sled', 'usl_net', 'rpi_net'} or len(candidates) != 3:
        raise ValueError('The three registered candidates must be preserved')
    result = []
    for c in candidates:
        gates = c['gates']
        if set(gates) != set(REQUIRED_GATES) or any(type(v) is not bool for v in gates.values()):
            raise ValueError('Each readiness gate must be an explicit boolean')
        missing = [g for g in REQUIRED_GATES if not gates[g]]
        result.append({'method': c['id'], 'ready_for_pilot': not missing,
                       'missing_gates': missing, 'status': c['status']})
    return result


def original_manifest(preflight: dict, expected_count: int = 100) -> list[dict]:
    if preflight.get('status') != 'preflight_passed':
        raise ValueError('A successful data preflight is required')
    rows = preflight['verified_cases']
    if len(rows) != expected_count or len({r['image_id'] for r in rows}) != expected_count:
        raise ValueError('The fixed sample must have the expected number of unique IDs')
    manifest = []
    for r in rows:
        path = Path(r['original'])
        if sha256(path) != r['source_sha256']:
            raise ValueError(f"Original input changed: {r['image_id']}")
        manifest.append({'image_id': r['image_id'], 'dataset': r['dataset'],
                         'image_path': str(path), 'image_sha256': r['source_sha256'],
                         'source_size': r['size']})
    return manifest
