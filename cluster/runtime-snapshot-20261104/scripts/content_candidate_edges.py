"""Generate exact and near-image candidate edges from precomputed fingerprints."""
import importlib.util
from pathlib import Path


_AUDIT_PATH = Path(__file__).with_name("audit_cross_role_content.py")
_SPEC = importlib.util.spec_from_file_location("_cross_role_content", _AUDIT_PATH)
_AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIT)
HashIndex = _AUDIT.HashIndex

_HASH_FIELDS = ("byte_sha256", "rgb_sha256")


def _validate(records, collection_name):
    if not isinstance(records, (list, tuple)):
        raise ValueError(f"{collection_name} must be a sequence of fingerprints")
    by_path = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Fingerprint must be a dictionary")
        path = record.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("Fingerprint path must be a nonempty string")
        if path in by_path:
            raise ValueError(f"Duplicate fingerprint path in {collection_name}: {path}")
        for field in _HASH_FIELDS:
            value = record.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Malformed {field} for {path}")
        value = record.get("phash")
        if type(value) is not int or value < 0 or value >= (1 << 63):
            raise ValueError(f"Malformed phash for {path}")
        by_path[path] = record
    return by_path


def hash_edges(query_records, reference_records, limit=250000):
    """Return deduplicated fingerprint matches; no semantic identity is inferred."""
    if type(limit) is not int or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    queries = _validate(query_records, "query_records")
    references = _validate(reference_records, "reference_records")
    indices = {field: {} for field in _HASH_FIELDS}
    phash_index = HashIndex()
    phash_paths = {}
    for path, record in references.items():
        for field in _HASH_FIELDS:
            indices[field].setdefault(record[field], set()).add(path)
        phash_index.add(record["phash"])
        phash_paths.setdefault(record["phash"], set()).add(path)

    edges = {}
    for left_path, record in queries.items():
        candidates = {field: set() for field in _HASH_FIELDS}
        for field in _HASH_FIELDS:
            candidates[field].update(indices[field].get(record[field], ()))
        near_hashes = phash_index.query(record["phash"], 4)
        candidates["phash"] = {path for value in near_hashes for path in phash_paths[value]}
        right_paths = set().union(*candidates.values())
        for right_path in right_paths:
            if left_path == right_path:
                continue
            pair = tuple(sorted((left_path, right_path)))
            methods = []
            if right_path in candidates["byte_sha256"]:
                methods.append("byte_exact")
            if right_path in candidates["rgb_sha256"]:
                methods.append("rgb_exact")
            distance = (record["phash"] ^ references[right_path]["phash"]).bit_count()
            if distance <= 4:
                methods.append("phash_le4")
            if pair in edges:
                prior = edges[pair]
                prior["methods"] = sorted(set(prior["methods"]) | set(methods))
                prior["phash_distance"] = min(prior["phash_distance"], distance)
            else:
                edges[pair] = {"left_path": pair[0], "right_path": pair[1],
                               "methods": sorted(methods), "phash_distance": distance}
                if len(edges) > limit:
                    raise ValueError(f"Candidate edge count exceeds limit ({limit})")
    return [edges[pair] for pair in sorted(edges)]
