"""SkinDisNet content candidates against canonical pools, without diagnosis access."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from audit_cross_role_content import fingerprint
from content_candidate_edges import hash_edges
from gpu_content_similarity import cosine_edges

ROOT = Path(__file__).resolve().parents[1]
COSINE_THRESHOLD = .995


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify_stats(records):
    for row in records:
        state = Path(row["path"]).stat()
        if state.st_size != row["size"] or state.st_mtime_ns != row["mtime_ns"]:
            raise ValueError("Pool image changed since recorded fingerprint: " + row["path"])


def load_inputs(config):
    for path, sha in config["input_sha256"].items():
        if digest(path) != sha:
            raise ValueError("Content audit input changed: " + path)
    plan = json.loads(Path(config["pool_plan"]).read_text())
    for row in plan["manifests"]:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("Canonical pool manifest changed")
    observations = defaultdict(list)
    for row in plan["records"]:
        path = str(Path(row["image_path"]).resolve())
        observations[path].append({"role": row["role"], "image_id": row["image_id"]})
    cached = jsonl(config["pool_fingerprints"])
    if any("error" in row for row in cached) or {row["path"] for row in cached} != set(observations):
        raise ValueError("Pool fingerprint cache is incomplete or misaligned")
    verify_stats(cached)
    dataset = Path(config["dataset_root"]).resolve()
    inventory = jsonl(dataset / "image-inventory.jsonl")
    if len(inventory) != 1710 or len({row["image_id"] for row in inventory}) != 1710:
        raise ValueError("Expected1710 unique preprocessed source file identities")
    skin, fingerprints = [], []
    for row in inventory:
        path = (dataset / row["relative_path"]).resolve()
        if not path.is_relative_to(dataset) or path.suffix.casefold() != ".jpg":
            raise ValueError("Invalid uploaded image path")
        record = fingerprint(str(path))
        if "error" in record or record["byte_sha256"] != row["sha256"]:
            raise ValueError("Uploaded image fingerprint mismatch: " + str(path))
        fingerprints.append(record)
        skin.append({"image_id": row["image_id"], "image_path": str(path),
                     "source_dataset": "SkinDisNet-preprocessed", "source_patient_identity": "unknown"})
    return plan, observations, cached, skin, fingerprints


def save_candidates(path, edges, scope, skin_ids, observations, excluded):
    role_hits = defaultdict(lambda: defaultdict(set))
    with path.open("x") as stream:
        for edge in edges:
            left, right = edge["left_path"], edge["right_path"]
            ids = [skin_ids[p] for p in (left, right) if p in skin_ids]
            excluded.update(ids)
            pool_members = observations.get(left, []) + observations.get(right, [])
            stream.write(json.dumps({"scope": scope, **edge, "new_source_ids": ids,
                                     "pool_members": pool_members}) + "\n")
            for method in edge["methods"]:
                for member in pool_members:
                    role_hits[method][member["role"]].update(ids)
    return {method: {role: len(ids) for role, ids in roles.items()} for method, roles in role_hits.items()}


def extract_embeddings(paths, output, config):
    frozen = Path(json.loads((ROOT / "config/paths.json").read_text())["frozen_experiment_root"])
    sys.path[:0] = [str(ROOT / "scripts"), str(frozen / "scripts"), str(frozen / "code/src")]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from evaluate_final_refit import load_encoder
    from extract_preprocessing_variants import extract
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or digest(config["public_checkpoint"]) != config["public_checkpoint_sha256"]:
        raise ValueError("Available GPU and pinned public encoder required")
    matrix = np.lib.format.open_memmap(output / "public-dinov2-content-embeddings.npy",
                                       mode="w+", dtype=np.float32, shape=(len(paths), 768))
    model = load_encoder(Path(config["public_checkpoint"]))
    for start in range(0, len(paths), 512):
        rows = [{"image_path": path} for path in paths[start:start + 512]]
        features = extract(model, rows, 224, False)
        if features.shape != (len(rows), 768) or not np.isfinite(features).all():
            raise ValueError("Invalid content embedding batch")
        matrix[start:start + len(rows)] = features
        matrix.flush()
        print(json.dumps({"event": "content_embeddings", "completed": start + len(rows), "total": len(paths)}), flush=True)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("Content encoder must have no gradients")
    del model
    torch.cuda.empty_cache()
    return matrix


def known_id_agreement(plan, index, matrix):
    import torch
    import torch.nn.functional as functional
    roles = defaultdict(dict)
    for row in plan["records"]:
        roles[row["role"]][row["image_id"]] = index[str(Path(row["image_path"]).resolve())]
    pairs = []
    for source, ham in (("source_fit", "ham_fit"), ("source_dev", "ham_dev")):
        pairs.extend((roles[source][key], roles[ham][key]) for key in roles[source].keys() & roles[ham].keys())
    left, right = np.asarray(pairs).T
    with torch.inference_mode():
        a = functional.normalize(torch.as_tensor(np.asarray(matrix[left]), device="cuda:0"), dim=1)
        b = functional.normalize(torch.as_tensor(np.asarray(matrix[right]), device="cuda:0"), dim=1)
        scores = (a * b).sum(1).cpu().numpy()
    return {"raw_ID_pairs": len(pairs), "cosine_ge995": int((scores >= COSINE_THRESHOLD).sum()),
            "minimum": float(scores.min()), "median": float(np.median(scores)), "maximum": float(scores.max()),
            "meaning": "Known raw-ID link agreement, not adjudicated clinical sensitivity or patient linkage"}


def run(protocol):
    started = time.monotonic()
    config = json.loads(Path(protocol).read_text())
    output = Path(config["output_directory"])
    if not output.resolve().is_relative_to((ROOT / "results").resolve()):
        raise ValueError("Audit output must remain in the isolated results directory")
    output.mkdir(exist_ok=False)
    plan, observations, pool_fp, skin, skin_fp = load_inputs(config)
    pool_fp = sorted(pool_fp, key=lambda row: row["path"])
    skin_ids = {row["image_path"]: row["image_id"] for row in skin}
    excluded = set()
    cross_hash = hash_edges(skin_fp, pool_fp)
    internal_hash = hash_edges(skin_fp, skin_fp)
    role_hits = save_candidates(output / "cross-pool-hash-candidates.jsonl", cross_hash,
                                "SkinDisNet versus canonical pools", skin_ids, observations, excluded)
    save_candidates(output / "within-skindisnet-hash-candidates.jsonl", internal_hash,
                    "Within SkinDisNet preprocessed", skin_ids, observations, excluded)
    paths = [row["path"] for row in pool_fp] + [row["image_path"] for row in skin]
    if len(set(paths)) != len(paths):
        raise ValueError("Expected unique physical content paths")
    with (output / "embedding-paths.jsonl").open("x") as stream:
        for path in paths:
            stream.write(json.dumps({"path": path, "source": "SkinDisNet" if path in skin_ids else "canonical-pool"}) + "\n")
    matrix = extract_embeddings(paths, output, config)
    n = len(pool_fp)
    edges = cosine_edges(matrix[n:], matrix[:n], threshold=COSINE_THRESHOLD)
    cross_embedding = [{"left_path": paths[n + e["query_index"]], "right_path": paths[e["reference_index"]],
                        "methods": ["embedding_cos_ge995"], "cosine": e["cosine"]} for e in edges]
    internal = cosine_edges(matrix[n:], matrix[n:], threshold=COSINE_THRESHOLD, same_collection=True)
    internal_embedding = [{"left_path": paths[n + e["query_index"]], "right_path": paths[n + e["reference_index"]],
                           "methods": ["embedding_cos_ge995"], "cosine": e["cosine"]} for e in internal]
    embedding_hits = save_candidates(output / "cross-pool-embedding-candidates.jsonl", cross_embedding,
                                     "SkinDisNet versus canonical pools", skin_ids, observations, excluded)
    role_hits.update(embedding_hits)
    save_candidates(output / "within-skindisnet-embedding-candidates.jsonl", internal_embedding,
                    "Within SkinDisNet preprocessed", skin_ids, observations, excluded)
    agreement = known_id_agreement(plan, {path: i for i, path in enumerate(paths)}, matrix)
    clean = [row for row in skin if row["image_id"] not in excluded]
    with (output / "skindisnet-preprocessed-eligible.jsonl").open("x") as stream:
        for row in clean:
            stream.write(json.dumps(row) + "\n")
    with (output / "excluded-skindisnet-file-ids.json").open("x") as stream:
        json.dump(sorted(excluded), stream)
    verify_stats(pool_fp)
    verify_stats(skin_fp)
    for path, sha in config["input_sha256"].items():
        if digest(path) != sha:
            raise ValueError("Audit input changed during execution")
    if any(digest(row["path"]) != row["sha256"] for row in plan["manifests"]):
        raise ValueError("Canonical manifest changed during content audit")
    summary = {"status": "COMPLETE", "new_source_images": len(skin), "canonical_physical_paths": len(pool_fp),
               "hash_cross_edges": len(cross_hash), "hash_internal_edges": len(internal_hash),
               "embedding_cross_edges": len(cross_embedding), "embedding_internal_edges": len(internal_embedding),
               "thresholds": {"phash_hamming_max": 4, "public_dinov2_cosine_min": COSINE_THRESHOLD},
               "new_source_excluded_unique_files": len(excluded), "new_source_eligible_files": len(clean),
               "candidate_skin_ids_by_method_and_canonical_role": role_hits, "known_raw_ID_agreement": agreement,
               "combined_phone_count_with_PAD1134": len(clean) + 1134,
               "planning_gate_at_least2000_combined_phone_images": len(clean) + 1134 >= 2000,
               "eligible_manifest_sha256": digest(output / "skindisnet-preprocessed-eligible.jsonl"),
               "protocol_sha256": digest(protocol), "allocated_gpu_hours_upper_bound": (time.monotonic() - started) / 3600,
               "duration_seconds": time.monotonic() - started,
               "exclusion_policy": "Conservatively exclude new-source endpoints of hash/perceptual/embedding candidates. Canonical pools and existing derived base overlay are unchanged for a matched dataset contrast.",
               "limits": ["Candidates are not confirmed duplicate patients or prevalence estimates.", "Pool fingerprints reused only after size/mtime and manifest checks,not fresh whole-pool byte hashing.", "Unknown SkinDisNet patient identities and transformed-content false negatives remain possible.", "No diagnosis access, class mapping, encoder training or test-based threshold selection."]}
    with (output / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    run(parser.parse_args().protocol)
