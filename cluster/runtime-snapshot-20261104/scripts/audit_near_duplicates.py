"""Development-only similarity candidates, not adjudicated duplicate identities."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def phash(image):
    pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    basis = np.cos(np.pi * np.arange(8)[:, None] * (2 * np.arange(32) + 1) / 64)
    low = (basis @ pixels @ basis.T).ravel()[1:]
    return low > np.median(low)


def candidates(hashes, features, ids, groups):
    hashes, features = np.asarray(hashes), np.asarray(features, dtype=float)
    ids, groups = np.asarray(ids, dtype=str), np.asarray(groups, dtype=str)
    if (features.ndim != 2 or hashes.shape != (len(ids), 63)
            or features.shape[0] != len(ids) or groups.shape != ids.shape
            or len(set(ids)) != len(ids) or np.any(groups == "")
            or not np.isfinite(features).all()):
        raise ValueError("Expected unique aligned IDs, nonempty groups and finite features")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero feature norm")
    unit = features / norms
    out = []
    for i in range(len(ids)):
        distances = np.count_nonzero(hashes[i + 1:] != hashes[i], axis=1)
        cosines = unit[i + 1:] @ unit[i]
        mask = ((distances <= 4) | (cosines >= .995)) & (groups[i + 1:] != groups[i])
        for offset in np.flatnonzero(mask):
            j = i + 1 + offset
            out.append({"left_id": str(ids[i]), "right_id": str(ids[j]),
                        "left_group": str(groups[i]), "right_group": str(groups[j]),
                        "phash_distance": int(distances[offset]), "cosine": float(cosines[offset]),
                        "phash_candidate": bool(distances[offset] <= 4),
                        "embedding_candidate": bool(cosines[offset] >= .995)})
    return out


def run(root):
    root = Path(root).resolve()
    protocol = json.loads((root / "config/clinical-development-protocol-v1.json").read_text())
    manifest = root / "inputs/pad-adapt-image-only.jsonl"
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if manifest_hash != protocol["cohort"]["manifest_sha256"]:
        raise ValueError("Manifest digest changed")
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    ids = np.asarray([r["image_id"] for r in rows], dtype=str)
    groups = np.asarray([r["identity_component_id"] for r in rows], dtype=str)
    if len(ids) != protocol["cohort"]["images"] or len(set(groups)) != protocol["cohort"]["patients"]:
        raise ValueError("Cohort cardinality changed")
    cache = root / "results/preprocessing-variants-v1/S0_squash224.npz"
    with np.load(cache, allow_pickle=False) as data:
        if not np.array_equal(ids, data["image_ids"].astype(str)):
            raise ValueError("Feature image IDs are misaligned")
        features = data["features"]
    hashes = []
    for row in rows:
        with Image.open(row["image_path"]) as image:
            hashes.append(phash(image))
    pairs = candidates(np.asarray(hashes), features, ids, groups)
    output = root / "results/near-duplicate-audit-v1.json"
    report = {"scope": "PAD target_adapt only, label-free candidate audit",
              "manifest_sha256": manifest_hash, "feature_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
              "images": len(ids), "groups": len(set(groups)),
              "phash": "Grayscale 32x32 LANCZOS, unnormalized DCT-II low 8x8, DC excluded, median threshold",
              "thresholds": {"max_hamming_63_bits": 4, "min_embedding_cosine": .995},
              "cross_group_candidates": pairs,
              "candidate_count": len(pairs),
              "affected_groups": sorted({p[k] for p in pairs for k in ("left_group", "right_group")}),
              "adjudication": "Candidates only. Similarity does not prove shared lesion/patient. No identity changes or automatic exclusions.",
              "remaining": "Sensitivity analysis after explicit conservative candidate-group exclusion; no clinical validation claim"}
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"output": str(output), "candidate_count": len(pairs),
                      "affected_groups": len(report["affected_groups"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
