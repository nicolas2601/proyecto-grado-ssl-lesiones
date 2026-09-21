"""Versioned, source-preserving inventory contracts for the FST pool."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

SCHEMA_VERSION = "1.0"
EXPECTED_COUNTS = {
    "ham10000": 10_015,
    "isic2019": 25_331,
    "isic2020": 33_126,
}
SPLITS = ("train", "validation", "test")
SPLIT_RATIOS = (0.70, 0.15, 0.15)


def sha256_file(path: Path) -> str:
    """Return the content digest without loading an image into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as image_file:
        for block in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_pool_manifest(
    dataset_images: Mapping[str, Iterable[Path]],
    *,
    source_revisions: Mapping[str, str],
    expected_counts: Mapping[str, int] = EXPECTED_COUNTS,
) -> dict:
    """Build JSON-ready inventory rows without collapsing same-named images.

    The composite ``(dataset, image_id, source_revision)`` key is intentional:
    ISIC image identifiers can occur in multiple mounted sources and must retain
    their provenance rather than overwrite one another in a stem-keyed index.
    """
    images: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    counts = {dataset: 0 for dataset in expected_counts}

    for dataset, paths in dataset_images.items():
        if dataset not in source_revisions:
            raise ValueError(f"missing source revision for {dataset!r}")
        revision = source_revisions[dataset]
        for raw_path in paths:
            path = Path(raw_path)
            key = (dataset, path.stem, revision)
            if key in seen:
                raise ValueError(f"duplicate pool identity: {key!r}")
            seen.add(key)
            counts[dataset] = counts.get(dataset, 0) + 1
            images.append(
                {
                    "dataset": dataset,
                    "image_id": path.stem,
                    "source_revision": revision,
                    "sha256": sha256_file(path),
                    "duplicate_group": None,
                    "patient_or_lesion_group": None,
                    "split": None,
                    "coverage_status": "unknown",
                }
            )

    expected_total = sum(expected_counts.values())
    total_images = len(images)
    return {
        "schema_version": SCHEMA_VERSION,
        "total_images": total_images,
        "images": images,
        "coverage": {
            "expected_counts": dict(expected_counts),
            "observed_counts": counts,
            "expected_total": expected_total,
            "complete": total_images == expected_total
            and all(counts.get(dataset, 0) == expected for dataset, expected in expected_counts.items()),
        },
    }


def write_pool_manifest(manifest: Mapping[str, object], path: Path) -> None:
    """Persist the primary pool contract as deterministic UTF-8 JSON."""
    Path(path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def split_hash(images: Iterable[Mapping[str, object]]) -> str:
    """Hash the frozen assignment using source identity, group and split."""
    rows = [
        {
            "dataset": row["dataset"],
            "image_id": row["image_id"],
            "source_revision": row["source_revision"],
            "duplicate_group": row.get("duplicate_group"),
            "patient_or_lesion_group": row.get("patient_or_lesion_group"),
            "split": row.get("split"),
        }
        for row in images
    ]
    rows.sort(key=lambda row: (row["dataset"], row["image_id"], row["source_revision"]))
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _union(parent: list[int], left: int, right: int) -> None:
    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    left_root, right_root = root(left), root(right)
    if left_root != right_root:
        parent[right_root] = left_root


def _split_for_component(component_key: str, seed: int) -> str:
    sample = int.from_bytes(
        hashlib.sha256(f"{seed}:{component_key}".encode("utf-8")).digest()[:8], "big"
    ) / 2**64
    cumulative = 0.0
    for split, ratio in zip(SPLITS, SPLIT_RATIOS, strict=True):
        cumulative += ratio
        if sample < cumulative:
            return split
    return SPLITS[-1]


def freeze_grouped_splits(
    records: Iterable[Mapping[str, object]],
    *,
    seed: int = 42,
    perceptual_distance: int = 4,
) -> dict:
    """Assign deterministic splits without separating known linked images.

    Exact-content hashes and patient/lesion identifiers form connected
    components. A component, rather than an individual image, receives one
    split so known duplicates and clinical groups cannot leak across splits.
    """
    images = [dict(record) for record in records]
    if perceptual_distance < 0:
        raise ValueError("perceptual_distance must be non-negative")
    parent = list(range(len(images)))
    representatives: dict[tuple[str, str], int] = {}
    exact_hashes: dict[str, list[int]] = {}
    perceptual_buckets: dict[tuple[int, str], list[int]] = {}
    perceptual_edges: list[tuple[int, int]] = []

    for index, row in enumerate(images):
        for key in (
            ("sha256", str(row.get("sha256", ""))),
            ("clinical", str(row["patient_or_lesion_group"]))
            if row.get("patient_or_lesion_group")
            else None,
        ):
            if key is None or not key[1]:
                continue
            if key in representatives:
                _union(parent, index, representatives[key])
            else:
                representatives[key] = index
        exact_hashes.setdefault(str(row.get("sha256", "")), []).append(index)

        perceptual_hash = row.get("perceptual_hash")
        if perceptual_hash is None:
            continue
        try:
            perceptual_value = int(str(perceptual_hash), 16)
        except ValueError as exc:
            raise ValueError(f"invalid perceptual hash for {row['image_id']!r}") from exc
        candidate_indexes: set[int] = set()
        normalized_hash = f"{perceptual_value:016x}"
        for position in range(4):
            bucket_key = (position, normalized_hash[position * 4 : position * 4 + 4])
            candidate_indexes.update(perceptual_buckets.get(bucket_key, []))
            perceptual_buckets.setdefault(bucket_key, []).append(index)
        for candidate in candidate_indexes:
            candidate_value = int(str(images[candidate]["perceptual_hash"]), 16)
            if (perceptual_value ^ candidate_value).bit_count() <= perceptual_distance:
                _union(parent, index, candidate)
                perceptual_edges.append((index, candidate))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    components: dict[int, list[int]] = {}
    for index in range(len(images)):
        components.setdefault(root(index), []).append(index)

    perceptual_component_roots = {
        root(index) for edge in perceptual_edges for index in edge
    }

    for indexes in components.values():
        component_key = "|".join(
            sorted(
                f"{images[index]['dataset']}:{images[index]['image_id']}:{images[index]['source_revision']}"
                for index in indexes
            )
        )
        assigned_split = _split_for_component(component_key, seed)
        for index in indexes:
            images[index]["split"] = assigned_split
            sha256 = str(images[index].get("sha256", ""))
            images[index]["duplicate_group"] = (
                f"exact:{sha256}" if len(exact_hashes[sha256]) > 1 else None
            )
            if images[index]["duplicate_group"] is None and root(index) in perceptual_component_roots:
                images[index]["duplicate_group"] = f"perceptual:{root(index)}"

    limitations = []
    if any(
        row.get("dataset") == "isic2019" and not row.get("patient_or_lesion_group")
        for row in images
    ):
        limitations.append(
            "ISIC2019 lacks reliable patient/lesion groups for some images; "
            "only content-derived duplicate components constrain those assignments."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "images": images,
        "split_hash": split_hash(images),
        "limitations": limitations,
    }


def validate_split_isolation(images: Iterable[Mapping[str, object]]) -> list[str]:
    """Return any duplicate or clinical groups assigned to multiple splits."""
    group_splits: dict[str, set[str]] = {}
    for row in images:
        split = row.get("split")
        if split is None:
            continue
        for field in ("duplicate_group", "patient_or_lesion_group"):
            group = row.get(field)
            if group:
                key = f"{field}:{group}"
                group_splits.setdefault(key, set()).add(str(split))
    return sorted(key for key, splits in group_splits.items() if len(splits) > 1)
