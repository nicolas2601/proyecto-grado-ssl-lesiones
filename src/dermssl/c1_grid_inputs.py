"""Everything the C1 grid reads before it fits anything, checked on arrival.

Labels come from the v6 bank cache and are verified against the split's ids
by fingerprint; masks come from an artifact whose manifest and sha256 are
verified; ``--limit`` subsamples ids and labels with the SAME stride. The
point of gathering it here is that each of these checks runs before a single
backbone is built, so a mismatch costs seconds instead of a GPU session.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dermssl import c1_mask_export as mx
from dermssl import c1_mask_provenance as mp
from dermssl import c1_split_identity as si
from dermssl.feature_cache import FeatureCache, _digest, fingerprint_ids


@dataclass(frozen=True)
class GridInputs:
    """The verified inputs every cell of one launch shares."""

    mask_source: mp.MaskSource
    specs: list
    train_ids: list[str]
    val_ids: list[str]
    train_labels: np.ndarray
    val_labels: np.ndarray
    n_classes: int
    labels_sha256: str
    train_masks: np.ndarray
    val_masks: np.ndarray

    def spec_of(self, backbone: str) -> tuple[str, str]:
        return next((arch, source) for name, arch, source in self.specs if name == backbone)

    def backbone_names(self) -> list[str]:
        return [name for name, _, _ in self.specs]


def stride(n: int, limit: int | None) -> np.ndarray:
    """Indices of an evenly spaced subsample of ``limit`` out of ``n``."""
    if limit is None or limit >= n:
        return np.arange(n)
    return np.linspace(0, n - 1, limit).round().astype(int)


def _cache_key(cache_dir: Path, backbone: str, split: str, expected: dict | None) -> str:
    hits = sorted(cache_dir.glob(f"v1__{backbone}__{split}__r*__eval__*.features.npy"))
    if len(hits) != 1:
        raise SystemExit(
            f"ABORT: esperaba 1 cache de {backbone}/{split}, encontre {len(hits)}"
        )
    key = hits[0].name[: -len(".features.npy")]
    if expected is None:
        return key
    want = fingerprint_ids(expected[split])
    have = key.rsplit("__", 1)[-1]
    if have != want:
        raise SystemExit(
            f"ABORT: el cache de {split} tiene fingerprint {have} y los ids del split "
            f"dan {want}. Las etiquetas quedarian contra otras imagenes"
        )
    return key


def load_bank_labels(cache_dir: Path, backbone: str, expected: dict | None = None):
    """Train and val labels straight from the v6 bank cache.

    Taken rather than re-derived: the cache stores them aligned to the ordered
    id list its key fingerprints, and that fingerprint is what this split was
    verified against. Re-deriving from the metadata would be a second
    implementation of "which class is this row", and the two only have to
    disagree once.
    """
    cache = FeatureCache(cache_dir)
    out = []
    for split in ("train", "val"):
        key = _cache_key(cache_dir, backbone, split, expected)
        entry = cache.load(key)
        if entry is None:
            raise SystemExit(f"ABORT: no se pudo cargar el cache {key}")
        out.append(np.asarray(entry.labels))
    return out[0], out[1]


def _read_json(path: Path) -> dict:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def _lesion_masks(masks_path: Path, mask_source, train_ids, val_ids):
    position = {image_id: i for i, image_id in enumerate(mask_source.image_ids)}
    missing = [i for i in train_ids + val_ids if i not in position]
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} imagenes del split no estan en el artefacto")

    all_masks = np.load(masks_path.expanduser(), mmap_mode="r")
    train_masks = np.stack([np.asarray(all_masks[position[i]]) == mx.LESION_LABEL for i in train_ids])
    val_masks = np.stack([np.asarray(all_masks[position[i]]) == mx.LESION_LABEL for i in val_ids])
    return train_masks, val_masks


def load_grid_inputs(*, cache_dir: Path, masks: Path, masks_manifest: Path, split_index: Path,
                     backbones, parse_spec, limit: int | None) -> GridInputs:
    """Read and cross-check masks, split and labels, in the order that fails cheapest."""
    mask_source = mp.MaskSource.from_manifest(_read_json(masks_manifest))
    mx.verify_artifact(masks.expanduser(), mask_source.masks_sha256)

    index = _read_json(split_index)
    full_train_ids = list(index["train_ids"])
    full_val_ids = list(index["val_ids"])
    specs = [parse_spec(raw) for raw in backbones]

    # Review P1-1: labels come from the bank cache, features are extracted in
    # the order of the split index. Nothing guaranteed the two orders agree,
    # so it is checked HERE, on the full lists, before anything is fitted.
    train_labels, val_labels = load_bank_labels(
        cache_dir.expanduser(), specs[0][0],
        expected={"train": full_train_ids, "val": full_val_ids},
    )

    # --limit takes an evenly spaced subsample, not a prefix: the split lists
    # are grouped, and the first 40 train ids were measured on the pod to hold
    # a single class -- a probe cannot even be fitted on that. Labels are
    # sliced with the SAME stride so they stay aligned to their ids.
    train_pick = stride(len(full_train_ids), limit)
    val_pick = stride(len(full_val_ids), limit)
    train_ids = [full_train_ids[i] for i in train_pick]
    val_ids = [full_val_ids[i] for i in val_pick]
    train_labels = np.asarray(train_labels)[train_pick]
    val_labels = np.asarray(val_labels)[val_pick]
    n_classes = int(max(train_labels.max(), val_labels.max())) + 1
    print(f"etiquetas del banco: {len(train_labels)} train / {len(val_labels)} val, "
          f"{n_classes} clases (orden verificado por fingerprint)")

    labels_sha = _digest(np.concatenate([np.asarray(train_labels), np.asarray(val_labels)]))
    train_masks, val_masks = _lesion_masks(masks, mask_source, train_ids, val_ids)
    return GridInputs(
        mask_source=mask_source, specs=specs, train_ids=train_ids, val_ids=val_ids,
        train_labels=train_labels, val_labels=val_labels, n_classes=n_classes,
        labels_sha256=labels_sha, train_masks=train_masks, val_masks=val_masks,
    )


def provenance_factory(inputs: GridInputs, *, revision: str, probe: str):
    """``backbone -> provenance`` for the cell store.

    Recomputed from the id lists actually used, not copied from fields of the
    JSON that could disagree with those lists (review P1-1).
    """
    spec_by_name = {name: f"{arch}:{source}" for name, arch, source in inputs.specs}

    def provenance_for(backbone: str) -> dict:
        return {
            "code_revision": revision,
            "masks_sha256": inputs.mask_source.masks_sha256,
            "c1_train_ids_sha256": si.ids_sha256(inputs.train_ids),
            "c1_val_ids_sha256": si.ids_sha256(inputs.val_ids),
            "probe": probe,
            "backbone_spec": spec_by_name[backbone],
            "labels_sha256": inputs.labels_sha256,
        }

    return provenance_for
