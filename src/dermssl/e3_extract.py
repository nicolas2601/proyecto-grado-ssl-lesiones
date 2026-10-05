"""Render the E3 conditions and extract their features, one split at a time (docs/33 section 3).

``extract_cell`` walks the ids in order, in batches. For each image it renders the requested
conditions (``e3_render``), runs the injected extractor on each condition's uint8 batch and
stores ``<split>__<condition>.npy`` as ``(n_views, n, dim)`` float32, the layout of
``torch_v71.extract_split``. It also stores the mask descriptors
(``<split>__descriptors.npy``, ``(n, 7)``, NaN for an empty mask) and the fallback flags
(``<split>__fallbacks.npy``, ``(n, len(CONDITIONS))`` bool), and returns the meta the cell
JSON keeps: shapes, sha256 of every array and fallback counts per condition.

The extractor is ``extract_fn(images_nhwc_uint8, n_views) -> (n_views, n, dim)``; on the
pod it wraps ``torch_v71.extract_split(...)["cls4"]`` (the B71 feature), in tests a fake.
Nothing here reads a label.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from dermssl import e3_render as render
from dermssl import e3_store as store

LoadItem = Callable[[str], tuple[np.ndarray, np.ndarray]]
ExtractFn = Callable[[np.ndarray, int], np.ndarray]


def _check_conditions(conditions: Sequence[str]) -> None:
    unknown = [c for c in conditions if c not in render.CONDITIONS]
    if unknown:
        raise ValueError(f"condiciones desconocidas {unknown}; se esperaba un subconjunto de {render.CONDITIONS}")


def _render_one(image_id: str, load_item: LoadItem, resize_crop, hair_fn):
    image, mask = load_item(image_id)
    images, fallbacks = render.render(image, mask, resize_crop=resize_crop, hair_fn=hair_fn)
    return images, fallbacks, render.describe(image, mask, hair_fn)


def extract_cell(split: str, ids: Sequence[str], conditions: Sequence[str], *, load_item: LoadItem,
                 extract_fn: ExtractFn, resize_crop, out_dir: Path, batch_size: int, n_views: int,
                 hair_fn=None, workers: int = 1) -> dict:
    """Extract ``conditions`` for ``ids`` and write the arrays; returns the cell's meta."""
    _check_conditions(conditions)
    feats: dict[str, list] = {c: [] for c in conditions}
    descriptors, flags = [], []
    with ThreadPoolExecutor(max(1, workers)) as pool:
        for start in range(0, len(ids), batch_size):
            batch = list(ids[start:start + batch_size])
            rendered = list(pool.map(lambda i: _render_one(i, load_item, resize_crop, hair_fn), batch))
            for condition in conditions:
                stacked = np.stack([images[condition] for images, _, _ in rendered])
                feats[condition].append(np.asarray(extract_fn(stacked, n_views), dtype=np.float32))
            descriptors += [d for _, _, d in rendered]
            flags += [[c in fb for c in render.CONDITIONS] for _, fb, _ in rendered]
    return _write(split, out_dir, conditions, feats, np.stack(descriptors), np.array(flags, dtype=bool), n_views)


def _write(split, out_dir, conditions, feats, descriptors, flags, n_views) -> dict:
    paths = {c: store.save_array(out_dir, split, c, np.concatenate(feats[c], axis=1)) for c in conditions}
    paths["descriptors"] = store.save_array(out_dir, split, "descriptors", descriptors)
    paths["fallbacks"] = store.save_array(out_dir, split, "fallbacks", flags)
    return {
        "split": split, "n": int(len(flags)), "n_views": n_views, "conditions": list(conditions),
        "shapes": {name: list(np.load(p, mmap_mode="r").shape) for name, p in paths.items()},
        "sha256": {name: store.file_sha256(p) for name, p in paths.items()},
        "fallbacks": {c: int(flags[:, i].sum()) for i, c in enumerate(render.CONDITIONS)},
        "descriptors_nan_rows": int(np.isnan(descriptors).any(axis=1).sum()),
    }
