"""Extract the F4 inputs of one backbone and the mask descriptors, one split at a time.

``extract_model_split`` writes ``<out>/<model>/<split>__full.npy`` and ``__crop_view.npy``
(``(n, dim)`` float32, ids in order) plus ``__fallbacks.npy`` (the crop that fell back to
the full view). ``describe_split`` writes ``<out>/<split>__descriptors.npy`` (``(n, 7)``,
the E3 ``describe``: it does not depend on the backbone, so it is computed once). Nothing
here reads a label.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from dermssl import e3_render as render
from dermssl import e3_store as store
from dermssl.e3b import views

LoadItem = Callable[[str], tuple[np.ndarray, np.ndarray]]


def _pair(image_id: str, load_item: LoadItem, resize_crop):
    image, mask = load_item(image_id)
    return views.render_pair(image, mask, resize_crop)


def extract_model_split(backbone, split: str, ids: Sequence[str], load_item: LoadItem, out_dir: Path, *,
                        batch_size: int, workers: int = 1) -> dict:
    """Features of the full view and the crop view of every id; returns the cell's meta."""
    full, crop, flags = [], [], []
    with ThreadPoolExecutor(max(1, workers)) as pool:
        for start in range(0, len(ids), batch_size):
            rendered = list(pool.map(lambda i: _pair(i, load_item, backbone.resize_crop),
                                     ids[start:start + batch_size]))
            full.append(backbone.extract(np.stack([r[0] for r in rendered])))
            crop.append(backbone.extract(np.stack([r[1] for r in rendered])))
            flags += [r[2] for r in rendered]
    root = Path(out_dir) / backbone.name
    arrays = {"full": np.concatenate(full), "crop_view": np.concatenate(crop),
              "fallbacks": np.array(flags, dtype=bool)}
    paths = {name: store.save_array(root, split, name, array) for name, array in arrays.items()}
    return {"model": backbone.name, "split": split, "n": len(ids), "dim": backbone.dim,
            "weights_sha256": backbone.weights_sha256, "crop_fallbacks": int(arrays["fallbacks"].sum()),
            "shapes": {n: list(a.shape) for n, a in arrays.items()},
            "sha256": {n: store.file_sha256(p) for n, p in paths.items()}}


def _describe(image_id: str, load_item: LoadItem, hair_fn) -> np.ndarray:
    image, mask = load_item(image_id)
    return render.describe(image, mask, hair_fn)


def describe_split(split: str, ids: Sequence[str], load_item: LoadItem, hair_fn, out_dir: Path, *,
                   workers: int = 1) -> dict:
    with ThreadPoolExecutor(max(1, workers)) as pool:
        rows = np.stack(list(pool.map(lambda i: _describe(i, load_item, hair_fn), ids)))
    path = store.save_array(Path(out_dir), split, "descriptors", rows)
    return {"split": split, "n": len(ids), "shape": list(rows.shape), "sha256": store.file_sha256(path),
            "nan_rows": int(np.isnan(rows).any(axis=1).sum())}
