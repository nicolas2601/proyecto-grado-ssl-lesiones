"""The resumable cells of E3b: descriptors, per-backbone extraction and per-backbone scoring.

Every cell is one JSON under ``<out>/cells`` (``e3_store.CellStore``: skipped when it exists
with the same key sha256, two retries, failures listed while the rest goes on). The keys
carry the ids, the mask and weight identities and the protocol, so a cell computed for
other inputs is set aside, never mixed in. Validation labels are read only inside the eval
cells, after the pinned-selection gate has passed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dermssl import e3_store as st
from dermssl import feature_cache as fc
from dermssl.e3b import b71, scoring
from dermssl.e3b import extraction as xt
from dermssl.e3b import protocol as proto
from dermssl.e3b.runtime import SPLITS, Runtime, Settings

PROBA_DIR = "probas"


def ids_key(rt: Runtime) -> dict:
    return {s: fc.fingerprint_ids(rt.ids[s]) for s in SPLITS}


def base_key(rt: Runtime, settings: Settings) -> dict:
    return {"ids_sha256": ids_key(rt), "protocol": proto.PROTOCOL, "limit": settings.limit, **rt.fingerprint}


def proba_path(settings: Settings, name: str) -> Path:
    return settings.out_dir / PROBA_DIR / f"{name}.npz"


def run_cell(cells: st.CellStore, name: str, key: dict, fn, sidecar, *, artifact: Path | None = None) -> str:
    """``cells.run`` plus the sidecar of a fresh result; a cell whose ``artifact`` vanished is recomputed."""
    status = cells.run(name, key, fn)
    if status == "skipped" and artifact is not None and not artifact.is_file():
        cells.path(name).unlink()
        status = cells.run(name, key, fn)
    print(f"[e3b] {name}: {status}")
    if status == "done":
        sidecar(cells.path(name))
    return status


def descriptor_cells(cells, rt: Runtime, settings: Settings, sidecar) -> None:
    feats = settings.out_dir / "features"
    for split in SPLITS:
        key = {"stage": "e3b_descriptors", "split": split, **base_key(rt, settings), "hair_flag": rt.hair_fn is not None}
        run_cell(cells, f"e3b__descriptors__{split}", key, lambda s=split: _describe(rt, settings, s, feats), sidecar)


def _describe(rt: Runtime, settings: Settings, split: str, feats: Path) -> dict:
    meta = xt.describe_split(split, rt.ids[split], rt.load_item, rt.hair_fn, feats, workers=settings.workers)
    if settings.e3_dir is not None and not settings.smoke:
        b71.check_descriptors(settings.e3_dir, split, np.load(feats / f"{split}__descriptors.npy"))
        meta["matches_e3_descriptors"] = True
    return meta


def extract_cells(cells, rt: Runtime, settings: Settings, model: str, get_backbone, sidecar) -> bool:
    """Extract train and val for ``model``; ``True`` when both cells exist afterwards.

    ``get_backbone()`` loads the weights and is called only when a cell has to run, so resuming a
    finished model never touches the weights. The key names the weights' source; their sha256 is
    recorded in the result.
    """
    for split in SPLITS:
        key = {"stage": "e3b_extract", "model": model, "split": split, **base_key(rt, settings),
               "weights_source": proto.WEIGHTS_SOURCE[model]}
        run_cell(cells, f"e3b__extract__{model}__{split}", key,
                 lambda s=split: _extract(rt, settings, get_backbone(), s), sidecar)
    return all(cells.path(f"e3b__extract__{model}__{s}").is_file() for s in SPLITS)


def _extract(rt: Runtime, settings: Settings, backbone, split: str) -> dict:
    feats = settings.out_dir / "features"
    meta = xt.extract_model_split(backbone, split, rt.ids[split], rt.load_item, feats,
                                  batch_size=settings.batch_size, workers=settings.workers)
    if settings.e3_dir is not None and not settings.smoke:
        b71.check_fallbacks(settings.e3_dir, split, np.load(feats / backbone.name / f"{split}__fallbacks.npy"))
    return meta


def _load_features(settings: Settings, model: str, split: str) -> dict:
    reader = st.FeatureReader(settings.out_dir / "features" / model)
    return {c: np.asarray(reader.load(split, c)) for c in proto.CONDITIONS}


def eval_cell(cells, rt: Runtime, settings: Settings, model: str, sidecar) -> str:
    shas = {s: cells.read(f"e3b__extract__{model}__{s}")["result"]["sha256"] for s in SPLITS}
    desc = {s: cells.read(f"e3b__descriptors__{s}")["result"]["sha256"] for s in SPLITS}
    key = {"stage": "e3b_eval", "model": model, "feature_sha256": shas, "descriptor_sha256": desc,
           "n_resamples": settings.n_resamples, **base_key(rt, settings)}
    return run_cell(cells, f"e3b__eval__{model}", key, lambda: _score(cells, rt, settings, model), sidecar,
                    artifact=proba_path(settings, model))


def _score(cells, rt: Runtime, settings: Settings, model: str) -> dict:
    feats = settings.out_dir / "features"
    y_train, y_val = rt.labels_of("train"), rt.labels_of("val")
    probas, info = scoring.arm_probas(
        _load_features(settings, model, "train"), _load_features(settings, model, "val"),
        np.load(feats / "train__descriptors.npy"), np.load(feats / "val__descriptors.npy"), y_train, rt.n_classes)
    path = proba_path(settings, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **probas)
    groups = rt.groups_of("val")
    rows = {arm: scoring.score_row(y_val, p, groups, rt.classes, mel=rt.mel, n_resamples=settings.n_resamples)
            for arm, p in probas.items()}
    extract = {s: cells.read(f"e3b__extract__{model}__{s}")["result"] for s in SPLITS}
    return {"model": model, "rows": rows, "descriptor_n_filled": info["n_filled"],
            "probas_sha256": st.file_sha256(path), "dim": extract["val"]["dim"],
            "weights_sha256": extract["val"]["weights_sha256"],
            "crop_fallbacks": {s: extract[s]["crop_fallbacks"] for s in SPLITS}}
