"""Resumable cells of the ISIC evaluation: one JSON per (dataset, config, pool variant), then the contrasts.

A cell is skipped when its JSON exists with the same key sha256 (dataset, config, split digests, weights
digest, resamples, ``--limit``, pool label); it is retried twice and, if it keeps failing, listed in
``store.failures`` while the other cells go on (``e3_store.CellStore``). The val probabilities of each
cell are kept in ``<cell>.probas.npz`` next to its JSON so the paired contrasts can be computed later,
whichever order the cells ran in.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from dermssl.isic_eval import evaluate as ev
from dermssl.isic_eval import protocol as pr
from dermssl.isic_eval import scoring
from dermssl.isic_eval.splits import EvalDataset


class ProbasIntegrityError(ValueError):
    """A legacy ``.probas.npz`` (object arrays) does not match the sha256 its cell registered."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path, *, allow_pickle: bool) -> dict:
    with np.load(path, allow_pickle=allow_pickle) as saved:
        return {name: saved[name] for name in saved.files}


def load_probas(path: Path, expected_sha: str) -> dict:
    """The arrays of a cell's ``.probas.npz``, groups as strings.

    Files written now need no pickle. The first pod run saved the patient groups of ISIC 2020 as an
    object array, which only loads with ``allow_pickle=True``: that is enabled for such a file ONLY after
    its sha256 equals the one its cell registered, and the groups are converted to ``str`` at once.
    """
    path = Path(path)
    try:
        loaded = _read(path, allow_pickle=False)
    except ValueError as exc:
        if "Object arrays" not in str(exc):
            raise
        if _sha(path) != expected_sha:
            raise ProbasIntegrityError(
                f"{path.name}: sha256 distinto del registrado ({expected_sha[:12]}); no se carga con pickle") from exc
        loaded = _read(path, allow_pickle=True)
    if "groups" in loaded:
        loaded["groups"] = loaded["groups"].astype(str)
    return loaded


def cell_key(ds: EvalDataset, config: str, pool_excludes_val: bool, weights_sha: str, limit, n_resamples) -> dict:
    return {"dataset": ds.name, "cell": pr.cell_name(config, pool_excludes_val), "config_id": pr.CONFIGS[config],
            "transductivo": pr.transductive(config, pool_excludes_val), "weights_sha256": weights_sha,
            "val_ids_sha256": ds.identity["val_ids_sha256"], "train_ids_sha256": ds.identity["train_ids_sha256"],
            "n_val": len(ds.val_idx), "n_train": len(ds.train_idx), "limit": limit, "n_resamples": n_resamples}


def _save_probas(path: Path, ds: EvalDataset, proba: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, y=ds.y_val, proba=proba, val_ids=np.array(ds.val_ids),
                        groups=np.array([]) if ds.val_groups is None else np.asarray(ds.val_groups).astype(str))
    tmp.replace(path)


def run_cell(store, ds: EvalDataset, config: str, extract: ev.Extractor, *, pool_excludes_val: bool,
             weights_sha: str, limit=None, n_resamples: int = scoring.N_RESAMPLES,
             keep_features: bool = False, provenance: dict | None = None) -> str:
    key = cell_key(ds, config, pool_excludes_val, weights_sha, limit, n_resamples)
    name = key["cell"]

    def compute() -> dict:
        cache = store.root / f"{name}.features.tmp.npz"
        block, proba = ev.evaluate_config(ds, config, extract, cache=cache, n_resamples=n_resamples)
        _save_probas(store.root / f"{name}.probas.npz", ds, proba)
        if not keep_features:
            cache.unlink(missing_ok=True)
        return {**(provenance or {}), **{k: key[k] for k in ("dataset", "config_id", "transductivo", "n_train", "n_val", "limit")},
                "cell": name, "pool_excludes_val": None if pr.is_public(config) else pool_excludes_val,
                "protocol": pr.parse(config), "split_identity": ds.identity,
                "probe": "logreg C=1 standardised, fitted on the dataset TRAIN split; no selection on val",
                "metrics": block, "probas_sha256": _sha(store.root / f"{name}.probas.npz")}

    return store.run(name, key, compute)


def _contrast_inputs(store, arm: str, control: str):
    docs = [store.read(c)["result"] for c in (arm, control)]
    loaded = [load_probas(store.root / f"{c}.probas.npz", doc["probas_sha256"])
              for c, doc in zip((arm, control), docs)]
    if not np.array_equal(loaded[0]["val_ids"], loaded[1]["val_ids"]):
        raise ValueError(f"{arm} and {control} were scored on different val images")
    return docs, loaded


def run_contrast(store, contrasts_store, arm: str, control: str, *, n_resamples: int) -> str | None:
    """Paired contrast ``arm - control`` once both cells exist; ``None`` when one is missing."""
    if not all(store.path(c).is_file() and (store.root / f"{c}.probas.npz").is_file() for c in (arm, control)):
        return None
    docs, (a, c) = _contrast_inputs(store, arm, control)
    key = {"arm": arm, "control": control, "arm_probas_sha256": docs[0]["probas_sha256"],
           "control_probas_sha256": docs[1]["probas_sha256"], "n_resamples": n_resamples,
           "metrics": list(scoring.CONTRAST_METRICS)}
    groups = a["groups"] if a["groups"].size else None
    return contrasts_store.run(f"{arm}__vs__{control}", key, lambda: {
        "paired": scoring.contrasts(arm, control, a["y"], a["proba"], c["proba"], groups=groups,
                                    n_resamples=n_resamples),
        "unit": "patient_id" if groups is not None else "image"})


def run_all_contrasts(store, contrasts_store, *, n_resamples: int) -> dict[str, str]:
    """Every registered pair whose two cells are on disk: ``name -> done|skipped|failed``."""
    out = {}
    for arm, control in pr.contrast_pairs():
        name = f"{arm}__vs__{control}"
        try:
            status = run_contrast(store, contrasts_store, arm, control, n_resamples=n_resamples)
        except ProbasIntegrityError as exc:  # one corrupt legacy cell fails its contrasts, not the others
            contrasts_store.failures.append((name, f"{type(exc).__name__}: {exc}"))
            status = "failed"
        if status is not None:
            out[name] = status
    return out
