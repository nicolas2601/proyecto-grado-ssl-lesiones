"""Per-image val probabilities rebuilt from a saved fine-tuning checkpoint.

The fine-tuning runs saved only metrics, not per-image probabilities. This
module runs INFERENCE ONLY (no retraining) from the best-epoch checkpoint with
the same evaluation pass as ``torch_run.run`` (``predict_proba``: same val
split, transform, effective batch, no autocast, no TTA) and refuses to write
anything unless every metric recomputed from those probabilities equals the one
recorded in the run's JSON within 1e-4.

The checkpoint holds ``modelo``, ``dataset``, ``smoke``, ``clases`` and
``state_dict`` - no epoch. The best epoch is therefore taken from the JSON
(``mejor_epoca``) and the pairing evidence is "recomputed metrics == recorded
metrics", plus the sha256 of the checkpoint stored next to the arrays.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from dermssl.finetune import torch_run
from dermssl.finetune.config import FinetuneConfig

TOLERANCE = 1e-4


class ValProbasError(ValueError):
    """Base of the reasons a model cannot be paired."""


class MetricMismatchError(ValProbasError):
    """Recomputed metrics differ from the ones recorded by the run."""


class CheckpointMismatchError(ValProbasError):
    """The checkpoint is not the one of this (model, dataset, classes)."""


class IdsMismatchError(ValProbasError):
    """The val ids are not exactly the reference val ids."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def core_metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """The five scalar metrics the paired contrasts read."""
    pred = proba.argmax(1)
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "auc_ovr_macro": float(roc_auc_score(y, proba[:, 1])),
            "ece": torch_run.ece(y, proba)}


def differing_metrics(recomputed, recorded, path: str = "") -> list[str]:
    """Paths of every numeric leaf that differs by more than TOLERANCE (or is missing)."""
    if isinstance(recorded, dict):
        out: list[str] = []
        for key, value in recorded.items():
            sub = f"{path}.{key}" if path else key
            if not isinstance(recomputed, dict) or key not in recomputed:
                out.append(sub)
                continue
            out += differing_metrics(recomputed[key], value, sub)
        return out
    if isinstance(recorded, list):
        if len(recomputed) != len(recorded):
            return [path]
        return [p for k, (a, b) in enumerate(zip(recomputed, recorded))
                for p in differing_metrics(a, b, f"{path}[{k}]")]
    return [] if abs(float(recomputed) - float(recorded)) <= TOLERANCE else [path]


def load_reference_ids(path: Path) -> list[str]:
    with np.load(path, allow_pickle=False) as saved:
        return saved["val_ids"].astype(str).tolist()


def _load_checkpoint(ckpt: Path, cfg: FinetuneConfig, classes: list[str]) -> dict:
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if (state.get("modelo"), state.get("dataset")) != (cfg.model, cfg.dataset):
        raise CheckpointMismatchError(
            f"{ckpt.name}: holds {state.get('modelo')}/{state.get('dataset')}, "
            f"expected {cfg.model}/{cfg.dataset}")
    if list(state.get("clases", [])) != list(classes):
        raise CheckpointMismatchError(f"{ckpt.name}: classes {state.get('clases')} != {classes}")
    return state


def _check_ids(val_ids: list[str], reference: list[str] | None) -> bool | None:
    """Require the same id SET as the reference; return whether the order is also equal."""
    if reference is None:
        return None
    if len(val_ids) != len(reference) or set(val_ids) != set(reference):
        raise IdsMismatchError(
            f"val ids differ from the reference: {len(val_ids)} vs {len(reference)} ids, "
            f"{len(set(val_ids) ^ set(reference))} not shared")
    return val_ids == reference


def write_probas(cfg: FinetuneConfig, ckpt: Path, json_path: Path, out: Path,
                 reference_val_ids: list[str] | None = None) -> dict:
    """Infer the val split from ``ckpt``, verify against ``json_path``, write ``out`` (npz).

    Raises a ``ValProbasError`` subclass, writing nothing, if the model cannot be paired.
    """
    ckpt, out = Path(ckpt), Path(out)
    recorded_doc = json.loads(Path(json_path).read_text())
    data = torch_run.load_split(cfg)
    order_equal = _check_ids(data.va_ids, reference_val_ids)
    if recorded_doc["n_val"] != len(data.va_ids):
        raise IdsMismatchError(f"n_val {len(data.va_ids)} != JSON n_val {recorded_doc['n_val']}")

    state = _load_checkpoint(ckpt, cfg, data.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _, interp = torch_run.build_net(cfg.model, data.n_classes, device, pretrained=False)
    net.load_state_dict(state["state_dict"])
    _, tf_eval = torch_run.loading.build_worker_transforms(interp)
    proba = torch_run.predict_proba(net, torch_run.make_val_loader(data, cfg, tf_eval), device)

    metrics = torch_run.all_metrics(data.y_va, proba, data.classes, cfg.seed)
    bad = differing_metrics(metrics, recorded_doc["probes"]["finetune"])
    if bad:
        raise MetricMismatchError(f"{cfg.model}: metrics differ from the JSON beyond "
                                  f"{TOLERANCE}: {', '.join(bad)}")

    sha = sha256_file(ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(
        tmp, y=data.y_va.astype(np.int64), proba=proba.astype(np.float64),
        val_ids=np.array(data.va_ids),
        groups=np.array([data.id2group[i] for i in data.va_ids]).astype(str),
        checkpoint_sha256=np.array(sha), model=np.array(cfg.model))
    tmp.replace(out)
    return {"model": cfg.model, "status": "pareado", "checkpoint_sha256": sha,
            "mejor_epoca_json": recorded_doc.get("mejor_epoca"), "checkpoint_epoch": None,
            "n_val": len(data.va_ids), "ids_order_equal_reference": order_equal,
            "probas_sha256": sha256_file(out)}


CHECKPOINT_IDENTITY_NOTE = ("checkpoint_epoch=null; identidad del checkpoint inferida por "
                            "coincidencia de métricas")
LABEL = ("probas de val reconstruidas por inferencia desde el checkpoint de la mejor epoca; "
         "verificadas contra las metricas registradas")


def pair_models(models, make_cfg, ckpt_root: Path, json_root: Path, out_dir: Path,
                reference_val_ids: list[str] | None, suffix: str = "isic2020") -> dict:
    """Run :func:`write_probas` per model; a model that cannot be paired is recorded, not fatal.

    Environment problems (wrong scikit-learn, a split that is not the committed one) still raise:
    they affect every model. Returns the sidecar document.
    """
    records = []
    for model in models:
        folder = Path(ckpt_root) / model
        ckpt = folder / f"ft_{model}_{suffix}.pth"
        json_path = Path(json_root) / model / f"results_{model}_ft_{suffix}.json"
        out = Path(out_dir) / f"{model}.probas.npz"
        try:
            records.append(write_probas(make_cfg(model), ckpt, json_path, out, reference_val_ids))
        except (ValProbasError, FileNotFoundError) as exc:
            records.append({"model": model, "status": "no pareado",
                            "reason": f"{type(exc).__name__}: {exc}"})
    records = [{**rec, "checkpoint_identity_note": CHECKPOINT_IDENTITY_NOTE} for rec in records]
    sidecar = {"label": LABEL, "checkpoint_identity_note": CHECKPOINT_IDENTITY_NOTE,
               "tolerance": TOLERANCE, "models": records}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"ft_{suffix}_probas.sidecar.json").write_text(json.dumps(sidecar, indent=2))
    return sidecar
