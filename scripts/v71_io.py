"""Input/output side of ``v71_run.py``: split, holdout, images, weights, unit store.

Kept out of the stage logic so the stages read as the plan in docs/29 and the
seams a test needs (the hub model, the image decoder) sit in one place.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import cluster_frozen_bank as bank
from c1_build_val_index import rebuild_split
from dermssl import feature_cache as fc
from dermssl import v71_plan as plan
from dermssl import v71_protocol as vp

ARCH = "dinov2_vitb14"
WEIGHTS_FILE = "simsiam_dinov2_backbone_final.pt"
FINAL_DIRS = {("A", "s42"): "final_A", ("A", "s43"): "final_A_s43",
              ("B", "s42"): "final_B", ("B", "s43"): "final_B_s43"}


def write_json(path: Path, payload) -> None:
    """Atomic: a crash leaves the old file or none, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(bank.json_safe(payload), indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_context(args) -> dict:
    """C1 split (checked against its index), the docs/28 holdout (checked), labels, groups."""
    split, meta = rebuild_split(args.ham_root.expanduser())
    index = read_json(args.split_index.expanduser())
    if list(index["train_ids"]) != list(split["train_ids"]) or list(index["val_ids"]) != list(split["val_ids"]):
        raise SystemExit("ABORT: el indice de C1 no coincide con el split reconstruido")
    rows = meta.set_index("image_id")
    train_ids = list(split["train_ids"])
    labels = [split["cls2idx"][d] for d in rows.loc[train_ids, "dx"]]
    holdout = read_json(args.holdout.expanduser())
    try:
        holdout_sha = plan.check_holdout(holdout, train_ids, labels, list(rows.loc[train_ids, "lesion_id"]),
                                         val_ids=split["val_ids"])
    except ValueError as exc:
        raise SystemExit(f"ABORT: {exc}") from exc
    id2label = split["id2label"]
    ids = {"fit": holdout["fit_ids"], "holdout": holdout["holdout_ids"],
           "train": train_ids, "val": list(split["val_ids"])}
    if args.limit:
        for key in ids:
            ids[key] = plan.limited_ids(ids[key], [id2label[i] for i in ids[key]], args.limit)
    run_fingerprint = fc.fingerprint_ids([*ids["fit"], "|", *ids["holdout"]])  # changes with --limit
    return {"split": split, "ids": ids, "id2label": id2label, "holdout_sha": holdout_sha,
            "run_fingerprint": run_fingerprint,
            "classes": split["classes"], "mel": split["classes"].index("mel"),
            "image_index": bank.index_images([args.image_dir.expanduser()])}


def labels_of(ctx: dict, ids) -> np.ndarray:
    return np.array([ctx["id2label"][i] for i in ids])


class ImageStore:
    """uint8 arrays after the bank's resize + centre crop, decoded once per (ids, resolution)."""

    def __init__(self, root: Path, image_index: dict, workers: int):
        self.root, self.image_index, self.workers = root, image_index, workers

    def _decode(self, image_id: str, resolution: int) -> np.ndarray:
        from dermssl import torch_v71 as tv  # noqa: PLC0415

        image = bank.load_rgb_or_placeholder(self.image_index[image_id], resolution)
        return tv.resize_crop_uint8(image, resolution)

    def array(self, ids, resolution: int, tag: str) -> np.ndarray:
        path = self.root / f"{tag}_{resolution}_{fc.fingerprint_ids(list(ids))}.npy"
        if path.exists():
            return np.load(path, mmap_mode="r")
        missing = [i for i in ids if i not in self.image_index]
        if missing:
            raise SystemExit(f"ABORT: {len(missing)} imagenes sin archivo, p.ej. {missing[:3]}")
        start = time.time()
        with ThreadPoolExecutor(self.workers) as pool:
            arrays = list(pool.map(lambda i: self._decode(i, resolution), ids))
        out = np.stack(arrays)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npy")
        np.save(tmp, out)
        os.replace(tmp, path)
        print(f"[decode] {tag}@{resolution}: {len(ids)} imgs en {time.time() - start:.0f}s")
        return out


def hub_model(arch: str, pretrained: bool):
    """The one place a DINOv2 module is built (tests replace it with a tiny one)."""
    import torch  # noqa: PLC0415

    return torch.hub.load("facebookresearch/dinov2", arch, pretrained=pretrained)


class WeightBank:
    """public, the four finals, the soups, and every WiSE point, built on demand on CPU."""

    def __init__(self, weights_dir: Path, arch: str = ARCH):
        self.weights_dir, self.arch = weights_dir, arch
        self._cache: dict = {}
        self._model = None

    def final_path(self, arm: str, seed: str) -> Path:
        return self.weights_dir / FINAL_DIRS[(arm, seed)] / WEIGHTS_FILE

    def run_manifests(self) -> dict:
        return {f"{arm}_{seed}": read_json(self.final_path(arm, seed).parent / "run.json")
                for arm, seed in FINAL_DIRS}

    def _state(self, key: str) -> dict:
        import torch  # noqa: PLC0415

        from dermssl import torch_v71 as tv  # noqa: PLC0415

        if key not in self._cache:
            if key == vp.PUBLIC:
                state = hub_model(self.arch, pretrained=True).state_dict()
            elif key.endswith("_soup"):
                arm = key[0]
                state = tv.average_states(self._state(f"{arm}_s42"), self._state(f"{arm}_s43"))
            else:
                arm, seed = key.split("_")
                state = torch.load(self.final_path(arm, seed), map_location="cpu", weights_only=True)
            self._cache[key] = {k: v.detach().cpu().clone() for k, v in state.items()}
        return self._cache[key]

    def state(self, weights: str) -> dict:
        from dermssl import torch_v71 as tv  # noqa: PLC0415

        cfg = vp.parse_weights(weights)
        if cfg["arm"] == vp.PUBLIC:
            return self._state(vp.PUBLIC)
        return tv.interpolate_state(self._state(vp.PUBLIC), self._state(f"{cfg['arm']}_{cfg['source']}"),
                                    cfg["alpha"])

    def direct_state(self, arm: str) -> dict:
        """The seed-42 final read straight from disk, for the pre-test."""
        return self._state(f"{arm}_s42")

    def model(self, state: dict):
        """One module reused for every weight set: load_state_dict is strict."""
        if self._model is None:
            self._model = hub_model(self.arch, pretrained=False)
        self._model.load_state_dict(state, strict=True)
        return self._model.eval()


def direct_public_model(arch: str = ARCH):
    """Public DINOv2 straight from the hub, as the bank builds it (pre-test reference)."""
    return hub_model(arch, pretrained=True).eval()


def direct_final_model(path: Path, arch: str = ARCH):
    """A final loaded the bank's way: fresh hub module + state from disk."""
    import torch  # noqa: PLC0415

    model = hub_model(arch, pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()
