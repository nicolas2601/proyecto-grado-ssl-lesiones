"""Dataset side of ``e5_fitzpatrick.py``: ISIC through its verified loaders, HAM10000 with its masks.

Nothing here decodes an image. The ISIC datasets come from ``dermssl.isic_eval.splits`` (the rebuilt
split is compared with the committed one and the run refuses on a difference); HAM10000 comes from the
C1 split (``c1_build_val_index.rebuild_split``) and its native 450 x 600 student_r3 masks, verified against
their manifest as E3 does. A HAM image without a mask aborts the run (a silent drop would change the counts).
ISIC reads the 224 x 224 pool (``salidas/tone/inputs``: ``originals/``, ``masks/``, ``mask_manifest.json``): each
image is ``eligible``, ``quarantine`` (the manifest rejected its mask) or ``absent`` (not in the pool); the last
two fall back to the border estimator. Nothing is invented for an image without a mask.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from dermssl import v71_plan as plan  # noqa: E402
from dermssl.e5_analyze import Task  # noqa: E402
from dermssl.e5_document import DatasetSpec  # noqa: E402
from dermssl.isic_eval import splits as sp  # noqa: E402

HAM = "ham10000"
DATASETS = (HAM, *sorted(sp.LOADERS))
SPLITS = ("train", "val")
# Pool folders searched, in order, for the images of each evaluation dataset: the ISIC 2019 evaluation split also
# holds HAM10000 images (same ISIC_xxxx ids), whose pool masks live in the two HAM passes.
POOL_DATASETS = {"isic2019": ("isic2019", "ham10000_p1", "ham10000_p2"), "isic2020": ("isic2020",)}
POOL_KIND = "pseudo-máscaras automáticas (student_r3, sin etiquetas expertas) del pool de `salidas/tone/inputs`"
ABSENT = {"mask_state": "absent", "pool_image": None, "pool_mask": None, "mask_sha256": None}


class MaskCoverageError(KeyError):
    """A HAM10000 image has no mask in the manifest."""


class MissingImageError(FileNotFoundError):
    """A declared image has no file under the image root."""


def check_mask_args(dataset: str, *, masks, masks_manifest, pool_root) -> None:
    """HAM10000 reads its native C1 masks; ISIC reads the pool masks. Each source is refused for the other."""
    if dataset == HAM:
        if masks is None or masks_manifest is None:
            raise SystemExit("ABORT: ham10000 necesita --masks y --masks-manifest (máscaras nativas 450 x 600 de C1)")
        if pool_root is not None:
            raise SystemExit("ABORT: --pool-root es solo para ISIC; HAM10000 usa sus máscaras nativas de C1")
        return
    if pool_root is None:
        raise SystemExit(f"ABORT: {dataset} necesita --pool-root (salidas/tone/inputs: originals/, masks/, mask_manifest.json)")
    if masks is not None or masks_manifest is not None:
        raise SystemExit(f"ABORT: --masks / --masks-manifest son las máscaras nativas de HAM10000, no de {dataset}")


@dataclass(frozen=True)
class Pool:
    """The 224 x 224 pool (``salidas/tone/inputs``): image and mask share a grid; the manifest says which masks are usable."""

    root: Path
    entries: dict  # (pool dataset, image_id) -> manifest entry
    manifest_sha256: str

    def _files(self, ds: str, image_id: str) -> tuple[str, str]:
        return (str(self.root / "originals" / ds / f"{image_id}.png"), str(self.root / "masks" / ds / f"{image_id}.png"))

    def fields(self, dataset: str, image_id: str) -> dict:
        """Task fields of one image: its mask state (eligible / quarantine / absent) and the pool files to read."""
        for ds in POOL_DATASETS[dataset]:
            entry = self.entries.get((ds, image_id))
            if entry is None:
                continue
            image, mask = self._files(ds, image_id)
            eligible = bool(entry.get("eligible"))
            return {"mask_state": "eligible" if eligible else "quarantine", "pool_image": image,
                    "pool_mask": mask if eligible else None, "mask_sha256": entry.get("mask_sha256")}
        return dict(ABSENT)

    def found_in(self, dataset: str, image_ids) -> dict:
        names = POOL_DATASETS[dataset]
        found = [next((d for d in names if (d, i) in self.entries), None) for i in image_ids]
        return {d: found.count(d) for d in names if d in found}


def load_pool(root: Path) -> Pool:
    """Read ``<root>/mask_manifest.json`` (a list of dicts keyed by dataset and image_id)."""
    root = Path(root).expanduser()
    path = root / "mask_manifest.json"
    if not path.is_file():
        raise SystemExit(f"ABORT: no {path}")
    raw = path.read_bytes()
    entries = {(e["dataset"], e["image_id"]): e for e in _manifest_entries(json.loads(raw))}
    return Pool(root, entries, hashlib.sha256(raw).hexdigest())


def _manifest_entries(payload) -> list[dict]:
    """The entry list of ``mask_manifest.json``: a dict with the list under ``masks`` (real), or a bare list (legacy)."""
    entries = payload.get("masks") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise SystemExit("ABORT: mask_manifest.json debe ser una lista de entradas o un dict con la lista en 'masks'")
    for entry in entries:
        if not isinstance(entry, dict) or "dataset" not in entry or "image_id" not in entry:
            raise SystemExit(f"ABORT: mask_manifest.json: entrada sin 'dataset' / 'image_id': {str(entry)[:80]}")
    return entries


def limit_tasks(tasks: list[Task], n: int | None) -> list[Task]:
    """Smoke subset: ``n`` images per split, round-robin over the classes, original order kept."""
    if n is None:
        return tasks
    keep: set[str] = set()
    for split in SPLITS:
        mine = [t for t in tasks if t.split == split]
        keep.update(plan.limited_ids([t.image_id for t in mine], [t.label for t in mine], n))
    return [t for t in tasks if t.image_id in keep]


def spec_from_eval_dataset(ds: sp.EvalDataset, limit: int | None = None, pool: Pool | None = None) -> DatasetSpec:
    """An ISIC dataset as loaded and verified by ``dermssl.isic_eval.splits``; with ``pool``, each task gets its mask state."""
    split_of = {int(k): "train" for k in ds.train_idx} | {int(k): "val" for k in ds.val_idx}
    extra = (lambda i: {}) if pool is None else (lambda i: pool.fields(ds.name, i))
    tasks = [Task(image_id=ds.ids[k], path=str(ds.paths[ds.ids[k]]), split=split_of[k],
                  label=ds.classes[int(ds.labels[k])], **extra(ds.ids[k])) for k in sorted(split_of)]
    tasks = limit_tasks(tasks, limit)
    masks = None if pool is None else {
        "kind": POOL_KIND, "manifest_sha256": pool.manifest_sha256, "n_entries": len(pool.entries),
        "found_in": pool.found_in(ds.name, [t.image_id for t in tasks]),
        "reading": "máscara PNG del pool (> 0 = lesión) e imagen del pool, ambas 224 x 224"}
    return DatasetSpec(name=ds.name, classes=tuple(ds.classes), splits=SPLITS, tasks=tasks,
                       provenance={"split": dict(ds.identity)}, masks=masks)


def spec_from_ham(split: dict, paths: dict, positions: dict, *, masks_info: dict, provenance: dict,
                  limit: int | None = None) -> DatasetSpec:
    """HAM10000 from the C1 split dict, the image index and ``image_id -> mask position``."""
    classes = tuple(split["classes"])
    tasks, no_file, no_mask = [], [], []
    for name, ids in (("train", split["train_ids"]), ("val", split["val_ids"])):
        for image_id in ids:
            if image_id not in paths:
                no_file.append(image_id)
            elif image_id not in positions:
                no_mask.append(image_id)
            else:
                tasks.append(Task(image_id, str(paths[image_id]), name, classes[split["id2label"][image_id]],
                                  positions[image_id], mask_state="c1_native"))
    if no_file:
        raise MissingImageError(f"{len(no_file)} images without a file, e.g. {no_file[:3]}")
    if no_mask:
        raise MaskCoverageError(f"{len(no_mask)} images without a mask, e.g. {no_mask[:3]}")
    return DatasetSpec(name=HAM, classes=classes, splits=SPLITS, tasks=limit_tasks(tasks, limit),
                       provenance=provenance, masks=masks_info)


def load_masks(masks: Path, manifest: Path) -> tuple[dict, dict]:
    """``(positions, info)``: student_r3 only, the file verified against the sha256 of its manifest."""
    import e3_io  # noqa: PLC0415
    import v71_io  # noqa: PLC0415

    from dermssl import c1_mask_export as mx  # noqa: PLC0415
    from dermssl import c1_mask_provenance as mp  # noqa: PLC0415

    raw = v71_io.read_json(manifest.expanduser())
    source = mp.MaskSource.from_manifest(raw)
    e3_io.require_student_r3(source)
    mx.verify_artifact(masks.expanduser(), source.masks_sha256)
    info = {"producer": source.producer, "segmentation_run": source.segmentation_run,
            "segmentation_commit": source.segmentation_commit, "masks_sha256": source.masks_sha256,
            "checkpoint_sha256": raw["checkpoint_sha256"], "n_images": len(source.image_ids),
            "mask_shape": list(source.mask_shape), "label_convention": "1 = lesion; 255 (uncertain) = not lesion"}
    return {image_id: k for k, image_id in enumerate(source.image_ids)}, info


def _ham_spec(args) -> DatasetSpec:
    import cluster_frozen_bank as bank  # noqa: PLC0415
    from c1_build_val_index import rebuild_split  # noqa: PLC0415

    from dermssl import c1_split_identity as si  # noqa: PLC0415

    root = args.data_root.expanduser()
    split, _ = rebuild_split(root)
    positions, info = load_masks(args.masks, args.masks_manifest)
    identity = si.split_identity(train_ids=split["train_ids"], val_ids=split["val_ids"])
    return spec_from_ham(split, bank.index_images([(args.image_dir or root).expanduser()]), positions,
                         masks_info=info, provenance={"split": identity}, limit=args.limit)


def _isic_spec(args) -> DatasetSpec:
    loader, csv_name = sp.LOADERS[args.dataset]
    csv = next(args.data_root.expanduser().rglob(csv_name), None)
    if csv is None:
        raise SystemExit(f"ABORT: no {csv_name} under {args.data_root}")
    try:
        ds = loader(csv, (args.image_dir or args.data_root).expanduser())
    except sp.SplitMismatchError as exc:
        raise SystemExit(f"ABORT: {exc}") from exc
    return spec_from_eval_dataset(ds, args.limit, load_pool(args.pool_root))


def load_spec(args) -> DatasetSpec:
    """The dataset named by ``args.dataset``, verified, with its tasks in a fixed order."""
    check_mask_args(args.dataset, masks=args.masks, masks_manifest=args.masks_manifest, pool_root=args.pool_root)
    return _ham_spec(args) if args.dataset == HAM else _isic_spec(args)


def pool_gaps(spec: DatasetSpec) -> dict:
    """Which tasks have no pool PNG: eligible ones are a corrupt pool; the others are read from the source."""
    eligible_missing, from_source, from_pool = [], 0, 0
    for t in spec.tasks:
        if t.mask_state == "eligible" and not (Path(t.pool_image).is_file() and Path(t.pool_mask).is_file()):
            eligible_missing.append(t.image_id)
        elif t.mask_state in ("quarantine", "absent"):
            if t.pool_image is not None and Path(t.pool_image).is_file():
                from_pool += 1
            else:
                from_source += 1
    return {"eligible_missing": eligible_missing, "fallback_from_source": from_source, "fallback_from_pool": from_pool}


def require_intact_pool(spec: DatasetSpec) -> dict:
    """Abort before any work if an ELIGIBLE entry lacks its PNG (a corrupt pool); return the gap counts."""
    gaps = pool_gaps(spec)
    if gaps["eligible_missing"]:
        raise SystemExit(f"ABORT: {len(gaps['eligible_missing'])} entradas elegibles sin PNG en el pool (corrupción "
                         f"del pool), p. ej. {gaps['eligible_missing'][:3]}")
    return gaps
