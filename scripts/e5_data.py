"""Dataset side of ``e5_fitzpatrick.py``: ISIC through its verified loaders, HAM10000 with its masks.

Nothing here decodes an image. The ISIC datasets come from ``dermssl.isic_eval.splits`` (the rebuilt
split is compared with the committed one and the run refuses on a difference); HAM10000 comes from the
C1 split (``c1_build_val_index.rebuild_split``) and its student_r3 masks, verified against their manifest
exactly as E3 does. A HAM image without a mask aborts the run (a silent drop would change the counts);
ISIC has no masks, so asking for them is refused rather than invented.
"""

from __future__ import annotations

import sys
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


class MaskCoverageError(KeyError):
    """A HAM10000 image has no mask in the manifest."""


class MissingImageError(FileNotFoundError):
    """A declared image has no file under the image root."""


def check_mask_args(dataset: str, *, masks, masks_manifest) -> None:
    """HAM10000 needs both mask arguments; the ISIC datasets must not get any (they have no masks)."""
    if dataset == HAM:
        if masks is None or masks_manifest is None:
            raise SystemExit("ABORT: ham10000 necesita --masks y --masks-manifest (el anillo perilesional usa la máscara)")
    elif masks is not None or masks_manifest is not None:
        raise SystemExit(f"ABORT: {dataset} no tiene máscaras; el estimador por anillo se abstiene, no se inventan")


def limit_tasks(tasks: list[Task], n: int | None) -> list[Task]:
    """Smoke subset: ``n`` images per split, round-robin over the classes, original order kept."""
    if n is None:
        return tasks
    keep: set[str] = set()
    for split in SPLITS:
        mine = [t for t in tasks if t.split == split]
        keep.update(plan.limited_ids([t.image_id for t in mine], [t.label for t in mine], n))
    return [t for t in tasks if t.image_id in keep]


def spec_from_eval_dataset(ds: sp.EvalDataset, limit: int | None = None) -> DatasetSpec:
    """An ISIC dataset as loaded and verified by ``dermssl.isic_eval.splits``; no masks."""
    split_of = {int(k): "train" for k in ds.train_idx} | {int(k): "val" for k in ds.val_idx}
    tasks = [Task(image_id=ds.ids[k], path=str(ds.paths[ds.ids[k]]), split=split_of[k],
                  label=ds.classes[int(ds.labels[k])]) for k in sorted(split_of)]
    return DatasetSpec(name=ds.name, classes=tuple(ds.classes), splits=SPLITS, tasks=limit_tasks(tasks, limit),
                       provenance={"split": dict(ds.identity)}, masks=None)


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
                                  positions[image_id]))
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
    return spec_from_eval_dataset(ds, args.limit)


def load_spec(args) -> DatasetSpec:
    """The dataset named by ``args.dataset``, verified, with its tasks in a fixed order."""
    check_mask_args(args.dataset, masks=args.masks, masks_manifest=args.masks_manifest)
    return _ham_spec(args) if args.dataset == HAM else _isic_spec(args)
