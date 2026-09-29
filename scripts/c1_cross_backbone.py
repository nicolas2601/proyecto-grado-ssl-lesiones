"""C1 contrasts between backbones and by crop retention, from stored cells.

CPU only: it reads the probabilities the grid already saved, verifies them
against their manifests, and bootstraps. Nothing is re-extracted, so the
numbers are the same rows the per-backbone table was computed on.

    python scripts/c1_cross_backbone.py --grid-dir $S/grid \
        --cache-dir $T/bank/cache --split-index $S/c1_split_index.json \
        --masks $S/masks/c1_pseudo_masks.npy \
        --masks-manifest $S/masks/c1_pseudo_masks_manifest.json \
        --out $S/prof8/c1_cross_backbone.json --wandb-project dermssl-segmentacion
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import c1_run_grid as grid  # noqa: E402

from dermssl import c1_cross_backbone as cb  # noqa: E402
from dermssl import c1_extraction as ex  # noqa: E402
from dermssl import c1_mask_export as mx  # noqa: E402
from dermssl import c1_mask_provenance as mp  # noqa: E402
from dermssl import c1_split_identity as si  # noqa: E402
from dermssl import paired_bootstrap as pb  # noqa: E402
from dermssl import run_environment  # noqa: E402
from dermssl.feature_cache import _digest  # noqa: E402
from dermssl.pretrain import tracking  # noqa: E402

CONDITIONS = ("none", "bbox_crop", "neutral_bg", "mask_pool")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for flag in ("--grid-dir", "--cache-dir", "--split-index", "--masks", "--masks-manifest", "--out"):
        parser.add_argument(flag, type=Path, required=True)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-group", default="v7-c1")
    return parser.parse_args()


def code_sha256() -> str:
    """This script and its module, hashed: it may run from an uncommitted copy."""
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(cb.__file__)):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_cells(grid_dir: Path, val_ids: list, labels_sha: str) -> tuple[dict, dict, dict]:
    """Probas, gate verdicts and manifests, each verified before use."""
    probas, passes, manifests = {}, {}, {}
    for backbone in (cb.PUBLIC, cb.PROPOSED):
        for condition in CONDITIONS:
            name = f"{backbone}__{condition}"
            stem = grid_dir / f"cell__{name}"
            manifest = json.loads(Path(f"{stem}.manifest.json").read_text(encoding="utf-8"))
            proba = np.load(f"{stem}.proba.npy")
            if manifest.get("proba_sha256") != _digest(proba):
                raise SystemExit(f"ABORT: {name}: la proba no coincide con su manifiesto")
            if manifest.get("labels_sha256") != labels_sha:
                raise SystemExit(f"ABORT: {name}: etiquetas distintas a las del banco")
            if manifest.get("c1_val_ids_sha256") != si.ids_sha256(val_ids):
                raise SystemExit(f"ABORT: {name}: otro orden de validacion")
            probas[name] = proba
            passes[name] = bool(manifest.get("gate", {}).get("gate_passes", False))
            manifests[name] = manifest
    cb.check_pairable(manifests)
    return probas, passes, manifests


def val_retention(masks_path: Path, manifest_path: Path, val_ids: list) -> np.ndarray:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = mp.MaskSource.from_manifest(manifest)
    mx.verify_artifact(masks_path, source.masks_sha256)
    position = {image_id: i for i, image_id in enumerate(source.image_ids)}
    masks = np.load(masks_path, mmap_mode="r")
    return np.array(
        [ex.mask_area_retained_after_crop(np.asarray(masks[position[i]]) == mx.LESION_LABEL)
         for i in val_ids]
    )


def log_tables(module, args, document: dict) -> None:
    run = module.init(project=args.wandb_project, group=args.wandb_group,
                      name="c1-cross-backbone", config={"code_sha256": document["code_sha256"]})
    columns = ["stratum", "arm", "control", "metric", "point_delta", "ci95_low", "ci95_high", "direction"]
    rows = [["all", *[r[c] for c in columns[1:]]] for r in document["results"]]
    for stratum in document["stratified"]:
        rows += [[stratum["stratum"], *[r[c] for c in columns[1:]]] for r in stratum["results"]]
    run.log({"c1_cross_backbone": module.Table(columns=columns, data=rows)})
    run.finish()
    print(f"W&B: {run.url}")


def main() -> int:
    args = parse_args()
    wandb_module = tracking.require_wandb(args.wandb_project)

    index = json.loads(args.split_index.expanduser().read_text(encoding="utf-8"))
    train_ids, val_ids = list(index["train_ids"]), list(index["val_ids"])
    train_labels, val_labels = grid.load_bank_labels(
        args.cache_dir.expanduser(), cb.PUBLIC, expected={"train": train_ids, "val": val_ids}
    )
    labels_sha = _digest(np.concatenate([np.asarray(train_labels), np.asarray(val_labels)]))
    y_true = np.asarray(val_labels)

    probas, passes, _ = load_cells(args.grid_dir.expanduser(), val_ids, labels_sha)
    records, skipped = cb.contrast_pairs(probas, passes, y_true, cb.ALL_CROSS_PAIRS)
    retained = val_retention(args.masks.expanduser(), args.masks_manifest.expanduser(), val_ids)

    document = pb.bootstrap_document(
        records, n_resamples=1000, rng_seed=20260923, probe="logreg", fraction=1.0,
        seed_used=0, n_val=len(y_true), val_order_sha256=index["c1_val_ids_sha256"],
    )
    document.update(
        not_contrasted=skipped,
        code_sha256=code_sha256(),
        retained_threshold=cb.RETAINED_THRESHOLD,
        stratified=cb.stratified_summary(probas, passes, y_true, retained),
        environment=run_environment.snapshot(args),
    )
    out = args.out.expanduser()
    out.write_text(json.dumps(document, indent=1), encoding="utf-8")
    print(f"escrito {out}: {len(records)} contrastes, estratos "
          f"{[(s['stratum'], s['n']) for s in document['stratified']]}")
    if wandb_module is not None:
        log_tables(wandb_module, args, document)
    return 0


if __name__ == "__main__":
    sys.exit(main())
