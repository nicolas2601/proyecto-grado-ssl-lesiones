"""Run the C1 grid: 6 backbones x 4 mask conditions, resumable.

What this does, per cell
------------------------
1. Extract features for (backbone, condition) over train and val, through the
   SAME geometry the bank uses, counting the degenerate cases per image.
2. Gate the cell on those counts (CLS fallback in mask_pool, whole-image grey
   in neutral_bg). A failing cell is recorded and does NOT enter the table.
3. Fit the probe, compute all four metrics, and log one W&B run for the cell.
4. Write the cell atomically so a relaunch skips it -- verified by sha, not by
   the filename.

Then, across cells: paired contrasts of every condition against (0), using
the same resampling indices for both sides.

Order
-----
The backbones of the professor's question run FIRST. A crash at cell 20 still
leaves that answer on disk, which is the difference between "we have the
answer to what was asked" and "we have nothing".

Usage
-----
    python scripts/c1_run_grid.py \\
        --cache-dir ~/work/proyectonp/salidas/tone/bank/cache \\
        --image-dir ~/work/proyectonp/eval_ham10000_images \\
        --masks ~/work/proyectonp/salidas/c1/masks/c1_pseudo_masks.npy \\
        --masks-manifest ~/work/proyectonp/salidas/c1/masks/c1_pseudo_masks_manifest.json \\
        --split-index ~/work/proyectonp/salidas/c1/c1_split_index.json \\
        --out-dir ~/work/proyectonp/salidas/c1/grid \\
        --backbones dinov2_public_vitb14=dinov2_vitb14:public \\
        --wandb-project dermssl-segmentacion
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from dermssl import c1_cell_gates as cg  # noqa: E402
from dermssl import c1_cell_store as cs  # noqa: E402
from dermssl import c1_extraction as ex  # noqa: E402
from dermssl import c1_mask_export as mx  # noqa: E402
from dermssl import c1_mask_provenance as mp  # noqa: E402
from dermssl import c1_split_identity as si  # noqa: E402
from dermssl import cli_args  # noqa: E402
from dermssl import metrics as _metrics  # noqa: E402
from dermssl import run_environment  # noqa: E402
from dermssl import paired_bootstrap as pb  # noqa: E402
from dermssl import probes  # noqa: E402
from dermssl.feature_cache import FeatureCache, _digest, fingerprint_ids  # noqa: E402
from dermssl.pretrain import tracking  # noqa: E402

#: The question the professor actually asked, run before the other sixteen.
PRIORITY_BACKBONES = ("dinov2_public_vitb14", "ssl_proposed_vitb14")

#: (0) first: every contrast is measured against it, so a grid that dies early
#: still has the baseline every other cell needs.
CONDITION_ORDER = ("none", "bbox_crop", "neutral_bg", "mask_pool")

METRICS = ("macro_f1", "balanced_accuracy", "auc", "ece")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--masks-manifest", type=Path, required=True)
    parser.add_argument("--split-index", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--backbones",
        nargs="+",
        required=True,
        help="NAME=ARCH:SOURCE, igual que cluster_frozen_bank",
    )
    parser.add_argument("--conditions", nargs="+", default=list(CONDITION_ORDER))
    parser.add_argument("--probe", default="logreg")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=cli_args.positive_int, default=None)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-group", default="v7-c1")
    return parser.parse_args()


def code_revision() -> str:
    import subprocess  # noqa: PLC0415

    try:
        head = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{head}-dirty" if dirty else head
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _stride(n: int, limit: int | None) -> np.ndarray:
    """Indices of an evenly spaced subsample of ``limit`` out of ``n``."""
    if limit is None or limit >= n:
        return np.arange(n)
    return np.linspace(0, n - 1, limit).round().astype(int)


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
        hits = sorted(cache_dir.glob(f"v1__{backbone}__{split}__r*__eval__*.features.npy"))
        if len(hits) != 1:
            raise SystemExit(
                f"ABORT: esperaba 1 cache de {backbone}/{split}, encontre {len(hits)}"
            )
        key = hits[0].name[: -len(".features.npy")]
        if expected is not None:
            want = fingerprint_ids(expected[split])
            have = key.rsplit("__", 1)[-1]
            if have != want:
                raise SystemExit(
                    f"ABORT: el cache de {split} tiene fingerprint {have} y los ids del split "
                    f"dan {want}. Las etiquetas quedarian contra otras imagenes"
                )
        entry = cache.load(key)
        if entry is None:
            raise SystemExit(f"ABORT: no se pudo cargar el cache {key}")
        out.append(np.asarray(entry.labels))
    return out[0], out[1]


def extract_cell(model, image_dir, image_ids, masks, mask_source, condition, device, tally):
    """Features for one (backbone, condition) over the given ids.

    The tally is filled here, where the per-image facts exist. Recomputing the
    degenerate counts afterwards from the features would be measuring a
    different thing: by then the fallback has already happened and looks like
    an ordinary row.
    """
    from PIL import Image  # noqa: PLC0415

    from dermssl import torch_c1_extraction as tex  # noqa: PLC0415

    rows = []
    for index, image_id in enumerate(image_ids):
        path = image_dir / f"{image_id}.jpg"
        if not path.exists():
            raise SystemExit(f"ABORT: falta la imagen {image_id} en {image_dir}")

        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        tracked = mp.TrackedMask(
            array=masks[index].astype(bool), source=mask_source, image_id=image_id
        )
        prepared = ex.build_eval_input(image, tracked, condition)
        tensor = tex.to_normalized_tensor(prepared.image).to(device)
        feature, used_fallback = tex.extract_feature(
            model, tensor, condition, prepared.mask
        )
        tally.record(
            used_cls_fallback=bool(used_fallback),
            empty_after_geometry=prepared.empty_after_geometry,
            used_bbox_fallback=bool(prepared.used_bbox_fallback),
        )
        rows.append(feature)
    return np.stack(rows)


def non_finite_reasons(train_features, val_features, proba, metrics) -> list[str]:
    reasons = []
    for name, array in (("features train", train_features), ("features val", val_features),
                        ("proba", proba)):
        bad = int((~np.isfinite(np.asarray(array, dtype=np.float64))).sum())
        if bad:
            reasons.append(f"{bad} valores no finitos en {name}")
    bad_metrics = [k for k, v in metrics.items() if not np.isfinite(float(v))]
    if bad_metrics:
        reasons.append(f"metricas no finitas: {bad_metrics}")
    return reasons


def evaluate_cell(train_features, train_labels, val_features, val_labels, probe, n_classes):
    proba = probes.run_probe(probe, train_features, train_labels, val_features, n_classes)
    summary = _metrics.summarize(val_labels, np.asarray(proba), n_classes)
    return np.asarray(proba), summary


def fit_and_save_cell(store, backbone, condition, *, train_features, train_labels, val_features,
                      val_labels, train_tally, val_tally, probe, n_classes, provenance) -> dict:
    """Fit the probe, gate the cell and save it; the train array hashed is the one fitted on."""
    # Review (both reviewers): the TRAIN side is gated too. The probe is fitted on
    # train, so a degraded train set contaminates every val number.
    val_passes, val_reasons = cg.evaluate_cell_gates(val_tally)
    train_passes, train_reasons = cg.evaluate_cell_gates(train_tally)
    reasons = [f"val: {r}" for r in val_reasons] + [f"train: {r}" for r in train_reasons]
    cell_summary = cg.summarise_cell(val_tally)

    proba, cell_metrics = evaluate_cell(
        train_features, train_labels, val_features, val_labels, probe, n_classes
    )
    # Review (opencode): a NaN in features, proba or a metric compares
    # False against every threshold and serialises into the JSON as if it
    # were a number. Non-finite anything fails the cell.
    finite_reasons = non_finite_reasons(train_features, val_features, proba, cell_metrics)
    reasons += finite_reasons
    passes = val_passes and train_passes and not finite_reasons
    cell_summary = {**cell_summary, "gate_passes": passes, "gate_reasons": reasons}
    for reason in reasons:
        print(f"  GATE: {reason}")
    print(f"  macro_f1 {cell_metrics['macro_f1']:.4f} | "
          f"bal_acc {cell_metrics['balanced_accuracy']:.4f} | "
          f"auc {cell_metrics['auc']:.4f} | ece {cell_metrics['ece']:.4f}")

    store.save(
        backbone, condition, val_features,
        proba=proba,
        train_features=train_features,
        provenance=provenance,
        extra={
            "gate": cell_summary,
            "train_gate": cg.summarise_cell(train_tally),
            "metrics": {k: float(v) for k, v in cell_metrics.items()},
            "n_classes": n_classes,
        },
    )
    return {"proba": proba, "metrics": cell_metrics, "gate": cell_summary, "passes": passes}


def main() -> int:
    args = parse_args()

    tracker_probe = tracking.require_wandb(args.wandb_project)
    if args.wandb_project and tracker_probe is None:
        raise SystemExit("ABORT: se pidio W&B y no esta disponible")

    revision = code_revision()
    if revision == "unknown" or revision.endswith("-dirty"):
        # Review P1-4: "unknown" matches every other "unknown", and HEAD of a
        # dirty tree is not the code that ran. Either way the skip would reuse
        # cells from different code.
        raise SystemExit(
            f"ABORT: code_revision={revision!r}. Commitea antes de correr (regla 4): sin "
            "una revision limpia, el skip reanudable reusaria celdas de otro codigo"
        )

    import torch  # noqa: PLC0415

    import cluster_frozen_bank as bank  # noqa: PLC0415

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    manifest = json.loads(args.masks_manifest.expanduser().read_text(encoding="utf-8"))
    mask_source = mp.MaskSource.from_manifest(manifest)
    mx.verify_artifact(args.masks.expanduser(), mask_source.masks_sha256)

    index = json.loads(args.split_index.expanduser().read_text(encoding="utf-8"))
    full_train_ids = list(index["train_ids"])
    full_val_ids = list(index["val_ids"])
    specs = [bank.parse_backbone_spec(raw) for raw in args.backbones]

    # Review P1-1: labels come from the bank cache, features are extracted in
    # the order of the split index. Nothing guaranteed the two orders agree,
    # so it is checked HERE, on the full lists, before anything is fitted.
    train_labels, val_labels = load_bank_labels(
        args.cache_dir.expanduser(), specs[0][0],
        expected={"train": full_train_ids, "val": full_val_ids},
    )

    # --limit takes an evenly spaced subsample, not a prefix: the split lists
    # are grouped, and the first 40 train ids were measured on the pod to hold
    # a single class -- a probe cannot even be fitted on that. Labels are
    # sliced with the SAME stride so they stay aligned to their ids.
    train_pick = _stride(len(full_train_ids), args.limit)
    val_pick = _stride(len(full_val_ids), args.limit)
    train_ids = [full_train_ids[i] for i in train_pick]
    val_ids = [full_val_ids[i] for i in val_pick]
    train_labels = np.asarray(train_labels)[train_pick]
    val_labels = np.asarray(val_labels)[val_pick]
    n_classes = int(max(train_labels.max(), val_labels.max())) + 1
    print(f"etiquetas del banco: {len(train_labels)} train / {len(val_labels)} val, "
          f"{n_classes} clases (orden verificado por fingerprint)")

    labels_sha = _digest(np.concatenate([np.asarray(train_labels), np.asarray(val_labels)]))
    spec_by_name = {name: f"{arch}:{source}" for name, arch, source in specs}

    def provenance_for(backbone: str) -> dict:
        # Recomputed from the id lists actually used, not copied from fields
        # of the JSON that could disagree with those lists (review P1-1).
        return {
            "code_revision": revision,
            "masks_sha256": mask_source.masks_sha256,
            "c1_train_ids_sha256": si.ids_sha256(train_ids),
            "c1_val_ids_sha256": si.ids_sha256(val_ids),
            "probe": args.probe,
            "backbone_spec": spec_by_name[backbone],
            "labels_sha256": labels_sha,
        }

    position = {image_id: i for i, image_id in enumerate(mask_source.image_ids)}
    missing = [i for i in train_ids + val_ids if i not in position]
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} imagenes del split no estan en el artefacto")

    all_masks = np.load(args.masks.expanduser(), mmap_mode="r")
    train_masks = np.stack([np.asarray(all_masks[position[i]]) == mx.LESION_LABEL for i in train_ids])
    val_masks = np.stack([np.asarray(all_masks[position[i]]) == mx.LESION_LABEL for i in val_ids])

    store = cs.CellStore(args.out_dir.expanduser())
    plan = store.plan(
        backbones=[name for name, _, _ in specs],
        conditions=args.conditions,
        provenance=provenance_for,
        priority_backbones=PRIORITY_BACKBONES,
    )
    print(f"celdas ya hechas: {len(plan['done'])} | pendientes: {len(plan['pending'])}")

    results: dict[tuple[str, str], dict] = {}
    for backbone, condition in plan["done"]:
        cell = store.load(backbone, condition, provenance=provenance_for(backbone))
        gate = cell.manifest.get("gate", {})
        results[(backbone, condition)] = {
            "proba": cell.proba,
            "gate": gate,
            "metrics": cell.manifest.get("metrics", {}),
            "passes": bool(gate.get("gate_passes", False)),
        }
        print(f"  REUSADA {backbone} / {condition}")

    for backbone, condition in plan["pending"]:
        arch, source = next((a, src) for n, a, src in specs if n == backbone)
        print(f"\n=== {backbone} / {condition} ===", flush=True)
        built = bank.build_frozen_backbone(arch, source, ex.RESOLUTION, device)
        model = built["model"] if isinstance(built, dict) else built

        train_tally = cg.CellTally(backbone=backbone, condition=condition)
        val_tally = cg.CellTally(backbone=backbone, condition=condition)
        train_features = extract_cell(
            model, args.image_dir.expanduser(), train_ids, train_masks,
            mask_source, condition, device, train_tally,
        )
        val_features = extract_cell(
            model, args.image_dir.expanduser(), val_ids, val_masks,
            mask_source, condition, device, val_tally,
        )

        done = fit_and_save_cell(
            store, backbone, condition, train_features=train_features, train_labels=train_labels,
            val_features=val_features, val_labels=val_labels, train_tally=train_tally,
            val_tally=val_tally, probe=args.probe, n_classes=n_classes,
            provenance=provenance_for(backbone),
        )
        if tracker_probe is not None:
            log_cell_to_wandb(
                tracker_probe, args, backbone, condition, provenance_for(backbone),
                done["metrics"], done["gate"], done["passes"],
            )
        results[(backbone, condition)] = done

    print("\n=== celdas ===")
    for (backbone, condition), payload in sorted(results.items()):
        print(f"  {backbone} / {condition}: {'PASA' if payload['passes'] else 'FRENA'}")

    write_outputs(args.out_dir.expanduser(), results, val_labels, si.ids_sha256(val_ids), args=args)
    return 0


def write_outputs(out_dir: Path, results: dict, val_labels, val_ids_sha256: str, args=None) -> None:
    """Both JSONs get an environment sidecar; their own bytes stay machine-independent."""
    contrasts = build_contrasts(results, val_labels, {"c1_val_ids_sha256": val_ids_sha256})
    (out_dir / "c1_bootstrap.json").write_text(
        json.dumps(contrasts, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    cells = {
        f"{backbone}__{condition}": {
            "backbone": backbone,
            "condition": condition,
            "gate_passes": payload["passes"],
            "metrics": {k: float(v) for k, v in payload["metrics"].items()},
            "gate": payload["gate"],
        }
        for (backbone, condition), payload in sorted(results.items())
    }
    (out_dir / "c1_cells.json").write_text(
        json.dumps(cells, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    for name in ("c1_bootstrap.json", "c1_cells.json"):
        run_environment.write_sidecar(out_dir / name, args)
    print(f"\ncontrastes: {len(contrasts['results'])} filas, celdas: {len(cells)} -> {out_dir}")


def log_cell_to_wandb(module, args, backbone, condition, provenance, metrics, gate, passes):
    """One W&B run per cell, logged whether the gate passed or not.

    A dashboard holding only the cells that cleared their gate is evidence for
    one side: the ones that degraded are exactly the ones someone will need to
    look up later.
    """
    tracker = tracking.WandbTracker(
        module,
        project=args.wandb_project,
        run=f"c1-{backbone}-{condition}",
        group=args.wandb_group,
        strict=True,
    )
    tracker.start(
        {
            "experiment": "c1",
            "backbone": backbone,
            "condition": condition,
            "probe": args.probe,
            **provenance,
        }
    )
    for key, value in metrics.items():
        tracker.log_metric(key, float(value))
    tracker.log_metric("gate_passes", 1.0 if passes else 0.0)
    if gate.get("fraction_cls_fallback") is not None:
        tracker.log_metric("fraction_cls_fallback", float(gate["fraction_cls_fallback"]))
    if gate.get("fraction_empty_after_geometry") is not None:
        tracker.log_metric(
            "fraction_empty_after_geometry", float(gate["fraction_empty_after_geometry"])
        )
    tracker.end()
    if tracker.run_url:
        print(f"  W&B: {tracker.run_url}")


def build_contrasts(results, val_labels, index) -> dict:
    """Every condition against (0), per backbone, paired.

    Only cells that PASSED their gate enter a contrast. A degraded cell still
    produces a number, and contrasting it would compare the baseline against
    something that is partly the baseline -- a tie that means nothing and
    reads like a finding.
    """
    records = []
    skipped = []
    for (backbone, condition), payload in sorted(results.items()):
        if condition == "none":
            continue
        baseline = results.get((backbone, "none"))
        if baseline is None:
            skipped.append(f"{backbone}/{condition}: no hay celda (0) con la que contrastar")
            continue
        if not payload["passes"] or not baseline["passes"]:
            skipped.append(f"{backbone}/{condition}: una de las dos celdas no paso su gate")
            continue

        for metric in METRICS:
            records.append(
                pb.contrast_record(
                    arm=f"{backbone}__{condition}",
                    control=f"{backbone}__none",
                    metric=metric,
                    probe="logreg",
                    fraction=1.0,
                    seed=0,
                    y_true=np.asarray(val_labels),
                    arm_proba=payload["proba"],
                    control_proba=baseline["proba"],
                )
            )

    document = pb.bootstrap_document(
        records,
        n_resamples=1000,
        rng_seed=20260923,
        probe="logreg",
        fraction=1.0,
        seed_used=0,
        n_val=int(len(val_labels)),
        val_order_sha256=index["c1_val_ids_sha256"],
    )
    # Recorded, not dropped: a contrast that could not be computed is a fact
    # about the grid, and an absent row would read as one that was never asked.
    document["not_contrasted"] = skipped
    return document


if __name__ == "__main__":
    sys.exit(main())
