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

Where the pieces live
---------------------
This script is the entry point: arguments, the clean-revision check, torch
and the backbone builder, W&B and the two result JSONs. The inputs and their
cross-checks are in ``dermssl.c1_grid_inputs``, one pending cell from backbone
to saved result in ``dermssl.c1_grid_extraction``, and fit/gate/save, resume
and the contrasts in ``dermssl.c1_grid_cells``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from dermssl import c1_cell_store as cs  # noqa: E402
from dermssl import c1_grid_cells as gc  # noqa: E402
from dermssl import c1_grid_extraction as gx  # noqa: E402
from dermssl import c1_grid_inputs as gi  # noqa: E402
from dermssl import c1_split_identity as si  # noqa: E402
from dermssl import cli_args  # noqa: E402
from dermssl import probe_threads  # noqa: E402
from dermssl import run_environment  # noqa: E402
from dermssl.pretrain import tracking  # noqa: E402


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
    parser.add_argument("--conditions", nargs="+", default=list(gc.CONDITION_ORDER))
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


def require_clean_revision() -> str:
    revision = code_revision()
    if revision == "unknown" or revision.endswith("-dirty"):
        # Review P1-4: "unknown" matches every other "unknown", and HEAD of a
        # dirty tree is not the code that ran. Either way the skip would reuse
        # cells from different code.
        raise SystemExit(
            f"ABORT: code_revision={revision!r}. Commitea antes de correr (regla 4): sin "
            "una revision limpia, el skip reanudable reusaria celdas de otro codigo"
        )
    return revision


def run_pending(args, store, inputs, pending, provenance_for, *, device, build_model,
                tracker_probe) -> dict:
    """Run the cells not on disk yet, in plan order; one W&B run per cell."""
    results = {}
    for backbone, condition in pending:
        provenance = provenance_for(backbone)
        done = gx.run_pending_cell(
            store, inputs, backbone, condition, build_model=build_model,
            image_dir=args.image_dir.expanduser(), device=device, probe=args.probe,
            provenance=provenance,
        )
        if tracker_probe is not None:
            log_cell_to_wandb(
                tracker_probe, args, backbone, condition, provenance,
                done["metrics"], done["gate"], done["passes"],
            )
        results[(backbone, condition)] = done
    return results


def main() -> int:
    args = parse_args()

    tracker_probe = tracking.require_wandb(args.wandb_project)
    if args.wandb_project and tracker_probe is None:
        raise SystemExit("ABORT: se pidio W&B y no esta disponible")
    revision = require_clean_revision()

    import torch  # noqa: PLC0415

    import cluster_frozen_bank as bank  # noqa: PLC0415

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    inputs = gi.load_grid_inputs(
        cache_dir=args.cache_dir, masks=args.masks, masks_manifest=args.masks_manifest,
        split_index=args.split_index, backbones=args.backbones,
        parse_spec=bank.parse_backbone_spec, limit=args.limit,
    )
    provenance_for = gi.provenance_factory(inputs, revision=revision, probe=args.probe)

    store = cs.CellStore(args.out_dir.expanduser())
    plan = store.plan(
        backbones=inputs.backbone_names(),
        conditions=args.conditions,
        provenance=provenance_for,
        priority_backbones=gc.PRIORITY_BACKBONES,
    )
    print(f"celdas ya hechas: {len(plan['done'])} | pendientes: {len(plan['pending'])}")

    results = gc.load_done_cells(store, plan["done"], provenance_for)
    results.update(run_pending(
        args, store, inputs, plan["pending"], provenance_for, device=device,
        build_model=bank.build_frozen_backbone, tracker_probe=tracker_probe,
    ))
    gc.print_cell_table(results)
    write_outputs(args.out_dir.expanduser(), results, inputs.val_labels,
                  si.ids_sha256(inputs.val_ids), args=args)
    return 0


def write_outputs(out_dir: Path, results: dict, val_labels, val_ids_sha256: str, args=None) -> None:
    """Both JSONs get an environment sidecar; their own bytes stay machine-independent."""
    contrasts = gc.build_contrasts(results, val_labels, {"c1_val_ids_sha256": val_ids_sha256})
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
        run_environment.write_sidecar(out_dir / name, args,
                                      loky_inner_threads=probe_threads.PROBE_BLAS_THREADS)
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


if __name__ == "__main__":
    sys.exit(main())
