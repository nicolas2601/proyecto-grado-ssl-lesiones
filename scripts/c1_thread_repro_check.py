r"""Diagnostic: which BLAS thread count reproduces the published C1 probas bit for bit?

C1 (``c1_run_grid.py``) fitted its logreg probe in-process with the library's
default thread count, never pinned. For each requested published cell this
script re-extracts the cell's train AND val features with the grid's own
``extract_cell`` (the grid stored only val features, so the train side has to
be recomputed on GPU), refuses the cell unless the re-extracted val features
hash to the cell's ``features_sha256`` and its split, labels, masks and
published macro-F1 match, then refits the probe through the grid's own
``evaluate_cell`` under every thread setting and compares with the published
``.proba.npy``. Nothing published is written to or refitted in place.

Pod (GPU for extraction, then CPU fits; run when the CPU is free, NO thread
env vars, so ``default`` is the library default C1 got)::

    P=$HOME/work/proyectonp; source $P/.venv/bin/activate
    export TORCH_HOME=$HOME/work/.torch; unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
    cd $P/repo_t4 && python scripts/c1_thread_repro_check.py \
      --cache-dir $P/salidas/tone/bank/cache --image-dir $P/eval_ham10000_images \
      --split-index $P/salidas/c1/c1_split_index.json \
      --grid student_r3=$P/salidas/c1_v7/grid --grid a2=$P/salidas/c1_v7/grid_a2 \
      --masks student_r3=$P/salidas/c1/masks --masks a2=$P/salidas/c1/masks_a2 \
      --cell student_r3:dinov2_public_vitb14:none --cell student_r3:v7_astar_s42:none \
      --cell a2:v7_bstar_s42:mask_pool \
      --scratch-dir $P/salidas/diag_hilos_c1/features --out-dir $P/salidas/diag_hilos_c1

Then copy ``c1_thread_repro.json``, its ``.environment.json`` and ``README.md``
to ``resultados/clasificacion/c1_v7/diagnostico_hilos/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import c1_run_grid as grid  # noqa: E402

from dermssl import c1_split_identity as si  # noqa: E402
from dermssl import c1_thread_repro as ctr  # noqa: E402
from dermssl import run_environment  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402
from dermssl.feature_cache import _digest  # noqa: E402
from dermssl.probe_convergence import check_out_dir  # noqa: E402

PUBLISHED_ROOT = REPO / "resultados/clasificacion/c1_v7"
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
DEFAULT_THREADS = ("default", "recorded", "1", "4", "8", "12", "24", "48")


def _pairs(values) -> dict:
    return {k: Path(v) for k, v in (item.split("=", 1) for item in values or [])}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for flag in ("--cache-dir", "--image-dir", "--split-index", "--out-dir", "--scratch-dir"):
        p.add_argument(flag, type=Path, required=True)
    p.add_argument("--grid", action="append", required=True, help="NAME=DIR of a published C1 grid")
    p.add_argument("--masks", action="append", required=True, help="NAME=DIR with c1_pseudo_masks.{npy,_manifest.json}")
    p.add_argument("--published", action="append", default=[], help="NAME=c1_cells.json (default: committed c1_v7/NAME)")
    p.add_argument("--cell", action="append", required=True, help="NAME:BACKBONE:CONDITION")
    p.add_argument("--threads", nargs="+", default=list(DEFAULT_THREADS))
    p.add_argument("--labels-backbone", default="dinov2_public_vitb14", help="bank cache the grid read labels from")
    p.add_argument("--device", default=None)
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--allow-thread-env", action="store_true")
    return p.parse_args(argv)


def load_labels(args, index):
    return grid.load_bank_labels(args.cache_dir.expanduser(), args.labels_backbone,
                                 expected={"train": index["train_ids"], "val": index["val_ids"]})


def masks_sha256(masks_dir: Path) -> str:
    from dermssl import c1_mask_export as mx  # noqa: PLC0415

    manifest = json.loads((masks_dir / "c1_pseudo_masks_manifest.json").read_text(encoding="utf-8"))
    mx.verify_artifact(masks_dir / "c1_pseudo_masks.npy", manifest["masks_sha256"])
    return manifest["masks_sha256"]


def extract_features(ctx, cell, manifest):
    """Train and val features exactly as ``c1_run_grid.main`` extracted them (GPU, no cudnn flags touched)."""
    import torch  # noqa: PLC0415

    import cluster_frozen_bank as bank  # noqa: PLC0415
    from dermssl import c1_cell_gates as cg  # noqa: PLC0415
    from dermssl import c1_extraction as ex  # noqa: PLC0415
    from dermssl import c1_mask_export as mx  # noqa: PLC0415
    from dermssl import c1_mask_provenance as mp  # noqa: PLC0415

    args, index = ctx["args"], ctx["index"]
    masks_dir = ctx["masks"][cell["grid"]]
    source = mp.MaskSource.from_manifest(json.loads(
        (masks_dir / "c1_pseudo_masks_manifest.json").read_text(encoding="utf-8")))
    position = {image_id: i for i, image_id in enumerate(source.image_ids)}
    all_masks = np.load(masks_dir / "c1_pseudo_masks.npy", mmap_mode="r")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    arch, weights = manifest["backbone_spec"].split(":", 1)
    built = bank.build_frozen_backbone(arch, weights, ex.RESOLUTION, device)
    model = built["model"] if isinstance(built, dict) else built
    out = []
    for split in ("train_ids", "val_ids"):
        ids = index[split]
        masks = np.stack([np.asarray(all_masks[position[i]]) == mx.LESION_LABEL for i in ids])
        tally = cg.CellTally(backbone=cell["backbone"], condition=cell["condition"])
        out.append(grid.extract_cell(model, args.image_dir.expanduser(), ids, masks, source,
                                     cell["condition"], device, tally))
    return out[0], out[1]


def _features(ctx, cell, manifest):
    scratch = ctx["args"].scratch_dir.expanduser()
    stem = scratch / f"{cell['grid']}__{cell['backbone']}__{cell['condition']}"
    paths = (Path(f"{stem}.train.npy"), Path(f"{stem}.val.npy"))
    if all(p.is_file() for p in paths):
        return np.load(paths[0]), np.load(paths[1]), "scratch"
    train, val = (np.asarray(a, dtype=np.float32) for a in extract_features(ctx, cell, manifest))
    scratch.mkdir(parents=True, exist_ok=True)
    for path, array in zip(paths, (train, val)):
        np.save(path, array)
    return train, val, "gpu_extraction"


def _pre_gate(ctx, cell, manifest, proba, published_f1) -> list[str]:
    """Everything checkable before spending GPU: is this pod cell the published one, on this split?"""
    index, problems = ctx["index"], []
    expected = {"proba_sha256": _digest(proba), "probe": "logreg", "labels_sha256": ctx["labels_sha"],
                "c1_train_ids_sha256": si.ids_sha256(index["train_ids"]),
                "c1_val_ids_sha256": si.ids_sha256(index["val_ids"]),
                "masks_sha256": masks_sha256(ctx["masks"][cell["grid"]]), "n_rows": len(index["val_ids"])}
    problems += [f"{k}: manifest {manifest.get(k)!r} != {v!r}" for k, v in expected.items() if manifest.get(k) != v]
    if manifest.get("metrics", {}).get("macro_f1") != published_f1:
        problems.append(f"macro_f1: manifest {manifest.get('metrics', {}).get('macro_f1')!r} != published "
                        f"{published_f1!r}")
    return problems


def _refits(ctx, train, val, n_classes, published, published_f1) -> tuple[list, bool | None]:
    y_tr, y_va = ctx["labels"]
    fit = lambda cap: ctr.fit_with_threads(grid.evaluate_cell, train, y_tr, val, y_va, "logreg",  # noqa: E731
                                           n_classes, threads=cap)
    rows, first_explicit = [], None
    for label, cap in ctx["settings"]:
        start = time.time()
        (proba, summary), pools = fit(cap)
        rows.append({"setting": label, "threads": cap, "resolved_threads": [p["num_threads"] for p in pools],
                     "pools": pools, "seconds": round(time.time() - start, 2),
                     **ctr.compare_proba(proba, published, refit_macro_f1=float(summary["macro_f1"]),
                                         published_macro_f1=published_f1)})
        if cap is not None and first_explicit is None:
            first_explicit = (cap, proba)
        print(f"    {label:>8}: bit_equal={rows[-1]['bit_equal']} max|d|={rows[-1]['max_abs_delta']:.2e} "
              f"flips={rows[-1]['argmax_flips']} ({rows[-1]['seconds']}s)", flush=True)
    if first_explicit is None:
        return rows, None
    # Control: the same explicit count twice must be bit-equal, or no comparison above means anything.
    (again, _), _ = fit(first_explicit[0])
    return rows, bool(np.array_equal(again, first_explicit[1]))


def run_cell(ctx, spec: str) -> dict:
    name, backbone, condition = spec.split(":")
    cell = {"name": spec, "grid": name, "backbone": backbone, "condition": condition}
    stem = ctx["grids"][name] / f"cell__{backbone.replace('/', '-')}__{condition}"
    manifest = json.loads(Path(f"{stem}.manifest.json").read_text(encoding="utf-8"))
    published = np.load(f"{stem}.proba.npy")
    published_f1 = ctx["published"][name][f"{backbone}__{condition}"]["metrics"]["macro_f1"]
    cell.update(cell_code_revision=manifest.get("code_revision"), backbone_spec=manifest.get("backbone_spec"))
    print(f"\n=== {spec} ===", flush=True)
    problems = _pre_gate(ctx, cell, manifest, published, published_f1)
    source = None
    if not problems:
        train, val, source = _features(ctx, cell, manifest)
        if _digest(val) != manifest["features_sha256"]:
            problems.append("val features re-extracted do not hash to the cell's features_sha256")
    cell.update(features_from=source, gate={"passed": not problems, "problems": problems})
    if problems:
        print("  GATE: " + "; ".join(problems), flush=True)
        return {**cell, "fits": [], "repeat_bit_equal": None}
    fits, repeat = _refits(ctx, train, val, int(manifest["n_classes"]), published, published_f1)
    return {**cell, "fits": fits, "repeat_bit_equal": repeat}


def recorded_threads(grids: dict) -> dict:
    """Thread count the grids' environment sidecars recorded, if any grid has one."""
    found = {}
    for name, gdir in grids.items():
        env = run_environment.read_sidecar(gdir / "c1_cells.json")
        if env is None:
            continue
        values = [env.get("env", {}).get(k) for k in THREAD_ENV] + \
                 [p.get("num_threads") for p in env.get("threadpools") or []]
        found[name] = next((int(v) for v in values if v not in (None, "")), None)
    unique = set(found.values()) - {None}
    if not found:
        return {"value": None, "source": "no environment sidecar next to any grid (C1-v7 predates sidecars)"}
    if len(unique) != 1:
        return {"value": None, "source": f"sidecars disagree or record nothing: {found}"}
    return {"value": unique.pop(), "source": f"environment sidecar of {sorted(found)}"}


def _revision(args) -> str:
    try:
        return rp.clean_revision(REPO)
    except rp.DirtyTreeError as exc:
        if args.allow_dirty:
            return "dirty"
        raise SystemExit(f"ABORT: {exc}") from exc


def _context(args) -> dict:
    grids, masks = {k: v.expanduser() for k, v in _pairs(args.grid).items()}, _pairs(args.masks)
    masks = {k: v.expanduser() for k, v in masks.items()}
    published = {name: PUBLISHED_ROOT / name / "c1_cells.json" for name in grids} | _pairs(args.published)
    protected = [*grids.values(), *masks.values(), args.cache_dir, REPO / "resultados/clasificacion/c1",
                 *(p.expanduser().parent for p in published.values())]
    for target in (args.out_dir, args.scratch_dir):
        check_out_dir(target, protected=protected)
    set_env = [k for k in THREAD_ENV if os.environ.get(k)]
    if set_env and not args.allow_thread_env:
        raise SystemExit(f"ABORT: {set_env} set; 'default' must be the library default C1 ran with. unset them")
    index = json.loads(args.split_index.expanduser().read_text(encoding="utf-8"))
    labels = load_labels(args, index)
    recorded = recorded_threads(grids)
    return {"args": args, "index": index, "grids": grids, "masks": masks, "labels": labels,
            "labels_sha": _digest(np.concatenate([np.asarray(labels[0]), np.asarray(labels[1])])),
            "published": {k: json.loads(v.expanduser().read_text(encoding="utf-8")) for k, v in published.items()},
            "recorded": recorded, "settings": ctr.parse_settings(args.threads, recorded=recorded["value"])}


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out_dir = args.out_dir.expanduser()
    revision, ctx = _revision(args), _context(args)
    cells = [run_cell(ctx, spec) for spec in args.cell]
    doc = {"code_revision": revision, "settings": [{"label": lb, "threads": n} for lb, n in ctx["settings"]],
           "recorded_threads": ctx["recorded"], "cells": cells, "summary": ctr.summarize(cells)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / ctr.OUT_JSON
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    run_environment.write_sidecar(out, args)
    (args.out_dir / "README.md").write_text(ctr.render_readme(doc), encoding="utf-8")
    print(f"\n-> {out}: {doc['summary']['verdict']} (reproducen: {doc['summary']['reproducing_settings']})")
    if not doc["summary"]["all_gates_passed"]:
        print("!!! GATE: alguna celda no es la publicada; ver gate en el JSON", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
