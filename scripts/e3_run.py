r"""E3 (exploratory): segmentation variants F1-F5 on top of the frozen B71 (docs/33 section 3).

Three stages, each resumable (one JSON per cell, skipped when it exists with the same
key sha256; 2 retries; failures listed at the end; the rest goes on):

1. ``extract``: for each split, render the conditions (full view, crops, shifted crops,
   perilesional ring; the mask is ``student_r3``, never the reference one) and extract the
   B71 feature (``B_s42_a1.0@224|cls4``, D4 TTA of 8 views on holdout/val, identity on
   fit/train) with ``--batch-size`` (default 16) on ``--device``. Writes
   ``features/<split>__<condition>.npy`` plus the descriptors and fallback flags. ``fit``
   and ``holdout`` need nothing else; ``train`` and ``val`` are extracted only for the
   conditions of the selection's ``val_plan`` and need ``--selection`` + its pin.
2. ``select``: HAM10000 holdout ONLY. Probe fitted on ``fit``, every variant scored on
   ``holdout``, the registered adoption rule applied; writes ``e3_selection.json`` and
   prints its sha256. It opens no ``val`` or ``train`` array and no val label. Commit that
   file and pin its sha256 before the next stage.
3. ``final``: REFUSES to run without the selection file and ``--selection-sha256``, on a
   dirty tree, or when ``val_read.lock`` exists. Reads val exactly once (probe fitted on the
   full train), paired bootstrap of 1000 resamples, seed 20260923, grouped by ``lesion_id``,
   against B71 and the no-mask reference; writes ``e3_final_eval.json``.

Pod (RUNBOOK_CLUSTER; same clone, venv and TORCH_HOME as ``v71_run.py``)::

    P=$HOME/work/proyectonp; C1=$P/salidas/c1; E3=$P/salidas/e3
    COMMON="--ham-root $HOME/work/datos/ham10000 --image-dir $P/eval_ham10000_images \
      --split-index $C1/c1_split_index.json --holdout $C1/selection/selection_holdout.json \
      --weights-dir $P/salidas/v7 --masks $C1/masks/c1_pseudo_masks.npy \
      --masks-manifest $C1/masks/c1_pseudo_masks_manifest.json \
      --tone-manifest $P/salidas/final/fst/fst_pseudo_manifest.csv --out-dir $E3"
    python scripts/e3_run.py $COMMON --stage extract --splits fit holdout --limit 70   # smoke
    python scripts/e3_run.py $COMMON --stage extract --splits fit holdout --batch-size 16 --device cuda
    python scripts/e3_run.py $COMMON --stage select
    #   commit $E3/e3_selection.json, then:
    python scripts/e3_run.py $COMMON --stage extract --splits train val \
      --selection $E3/e3_selection.json --selection-sha256 <sha printed by select>
    python scripts/e3_run.py $COMMON --stage final \
      --selection $E3/e3_selection.json --selection-sha256 <sha printed by select>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from dermssl import e3_extract as xt  # noqa: E402
from dermssl import e3_render as render  # noqa: E402
from dermssl import e3_stages as sg  # noqa: E402
from dermssl import e3_store as st  # noqa: E402
from dermssl import e3_variants as var  # noqa: E402
from dermssl import feature_cache as fc  # noqa: E402
from dermssl import probe_threads  # noqa: E402
from dermssl import run_environment  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402
from dermssl import v71_protocol as vp  # noqa: E402

VIEWS = {"fit": 1, "train": 1, "holdout": 8, "val": 8}  # D4 views kept per split (v7.1: TTA only on eval)
GATED_SPLITS = frozenset({"train", "val"})
SELECTION_FILE, FINAL_FILE = "e3_selection.json", "e3_final_eval.json"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", choices=("extract", "select", "final"), required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    for flag in ("--ham-root", "--image-dir", "--split-index", "--holdout", "--weights-dir", "--masks",
                 "--masks-manifest", "--tone-manifest"):
        p.add_argument(flag, type=Path, default=None)
    p.add_argument("--splits", nargs="+", choices=tuple(VIEWS), default=["fit", "holdout"])
    p.add_argument("--conditions", nargs="+", choices=render.CONDITIONS, default=list(render.CONDITIONS))
    p.add_argument("--variants", nargs="+", choices=var.VARIANTS, default=list(var.VARIANTS))
    p.add_argument("--selection", type=Path, default=None, help="default: <out-dir>/e3_selection.json")
    p.add_argument("--selection-sha256", default="")
    p.add_argument("--nomask-grid", type=Path, default=None, help="v7.1 holdout_grid.json: its B-arm 224 rows join the no-mask bar")
    p.add_argument("--ci-resamples", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--decode-workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--allow-dirty", action="store_true", help="smoke only; never for train/val/final")
    return p.parse_args(argv)


def _revision(args, *, strict: bool) -> str:
    try:
        return rp.clean_revision(REPO)
    except rp.DirtyTreeError as exc:
        if args.allow_dirty and not strict:
            return "dirty"
        raise st.E3Error(str(exc)) from exc


def _is_committed(path: Path) -> bool:
    """The selection file is tracked and unchanged in the clone (informational, recorded in the final JSON)."""
    rel = str(path.resolve())
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", rel], capture_output=True)
    clean = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", rel], capture_output=True, text=True)
    return tracked.returncode == 0 and not clean.stdout.strip()


def _sidecar(path: Path, args) -> None:
    run_environment.write_sidecar(path, args, probe_blas_threads=probe_threads.PROBE_BLAS_THREADS)


def _report(cells: st.CellStore) -> int:
    for name, error in cells.failures:
        print(f"[FALLO] {name}: {error}")
    return 1 if cells.failures else 0


def _feature_shas(cells: st.CellStore, splits) -> dict:
    out = {}
    for split in splits:
        if not cells.path(f"extract__{split}").is_file():
            raise st.MissingFeaturesError(f"falta la celda extract__{split}: corre `extract --splits {split}`")
        out[split] = cells.read(f"extract__{split}")["result"]["sha256"]
    return out


def _gated_conditions(args) -> list[str]:
    selection, _ = st.require_selection(args.selection or args.out_dir / SELECTION_FILE, args.selection_sha256)
    if not selection.get("complete"):
        raise st.SelectionPinError("la seleccion esta incompleta: no se extraen train/val")
    needed = {c for name in selection["val_plan"] for c in var.required_conditions(name)}
    return [c for c in render.CONDITIONS if c in needed]


def stage_extract(args, rt, revision: str) -> int:
    cells, feats = st.CellStore(args.out_dir / "cells"), args.out_dir / "features"
    gated = _gated_conditions(args) if GATED_SPLITS & set(args.splits) else []
    for split in args.splits:
        conditions = gated if split in GATED_SPLITS else args.conditions
        ids = rt.ids[split]
        key = {"stage": "extract", "split": split, "conditions": conditions, "n_views": VIEWS[split],
               "ids_sha256": fc.fingerprint_ids(ids), **rt.fingerprint}
        status = cells.run(f"extract__{split}", key, lambda s=split, c=conditions, i=ids: xt.extract_cell(
            s, i, c, load_item=rt.load_item, extract_fn=rt.extract_fn, resize_crop=rt.resize_crop,
            out_dir=feats, batch_size=args.batch_size, n_views=VIEWS[s], hair_fn=rt.hair_fn, workers=args.decode_workers))
        print(f"[extract] {split}: {status}")
        if status == "done":
            _sidecar(cells.path(f"extract__{split}"), args)
    return _report(cells)


def _nomask_rows(path: Path | None) -> dict:
    """B-arm, 224 rows of the v7.1 holdout grid as extra no-mask bars (``macro_f1`` and ``ece`` only)."""
    if path is None:
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    keep = {cid: row for cid, row in rows.items()
            if vp.arm_of(vp.parse_config_id(cid)["weights"]) == "B" and vp.parse_config_id(cid)["resolution"] == 224}
    return {f"v71:{cid}": {"macro_f1": r["macro_f1"], "ece": r["ece"]} for cid, r in keep.items()}


def _run_for(args, rt, cells, splits) -> sg.Run:
    return sg.Run(feats_dir=args.out_dir / "features", n_classes=rt.n_classes, mel=rt.mel, labels_of=rt.labels_of,
                  groups_of=rt.groups_of, codes_of=rt.codes_of,
                  fingerprint={**rt.fingerprint, "feature_sha256": _feature_shas(cells, splits)})


def stage_select(args, rt, revision: str) -> int:
    cells = st.CellStore(args.out_dir / "cells")
    run = _run_for(args, rt, cells, ("fit", "holdout"))
    doc = sg.select_stage(run, cells, variants=args.variants, n_resamples=args.ci_resamples,
                          nomask_rows=_nomask_rows(args.nomask_grid))
    for name in args.variants:
        if cells.path(f"select__{name}").is_file():
            _sidecar(cells.path(f"select__{name}"), args)
    if _report(cells) or not doc["complete"]:
        print("[ABORT] seleccion incompleta: no se escribe e3_selection.json")
        return 1
    path = args.out_dir / SELECTION_FILE
    st.write_json_atomic(path, {**doc, "code_revision": revision})
    _sidecar(path, args)
    print(f"decision: {doc['decision']} (ganador: {doc['winner']}); val_plan: {doc['val_plan']}")
    print(f"-> {path}\nsha256 {st.file_sha256(path)}  <- commitear el archivo y fijar este sha256 en --selection-sha256")
    return 0


def stage_final(args, rt, revision: str) -> int:
    path = args.selection or args.out_dir / SELECTION_FILE
    cells = st.CellStore(args.out_dir / "cells")
    run = _run_for(args, rt, cells, ("train", "val"))
    doc = sg.final_stage(run, selection_path=path, pinned_sha256=args.selection_sha256,
                         lock=st.ValLock(args.out_dir), n_resamples=args.ci_resamples)
    out = args.out_dir / FINAL_FILE
    st.write_json_atomic(out, {**doc, "code_revision": revision, "selection_committed": _is_committed(path)})
    _sidecar(out, args)
    print(f"decision: {doc['decision']}; filas: {list(doc['rows'])}\n-> {out}")
    return 0


def main(argv=None, *, runtime=None, revision: str | None = None) -> int:
    args = parse_args(argv)
    args.out_dir = args.out_dir.expanduser()
    try:
        if args.stage == "final":  # the cheap refusals come before any weight or image is loaded
            st.require_selection(args.selection or args.out_dir / SELECTION_FILE, args.selection_sha256)
        strict = args.stage == "final" or bool(GATED_SPLITS & set(args.splits))
        revision = revision or _revision(args, strict=strict)
        if runtime is None:
            import e3_io  # noqa: PLC0415

            runtime = e3_io.build_runtime(args)
        stage = {"extract": stage_extract, "select": stage_select, "final": stage_final}[args.stage]
        return stage(args, runtime, revision)
    except st.E3Error as exc:
        print(f"ABORT: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
