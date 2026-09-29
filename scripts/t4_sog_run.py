r"""T4 (docs/31): Shades of Gray p = 6 before DINOv2, on the v7.1 probe protocol.

Three stages, each writing the file the next one reads (the v7.1 pattern):

1. ``grid``: for identity and then SoG, the 60 v7.1 cells of public, A* (the
   A soup) and B* (B s42) on the docs/28 holdout at 224, reusing
   ``v71_run.run_unit``. Before any SoG unit, the identity cells of the three
   fixed protocols must reproduce the committed v7.1 holdout grid within 1e-6,
   or the run aborts. Units and arrays of each input live apart and carry the
   input in their fingerprint. -> ``t4_holdout_grid.json``
2. ``select``: the SoG winner per arm (docs/28 §4.1) and the holdout contrasts
   SoG - identity at the fixed protocols. -> ``t4_selection.json``
3. ``val``: the ONLY read of validation, ``--split val --single-evaluation``,
   with the v7.1 guards, the docs/31 registration guard and the registered
   luminance manifest. -> ``t4_final_eval.json`` (P1-P3 verdicts included).
   ``--split holdout`` rehearses the same code on the holdout.

``--dry-run`` prints the projected cost and touches nothing.

Pod (RUNBOOK_CLUSTER; clean clone; TORCH_HOME with the dinov2 hub repo)::

    cd ~/work/proyectonp/repo_v7 && git pull && source ~/work/proyectonp/.venv/bin/activate
    P=$HOME/work/proyectonp; export TORCH_HOME=$HOME/work/.torch OMP_NUM_THREADS=4
    C1=$P/salidas/c1; SEL=$C1/selection
    COMMON="--ham-root $HOME/work/datos/ham10000 --image-dir $P/eval_ham10000_images \
      --split-index $C1/c1_split_index.json --holdout $SEL/selection_holdout.json \
      --weights-dir $P/salidas/v7 --tone-manifest $P/salidas/final/fst/fst_pseudo_manifest.csv \
      --out-dir $P/salidas/t4_sog"
    python scripts/t4_sog_run.py $COMMON --stage grid --dry-run
    python scripts/t4_sog_run.py $COMMON --stage grid --probe-jobs 10 --boot-jobs 24 --decode-workers 16
    python scripts/t4_sog_run.py $COMMON --stage select
    python scripts/t4_sog_run.py $COMMON --stage val --split val --single-evaluation \
      --exclude-ids-file $SEL/pretrain_exclude_ids_finals.txt

Then copy ``t4_holdout_grid.json``, ``t4_selection.json`` and ``t4_final_eval.json``
to ``resultados/clasificacion/t4_sog/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import t4_sog_val  # noqa: E402
import v71_io as io  # noqa: E402
import v71_run  # noqa: E402
from t4_sog_val import input_args, input_ctx, store_for, unit_probas  # noqa: E402, F401 - unit_probas is API

from dermssl import pad_ufes_guard as guard  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402
from dermssl import t4_sog as t4  # noqa: E402

V71_GRID = REPO / "resultados/clasificacion/v71/holdout_grid.json"
V71_FINAL = REPO / "resultados/clasificacion/v71/v71_final_eval.json"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for flag in ("--ham-root", "--image-dir", "--split-index", "--holdout", "--weights-dir", "--out-dir"):
        p.add_argument(flag, type=Path, required=True)
    p.add_argument("--tone-manifest", type=Path, required=True, help="fst_pseudo_manifest.csv (docs/31 §4)")
    p.add_argument("--stage", choices=("grid", "select", "val"), required=True)
    p.add_argument("--split", choices=("holdout", "val"), default="holdout", help="val stage only")
    p.add_argument("--single-evaluation", action="store_true", help="required for --split val")
    p.add_argument("--exclude-ids-file", type=Path, default=None, help="required for --split val")
    p.add_argument("--v71-grid", type=Path, default=V71_GRID, help="committed v7.1 holdout grid")
    p.add_argument("--v71-final", type=Path, default=V71_FINAL, help="committed v7.1 val read")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="smoke: skips the reproduction guard")
    p.add_argument("--allow-dirty", action="store_true", help="smoke/rehearsal only; never for val")
    p.add_argument("--keep-features", action="store_true")
    p.add_argument("--ci-resamples", type=int, default=1000)
    p.add_argument("--probe-jobs", type=int, default=10)
    p.add_argument("--boot-jobs", type=int, default=16)
    p.add_argument("--decode-workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def check_registration(repo: Path) -> str:
    """docs/31 was committed, unchanged since, before this script existed."""
    try:
        return guard.check_registration(repo, commit=t4.REGISTRATION_COMMIT, doc=t4.REGISTRATION_DOC,
                                        script=t4.RUN_SCRIPT, start=t4.BLOCK_START, end=t4.BLOCK_END)
    except guard.RegistrationError as exc:
        raise SystemExit(f"ABORT: registro de docs/31: {exc}") from exc


def print_cost(ctx: dict) -> None:
    ids = ctx["ids"]
    cost = t4.cost_projection(n_fit=len(ids["fit"]), n_hold=len(ids["holdout"]),
                              n_train=len(ids["train"]), n_val=len(ids["val"]))
    b = cost["cpu_hours_breakdown"]
    print(f"plan T4: {cost['n_units']} unidades (entradas {t4.INPUTS} x pesos {tuple(t4.WEIGHTS.values())}); "
          f"forwards grid {cost['forwards']['grid']}, val {cost['forwards']['val']}")
    print(f"costo proyectado: {cost['gpu_hours']:.2f} GPU-h; {cost['cpu_hours']:.2f} CPU-h ("
          + ", ".join(f"{k} {v:.2f}" for k, v in b.items()) + f"); supuestos {cost['assumptions']}")


def _grid_input(args, ctx, weights, device, revision, name: str) -> dict:
    sub_args, sub_ctx, store = input_args(args, name), input_ctx(ctx, name), store_for(args, ctx, name)
    results = [v71_run.run_unit(sub_args, sub_ctx, store, weights, device, revision, w, t4.vp.SELECTION_RESOLUTION)
               for w in t4.WEIGHTS.values()]
    return {cid: row for result in results for cid, row in result["rows"].items()}


def _reproduction(args, identity_rows: dict) -> dict:
    if args.limit:
        return {"status": "skipped", "reason": "--limit cambia los ids", "passed": True}
    report = t4.reproduction_check(identity_rows, io.read_json(args.v71_grid.expanduser())["rows"])
    report["reference"] = str(args.v71_grid)
    if not report["passed"]:
        io.write_json(args.out_dir / "reproduction_failed.json", report)
        raise SystemExit("ABORT: sin SoG no se reproduce la grilla v7.1 commiteada (tol 1e-6); ver "
                         f"{args.out_dir / 'reproduction_failed.json'}. No se extrae nada con SoG.")
    return report


def stage_grid(args, ctx, revision, registration) -> None:
    device, weights = v71_run._device(args), io.WeightBank(args.weights_dir.expanduser())
    identity = _grid_input(args, ctx, weights, device, revision, t4.IDENTITY)
    reproduction = _reproduction(args, identity)
    print(f"reproduccion v7.1 sin SoG: {reproduction.get('status', 'ok' if reproduction['passed'] else 'falla')}")
    sog = _grid_input(args, ctx, weights, device, revision, t4.SOG)
    io.write_json(args.out_dir / "t4_holdout_grid.json", {
        "code_revision": revision, "registration_commit": registration, "holdout_ids_sha256": ctx["holdout_sha"],
        "limit": args.limit, "reproduction": reproduction, "sog_p": t4.SOG_P,
        "inputs": {t4.IDENTITY: identity, t4.SOG: sog}})
    print(f"-> {args.out_dir / 't4_holdout_grid.json'} ({len(identity) + len(sog)} celdas)")


def _early_guards(args) -> None:
    if args.stage == "val" and args.split == "val":
        if not args.single_evaluation:
            raise SystemExit("ABORT: --split val exige --single-evaluation (la validacion se toca una vez)")
        sha = rp.file_sha256(args.tone_manifest.expanduser())
        if sha != t4.MANIFEST_SHA256:
            raise SystemExit(f"ABORT: el manifiesto de luminancia {sha[:12]} no es el registrado en docs/31 "
                             f"({t4.MANIFEST_SHA256[:12]})")


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out_dir = args.out_dir.expanduser()
    _early_guards(args)
    ctx = io.load_context(args)
    print_cost(ctx)
    if args.dry_run:
        return 0
    registration = check_registration(REPO) if args.stage != "select" else None
    revision = v71_run._revision(args)
    if args.stage == "grid":
        stage_grid(args, ctx, revision, registration)
    elif args.stage == "select":
        t4_sog_val.stage_select(args, ctx)
    else:
        t4_sog_val.stage_val(args, ctx, revision, v71_run._device(args), registration)
    return 0


if __name__ == "__main__":
    sys.exit(main())
