r"""HAM10000 clean-weights sensitivity (docs/33): B71-limpio and A71-limpio on the C1 val, exact v7.1 protocol.

Posterior sensitivity analysis, NOT a selection. Configurations and operating thresholds come from the
ORIGINAL v7.1 holdout selection (``resultados/clasificacion/v71``, read only); the thresholds were fitted with
the original weights and are applied unchanged to the clean ones (declared assumption). Nothing here reads val
to choose anything. The clean run has its OWN out-dir and its OWN ``val_read.lock`` (the E3 lock class) and
aborts when that lock exists; the original out-dirs and their guards are never touched.

* B71-limpio = ``B_s42_a1.0@224|cls4|balanced|tta`` on ``<weights-dir>/final_B_s42`` (a single final, as B71).
* A71-limpio = ``A_soup_a1.0@224|cls_attn|la0.5|tta`` on the soup of ``final_A_s42`` and ``final_A_s43``.
* Same probe (full train for argmax/rank metrics, the 80 % fit probe for thresholded ones), canonical compute
  (batch 64 + CUDA autocast; the real read refuses anything else), D4 TTA of 8 views.
* Controls: the public and original probabilities of the committed v71_final_eval (same ids, asserted).
* Paired bootstrap: 1000 resamples, seed 20260923, grouped by ``lesion_id``; MCC appended last.

Stages (each one is a separate process; the lock is taken by ``extract``, the first read of val):
``extract`` (guards, lock, features, probes -> probabilities) then ``val`` (metrics, contrasts, JSON, sidecar).
``--rehearsal`` runs the SAME code on the holdout (fit 80 % -> holdout), never opens val, takes no lock, and
writes only to an out-dir named ``*rehearsal*`` outside ``resultados/``. Its controls come from a holdout
rehearsal of the original stage (``v71_final_eval_holdout_rehearsal.json``) in ``--reference-dir``.
A crashed ``extract`` after the lock was taken cannot be resumed in the same out-dir: use a new one.

Pod: see ``scripts/pod/ham_clean.sh`` (clone must be clean; TORCH_HOME with the dinov2 hub repo).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import v71_io as io  # noqa: E402
import v71_run  # noqa: E402
import v71_val  # noqa: E402

from dermssl import c1_split_identity as si  # noqa: E402
from dermssl import e3_store as st  # noqa: E402
from dermssl import final_eval as fe  # noqa: E402
from dermssl import ham_clean as hc  # noqa: E402
from dermssl import probe_threads, run_environment  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402
from dermssl import v71_metrics as vm  # noqa: E402
from dermssl import v71_plan as plan  # noqa: E402
from dermssl.isic_eval import weights as wt  # noqa: E402

V71_DIR = REPO / "resultados/clasificacion/v71"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for flag in ("--ham-root", "--image-dir", "--split-index", "--holdout", "--weights-dir", "--out-dir"):
        p.add_argument(flag, type=Path, required=True)
    p.add_argument("--stage", choices=("extract", "val"), required=True)
    p.add_argument("--v71-dir", type=Path, default=V71_DIR, help="original selection artifacts (read only)")
    p.add_argument("--reference-dir", type=Path, default=None, help="controls (default: --v71-dir)")
    p.add_argument("--rehearsal", action="store_true", help="same code on the holdout; never opens val")
    p.add_argument("--single-evaluation", action="store_true", help="required for the real val read")
    p.add_argument("--allow-dirty", action="store_true", help="rehearsal only")
    p.add_argument("--ci-resamples", type=int, default=vm.BOOTSTRAP_RESAMPLES)
    p.add_argument("--probe-jobs", type=int, default=10)
    p.add_argument("--decode-workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=hc.CANONICAL_BATCH)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    args.limit, args.split = None, ("holdout" if args.rehearsal else "val")
    args.out_dir, args.v71_dir = args.out_dir.expanduser(), args.v71_dir.expanduser()
    args.reference_dir = (args.reference_dir or args.v71_dir).expanduser()
    return args


def check_flags(args) -> None:
    if args.rehearsal == args.single_evaluation:
        raise SystemExit("ABORT: exactamente uno de --rehearsal o --single-evaluation (la val se toca una vez)")
    if args.rehearsal and ("rehearsal" not in args.out_dir.name or (REPO / "resultados") in args.out_dir.resolve().parents):
        raise SystemExit("ABORT: el ensayo escribe solo en un --out-dir '*rehearsal*' fuera de resultados/")


def revision(args) -> str:
    try:
        return rp.clean_revision(REPO)
    except rp.DirtyTreeError as exc:
        if args.allow_dirty and args.rehearsal:
            return "dirty"
        raise SystemExit(f"ABORT: {exc}") from exc


def load_selection(args, ctx) -> dict:
    """The ORIGINAL selection, grid and entries (read only), checked against this holdout."""
    selection, grid = io.read_json(args.v71_dir / "selection.json"), io.read_json(args.v71_dir / "holdout_grid.json")
    if selection["holdout_ids_sha256"] != ctx["holdout_sha"] or grid["holdout_ids_sha256"] != ctx["holdout_sha"]:
        raise SystemExit("ABORT: la seleccion original es de otro holdout")
    entries = plan.val_entries(selection)
    try:
        clean = hc.clean_entries(entries)
    except hc.HamCleanError as exc:
        raise SystemExit(f"ABORT: {exc}") from exc
    sources = {name: rp.file_sha256(args.v71_dir / name) for name in ("selection.json", "holdout_grid.json")}
    return {"entries": entries, "clean": clean, "grid_rows": grid["rows"], "sources": sources}


def split_ids(args, ctx) -> tuple[list, list]:
    ids = ctx["ids"]
    return (ids["fit"], ids["holdout"]) if args.rehearsal else (ids["train"], ids["val"])


def guard_weights(args, ctx, bank) -> dict:
    """Pool label (shared with the ISIC bank), A-soup consistency and, for the real read, the C1 ids."""
    manifests = bank.run_manifests(hc.NEEDED_FINALS)
    try:
        pool = wt.check_pool_label(manifests, pool_excludes_val=True)
        fe.check_runs_consistent({k: v for k, v in manifests.items() if k.startswith("A_")})
        if not args.rehearsal:
            fe.check_val_protocol(ctx["ids"]["train"], ctx["ids"]["val"])
    except (wt.PoolLabelError, ValueError) as exc:
        raise SystemExit(f"ABORT: {exc}") from exc
    return {"pool_check": pool, "runs": {k: {f: r.get(f) for f in ("code_revision", "exclude_ids_sha256", "seed")}
                                         for k, r in manifests.items()}}


def stage_extract(args, ctx, device) -> None:
    out = args.out_dir / hc.EXTRACT_JSON
    if out.exists():
        raise SystemExit(f"ABORT: {out} ya existe; la extraccion no se repite")
    selection = load_selection(args, ctx)
    bank = io.WeightBank(args.weights_dir.expanduser(), clean=True)
    guards = guard_weights(args, ctx, bank)
    try:
        compute = hc.compute_block(device.type, args.batch_size, rehearsal=args.rehearsal)
    except hc.HamCleanError as exc:
        raise SystemExit(f"ABORT: {exc}") from exc
    rev = revision(args)
    if not args.rehearsal:
        try:
            st.ValLock(args.out_dir).acquire(selection["sources"]["selection.json"])
        except st.ValAlreadyReadError as exc:
            raise SystemExit(f"ABORT: {exc}") from exc
    fit_ids, eval_ids = split_ids(args, ctx)
    probas, probas80 = v71_val._score_entries(args, ctx, selection["clean"], fit_ids, eval_ids, device, bank=bank)
    npz = args.out_dir / "clean_probas.npz"
    np.savez_compressed(npz, **probas, **{f"{k}__fit80": v for k, v in probas80.items()})
    weights = {f"{arm}_{seed}": rp.file_sha256(bank.final_path(arm, seed)) for arm, seed in hc.NEEDED_FINALS}
    io.write_json(out, {"split": args.split, "code_revision": rev, "eval_ids_sha256": si.ids_sha256(eval_ids),
                        "selection_sources": selection["sources"], "weights_sha256": weights, "compute": compute,
                        "probas_sha256": rp.file_sha256(npz), **guards})
    run_environment.write_sidecar(out, args, loky_inner_threads=probe_threads.PROBE_BLAS_THREADS)
    print(f"extract listo ({args.split}) -> {out}")


def load_reference(args, eval_ids, y, entries) -> tuple[dict, dict]:
    name = "v71_final_eval_holdout_rehearsal" if args.rehearsal else "v71_final_eval"
    doc = io.read_json(args.reference_dir / f"{name}.json")
    saved = np.load(args.reference_dir / f"{name}.probas.npz")
    try:
        hc.check_reference(doc, saved["y_true"], split=args.split, eval_ids=eval_ids, y=y, entries=entries)
    except hc.HamCleanError as exc:
        raise SystemExit(f"ABORT: {exc}") from exc
    return doc, {label: saved[label].astype(np.float64) for label in (*hc.ORIGINAL, *hc.PUBLIC)}


def stage_val(args, ctx) -> None:
    out = args.out_dir / (hc.REHEARSAL_JSON if args.rehearsal else Path(hc.RESULT_JSON).name)
    extract_path = args.out_dir / hc.EXTRACT_JSON
    if out.exists():
        raise SystemExit(f"ABORT: {out} ya existe; la evaluacion no se repite")
    if not extract_path.is_file() or not (args.out_dir / "clean_probas.npz").is_file():
        raise SystemExit("ABORT: falta la etapa extract en este --out-dir")
    if not args.rehearsal and not st.ValLock(args.out_dir).taken():
        raise SystemExit("ABORT: no existe val_read.lock: la val solo se lee a traves de `extract`")
    extracted = io.read_json(extract_path)
    selection = load_selection(args, ctx)
    _, eval_ids = split_ids(args, ctx)
    if extracted["split"] != args.split or extracted["eval_ids_sha256"] != si.ids_sha256(eval_ids):
        raise SystemExit("ABORT: el extract es de otro split o de otros ids")
    y = io.labels_of(ctx, eval_ids)
    ref_doc, ref = load_reference(args, eval_ids, y, selection["entries"])
    saved = np.load(args.out_dir / "clean_probas.npz")
    clean = {k: saved[k].astype(np.float64) for k in selection["clean"]}
    try:
        probas = hc.contrast_probas(clean, ref)
    except hc.HamCleanError as exc:
        raise SystemExit(f"ABORT: {exc}") from exc
    n, mel = args.ci_resamples, ctx["mel"]
    results = {label: v71_val._entry_result(y, clean[label], saved[f"{label}__fit80"].astype(np.float64), cid,
                                            selection["grid_rows"], mel, n) for label, cid in selection["clean"].items()}
    groups = np.array([ctx["lesion_of"][i] for i in eval_ids])
    paired = [row for a, c in hc.CONTRASTS
              for row in v71_val._contrast(a, c, y, probas, mel, n, metrics=hc.CONTRAST_METRICS, groups=groups)]
    npz = out.with_suffix(".probas.npz")
    np.savez_compressed(npz, y_true=y, **probas, **{f"{k}__fit80": saved[f"{k}__fit80"] for k in clean})
    io.write_json(out, {
        **hc.declarations(rehearsal=args.rehearsal), "split": args.split, "code_revision": extracted["code_revision"],
        "probe": "logreg (C1, C=1)", "n_eval": len(eval_ids), "eval_ids_sha256": si.ids_sha256(eval_ids),
        "holdout_ids_sha256": ctx["holdout_sha"], "entries": selection["clean"],
        "controls": {k: selection["entries"][k] for k in (*hc.ORIGINAL, *hc.PUBLIC)},
        "selection_sources": selection["sources"], "reference": {"dir": str(args.reference_dir),
                                                                  "code_revision": ref_doc.get("code_revision")},
        "results": results, "paired": paired, "probas_sha256": rp.file_sha256(npz),
        **{k: extracted[k] for k in ("pool_check", "runs", "weights_sha256", "compute")},
        "bootstrap": {"n_resamples": n, "rng_seed": vm.BOOTSTRAP_SEED, "paired": True, "grouped_by": hc.GROUPED_BY,
                      "note": "contrastes por lesion_id; los IC de cada entrada (results.ci95) son por fila, como en "
                              "v7.1, y los contrastes de v7.1 contra el publico tambien son por fila"},
        "note_operating_points": plan.OPERATING_POINT_PROTOCOL})
    run_environment.write_sidecar(out, args, loky_inner_threads=probe_threads.PROBE_BLAS_THREADS)
    for label, r in results.items():
        print(f"{label}: macro-F1 {r['macro_f1']:.4f}")
    print(f"-> {out}")


def main(argv=None) -> int:
    args = parse_args(argv)
    check_flags(args)
    ctx = io.load_context(args)
    if args.stage == "extract":
        stage_extract(args, ctx, v71_run._device(args))
    else:
        stage_val(args, ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
