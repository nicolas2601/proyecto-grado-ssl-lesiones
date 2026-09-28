r"""v7.1 (docs/29 §2.1-2.4 and §2.7): WiSE-FT, seed soup, probe protocol, TTA, one val read.

Three stages, each resumable and each writing the file the next one reads:

1. ``grid``: the §2.1 pre-test (alpha 0 = public and alpha 1 = the seed-42
   final, in the weights and in every holdout metric, tol 1e-6; abort
   otherwise), then every unit (21 weight sets x resolutions) on the docs/28
   internal holdout: 60 probe cells per unit with every metric and its CI.
   Selection ranges over 224 only; 448 is the §2.3 diagnostic at fixed weights.
   A unit is finished when ``units/<weights>@<res>/result.json`` exists; its
   features are kept on disk until then, so a crash never re-extracts a unit
   whose probes were running. Cached files from another holdout, ``--limit``
   or code revision are refused, never mixed in. -> ``holdout_grid.json``
2. ``select``: docs/28 §4.1 per arm (A*, B*) and for public, plus the §2.7
   item 7 phase-2 condition and holdout contrasts. -> ``selection.json``
3. ``val``: the ONLY read of validation, ``--split val --single-evaluation``:
   probe fitted on the full C1 train (8011), scored on the 2004 val images, for
   each winner, public under the same protocol, and every v7 primary; paired
   bootstrap (seed 20260923, 1000) against public. -> ``v71_final_eval.json``.
   ``--split holdout`` is the rehearsal of the same code on the holdout.

``--dry-run`` prints the plan and the projected cost and touches nothing;
``--limit N`` runs everything on N ids per split (class round-robin).

Pod (RUNBOOK_CLUSTER; the clone must be clean, ``git status --porcelain``
empty; TORCH_HOME must hold the facebookresearch/dinov2 hub repo and weights)::

    cd ~/work/proyectonp/repo_v7 && git pull && source ~/work/proyectonp/.venv/bin/activate
    P=$HOME/work/proyectonp; export TORCH_HOME=$HOME/work/.torch OMP_NUM_THREADS=4
    C1=$P/salidas/c1; SEL=$C1/selection
    COMMON="--ham-root $HOME/work/datos/ham10000 --image-dir $P/eval_ham10000_images \
      --split-index $C1/c1_split_index.json --holdout $SEL/selection_holdout.json \
      --weights-dir $P/salidas/v7 --bank-cache $P/salidas/tone/bank/cache"
    python scripts/v71_run.py $COMMON --out-dir $P/salidas/v71 --stage grid --dry-run
    # smoke: every stage on 70 ids per split, val stage only as holdout rehearsal
    for S in grid select; do
      python scripts/v71_run.py $COMMON --out-dir $P/salidas/v71_smoke --limit 70 --stage $S || break
    done
    python scripts/v71_run.py $COMMON --out-dir $P/salidas/v71_smoke --limit 70 --stage val --split holdout
    # real run (relaunch the same line after a crash: finished units are skipped)
    python scripts/v71_run.py $COMMON --out-dir $P/salidas/v71 --stage grid \
      --probe-jobs 10 --boot-jobs 24 --decode-workers 16
    python scripts/v71_run.py $COMMON --out-dir $P/salidas/v71 --stage select
    python scripts/v71_run.py $COMMON --out-dir $P/salidas/v71 --stage val --split val \
      --single-evaluation --exclude-ids-file $SEL/pretrain_exclude_ids_finals.txt

Then copy ``holdout_grid.json``, ``selection.json`` and ``v71_final_eval.json``
to ``resultados/clasificacion/v71/`` and run ``scripts/build_v71_table.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import v71_io as io  # noqa: E402
import v71_val  # noqa: E402

from dermssl import run_provenance as rp  # noqa: E402
from dermssl import v71_grid as vg  # noqa: E402
from dermssl import v71_plan as plan  # noqa: E402
from dermssl import v71_protocol as vp  # noqa: E402

DEFAULT_WEIGHTS = (vp.PUBLIC, *vp.weight_configs("A"), *vp.weight_configs("B"))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for flag in ("--ham-root", "--image-dir", "--split-index", "--holdout", "--weights-dir", "--out-dir"):
        p.add_argument(flag, type=Path, required=True)
    p.add_argument("--stage", choices=("grid", "select", "val"), required=True)
    p.add_argument("--split", choices=("holdout", "val"), default="holdout", help="val stage only")
    p.add_argument("--single-evaluation", action="store_true", help="required for --split val")
    p.add_argument("--exclude-ids-file", type=Path, default=None, help="required for --split val")
    p.add_argument("--bank-cache", type=Path, default=None, help="bank FeatureCache dir (sanity check)")
    p.add_argument("--bank-cache-name", default="dinov2_public_vitb14")
    p.add_argument("--v7-reference", type=Path, default=REPO / "resultados/clasificacion/v7/final_eval.json")
    p.add_argument("--resolutions", type=int, nargs="+", default=list(vp.RESOLUTIONS))
    p.add_argument("--weights", nargs="+", default=list(DEFAULT_WEIGHTS), help="subset, for smokes")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--allow-dirty", action="store_true", help="smoke/rehearsal only; never for val")
    p.add_argument("--keep-features", action="store_true")
    p.add_argument("--ci-resamples", type=int, default=1000)
    p.add_argument("--probe-jobs", type=int, default=10)
    p.add_argument("--boot-jobs", type=int, default=16)
    p.add_argument("--decode-workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def print_cost(ctx: dict, args) -> None:
    ids = ctx["ids"]
    cost = plan.cost_projection(n_fit=len(ids["fit"]), n_hold=len(ids["holdout"]), n_train=len(ids["train"]),
                                n_val=len(ids["val"]), resolutions=args.resolutions, n_weights=len(args.weights))
    print(f"plan: {cost['n_units']} unidades, {cost['n_configs']} configuraciones en el holdout; "
          f"fit {len(ids['fit'])} / holdout {len(ids['holdout'])} / train {len(ids['train'])} / val {len(ids['val'])}")
    for stage, by_res in cost["forwards"].items():
        print(f"  forwards {stage}: " + ", ".join(f"{n} @{r}" for r, n in by_res.items()))
    b = cost["cpu_hours_breakdown"]
    print(f"costo proyectado: {cost['gpu_hours']:.2f} GPU-h de extraccion (TTA x8 y 448 incluidos); "
          f"{cost['cpu_hours']:.2f} CPU-h (probes {b['probes']:.2f}, bootstrap {b['bootstrap']:.2f}, "
          f"decode {b['decode']:.2f}); supuestos {cost['assumptions']}")


def _revision(args) -> str:
    try:
        return rp.clean_revision(REPO)
    except rp.DirtyTreeError as exc:
        if args.allow_dirty and not (args.stage == "val" and args.split == "val"):
            return "dirty"
        raise SystemExit(f"ABORT: {exc}") from exc


def _device(args):
    import torch  # noqa: PLC0415

    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    return torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _holdout_arrays(ctx, store, resolution):
    return (store.array(ctx["ids"]["fit"], resolution, "fit"),
            store.array(ctx["ids"]["holdout"], resolution, "holdout"))


def _cls_metrics(model, arrays, ctx, args, device) -> tuple[dict, np.ndarray]:
    from dermssl import probes  # noqa: PLC0415
    from dermssl import torch_v71 as tv  # noqa: PLC0415
    from dermssl import v71_metrics as vm  # noqa: PLC0415

    fit_x, hold_x = arrays
    y_fit, y_hold = io.labels_of(ctx, ctx["ids"]["fit"]), io.labels_of(ctx, ctx["ids"]["holdout"])
    fit = tv.extract_split(model, fit_x, device=device, batch_size=args.batch_size)["cls"][0]
    hold = tv.extract_split(model, hold_x, device=device, batch_size=args.batch_size)["cls"][0]
    proba = probes.logreg_predict_proba(fit, y_fit, hold, len(ctx["classes"]))
    return vm.metric_bundle(y_hold, proba, mel=ctx["mel"]), np.concatenate([fit, hold])


def _bank_cache_check(args, ctx, public_cls: np.ndarray) -> dict:
    """Secondary: public CLS here vs the bank's cached train features (autocast residue expected)."""
    from dermssl import c1_provenance as pv  # noqa: PLC0415
    from dermssl.feature_cache import FeatureCache, cache_key  # noqa: PLC0415

    if args.bank_cache is None or args.limit:
        return {"status": "skipped"}
    train_ids = list(ctx["split"]["train_ids"])
    key = cache_key(args.bank_cache_name, "train", 224, train_ids)
    cached = FeatureCache(args.bank_cache.expanduser()).load(key)
    if cached is None:
        print(f"[WARN] {key} no esta en la cache del banco; el chequeo secundario no corre")
        return {"status": "not_found", "key": key}
    row = {i: k for k, i in enumerate(train_ids)}
    ref = cached.features[[row[i] for i in ctx["ids"]["fit"] + ctx["ids"]["holdout"]]]
    cos = (ref * public_cls).sum(1) / np.linalg.norm(ref, axis=1) / np.linalg.norm(public_cls, axis=1)
    ok = bool(cos.min() >= pv.PUBLISHED_CACHE_MIN_COSINE)
    return {"status": "ok" if ok else "failed", "key": key, "min_cosine": float(cos.min()),
            "bound": pv.PUBLISHED_CACHE_MIN_COSINE}


def _check_cached(cached: dict, ctx: dict, revision: str, path: Path) -> None:
    """A cached file is reused only for this holdout, these ids and this code revision."""
    if (cached.get("holdout_ids_sha256"), cached.get("run_fingerprint")) != (ctx["holdout_sha"], ctx["run_fingerprint"]):
        raise SystemExit(f"ABORT: {path} es de otro holdout o de otro --limit; usar un --out-dir nuevo")
    if cached.get("code_revision") != revision:
        raise SystemExit(f"ABORT: {path} es de la revision {cached.get('code_revision')}, no {revision}; "
                         "usar un --out-dir nuevo")


def run_pretest(args, ctx, store, weights, device, revision) -> None:
    from dermssl import torch_v71 as tv  # noqa: PLC0415

    path = args.out_dir / "pretest.json"
    if path.exists():
        cached = io.read_json(path)
        _check_cached(cached, ctx, revision, path)
        if cached["passed"]:
            return
    arrays = _holdout_arrays(ctx, store, 224)
    public_metrics, public_cls = _cls_metrics(io.direct_public_model(), arrays, ctx, args, device)
    checks = []
    for arm in vp.ARMS:
        for alpha, reference_state, reference_model in (
            (0.0, weights.state(vp.PUBLIC), None),
            (1.0, weights.direct_state(arm), io.direct_final_model(weights.final_path(arm, "s42"))),
        ):
            state = weights.state(vp.weights_name(arm, "s42", alpha))
            direct = public_metrics if reference_model is None else _cls_metrics(reference_model, arrays, ctx, args, device)[0]
            wise = _cls_metrics(weights.model(state), arrays, ctx, args, device)[0]
            checks.append(plan.pretest_check(arm, alpha, direct, wise,
                                             weights_diff=tv.max_abs_diff(state, reference_state)))
    bank_check = _bank_cache_check(args, ctx, public_cls)
    passed = all(c["passed"] for c in checks) and bank_check["status"] != "failed"
    io.write_json(path, {"passed": passed, "checks": checks, "bank_cache_check": bank_check,
                         "holdout_ids_sha256": ctx["holdout_sha"], "run_fingerprint": ctx["run_fingerprint"],
                         "code_revision": revision})
    if not passed:
        raise SystemExit(f"ABORT: el pre-test de docs/29 §2.1 fallo; ver {path}")
    print("pre-test OK: alpha 0 = publico y alpha 1 = final en pesos y metricas (tol 1e-6)")


def run_unit(args, ctx, store, weights, device, revision, name: str, resolution: int) -> dict:
    from dermssl import torch_v71 as tv  # noqa: PLC0415

    udir = args.out_dir / "units" / f"{name}@{resolution}"
    if (udir / "result.json").exists():
        cached = io.read_json(udir / "result.json")
        _check_cached(cached, ctx, revision, udir / "result.json")
        return cached
    feats_path, start = udir / "features.npz", time.time()
    if feats_path.exists():
        saved = np.load(feats_path)
        fit_parts = {p: saved[f"fit_{p}"] for p in tv.PARTS}
        hold_parts = {p: saved[f"hold_{p}"] for p in tv.PARTS}
    else:
        print(f"[{name}@{resolution}] extraer")
        fit_x, hold_x = _holdout_arrays(ctx, store, resolution)
        model = weights.model(weights.state(name))
        fit_parts = tv.extract_split(model, fit_x, device=device, batch_size=args.batch_size)
        hold_parts = tv.extract_split(model, hold_x, device=device, batch_size=args.batch_size, n_views=tv.N_D4)
        udir.mkdir(parents=True, exist_ok=True)
        np.savez(udir / "features.tmp.npz", **{f"fit_{k}": v for k, v in fit_parts.items()},
                 **{f"hold_{k}": v for k, v in hold_parts.items()})
        (udir / "features.tmp.npz").replace(feats_path)
    extract_s = time.time() - start
    y_fit, y_hold = io.labels_of(ctx, ctx["ids"]["fit"]), io.labels_of(ctx, ctx["ids"]["holdout"])
    probas = vg.unit_probas(fit_parts, y_fit, hold_parts, len(ctx["classes"]), n_jobs=args.probe_jobs)
    rows = vg.score_unit(name, resolution, y_hold, probas, mel=ctx["mel"],
                         n_resamples=args.ci_resamples, n_jobs=args.boot_jobs)
    cells = list(rows)
    np.savez_compressed(udir / "probas.npz", y=y_hold,
                        **{f"p{k}": probas[key].astype(np.float32) for k, key in enumerate(probas)})
    n_fwd = len(ctx["ids"]["fit"]) + tv.N_D4 * len(ctx["ids"]["holdout"])
    result = {"weights": name, "resolution": resolution, "holdout_ids_sha256": ctx["holdout_sha"],
              "run_fingerprint": ctx["run_fingerprint"], "code_revision": revision, "cells": cells, "rows": rows,
              "timing": {"extract_s": extract_s, "img_per_s": n_fwd / max(extract_s, 1e-9),
                         "probe_and_ci_s": time.time() - start - extract_s}}
    io.write_json(udir / "result.json", result)
    if not args.keep_features:
        feats_path.unlink(missing_ok=True)
    t = result["timing"]
    print(f"[{name}@{resolution}] {t['img_per_s']:.0f} img/s, probes+IC {t['probe_and_ci_s']:.0f}s")
    return result


def unit_order(args) -> list[tuple[str, int]]:
    """224 before 448, public first: a crash still leaves the primary resolution answered."""
    return [(w, r) for r in sorted(args.resolutions) for w in args.weights]


def stage_grid(args, ctx, revision) -> None:
    device = _device(args)
    store = io.ImageStore(args.out_dir / "inputs", ctx["image_index"], args.decode_workers)
    weights = io.WeightBank(args.weights_dir.expanduser())
    run_pretest(args, ctx, store, weights, device, revision)
    results = [run_unit(args, ctx, store, weights, device, revision, w, r) for w, r in unit_order(args)]
    rows = {cid: row for r in results for cid, row in r["rows"].items()}
    io.write_json(args.out_dir / "holdout_grid.json", {
        "code_revision": revision, "holdout_ids_sha256": ctx["holdout_sha"], "n_configs": len(rows),
        "rule": "docs/29 §2.4 (docs/28 §4.1): max macro_f1, ties < 0.005 by lowest ece",
        "limit": args.limit, "rows": rows})
    print(f"-> {args.out_dir / 'holdout_grid.json'} ({len(rows)} configuraciones)")


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out_dir = args.out_dir.expanduser()
    if args.stage == "val" and args.split == "val" and not args.single_evaluation:
        raise SystemExit("ABORT: --split val exige --single-evaluation (la validacion se toca una vez)")
    ctx = io.load_context(args)
    print_cost(ctx, args)
    if args.dry_run:
        return 0
    revision = _revision(args)
    if args.stage == "grid":
        stage_grid(args, ctx, revision)
    elif args.stage == "select":
        v71_val.stage_select(args, ctx)
    else:
        v71_val.stage_val(args, ctx, revision, _device(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
