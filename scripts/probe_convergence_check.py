r"""Diagnostic: do the published v7.1 holdout-grid logreg fits converge within max_iter?

Refits, with the same code path and arguments and BLAS pinned to 4 threads
(the setting that reproduced v7.1 bit for bit), every fit of the three T4
identity units (public, A_soup_a1.0, B_s42_a1.0 at 224: 5 features x
{none, balanced} = 30 fits), and records ``n_iter_`` vs ``max_iter`` and any
``ConvergenceWarning``. Before trusting that, every refit cell must equal the
published ``probas.npz`` bit for bit and the committed grid's point metrics
exactly; otherwise the diagnosis is of other fits and the script exits 1
(the JSON is still written, with the failure inside).

Features come from ``--features-dir/<unit>/features.npz`` when a kept run has
them; otherwise they are extracted on GPU from the cached v7.1 inputs.

Pod::

    cd $P/repo_t4 && source $P/.venv/bin/activate
    export TORCH_HOME=$HOME/work/.torch OMP_NUM_THREADS=4
    python scripts/probe_convergence_check.py --ham-root $HOME/work/datos/ham10000 \
      --image-dir $P/eval_ham10000_images --split-index $P/salidas/c1/c1_split_index.json \
      --holdout $P/salidas/c1/selection/selection_holdout.json --weights-dir $P/salidas/v7 \
      --inputs-dir $P/salidas/v71/inputs --reference-units $P/salidas/v71/units \
      --out-dir $P/salidas/diag_convergencia --jobs 10

Then copy ``probe_convergence.json``, its ``.environment.json`` and ``README.md``
to ``resultados/clasificacion/t4/diagnostico_convergencia/``.
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

from dermssl import probe_convergence as pc  # noqa: E402
from dermssl import run_environment  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402
from dermssl import t4_sog as t4  # noqa: E402
from dermssl import v71_protocol as vp  # noqa: E402

V71_GRID = REPO / "resultados/clasificacion/v71/holdout_grid.json"
RESOLUTION = vp.SELECTION_RESOLUTION
OUT_JSON = "probe_convergence.json"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for flag in ("--ham-root", "--image-dir", "--split-index", "--holdout", "--weights-dir", "--inputs-dir",
                 "--reference-units", "--out-dir"):
        p.add_argument(flag, type=Path, required=True)
    p.add_argument("--features-dir", type=Path, default=None, help="units/ of a run kept with --keep-features")
    p.add_argument("--v71-grid", type=Path, default=V71_GRID)
    p.add_argument("--jobs", type=int, default=10, help="parallel fits; each pinned to BLAS_THREADS")
    p.add_argument("--decode-workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--limit", default=None, help=argparse.SUPPRESS)  # load_context reads it; never set
    return p.parse_args(argv)


def _cached_features(args, unit: str):
    path = args.features_dir.expanduser() / unit / "features.npz" if args.features_dir else None
    if path is None or not path.exists():
        return None
    saved = np.load(path)
    parts = sorted({k.split("_", 1)[1] for k in saved.files})
    return {p: saved[f"fit_{p}"] for p in parts}, {p: saved[f"hold_{p}"] for p in parts}


def _extract(args, ctx, weights, name: str):
    from dermssl import torch_v71 as tv  # noqa: PLC0415

    store = io.ImageStore(args.inputs_dir.expanduser(), ctx["image_index"], args.decode_workers)
    fit_x = store.array(ctx["ids"]["fit"], RESOLUTION, "fit")
    hold_x = store.array(ctx["ids"]["holdout"], RESOLUTION, "holdout")
    model, device = weights.model(weights.state(name)), _device(args)
    fit = tv.extract_split(model, fit_x, device=device, batch_size=args.batch_size)
    hold = tv.extract_split(model, hold_x, device=device, batch_size=args.batch_size, n_views=tv.N_D4)
    return fit, hold


def _device(args):
    import torch  # noqa: PLC0415

    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    return torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _reference(args, unit: str) -> tuple[dict, dict]:
    """``cid -> float32 proba`` of the published unit, and its recorded feature digests."""
    udir = args.reference_units.expanduser() / unit
    result = io.read_json(udir / "result.json")
    stored = np.load(udir / "probas.npz")
    return {cid: stored[f"p{k}"] for k, cid in enumerate(result["cells"])}, result.get("feature_sha256", {})


def _unit(args, ctx, weights, grid_rows, name: str) -> tuple[list, dict]:
    unit = f"{name}@{RESOLUTION}"
    start = time.time()
    cached = _cached_features(args, unit)
    fit_parts, hold_parts = cached if cached else _extract(args, ctx, weights, name)
    digests = t4.array_digests({**{f"fit_{k}": v for k, v in fit_parts.items()},
                                **{f"hold_{k}": v for k, v in hold_parts.items()}})
    y_fit, y_hold = io.labels_of(ctx, ctx["ids"]["fit"]), io.labels_of(ctx, ctx["ids"]["holdout"])
    cells, records = pc.unit_fits(unit, fit_parts, y_fit, hold_parts, len(ctx["classes"]), n_jobs=args.jobs)
    reference, ref_digests = _reference(args, unit)
    sanity = {"features_from": "features.npz" if cached else "gpu_extraction",
              "features_equal_published": (digests == ref_digests) if ref_digests else None,
              "probas": pc.compare_probas(name, RESOLUTION, cells, reference),
              "metrics": pc.compare_metrics(name, RESOLUTION, cells, grid_rows, y_hold, mel=ctx["mel"]),
              "seconds": time.time() - start}
    hits = sum(r["hit_max_iter"] for r in records)
    print(f"[{unit}] {len(records)} fits, {hits} en max_iter; probas iguales: {sanity['probas']['passed']}, "
          f"metricas iguales: {sanity['metrics']['passed']} ({sanity['seconds']:.0f}s)")
    return records, sanity


def _revision(args) -> str:
    try:
        return rp.clean_revision(REPO)
    except rp.DirtyTreeError as exc:
        if args.allow_dirty:
            return "dirty"
        raise SystemExit(f"ABORT: {exc}") from exc


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out_dir = args.out_dir.expanduser()
    pc.check_out_dir(args.out_dir, protected=[args.reference_units.parent, args.inputs_dir, REPO / "resultados"
                                              / "clasificacion" / "v71", REPO / "resultados" / "clasificacion"
                                              / "t4_sog", *([args.features_dir] if args.features_dir else [])])
    revision, ctx = _revision(args), io.load_context(args)
    grid_rows = io.read_json(args.v71_grid.expanduser())["rows"]
    weights = io.WeightBank(args.weights_dir.expanduser())
    records, sanity = [], {}
    for name in t4.WEIGHTS.values():
        unit_records, sanity[f"{name}@{RESOLUTION}"] = _unit(args, ctx, weights, grid_rows, name)
        records += unit_records
    passed = all(s["probas"]["passed"] and s["metrics"]["passed"] for s in sanity.values())
    doc = {"code_revision": revision, "holdout_ids_sha256": ctx["holdout_sha"], "max_iter": pc.MAX_ITER,
           "blas_threads": pc.BLAS_THREADS, "reference_units": str(args.reference_units),
           "sanity": {"passed": passed, "units": sanity}, "summary": pc.summarize(records), "fits": records}
    io.write_json(args.out_dir / OUT_JSON, doc)
    run_environment.write_sidecar(args.out_dir / OUT_JSON, args, blas_threads=pc.BLAS_THREADS)
    (args.out_dir / "README.md").write_text(pc.render_readme(doc), encoding="utf-8")
    s = doc["summary"]
    print(f"-> {args.out_dir / OUT_JSON}: {s['n_hit_max_iter']}/{s['n_fits']} en max_iter, "
          f"{s['n_convergence_warning']} con ConvergenceWarning")
    if not passed:
        print("!!! SANITY FALLO: los refits NO son los publicados bit a bit; el diagnostico no aplica a v7.1. "
              f"Ver sanity en {args.out_dir / OUT_JSON}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
