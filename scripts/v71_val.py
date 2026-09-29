"""Stages 2 and 3 of ``v71_run.py``: the per-arm selection and the single val read.

Selection reads only the holdout grid. The val stage reads validation once:
every configuration the selection named (``v71_plan.val_entries``) is fitted
on the full C1 train and scored on the 2004 val images, the holdout operating
thresholds are applied as they are, and each contrast is a paired bootstrap
with the published seed. ``--split holdout`` runs the same code on the holdout
(fit on the 80 %) so the path is rehearsed without touching validation.
"""

from __future__ import annotations

import numpy as np

import v71_io as io
from dermssl import c1_split_identity as si
from dermssl import final_eval as fe
from dermssl import paired_bootstrap as pb
from dermssl import run_environment
from dermssl import run_provenance as rp
from dermssl import v71_grid as vg
from dermssl import v71_metrics as vm
from dermssl import v71_plan as plan
from dermssl import v71_protocol as vp

CONTRAST_METRICS = (*fe.METRICS, "mel_recall", "mel_auc")
V7_ROLES = {"A_v7": "plain", "B_v7": "proposed", "public_v7": "public"}


def _unit_probas(args, cid: str) -> tuple[np.ndarray, np.ndarray]:
    cfg = vp.parse_config_id(cid)
    udir = args.out_dir / "units" / f"{cfg['weights']}@{cfg['resolution']}"
    cells = io.read_json(udir / "result.json")["cells"]
    saved = np.load(udir / "probas.npz")
    return saved["y"], saved[f"p{cells.index(cid)}"].astype(np.float64)


def _metric_fn(metric: str, mel: int):
    if metric in pb.METRIC_FUNCTIONS:
        return pb.METRIC_FUNCTIONS[metric]
    return lambda y, p: vm.metric_bundle(y, p, mel=mel)[metric]


def _contrast(arm, control, y, probas, mel, n_resamples, metrics=CONTRAST_METRICS) -> list[dict]:
    return [pb.contrast_record(arm=arm, control=control, metric=m, probe="logreg", fraction=1.0, seed=0,
                               y_true=y, arm_proba=probas[arm], control_proba=probas[control],
                               n_resamples=n_resamples, metric_fn=_metric_fn(m, mel))
            for m in metrics]


def _resolution_diagnostic(args, ctx, selection, grid) -> dict:
    """§2.3: each winner against itself at the other resolution, paired on the holdout."""
    out = {}
    for arm, (chosen, other) in plan.resolution_pairs(selection, grid["rows"]).items():
        y, chosen_p = _unit_probas(args, chosen)
        probas = {"chosen": chosen_p, "other_resolution": _unit_probas(args, other)[1]}
        out[arm] = {"chosen": chosen, "other_resolution": other,
                    "paired": _contrast("other_resolution", "chosen", y, probas, ctx["mel"],
                                        args.ci_resamples, metrics=("macro_f1", "ece", "mel_recall", "mel_auc"))}
    return out


def stage_select(args, ctx) -> None:
    out = args.out_dir / "selection.json"
    if out.exists():
        raise SystemExit(f"ABORT: {out} ya existe; una seleccion no se reescribe")
    grid = io.read_json(args.out_dir / "holdout_grid.json")
    if grid["holdout_ids_sha256"] != ctx["holdout_sha"]:
        raise SystemExit("ABORT: la grilla es de otro holdout")
    selection = plan.build_selection(grid["rows"])
    contrasts = {}
    for arm in vp.ARMS:
        winner = selection["winners"][arm]
        controls = {"public_same_protocol": vp.with_weights(winner, vp.PUBLIC),
                    "v7_primary": vp.primary_config(arm)}
        probas = {"winner": _unit_probas(args, winner)[1]}
        y = _unit_probas(args, winner)[0]
        rows = []
        for name, cid in controls.items():
            if cid not in grid["rows"]:
                continue
            probas[name] = _unit_probas(args, cid)[1]
            rows += _contrast("winner", name, y, probas, ctx["mel"], args.ci_resamples,
                              metrics=("macro_f1", "ece", "mel_recall", "mel_auc"))
        contrasts[arm] = {"controls": controls, "paired": rows}
    selection.update(code_revision=grid["code_revision"], holdout_ids_sha256=grid["holdout_ids_sha256"],
                     holdout_contrasts=contrasts, resolution_diagnostic=_resolution_diagnostic(args, ctx, selection, grid),
                     note_optimism=("El holdout ya eligio A3/B1 en los pilotos y aca elige entre "
                                    f"{grid['n_configs']} configuraciones: sus numeros tienen optimismo "
                                    "de seleccion; la lectura que vale es la de val (docs/29 §1, §2.7.6)."),
                     environment=run_environment.snapshot(args))
    io.write_json(out, selection)
    for arm, cid in selection["winners"].items():
        print(f"{arm}: {cid} (macro-F1 {selection['winner_rows'][arm]['macro_f1']:.4f})")
    print(f"fase 2 habilitada para proponer: {selection['phase2']['enabled']} -> {out}")


def _guard_val(args, ctx, weights) -> dict:
    """Every precondition of the one val read, checked before any extraction."""
    if args.exclude_ids_file is None:
        raise SystemExit("ABORT: --split val exige --exclude-ids-file")
    runs = weights.run_manifests()
    exclude_sha = rp.file_sha256(args.exclude_ids_file.expanduser())
    try:
        fe.check_val_protocol(ctx["ids"]["train"], ctx["ids"]["val"])
        fe.check_runs_consistent(runs)
    except ValueError as exc:
        raise SystemExit(f"ABORT: {exc}") from exc
    for name, run in runs.items():
        if run.get("exclude_ids_sha256") != exclude_sha:
            raise SystemExit(f"ABORT: {name}: el run.json no uso el exclude {exclude_sha[:12]}")
    return {"exclude_ids_sha256": exclude_sha,
            "runs": {k: {f: r.get(f) for f in ("code_revision", "exclude_ids_sha256", "seed")}
                     for k, r in runs.items()}}


def _fit_subset(parts: dict, ids, subset) -> dict:
    """The rows of ``subset`` inside features extracted for ``ids`` (no second forward)."""
    row = {image_id: k for k, image_id in enumerate(ids)}
    picks = [row[i] for i in subset]
    return {name: value[:, picks] for name, value in parts.items()}


def _score_entries(args, ctx, entries, fit_ids, eval_ids, device, store=None) -> tuple[dict, dict]:
    """``(full, fit80)``: each entry's val probas from the full-train probe and from the 80 % probe.

    ``store`` defaults to identity input; docs/31 passes a Shades of Gray one."""
    from dermssl import torch_v71 as tv  # noqa: PLC0415

    if store is None:
        store = io.ImageStore(args.out_dir / "inputs", ctx["image_index"], args.decode_workers)
    weights = io.WeightBank(args.weights_dir.expanduser())
    tags = ("fit", "holdout") if args.split == "holdout" else ("train", "val")
    fit80_ids = ctx["ids"]["fit"]
    y_fit, y_fit80 = io.labels_of(ctx, fit_ids), io.labels_of(ctx, fit80_ids)
    probas, probas80 = {}, {}
    for (name, res), need in plan.val_units(entries).items():
        model = weights.model(weights.state(name))
        fit_parts = tv.extract_split(model, store.array(fit_ids, res, tags[0]), device=device,
                                     batch_size=args.batch_size)
        eval_parts = tv.extract_split(model, store.array(eval_ids, res, tags[1]), device=device,
                                      batch_size=args.batch_size, n_views=tv.N_D4 if need["tta"] else 1)
        features = sorted(need["features"])
        cells = vg.unit_probas(fit_parts, y_fit, eval_parts, len(ctx["classes"]),
                               features=features, n_jobs=args.probe_jobs)
        cells80 = vg.unit_probas(_fit_subset(fit_parts, fit_ids, fit80_ids), y_fit80, eval_parts,
                                 len(ctx["classes"]), features=features, n_jobs=args.probe_jobs)
        for label, cid in entries.items():
            cfg = vp.parse_config_id(cid)
            if (cfg["weights"], cfg["resolution"]) == (name, res):
                key = (cfg["feature"], cfg["weighting"], cfg["tta"])
                probas[label], probas80[label] = cells[key], cells80[key]
        print(f"[val] {name}@{res} listo")
    return probas, probas80


def _entry_result(y, proba, proba80, cid, grid_rows, mel, n_resamples) -> dict:
    thresholds = grid_rows[cid]["thresholds"]
    return {"config": cid, "holdout_thresholds": thresholds,
            **vm.val_entry_metrics(y, proba_full=proba, proba_threshold=proba80, mel=mel,
                                   thresholds=thresholds, n_resamples=n_resamples)}


def _v7_reproduction(args, results) -> dict:
    if args.split != "val" or not args.v7_reference.exists():
        return {"status": "skipped"}
    ref = io.read_json(args.v7_reference)["results"]
    return {label: {m: results[label][m] - ref[role][m] for m in fe.METRICS}
            for label, role in V7_ROLES.items()}


def stage_val(args, ctx, revision, device) -> None:
    name = "v71_final_eval.json" if args.split == "val" else "v71_final_eval_holdout_rehearsal.json"
    out = args.out_dir / name
    if out.exists():
        raise SystemExit(f"ABORT: {out} ya existe; la evaluacion no se repite")
    selection = io.read_json(args.out_dir / "selection.json")
    if selection["holdout_ids_sha256"] != ctx["holdout_sha"]:
        raise SystemExit("ABORT: la seleccion es de otro holdout")
    guard = _guard_val(args, ctx, io.WeightBank(args.weights_dir.expanduser())) if args.split == "val" else {}
    grid_rows = io.read_json(args.out_dir / "holdout_grid.json")["rows"]
    fit_ids, eval_ids = ((ctx["ids"]["train"], ctx["ids"]["val"]) if args.split == "val"
                         else (ctx["ids"]["fit"], ctx["ids"]["holdout"]))
    entries = plan.val_entries(selection)
    probas, probas80 = _score_entries(args, ctx, entries, fit_ids, eval_ids, device)
    y = io.labels_of(ctx, eval_ids)
    results = {label: _entry_result(y, probas[label], probas80[label], cid, grid_rows, ctx["mel"],
                                    args.ci_resamples)
               for label, cid in entries.items()}
    paired = [row for a, c in plan.VAL_CONTRASTS
              for row in _contrast(a, c, y, probas, ctx["mel"], args.ci_resamples)]
    io.write_json(out, {
        "split": args.split, "code_revision": revision, "probe": "logreg (C1, C=1)",
        "n_fit": len(fit_ids), "n_eval": len(eval_ids), "eval_ids_sha256": si.ids_sha256(eval_ids),
        "holdout_ids_sha256": ctx["holdout_sha"], "grid_code_revision": selection["code_revision"],
        **guard, "entries": entries, "results": results, "paired": paired,
        "selection": {"winners": selection["winners"], "phase2": selection["phase2"]},
        "v7_reproduction_delta": _v7_reproduction(args, results),
        "bootstrap": {"n_resamples": args.ci_resamples, "rng_seed": vm.BOOTSTRAP_SEED, "paired": True},
        "n_fit_threshold_probe": len(ctx["ids"]["fit"]),
        "note_operating_points": plan.OPERATING_POINT_PROTOCOL,
        "environment": run_environment.snapshot(args),
    })
    np.savez_compressed(out.with_suffix(".probas.npz"), y_true=y, **probas,
                        **{f"{label}__fit80": p for label, p in probas80.items()})
    for label in ("A71", "B71"):
        r = results[label]
        print(f"{label} {entries[label]}: macro-F1 {r['macro_f1']:.4f}, mel sens@spec80 "
              f"{r['mel_sens_at_thr_spec80']:.3f}")
    print(f"-> {out}")
