"""Stages 2 and 3 of ``t4_sog_run.py`` (docs/31 §6): the SoG winners and the single val read.

Per-input helpers live here too: each input has its own units directory, its
own image store and its own run fingerprint, and nothing reads across them.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import cluster_frozen_bank as bank
import v71_io as io
import v71_val
from dermssl import c1_split_identity as si
from dermssl import final_eval as fe
from dermssl import paired_bootstrap as pb
from dermssl import run_environment
from dermssl import run_provenance as rp
from dermssl import t4_sog as t4
from dermssl import v71_metrics as vm

HOLDOUT_METRICS = ("macro_f1", "ece", "mel_recall", "mel_auc")
V71_LABELS = {"A": "A71", "B": "B71", "public": "public71"}
GAPS = ((t4.GAP_METRIC, t4.gap_dark_light), (t4.RANGE_METRIC, t4.gap_range))


def input_args(args, name: str):
    sub = copy.copy(args)
    sub.out_dir = args.out_dir / name
    return sub


def input_ctx(ctx: dict, name: str) -> dict:
    return {**ctx, "run_fingerprint": t4.run_fingerprint(ctx["ids"]["fit"], ctx["ids"]["holdout"], name)}


def store_for(args, ctx: dict, name: str):
    return io.ImageStore(args.out_dir / "inputs", ctx["image_index"], args.decode_workers, input_name=name)


def unit_probas(out_dir: Path, name: str, cid: str):
    """``(y, proba)`` of one holdout cell of one input."""
    return v71_val._unit_probas(SimpleNamespace(out_dir=Path(out_dir) / name), cid)


def _read_checked(path: Path, ctx: dict) -> dict:
    doc = io.read_json(path)
    if doc["holdout_ids_sha256"] != ctx["holdout_sha"]:
        raise SystemExit(f"ABORT: {path} es de otro holdout")
    return doc


def stage_select(args, ctx) -> None:
    out = args.out_dir / "t4_selection.json"
    if out.exists():
        raise SystemExit(f"ABORT: {out} ya existe; una seleccion no se reescribe")
    grid = _read_checked(args.out_dir / "t4_holdout_grid.json", ctx)
    if t4.SOG not in grid["inputs"]:
        raise SystemExit("ABORT: la grilla es solo identity (corrida de diagnostico); no hay nada que seleccionar")
    winners = t4.sog_winners(grid["inputs"][t4.SOG])
    contrasts = []
    for arm, cid in t4.FIXED_PROTOCOLS.items():
        y, plain = unit_probas(args.out_dir, t4.IDENTITY, cid)
        probas = {f"{arm}_id": plain, f"{arm}_sog": unit_probas(args.out_dir, t4.SOG, cid)[1]}
        contrasts += v71_val._contrast(f"{arm}_sog", f"{arm}_id", y, probas, ctx["mel"], args.ci_resamples,
                                       metrics=HOLDOUT_METRICS)
    io.write_json(out, {
        "rule": "docs/28 4.1 per arm over its 60 SoG cells at 224 (docs/31 6.2)", "winners": winners,
        "winner_rows": {arm: grid["inputs"][t4.SOG][cid] for arm, cid in winners.items()},
        "fixed_protocols": t4.FIXED_PROTOCOLS, "holdout_contrasts": contrasts,
        "code_revision": grid["code_revision"], "holdout_ids_sha256": grid["holdout_ids_sha256"],
        "note_optimism": "el holdout ya eligio los protocolos v7.1; la lectura que vale es la de val",
        "environment": run_environment.snapshot(args)})
    for arm, cid in winners.items():
        print(f"{arm}: ganador SoG {cid}")
    print(f"-> {out}")


def _score(args, ctx, entries, fit_ids, eval_ids, device) -> tuple[dict, dict]:
    probas, probas80 = {}, {}
    for name in t4.INPUTS:
        sub = {label: cid for label, (inp, cid) in entries.items() if inp == name}
        full, fit80 = v71_val._score_entries(input_args(args, name), ctx, sub, fit_ids, eval_ids, device,
                                             store=store_for(args, ctx, name))
        probas.update(full)
        probas80.update(fit80)
    return probas, probas80


def _packed_strata(args, ctx, eval_ids) -> tuple[np.ndarray, dict]:
    rows = bank.load_tone_manifest_rows(args.tone_manifest.expanduser())
    codes = t4.tercile_codes(bank.strata_labels_for(list(eval_ids), rows, "tone_tercile"))
    counts = {name: int((codes == c).sum()) for name, c in t4.TERCILES.items()}
    counts["unavailable"] = int((codes == t4.UNAVAILABLE_CODE).sum())
    return t4.pack_strata(io.labels_of(ctx, eval_ids), codes), counts


def _paired(y, packed, probas, mel, n_resamples) -> list[dict]:
    rows = []
    for arm, control in (*t4.PRIMARY_CONTRASTS, *t4.SECONDARY_CONTRASTS):
        rows += v71_val._contrast(arm, control, y, probas, mel, n_resamples)
        rows += [pb.contrast_record(arm=arm, control=control, metric=metric, probe="logreg", fraction=1.0, seed=0,
                                    y_true=packed, arm_proba=probas[arm], control_proba=probas[control],
                                    n_resamples=n_resamples, rng_seed=vm.BOOTSTRAP_SEED, metric_fn=fn)
                 for metric, fn in GAPS]
    return rows


def _v71_reproduction(args, results) -> dict:
    if args.split != "val" or not args.v71_final.exists():
        return {"status": "skipped"}
    ref = io.read_json(args.v71_final)["results"]
    return {arm: {m: results[f"{arm}_id"][m] - ref[label][m] for m in fe.METRICS}
            for arm, label in V71_LABELS.items()}


def stage_val(args, ctx, revision, device, registration) -> None:
    name = "t4_final_eval.json" if args.split == "val" else "t4_final_eval_holdout_rehearsal.json"
    out = args.out_dir / name
    if out.exists():
        raise SystemExit(f"ABORT: {out} ya existe; la evaluacion no se repite")
    selection = _read_checked(args.out_dir / "t4_selection.json", ctx)
    guard = v71_val._guard_val(args, ctx, io.WeightBank(args.weights_dir.expanduser())) if args.split == "val" else {}
    grid = io.read_json(args.out_dir / "t4_holdout_grid.json")
    fit_ids, eval_ids = ((ctx["ids"]["train"], ctx["ids"]["val"]) if args.split == "val"
                         else (ctx["ids"]["fit"], ctx["ids"]["holdout"]))
    entries = t4.val_entries(selection["winners"])
    probas, probas80 = _score(args, ctx, entries, fit_ids, eval_ids, device)
    y = io.labels_of(ctx, eval_ids)
    packed, counts = _packed_strata(args, ctx, eval_ids)
    results = {label: v71_val._entry_result(y, probas[label], probas80[label], cid, grid["inputs"][inp],
                                            ctx["mel"], args.ci_resamples)
               for label, (inp, cid) in entries.items()}
    paired = _paired(y, packed, probas, ctx["mel"], args.ci_resamples)
    io.write_json(out, {
        "split": args.split, "code_revision": revision, "registration_commit": registration,
        "probe": "logreg (C1, C=1)", "sog_p": t4.SOG_P, "n_fit": len(fit_ids), "n_eval": len(eval_ids),
        "eval_ids_sha256": si.ids_sha256(eval_ids), "holdout_ids_sha256": ctx["holdout_sha"],
        "tone_manifest_sha256": rp.file_sha256(args.tone_manifest.expanduser()), "tercile_counts": counts,
        **guard, "entries": {k: {"input": i, "config": c} for k, (i, c) in entries.items()},
        "results": results, "terciles": {label: t4.tercile_macro_f1(packed, p) for label, p in probas.items()},
        "paired": paired, "verdicts": t4.verdicts(paired), "v71_reproduction_delta": _v71_reproduction(args, results),
        "bootstrap": {"n_resamples": args.ci_resamples, "rng_seed": vm.BOOTSTRAP_SEED, "paired": True},
        "environment": run_environment.snapshot(args)})
    np.savez_compressed(out.with_suffix(".probas.npz"), y_true=y, tercile=packed[:, 1], **probas)
    for key, verdict in t4.verdicts(paired).items():
        print(f"{key}: {'se cumple' if verdict['holds'] else 'no se cumple'}")
    print(f"-> {out}")
