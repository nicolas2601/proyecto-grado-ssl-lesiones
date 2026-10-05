"""B71 reference cell, paired contrasts and the final summary / probabilities of E3b."""

from __future__ import annotations

import json

import numpy as np

from dermssl import e3_store as st
from dermssl import feature_cache as fc
from dermssl.e3b import b71, contrasts, scoring
from dermssl.e3b import protocol as proto
from dermssl.e3b.pipeline import SPLITS, base_key, ids_key, proba_path, run_cell
from dermssl.e3b.runtime import Runtime, Settings

B71_ARMS = ("B71", "B71+F4", "B71_tta", "B71+F4_tta")
SUMMARY_FILE, PROBAS_FILE, PROBAS_MANIFEST = "e3b_summary.json", "e3b_probas.npz", "e3b_probas.json"


class B71CheckError(st.E3Error):
    """The proposed arm recomputed here does not reproduce e3_final_eval.json."""


def b71_cell(cells, rt: Runtime, settings: Settings, sidecar) -> str:
    """The four B71 arms on val from E3's features; ``skipped-smoke`` when ``--limit`` is on."""
    if settings.smoke or settings.e3_dir is None:
        return "no-e3-dir" if settings.e3_dir is None else "skipped-smoke"
    key = {"stage": "e3b_b71", "n_resamples": settings.n_resamples, **base_key(rt, settings),
           "e3_cells": {s: cells_sha(settings, s) for s in SPLITS},
           "descriptor_sha256": {s: cells.read(f"e3b__descriptors__{s}")["result"]["sha256"] for s in SPLITS}}
    return run_cell(cells, "e3b__b71", key, lambda: _b71(rt, settings), sidecar, artifact=proba_path(settings, "B71"))


def cells_sha(settings: Settings, split: str) -> str:
    return st.file_sha256(settings.e3_dir / "cells" / f"extract__{split}.json")


def _b71(rt: Runtime, settings: Settings) -> dict:
    b71.verify_e3_features(settings.e3_dir, ids_key(rt), rt.fingerprint["masks_sha256"])
    feats = settings.out_dir / "features"
    probas = b71.proposed_probas(settings.e3_dir, rt.labels_of("train"), rt.n_classes,
                                 np.load(feats / "train__descriptors.npy"), np.load(feats / "val__descriptors.npy"))
    check = _check(settings, probas, rt)
    path = proba_path(settings, "B71")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **probas)
    y, groups = rt.labels_of("val"), rt.groups_of("val")
    rows = {arm: scoring.score_row(y, p, groups, rt.classes, mel=rt.mel, n_resamples=settings.n_resamples)
            for arm, p in probas.items()}
    return {"rows": rows, "check_against_e3_final": check, "probas_sha256": st.file_sha256(path)}


def _check(settings: Settings, probas: dict, rt: Runtime) -> dict | None:
    final = settings.e3_dir / "e3_final_eval.json"
    if not final.is_file():
        return None
    check = b71.check_against_e3_final(probas, json.loads(final.read_text()), rt.labels_of("val"), mel=rt.mel)
    if not check["matches"]:
        raise B71CheckError(f"las filas B71 registradas no reproducen e3_final_eval.json: {check['compared']}")
    return check


def _arm_probas(settings: Settings, models: list[str]) -> dict:
    """Probabilities of every arm on val; the B71 arms join when their file exists (not in a smoke run)."""
    out = {}
    for model in models:
        with np.load(proba_path(settings, model)) as data:
            out.update({model: data["full"], f"{model}+F4": data["F4"]})
    if proba_path(settings, "B71").is_file():
        with np.load(proba_path(settings, "B71")) as data:
            out.update({arm: data[arm] for arm in B71_ARMS})
    return out


def contrasts_cell(cells, rt: Runtime, settings: Settings, models: list[str], sidecar) -> list[dict]:
    """Paired contrasts of every finished baseline; ``[]`` when B71 (or the baselines) are missing."""
    if not models or not cells.path("e3b__b71").is_file():
        return []
    shas = {m: cells.read(f"e3b__eval__{m}")["result"]["probas_sha256"] for m in models}
    shas["B71"] = cells.read("e3b__b71")["result"]["probas_sha256"]
    key = {"stage": "e3b_contrasts", "probas_sha256": shas, "n_resamples": settings.n_resamples, **base_key(rt, settings)}
    run_cell(cells, "e3b__contrasts", key, lambda: _contrasts(rt, settings, models), sidecar)
    return cells.read("e3b__contrasts")["result"]["records"]


def _contrasts(rt: Runtime, settings: Settings, models: list[str]) -> dict:
    pairs = [pair for m in models for pair in contrasts.pairs_for(m)]
    records = contrasts.paired(rt.labels_of("val"), _arm_probas(settings, models), pairs, rt.groups_of("val"),
                               n_classes=rt.n_classes, mel=rt.mel, n_resamples=settings.n_resamples)
    return {"records": records}


def write_outputs(cells, rt: Runtime, settings: Settings, models: list[str], failures, sidecar) -> dict:
    """``e3b_summary.json`` (for the table builder) and ``e3b_probas.npz``, each with its sidecar."""
    done = [m for m in models if cells.path(f"e3b__eval__{m}").is_file()]
    records = contrasts_cell(cells, rt, settings, done, sidecar)
    summary = _summary(cells, rt, settings, models, done, failures, records)
    path = settings.out_dir / SUMMARY_FILE
    st.write_json_atomic(path, summary)
    sidecar(path)
    if done:
        _write_probas(rt, settings, done, sidecar)
    return summary


def _summary(cells, rt, settings, models, done, failures, records) -> dict:
    b71_doc = cells.read("e3b__b71")["result"] if cells.path("e3b__b71").is_file() else None
    results = {m: cells.read(f"e3b__eval__{m}")["result"] for m in done}
    return {
        "stage": "e3b", "winner": proto.ADOPTED_VARIANT, "selection_sha256": settings.selection_sha256,
        "protocol": proto.PROTOCOL, "smoke": settings.smoke, "code_revision": settings.revision,
        "fingerprint": rt.fingerprint, "classes": list(rt.classes), "n": {s: len(rt.ids[s]) for s in SPLITS},
        "models": {m: {k: v for k, v in r.items() if k not in ("model", "probas_sha256")} for m, r in results.items()},
        "proposed": None if b71_doc is None else {k: b71_doc[k] for k in ("rows", "check_against_e3_final")},
        "contrasts": records, "requested_models": list(models),
        "failures": [{"cell": n, "error": e} for n, e in failures],
        "complete": not failures and len(done) == len(models) and (settings.smoke or b71_doc is not None),
    }


def _write_probas(rt: Runtime, settings: Settings, done: list[str], sidecar) -> None:
    """The npz plus ``e3b_probas.json``: its sha256 and shapes, the file the sidecar (JSON only) describes."""
    path = settings.out_dir / PROBAS_FILE
    arms = _arm_probas(settings, done)
    np.savez(path, y_val=rt.labels_of("val"), val_ids=np.array(rt.ids["val"]), **arms)
    manifest = settings.out_dir / PROBAS_MANIFEST
    st.write_json_atomic(manifest, {"file": PROBAS_FILE, "sha256": st.file_sha256(path),
                                    "arrays": {name: list(p.shape) for name, p in arms.items()},
                                    "val_ids_sha256": fc.fingerprint_ids(rt.ids["val"])})
    sidecar(manifest)
