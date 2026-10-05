"""Contrasts-only rebuild of an E3b summary from its committed probabilities (CPU, no weights, no images).

The summaries committed before 1d7e447 hold paired contrasts for three metrics. The current code adds
MCC (always LAST in the metric dict, so the bootstrap draws and the values of the other metrics do not
move). This module recomputes the contrasts with the SAME code path as the run
(:func:`dermssl.e3b.contrasts.paired`, no second bootstrap) and refuses to write unless every
pre-existing record is reproduced identically: the MCC records are only added.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from dermssl import e3_store as st
from dermssl import feature_cache as fc
from dermssl import run_environment as env
from dermssl.e3b import contrasts

SUMMARY_FILE, PROBAS_FILE, PROBAS_MANIFEST = "e3b_summary.json", "e3b_probas.npz", "e3b_probas.json"
NOT_ARMS = ("y_val", "val_ids")
MCC = "mcc"


class RebuildMismatchError(st.E3Error):
    """The recomputed contrasts do not reproduce the committed ones, or an input is not the one the run used."""


def groups_from_metadata(csv_path: Path, ids: Sequence[str]) -> np.ndarray:
    """``lesion_id`` of every id, from ``HAM10000_metadata.csv`` (the grouping unit of the run's bootstrap)."""
    import pandas as pd  # noqa: PLC0415

    lesion = pd.read_csv(csv_path).set_index("image_id")["lesion_id"]
    missing = [i for i in ids if i not in lesion.index]
    if missing:
        raise RebuildMismatchError(f"{len(missing)} ids without lesion_id in {csv_path}, e.g. {missing[:3]}")
    return np.array([lesion[i] for i in ids])


def rebuild_records(y, probas: dict, groups, models: Sequence[str], *, classes: Sequence[str],
                    n_resamples: int) -> list[dict]:
    """The records of the run's own ``contrasts._contrasts``: same pairs, same order, same bootstrap."""
    pairs = [pair for m in models for pair in contrasts.pairs_for(m)]
    return contrasts.paired(y, probas, pairs, groups, n_classes=len(classes), mel=classes.index("mel"),
                            n_resamples=n_resamples)


def _key(record: dict) -> tuple:
    return record["arm"], record["control"], record["metric"]


def verify_preserved(old: list[dict], new: list[dict]) -> dict:
    """Every non-MCC record of ``new`` must equal ``old`` exactly (all fields, bit for bit), in order.

    An ``old`` that already carries MCC (a summary rebuilt before) must be reproduced in full.
    """
    kept = [r for r in new if r["metric"] != MCC]
    old_kept = [r for r in old if r["metric"] != MCC]
    if len(kept) != len(old_kept):
        raise RebuildMismatchError(f"{len(old_kept)} committed records vs {len(kept)} recomputed")
    bad = [_key(a) for a, b in zip(old_kept, kept, strict=True) if a != b]
    if bad:
        raise RebuildMismatchError(f"{len(bad)} of {len(old_kept)} records differ from the committed ones, "
                                   f"first: {bad[0]} (macro_f1/balanced_accuracy/mel_recall must be identical)")
    if len(old_kept) != len(old) and old != new:
        raise RebuildMismatchError("the committed MCC records differ from the recomputed ones")
    return {"compared": len(old_kept), "equal": len(old_kept), "mcc_added": sum(r["metric"] == MCC for r in new)}


def _load_inputs(directory: Path) -> tuple[dict, dict, dict]:
    path, manifest = directory / PROBAS_FILE, json.loads((directory / PROBAS_MANIFEST).read_text(encoding="utf-8"))
    if st.file_sha256(path) != manifest["sha256"]:
        raise RebuildMismatchError(f"{path.name} does not match the sha256 of its manifest")
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    if fc.fingerprint_ids(list(arrays["val_ids"])) != manifest["val_ids_sha256"]:
        raise RebuildMismatchError("val_ids do not match the manifest fingerprint")
    summary = json.loads((directory / SUMMARY_FILE).read_text(encoding="utf-8"))
    return summary, arrays, manifest


def _block(report: dict, manifest: dict, meta_sha256: str, n_resamples: int) -> dict:
    return {"what": "paired contrasts recomputed from e3b_probas.npz with the current code (MCC added); every "
                    "previous record reproduced identically", "records_reproduced": report["equal"],
            "mcc_records_added": report["mcc_added"], "n_resamples": n_resamples,
            "probas_sha256": manifest["sha256"], "groups_metadata_sha256": meta_sha256,
            "groups": "lesion_id from HAM10000_metadata.csv"}


def rebuild_dir(directory: Path, metadata_csv: Path, *, n_resamples: int = 1000, mel_name: str = "mel",
                models: Sequence[str] | None = None) -> dict:
    """Rewrite ``contrasts`` of ``<directory>/e3b_summary.json`` with MCC and refresh its sidecar.

    Nothing is written unless :func:`verify_preserved` passes. Returns its report.
    """
    directory, metadata_csv = Path(directory), Path(metadata_csv)
    summary, arrays, manifest = _load_inputs(directory)
    classes = list(summary["classes"])
    if mel_name not in classes:
        raise RebuildMismatchError(f"class {mel_name!r} not in {classes}")
    probas = {k: v for k, v in arrays.items() if k not in NOT_ARMS}
    groups = groups_from_metadata(metadata_csv, [str(i) for i in arrays["val_ids"]])
    new = rebuild_records(arrays["y_val"], probas, groups, list(models or summary["requested_models"]),
                          classes=classes, n_resamples=n_resamples)
    report = verify_preserved(summary["contrasts"], new)
    meta_sha = hashlib.sha256(metadata_csv.read_bytes()).hexdigest()
    path = directory / SUMMARY_FILE
    st.write_json_atomic(path, {**summary, "contrasts": new,
                                "contrasts_rebuild": _block(report, manifest, meta_sha, n_resamples)})
    env.write_sidecar(path)
    return report
