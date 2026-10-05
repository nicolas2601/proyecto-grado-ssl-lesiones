"""The proposed arm (B71, with and without F4) as references for E3b, from E3's feature directory.

E3 already extracted the B71 feature (``B_s42_a1.0@224|cls4``) for ``train`` (identity view)
and ``val`` (8 D4 views) for the conditions ``full`` and ``crop_view``. This module checks those
arrays belong to this run (same ids, same masks, same bytes as their cells say) and builds:

``B71`` / ``B71+F4``          first view only, balanced logreg: the protocol of the baselines;
``B71_tta`` / ``B71+F4_tta``  the registered protocol (probabilities averaged over the 8 D4 views).

The registered arms are checked against ``e3_final_eval.json`` when it is given: if E3b's
pipeline drifted from E3's, the numbers would not match and the run says so before spending
GPU time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dermssl import e3_descriptors as desc
from dermssl import e3_store as st
from dermssl import e3_variants as var
from dermssl import v71_metrics as vm

TOLERANCE = 1e-9
CHECKED = {"B71_tta": "B71", "B71+F4_tta": "F4"}  # our arm -> row name in e3_final_eval.json
SPLITS = ("train", "val")


class E3FeaturesMismatchError(st.E3Error):
    """The E3 feature directory is not the one of this run (ids, masks or bytes differ)."""


def _cell(e3_dir: Path, split: str) -> dict:
    path = Path(e3_dir) / "cells" / f"extract__{split}.json"
    if not path.is_file():
        raise E3FeaturesMismatchError(f"falta la celda de E3 {path}: corre `e3_run.py --stage extract --splits train val`")
    return st.CellStore(path.parent).read(f"extract__{split}")


def verify_e3_features(e3_dir: Path, ids_sha256: dict, masks_sha256: str) -> None:
    """Raise unless E3's train/val cells match the ids, the masks and the bytes on disk."""
    reader = st.FeatureReader(Path(e3_dir) / "features")
    for split in SPLITS:
        cell = _cell(e3_dir, split)
        if cell["key"].get("ids_sha256") != ids_sha256[split]:
            raise E3FeaturesMismatchError(f"{split}: los ids de E3 no son los de esta corrida (ids_sha256)")
        if cell["key"].get("masks_sha256") != masks_sha256:
            raise E3FeaturesMismatchError(f"{split}: las masks de E3 no son las de esta corrida (masks_sha256)")
        for condition in var.PARTS["F4"]:
            if st.file_sha256(reader.path(split, condition)) != cell["result"]["sha256"][condition]:
                raise E3FeaturesMismatchError(f"{split}/{condition}: el sha256 del arreglo no es el de su celda")


def proposed_probas(e3_dir: Path, y_train, n_classes: int, desc_train: np.ndarray, desc_val: np.ndarray) -> dict:
    """``{"B71", "B71+F4", "B71_tta", "B71+F4_tta"} -> (n_val, classes)``."""
    reader = st.FeatureReader(Path(e3_dir) / "features")
    train = {c: np.asarray(reader.load("train", c)) for c in var.PARTS["F4"]}
    val = {c: np.asarray(reader.load("val", c)) for c in var.PARTS["F4"]}
    z_train, z_val, _ = desc.standardize(desc_train, desc_val)
    out = {}
    for suffix, views in (("", slice(0, 1)), ("_tta", slice(None))):
        evaluated = {c: v[views] for c, v in val.items()}
        out[f"B71{suffix}"] = var.variant_proba("B71", {"full": train["full"]}, y_train,
                                                {"full": evaluated["full"]}, n_classes)
        out[f"B71+F4{suffix}"] = var.variant_proba("F4", train, y_train, evaluated, n_classes,
                                                   desc_fit=z_train, desc_eval=z_val)
    return out


def check_against_e3_final(probas: dict, final: dict, y_val, *, mel: int) -> dict:
    """Macro-F1 of the registered arms against the rows of ``e3_final_eval.json``."""
    compared = {}
    for arm, row in CHECKED.items():
        ours = vm.metric_bundle(y_val, probas[arm], mel=mel)["macro_f1"]
        theirs = final["rows"][row]["macro_f1"]
        compared[arm] = {"e3b": ours, "e3_final": theirs, "delta": ours - theirs}
    return {"matches": all(abs(c["delta"]) <= TOLERANCE for c in compared.values()), "compared": compared}
