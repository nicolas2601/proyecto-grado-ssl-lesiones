"""Constants and refusals of E3b (docs/33 section 3, E3b).

E3b applies the variant adopted by E3 (F4: full-image features, features of the lesion
box with a 10 % margin and the standardised mask descriptors) to the eight frozen public
baselines on the v7 HAM10000 split. It reads validation labels only because F4 was
selected on the holdout and pinned: :func:`require_pinned_selection` refuses to run
unless the selection file has the pinned sha256 and names the expected winner. Nothing
is chosen on validation: the probe is fixed (logreg, C = 1, standardised, balanced).
"""

from __future__ import annotations

from pathlib import Path

from dermssl import e3_store as st
from dermssl.finetune import protocol as bank

PINNED_SELECTION_SHA256 = "2446c05a18e6e76d16d9953d89a8d28547cde61a6c25cc64aa60fb92c0563777"
ADOPTED_VARIANT = "F4"
SELECTION_FILE = "resultados/banco_completo/e3/e3_selection.json"
MODELS = bank.MODELS
FEATURE_DIMS = bank.FEATURE_DIMS
CONDITIONS = ("full", "crop_view")  # the parts of F4 (e3_variants.PARTS["F4"])
ARMS = ("full", ADOPTED_VARIANT)
RESOLUTION = bank.RESOLUTION
PROTOCOL = {
    "views": "1 view, resize(int(224 * 1.14)) + centre crop 224, no TTA",
    "normalization": "ImageNet mean/std for every backbone, as the frozen bank",
    "feature": "GAP of the last conv block (CNN), CLS token (ViT)",
    "probe": "logreg C=1, standardised, class_weight=balanced, fitted on the v7 train split",
    "variant": "F4 = concat(full, bbox crop 10 % margin cut from the 224 view) + standardised mask descriptors",
    "masks": "student_r3",
    "bootstrap": {"n_resamples": 1000, "seed": 20260923, "grouped_by": "lesion_id"},
}


class WinnerMismatchError(st.E3Error):
    """The pinned selection does not adopt the variant E3b was written for."""


class FeatureWidthError(st.E3Error):
    """A backbone produced another feature width than the bank protocol fixes (the vgg19 trap)."""


def require_pinned_selection(path, pinned_sha256: str, winner: str = ADOPTED_VARIANT) -> dict:
    """The selection document, or an :class:`~dermssl.e3_store.E3Error`; never reads validation."""
    doc, _ = st.require_selection(Path(path), pinned_sha256)
    if doc.get("decision") != "adopt" or doc.get("winner") != winner:
        raise WinnerMismatchError(
            f"la seleccion fijada dice decision={doc.get('decision')!r}, ganador={doc.get('winner')!r}; "
            f"E3b solo corre con una adopcion de {winner!r}")
    return doc


def check_width(model: str, width: int) -> None:
    expected = FEATURE_DIMS[model]
    if width != expected:
        raise FeatureWidthError(
            f"{model}: {width} dims, el protocolo del banco fija {expected} (GAP de la ultima conv / CLS); "
            "el punto de extraccion cambio y la corrida no seria comparable")
