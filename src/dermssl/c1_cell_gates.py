"""Per-cell gates for the C1 grid: 6 backbones x 4 conditions.

Why a cell needs its own gate
-----------------------------
Both degradations this module watches produce a COMPLETE table of plausible
numbers. Nothing raises, nothing is missing, and the cell quietly stops
measuring what its label says:

* ``mask_pool`` pools the patch tokens inside the mask. When the mask selects
  no patch it falls back to the CLS token -- which is condition (0). Enough
  fallbacks and the cell is the baseline wearing another name, so a contrast
  against (0) compares the baseline with itself and reports a tie.
* ``neutral_bg`` greys everything outside the mask. On an empty mask that is
  the WHOLE image: the backbone sees a flat rectangle and the probe still
  returns a class.

Both are counted per image, reported with numerator AND denominator, and
gated at 1 % per cell (docs/27 §10.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dermssl.mask_conditions import CONDITIONS

#: Above this fraction the cell is no longer measuring its own condition.
MAX_FRACTION_DEGENERATE = 0.01

#: The only condition that pools patch tokens, and so the only one that CAN
#: fall back to CLS. Anywhere else a fallback means the condition being run is
#: not the condition being labelled.
POOLING_CONDITION = "mask_pool"

#: The only condition whose output is destroyed by an empty mask.
GREYING_CONDITION = "neutral_bg"

#: bbox_crop on an empty mask falls back to the BASELINE geometry, so that
#: image is identical to condition (0). Enough of them and the cell is (0)
#: under another name -- the same failure as mask_pool falling back to CLS,
#: and it needs its own gate: nothing else was counting it.
CROPPING_CONDITION = "bbox_crop"


@dataclass
class CellTally:
    """Per-image counters for one (backbone, condition) cell.

    Counts rather than rates, and the denominator travels with them: a rate
    over a denominator nobody states is how a gate ends up measured against
    the wrong population.
    """

    backbone: str
    condition: str
    n_images: int = field(default=0)
    n_cls_fallback: int = field(default=0)
    n_empty_after_geometry: int = field(default=0)
    # Split of n_cls_fallback by owner: the segmenter produced nothing, or the
    # lesion is too small for any 14x14 patch to be selected. Reported, never
    # gated -- the cell degrades the same either way, but which one dominates
    # decides who fixes it.
    n_fallback_from_empty_mask: int = field(default=0)
    n_fallback_from_small_mask: int = field(default=0)
    n_bbox_fallback: int = field(default=0)

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(
                f"condicion {self.condition!r} desconocida; las validas son "
                f"{sorted(CONDITIONS)}"
            )

    def record(
        self,
        *,
        used_cls_fallback: bool,
        empty_after_geometry: bool,
        used_bbox_fallback: bool = False,
    ) -> None:
        """Count one image. Both flags are keyword-only to stop them swapping.

        They are different failures with different owners -- an empty mask is
        the segmenter, a CLS fallback is the pooling -- and two positional
        bools of the same type would swap without a single test noticing.

        In ``mask_pool`` an empty mask MUST fall back: it cannot select a
        patch. Recording otherwise is refused rather than counted, because it
        means the caller is not reporting what the extraction did, and a tally
        that accepts an impossible combination will happily report it later as
        if it had been measured.
        """
        if (
            self.condition == POOLING_CONDITION
            and empty_after_geometry
            and not used_cls_fallback
        ):
            raise ValueError(
                "mascara vacia sin fallback a CLS en mask_pool: una mascara vacia no "
                "puede seleccionar ningun parche, asi que esta combinacion no la produce "
                "la extraccion"
            )

        if used_bbox_fallback and self.condition != CROPPING_CONDITION:
            raise ValueError(
                f"fallback de bbox registrado en {self.condition!r}: solo bbox_crop recorta, "
                "asi que la condicion que corrio no es la que dice la etiqueta"
            )

        self.n_images += 1
        self.n_bbox_fallback += int(used_bbox_fallback)
        self.n_cls_fallback += int(used_cls_fallback)
        self.n_empty_after_geometry += int(empty_after_geometry)

        # The two causes of a fallback, split by owner. Derived from the same
        # two facts rather than passed in: a third flag could disagree with
        # the other two, and then the breakdown would stop adding up to the
        # total without anything failing.
        if used_cls_fallback:
            if empty_after_geometry:
                self.n_fallback_from_empty_mask += 1
            else:
                self.n_fallback_from_small_mask += 1

    @property
    def fraction_cls_fallback(self) -> float | None:
        """``None`` on an empty cell: zero images satisfies any threshold."""
        if not self.n_images:
            return None
        return self.n_cls_fallback / self.n_images

    @property
    def fraction_bbox_fallback(self) -> float | None:
        if not self.n_images:
            return None
        return self.n_bbox_fallback / self.n_images

    @property
    def fraction_empty_after_geometry(self) -> float | None:
        if not self.n_images:
            return None
        return self.n_empty_after_geometry / self.n_images


def evaluate_cell_gates(tally: CellTally) -> tuple[bool, list[str]]:
    """``(passes, reasons)``; every failing condition is reported, not the first.

    Stopping at the first hides how bad the rest is, and the decision
    afterwards needs the whole picture.
    """
    reasons: list[str] = []

    if not tally.n_images:
        reasons.append(
            "cero imagenes en la celda: cualquier umbral se cumple de forma vacua, "
            "asi que esto no es un gate que paso sino uno que no se corrio"
        )
        return False, reasons

    fraction_fallback = tally.fraction_cls_fallback
    if tally.condition == POOLING_CONDITION:
        if fraction_fallback > MAX_FRACTION_DEGENERATE:
            reasons.append(
                f"fallback a CLS en {fraction_fallback:.1%} de las imagenes "
                f"({tally.n_cls_fallback}/{tally.n_images}), por encima del "
                f"{MAX_FRACTION_DEGENERATE:.0%}: la celda esta midiendo la condicion (0) "
                "con otro nombre"
            )
    elif tally.n_cls_fallback:
        reasons.append(
            f"la condicion {tally.condition!r} no agrupa parches y aun asi registro "
            f"{tally.n_cls_fallback} fallbacks a CLS: la condicion que corrio no es la "
            "que dice la etiqueta"
        )

    # Only neutral_bg is gated on empties. In mask_pool an empty mask IS a CLS
    # fallback -- it selects no patch -- so gating both would report one
    # failure as two and make the cell look twice as broken as it is. The
    # count is still reported by summarise_cell; what it is not is a second
    # gate on the same event.
    fraction_empty = tally.fraction_empty_after_geometry
    if tally.condition == GREYING_CONDITION and fraction_empty > MAX_FRACTION_DEGENERATE:
        reasons.append(
            f"mascara vacia tras la geometria en {fraction_empty:.1%} de las imagenes "
            f"({tally.n_empty_after_geometry}/{tally.n_images}): en esas la imagen queda "
            f"gris entera y el probe igual devuelve clase, por encima del "
            f"{MAX_FRACTION_DEGENERATE:.0%}"
        )

    fraction_bbox = tally.fraction_bbox_fallback
    if tally.condition == CROPPING_CONDITION and fraction_bbox > MAX_FRACTION_DEGENERATE:
        reasons.append(
            f"bbox_crop cayo a la geometria base en {fraction_bbox:.1%} de las imagenes "
            f"({tally.n_bbox_fallback}/{tally.n_images}): esas son identicas a la condicion "
            f"(0) y el contraste se aplasta hacia un empate, por encima del "
            f"{MAX_FRACTION_DEGENERATE:.0%}"
        )

    return not reasons, reasons


def summarise_cell(tally: CellTally) -> dict:
    """Everything needed to read the cell later, JSON-serialisable for W&B.

    A count that is structurally impossible is reported as "not applicable"
    with the reason, never as ``0``: a zero looks like a measurement that came
    out clean, and that is how "0 CLS fallbacks" once got read as evidence
    about a condition that cannot produce them.
    """
    passes, reasons = evaluate_cell_gates(tally)
    pools = tally.condition == POOLING_CONDITION

    summary: dict = {
        "backbone": tally.backbone,
        "condition": tally.condition,
        "n_images": tally.n_images,
        "n_empty_after_geometry": tally.n_empty_after_geometry,
        "fraction_empty_after_geometry": tally.fraction_empty_after_geometry,
        "gate_passes": passes,
        "gate_reasons": reasons,
    }
    if tally.condition == CROPPING_CONDITION:
        summary["n_bbox_fallback"] = tally.n_bbox_fallback
        summary["fraction_bbox_fallback"] = tally.fraction_bbox_fallback

    if pools:
        summary["n_cls_fallback"] = tally.n_cls_fallback
        summary["fraction_cls_fallback"] = tally.fraction_cls_fallback
        summary["n_fallback_from_empty_mask"] = tally.n_fallback_from_empty_mask
        summary["n_fallback_from_small_mask"] = tally.n_fallback_from_small_mask
    else:
        summary["cls_fallback"] = (
            f"not_applicable: {tally.condition} no agrupa parches"
        )

    return summary
