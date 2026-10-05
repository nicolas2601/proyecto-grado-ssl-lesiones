"""E3 selection on the HAM10000 holdout only, and the registered adoption rule (docs/33 section 3).

Rule, verbatim from the registered plan: a variant is adopted when

* macro-F1 >= (best no-mask variant's macro-F1) - 0.01 on the holdout, AND
* its delta against the reference is > 0 in melanoma recall OR in the macro-F1 of the
  darkest L* tercile (``tone_tercile == "dark"`` of the pinned ``fst_pseudo_manifest.csv``).

The best no-mask row is the maximum over ``nomask`` (default: B71 alone; the v7.1 grid rows
can be added by the caller). The delta is always against ``reference`` (B71, the proposed
model). Among adopted variants the winner is the docs/28 4.1 rule (macro-F1, ECE inside a
0.005 tie). A NaN never counts as a gain. Nothing here can read validation: it receives the
holdout labels and probabilities only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from dermssl import c1_luminance as lum
from dermssl import luminance_sampling as ls
from dermssl import selection_holdout as sh
from dermssl import t4_sog
from dermssl import v71_grid as vg

MACRO_F1_MARGIN = 0.01
REFERENCE = "B71"
MANIFEST_SHA256 = t4_sog.MANIFEST_SHA256  # 57df1e65... registered in docs/31 and docs/33
DARK = t4_sog.TERCILES["dark"]
RULE = ("macro_f1 >= best no-mask macro_f1 - 0.01 on the holdout AND delta > 0 vs the reference in "
        "melanoma recall OR in the macro-F1 of the darkest L* tercile (docs/33 E3)")


class SelectionError(ValueError):
    """The inputs of the selection are inconsistent (missing reference, no rows)."""


def tercile_codes(ids: Sequence[str], manifest, expected_sha256: str = MANIFEST_SHA256) -> np.ndarray:
    """0/1/2 (dark/medium/light) per id, -1 when unavailable; refuses a manifest with another sha256."""
    rows = [{"index": i, "image_id": image_id} for i, image_id in enumerate(ids)]
    return t4_sog.tercile_codes(ls.terciles_by_row(rows, manifest, expected_sha256))


def _dark_macro_f1(y, proba, codes, mel: int, n_classes: int) -> tuple[float, int]:
    dark = np.asarray(codes) == DARK
    if not dark.any():
        return float("nan"), 0
    fns = lum.metric_functions(np.asarray(y)[dark], n_classes=n_classes, mel=mel)
    return float(fns["macro_f1"](np.asarray(y)[dark], np.asarray(proba)[dark])), int(dark.sum())


def score_variants(y, probas: Mapping[str, np.ndarray], codes, *, mel: int, n_resamples: int,
                   ) -> dict[str, dict]:
    """``name -> row``: the v7.1 metric bundle with its CI, plus the dark-tercile macro-F1."""
    rows = {}
    for name, proba in probas.items():
        proba = np.asarray(proba, dtype=float)
        row = vg.score_cell(y, proba, mel=mel, n_resamples=n_resamples)
        row["dark_macro_f1"], row["n_dark"] = _dark_macro_f1(y, proba, codes, mel, proba.shape[1])
        rows[name] = row
    return rows


def _gain(candidate: float, reference: float) -> bool:
    return not (math.isnan(candidate) or math.isnan(reference)) and candidate - reference > 0


def _judge(row: dict, ref: dict, bar: float) -> dict:
    d_mel = float(row["mel_recall"]) - float(ref["mel_recall"])
    d_dark = float(row["dark_macro_f1"]) - float(ref["dark_macro_f1"])
    passes_f1 = round(float(row["macro_f1"]) - bar, 12) >= 0
    passes_delta = _gain(float(row["mel_recall"]), float(ref["mel_recall"])) or \
        _gain(float(row["dark_macro_f1"]), float(ref["dark_macro_f1"]))
    return {"macro_f1": row["macro_f1"], "passes_macro_f1": passes_f1, "delta_mel_recall": d_mel,
            "delta_dark_macro_f1": d_dark, "passes_delta": passes_delta,
            "adopted": bool(passes_f1 and passes_delta)}


def adoption(rows: Mapping[str, dict], *, reference: str = REFERENCE,
             nomask: Sequence[str] = (REFERENCE,), margin: float = MACRO_F1_MARGIN) -> dict:
    """The registered rule over holdout ``rows``; the winner is ``None`` when nothing is adopted."""
    missing = [n for n in (reference, *nomask) if n not in rows]
    if missing:
        raise SelectionError(f"faltan filas de referencia en el holdout: {missing}")
    best = max(nomask, key=lambda n: rows[n]["macro_f1"])
    bar = float(rows[best]["macro_f1"]) - margin
    candidates = [n for n in rows if n not in set(nomask) | {reference}]
    judged = {n: _judge(rows[n], rows[reference], bar) for n in candidates}
    adopted = [n for n in candidates if judged[n]["adopted"]]
    winner = sh.select_winner(dict(rows), candidates=adopted) if adopted else None
    return {"rule": RULE, "reference": reference, "nomask": list(nomask), "best_nomask": best,
            "macro_f1_bar": bar, "variants": judged, "adopted": adopted, "winner": winner,
            "decision": "adopt" if winner else "no_adoption"}
