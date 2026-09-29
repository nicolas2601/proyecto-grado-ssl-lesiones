"""docs/30 7: the descriptive second PAD-UFES-20 pass, A71 and seed 44.

Same readings, bootstrap and phototype strata as the first pass
(:mod:`pad_ufes_readings`), computed by its functions, with two differences
the addendum registers: there is no prediction, so no verdict is produced,
and each paired contrast is labelled only by its interval, "excludes 0" or
"indistinguishable" (never "mejora"). :func:`assert_descriptive` refuses a
document that carries anything else.
"""

from __future__ import annotations

import re

import numpy as np

from dermssl import pad_ufes as pu
from dermssl import pad_ufes_readings as pr
from dermssl import v71_protocol as vp

SEED = 44
A71, PUBLIC_A71 = "A71", "public_A71"
#: The v7.1 winner for A (resultados/clasificacion/v71/selection.json) and public on its protocol.
A71_CONFIG = "A_soup_a1.0@224|cls_attn|la0.5|tta"
PUBLIC_A71_CONFIG = vp.with_weights(A71_CONFIG, vp.PUBLIC)
SELECTION_SHA256 = "24e24e4ee6b97717e1bb79ac4e7dfa6447c69c7e40b222a5c40b02cb49af0ec1"
EXCLUDES_ZERO, INDISTINGUISHABLE = "excludes 0", "indistinguishable"
SCOPE = ("docs/30 7: descriptive only. No prediction, and no number here enters any "
         "decision rule. Contrasts are labelled by their paired interval only.")
_FORBIDDEN_KEYS = frozenset({"verdict", "verdicts", "mejora_robusta", "direction"})
_FORBIDDEN = re.compile(r"\bmejora|\bgana|PAD-P|arm_better|control_better", re.IGNORECASE)


def _cell(config: str) -> tuple:
    cfg = vp.parse_config_id(config)
    return cfg["feature"], cfg["weighting"], cfg["tta"]


A71_CELL = _cell(A71_CONFIG)


def roles() -> list[str]:
    return [*pr.roles((SEED,)), A71, PUBLIC_A71]


def contrast_pairs() -> list[tuple[str, str]]:
    return [(A71, PUBLIC_A71), *pr.contrast_pairs((SEED,))]


def describe(rows: list[dict]) -> list[dict]:
    """Each paired row without its direction, labelled by whether its interval excludes 0."""
    return [{**{k: v for k, v in row.items() if k != "direction"},
             "reading": EXCLUDES_ZERO if row["excludes_zero"] else INDISTINGUISHABLE}
            for row in rows]


def evaluate_reading(reading: str, ham_labels, groups, probas7: dict, pairs) -> dict:
    """:func:`pad_ufes_readings.evaluate_reading` without verdicts."""
    mask, y = pr.targets(ham_labels, reading)
    probas = {name: pr.restrict(np.asarray(p)[mask], reading) for name, p in probas7.items()}
    result = pr.evaluate_subset(y, probas, np.asarray(groups)[mask], pairs)
    result["paired"] = describe(result["paired"])
    result["classes"] = list(pu.READINGS[reading])
    result["outside_rate"] = {name: pr.outside_rate(np.asarray(p)[mask], reading)
                              for name, p in probas7.items()}
    return result


def fst_strata(ham_labels, groups, fst_bins, probas7: dict, pairs) -> dict:
    strata = pr.fst_strata(ham_labels, groups, fst_bins, probas7, pairs)
    return {name: {**entry, "paired": describe(entry["paired"])} if "paired" in entry else entry
            for name, entry in strata.items()}


def build_document(probas7: dict, pad: dict, provenance: dict) -> dict:
    pairs = contrast_pairs()
    args = (pad["ham_labels"], pad["groups"])
    document = {
        **provenance,
        "scope": SCOPE,
        "contrasts": [list(pair) for pair in pairs],
        "bootstrap": {**pr.BOOTSTRAP, "paired": True, "resampling_unit": "patient_id"},
        "readings": {name: evaluate_reading(name, *args, probas7, pairs) for name in pu.READINGS},
        "fst_strata": {"reading": "primary", "exploratory": True, "min_n": pr.MIN_BIN_N,
                       "note": pu.fst_note(pu.fst_composition(pad["fst"])),
                       "bins": fst_strata(*args, pad["fst_bins"], probas7, pairs)},
    }
    assert_descriptive(document)
    return document


def _offences(node, path: str = "") -> list[str]:
    if isinstance(node, dict):
        found = [f"{path}/{k}" for k in node if k in _FORBIDDEN_KEYS or _FORBIDDEN.search(str(k))]
        return found + [o for k, v in node.items() for o in _offences(v, f"{path}/{k}")]
    if isinstance(node, (list, tuple)):
        return [o for i, v in enumerate(node) for o in _offences(v, f"{path}[{i}]")]
    if isinstance(node, str) and _FORBIDDEN.search(node):
        return [f"{path}={node!r}"]
    return []


def assert_descriptive(document) -> None:
    """Refuse a verdict, a direction or a "mejora" anywhere in the addendum's output."""
    offences = _offences(document)
    if offences:
        raise ValueError(f"la adenda es solo descriptiva (docs/30 7) y trae: {offences[:5]}")
