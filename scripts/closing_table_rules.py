"""Verdict rules and row plumbing for resultados/TABLA_CIERRE.md.

Every verdict is one word of a closed vocabulary, decided here by rule from
the committed numbers. A registered item whose data file is not in the repo
renders as pending; nothing is filled in by hand.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]

BETTER, SAME, WORSE = "mejora", "no cambia", "empeora"
MET, NOT_MET = "cumplida", "no cumplida"
PENDING = "pendiente / sin datos en el repo"
# An action with no numeric prediction: the artifact exists, nothing was measured as better.
DONE = "hecho (no es mejora medida)"
# A threshold met in a way that is not an improvement (e.g. coverage above the
# registered floor because the method over-segments); the detail follows.
MEETS_THRESHOLD = "cumple umbral"
# T2's registered middle verdict (docs/27): the interval excludes zero but not
# the -0,02 margin. Only the exploratory rows use it, copied from their JSON.
DISTINGUISHABLE = "distinguible pero no clara"
# Verdict words an exploratory row may copy from its own JSON.
REGISTERED = (MET, NOT_MET, DISTINGUISHABLE)

# docs/27 FASE 1, S4-UNet: Dice(S4) >= Dice(student_r3) - 0,01, registered
# before the run; read on the paired interval of the difference.
S4_MAX_DICE_LOSS = 0.01
_SEED_SUFFIX = re.compile(r"_s\d+$")

# The per-image "FST" of this project is ITA with fixed Groh cuts
# (src/dermssl/fitzpatrick.py) and does not beat a trivial baseline against
# expert FST, so every row that uses it is labelled and annotated by rule.
ITA_LABEL = "estimación ITA no validada (no es FST)"
ITA_NOTE = ("Nota: estimación ITA no validada (no es FST). Contra FST experto acierta 28,5–30,8 % "
            "frente a 32,9 % de la baseline trivial en Fitzpatrick17k-C (docs/20), y en PAD-UFES-20 el "
            "ITA con umbrales fijos acierta 22,9 % frente a 51,3 % de la baseline de mayoría "
            "(Alipour et al. 2026, PMC13459198).")
LUMINANCE_TERCILES = "terciles de luminancia"
_ITA_ESTIMATE = re.compile(r"\bFST\b|fst_|\bITA\b|[Tt]ercil|Fitzpatrick")

Builder = Callable[[dict[Path, Any]], tuple[str, str]]


@dataclass(frozen=True)
class RowSpec:
    origen: str
    pedido: str
    hecho: str
    sources: tuple[Path, ...]
    build: Builder | None
    links: tuple[Path, ...] = ()


@dataclass(frozen=True)
class Row:
    origen: str
    pedido: str
    hecho: str
    resultado: str
    veredicto: str
    fuente: str


def paired_verdict(low: float, high: float, higher_is_better: bool = True) -> str:
    """``mejora``/``empeora`` only when the 95 % paired interval excludes zero."""
    if low > 0:
        return BETTER if higher_is_better else WORSE
    if high < 0:
        return WORSE if higher_is_better else BETTER
    return SAME


def met(ok: bool) -> str:
    return MET if ok else NOT_MET


def s4_threshold_met(dice_delta: dict) -> bool:
    return dice_delta["ci95_low"] >= -S4_MAX_DICE_LOSS


def below_without_overlap(candidate: dict, reference: dict) -> bool:
    return candidate["ci95_high"] < reference["ci95_low"]


def seeds_verdict(verdicts: dict[int, str]) -> str:
    return met(all(v == MET for v in verdicts.values()))


def _arm_family(arm: str) -> tuple[str, str]:
    backbone, condition = arm.split("__")
    return _SEED_SUFFIX.sub("", backbone), condition


def mask_helps(results: list[dict], metric: str = "macro_f1") -> bool:
    """docs/28 §5.2: a condition helps if it beats "sin máscara" with the CI
    above zero in every seed of the same arm (a single run for the public)."""
    families: dict[tuple[str, str], list[bool]] = {}
    for row in results:
        if row["metric"] != metric:
            continue
        families.setdefault(_arm_family(row["arm"]), []).append(row["direction"] == "arm_better")
    return any(all(wins) for wins in families.values())


def uses_ita_estimate(*texts: str) -> bool:
    return any(_ITA_ESTIMATE.search(t) for t in texts)


def with_ita_note(resultado: str, *texts: str) -> str:
    """Append ``ITA_NOTE`` when the row names the per-image estimate."""
    if ITA_NOTE in resultado or not uses_ita_estimate(resultado, *texts):
        return resultado
    return f"{resultado} {ITA_NOTE}"


def _fuente(paths: tuple[Path, ...]) -> str:
    if not paths:
        return "—"
    return "<br>".join(f"[`{p.parent.name}/{p.name}`](../{p.as_posix()})" for p in paths)


def build_row(spec: RowSpec, root: Path = REPO) -> Row:
    present = all((root / p).is_file() for p in spec.sources)
    if spec.build is None or not spec.sources or not present:
        resultado = veredicto = PENDING
    else:
        data = {p: json.loads((root / p).read_text(encoding="utf-8")) for p in spec.sources}
        resultado, veredicto = spec.build(data)
    resultado = with_ita_note(resultado, spec.pedido, spec.hecho)
    return Row(spec.origen, spec.pedido, spec.hecho, resultado, veredicto,
               _fuente(spec.links or spec.sources))
