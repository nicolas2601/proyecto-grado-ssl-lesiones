"""Acceptance gates: the explicit pass/fail contract of a run.

Why this module exists
----------------------
This logic used to live inside the notebook, and that is exactly why it shipped
with a bug: the SSL quality gate averaged the SSL backbone together
with the label-trained baseline it is supposed to beat, so a run where DINOv2 scored
0.8005 was reported as 0.794 and failed a 0.80 threshold it had actually cleared.
Untested code in a notebook cell is still untested code. Moving the gates here
puts them under ``tests/``.

Three states, not two
---------------------
``pass`` / ``fail`` / ``skip``. The third one matters: in ``SMOKE`` mode the
notebook states up front that the numbers are not valid results, so enforcing a
*quality* threshold on them is incoherent. Quality gates are skipped in smoke
runs and enforced in real ones. Pipeline gates (does it run at all, is the split
clean, is the backbone frozen) are enforced always - those are exactly what a
smoke run is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

PASS = "pass"
FAIL = "fail"
SKIP = "skip"

DEFAULT_KNN_AUC_THRESHOLD = 0.80
SSL_PREFIX = "ssl_"


@dataclass
class Gate:
    """One acceptance criterion and its outcome."""

    name: str
    criterion: str
    value: object
    status: str

    @property
    def passed(self) -> bool:
        return self.status == PASS

    @property
    def blocking(self) -> bool:
        """A skipped gate does not block; a failed one does."""
        return self.status == FAIL

    def render(self) -> str:
        label = {PASS: "PASA", FAIL: "FALLA", SKIP: "OMITIDA"}[self.status]
        return f"[{label}] {self.name}: {self.value}  ({self.criterion})"

    def as_dict(self) -> dict:
        return {
            "gate": self.name,
            "criterio": self.criterion,
            "valor": self.value,
            "estado": self.status,
            "pasa": self.passed,
        }


def _decide(ok: bool, *, skip: bool = False) -> str:
    if skip:
        return SKIP
    return PASS if ok else FAIL


def ssl_knn_auc(
    rows: Sequence[dict],
    fraction: float | None = None,
    probe: str = "knn",
    ssl_prefix: str = SSL_PREFIX,
) -> float:
    """Mean k-NN AUC of the **SSL backbones only**.

    The bug this function replaces averaged every backbone together. Mixing the
    label-trained baseline into the SSL score makes the metric answer a
    question nobody asked, and drags it toward the baseline.

    ``fraction=None`` means "the largest label fraction present". Rows with a NaN
    AUC (an undefined class, a failed probe) are ignored rather than propagated.
    """
    candidates = [
        r
        for r in rows
        if str(r.get("backbone", "")).startswith(ssl_prefix)
        and r.get("probe") == probe
        and r.get("auc") == r.get("auc")  # drop NaN
    ]
    if not candidates:
        return float("nan")
    target = max(r["fraction"] for r in candidates) if fraction is None else fraction
    selected = [r["auc"] for r in candidates if r["fraction"] == target]
    if not selected:
        return float("nan")
    return float(sum(selected) / len(selected))


def has_architecture_control(backbones: Iterable[str], ssl_prefix: str = SSL_PREFIX) -> bool:
    """True when a label-trained backbone shares the family of an SSL one.

    Without it, a win by DINOv2 (a ViT) over ResNet-50 (a CNN) cannot be
    attributed to SSL pretraining rather than to the architecture. The check is
    name based: an SSL ViT needs a label-trained ViT alongside it.
    """
    names = list(backbones)
    ssl_vits = [n for n in names if n.startswith(ssl_prefix) and "vit" in n.lower()]
    sup_vits = [n for n in names if not n.startswith(ssl_prefix) and "vit" in n.lower()]
    if ssl_vits:
        return bool(sup_vits)
    return True  # no SSL ViT in the run: nothing to control for


def evaluate_gates(
    *,
    backend: str,
    ham_images: int,
    lesion_overlap: int,
    rows_total: int,
    rows_ok: int,
    frozen_verified: bool,
    backbones: Sequence[str],
    grid_rows: Sequence[dict],
    smoke: bool,
    knn_auc_threshold: float = DEFAULT_KNN_AUC_THRESHOLD,
) -> list[Gate]:
    """Build the full gate report for a run.

    Pipeline gates are enforced in every mode. Quality gates are skipped under
    ``smoke`` because a smoke run deliberately uses a tiny subset.
    """
    auc = ssl_knn_auc(grid_rows)
    gates = [
        Gate(
            "G0 resolucion de datos",
            "backend resuelto y HAM10000 localizado",
            f"{backend} / {ham_images} imgs",
            _decide(bool(backend) and ham_images > 0),
        ),
        Gate(
            "G0 anti-leakage",
            "0 lesiones compartidas train/val",
            lesion_overlap,
            _decide(lesion_overlap == 0),
        ),
        Gate(
            "G1 pipeline end-to-end",
            "toda la rejilla corre sin fallos",
            f"{rows_ok}/{rows_total}",
            _decide(rows_total > 0 and rows_ok == rows_total),
        ),
        Gate(
            "G1 backbone congelado",
            "0 parametros entrenables en todo backbone usado",
            "verificado por assert en la extraccion" if frozen_verified else "NO verificado",
            _decide(frozen_verified),
        ),
        Gate(
            "G2 calidad SSL sin entrenar nada",
            f"AUC del probe kNN del backbone SSL > {knn_auc_threshold}",
            round(auc, 4) if auc == auc else "n/a",
            _decide(auc == auc and auc > knn_auc_threshold, skip=smoke),
        ),
        Gate(
            "G2 control de arquitectura",
            "cada ViT auto-supervisado tiene un ViT supervisado al lado",
            [b for b in backbones if not b.startswith(SSL_PREFIX)] or "ninguno",
            _decide(has_architecture_control(backbones), skip=smoke),
        ),
    ]
    return gates


FAIRNESS_METRIC = "macro_f1"
DEFAULT_FAIRNESS_MAX_GAP = 0.15
DEFAULT_FAIRNESS_MIN_BUCKET_N = 30


def fairness_gate(
    summary: dict,
    *,
    max_gap: float = DEFAULT_FAIRNESS_MAX_GAP,
    min_bucket_n: int = DEFAULT_FAIRNESS_MIN_BUCKET_N,
) -> Gate:
    """G3: how much macro-F1 varies across Fitzpatrick skin-type buckets.

    Standalone by design (D1 in design.md): composes over ``evaluate_gates``
    instead of extending it, so this module stays stdlib-only and its
    signature never changes. Caller concatenates::

        gates = evaluate_gates(...) + [fairness_gate(strat)]

    ``summary`` is the output of ``metrics.stratified_summary`` - this
    function never touches FST/ITA math, only the numbers already computed.
    Mirrors the G2 smoke-skip precedent: fewer than two buckets with at least
    ``min_bucket_n`` samples means there is nothing trustworthy to compare, so
    the gate SKIPs - it never FAILs on thin or absent data (spec.md,
    fairness-evaluation req 3). ``max_gap=0.15`` is the reference target from
    docs/07 and docs/11 ("Brecha de fairness (Fitzpatrick) <= 0.15") - a
    secondary KPI, not the thesis's primary acceptance criterion.
    """
    qualifying = {
        bucket: entry[FAIRNESS_METRIC]
        for bucket, entry in summary.items()
        if entry.get("status") == "OK" and entry.get("count", 0) >= min_bucket_n
    }
    criterion = f"brecha de {FAIRNESS_METRIC} entre fototipos <= {max_gap}"
    if len(qualifying) < 2:
        omitted = sorted(str(bucket) for bucket in summary if bucket not in qualifying)
        return Gate(
            "G3 equidad por fototipo",
            criterion,
            f"menos de 2 fototipos con >= {min_bucket_n} muestras "
            f"(omitidos: {', '.join(omitted) if omitted else 'sin datos'})",
            SKIP,
        )
    gap = max(qualifying.values()) - min(qualifying.values())
    return Gate("G3 equidad por fototipo", criterion, round(gap, 4), _decide(gap <= max_gap))


def report(gates: Sequence[Gate]) -> str:
    """Human-readable block, plus an unambiguous verdict line."""
    lines = ["=== REPORTE DE COMPUERTAS ==="]
    lines.extend(f"  {g.render()}" for g in gates)
    blocking = [g for g in gates if g.blocking]
    skipped = [g for g in gates if g.status == SKIP]
    if blocking:
        lines.append("")
        lines.append(f"HAY {len(blocking)} COMPUERTA(S) EN FALLA - revisar arriba")
    else:
        note = f" ({len(skipped)} omitida(s) por SMOKE)" if skipped else ""
        lines.append("")
        lines.append(f"TODAS LAS COMPUERTAS EXIGIBLES PASAN{note}")
    return "\n".join(lines)


def all_blocking_pass(gates: Sequence[Gate]) -> bool:
    """True when nothing failed. Skipped gates do not block."""
    return not any(g.blocking for g in gates)
