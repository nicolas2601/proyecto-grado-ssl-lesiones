"""Diagnostic: which BLAS thread count reproduces the published C1 probas bit for bit?

The lbfgs logreg probe (``probes.fit_logreg``) is not bit-stable across BLAS
thread counts: on the 48-CPU pod, 4 vs 8 threads moved v7.1 macro-F1 by up to
0.014 (resultados/clasificacion/t4/diagnostico_reproduccion/). v7.1 and T4 are
pinned by ``probe_threads``; C1 was not -- its grid fitted in-process with
whatever the library default resolved to, as the published cells were. Its
verdict (default and 4..48 reproduce every checked cell, only 1 differs) is
why C1 now fits under ``PROBE_BLAS_THREADS`` too. This module refits published C1
cells under explicit thread caps and compares each refit with the published
``.proba.npy``. It changes no protocol and rewrites nothing published.
"""

from __future__ import annotations

import numpy as np

DEFAULT = "default"
RECORDED = "recorded"
POOL_KEYS = ("user_api", "internal_api", "num_threads")


def parse_settings(raw, *, recorded: int | None) -> list[tuple[str, int | None]]:
    """``[(label, cap)]``: ``default`` is no cap, ``recorded`` the environment's value, else an int.

    ``recorded`` is dropped when nothing was recorded: a label with no value
    would read as a setting that was tried.
    """
    out = []
    for label in raw:
        if label == DEFAULT:
            out.append((DEFAULT, None))
        elif label == RECORDED:
            if recorded is not None:
                out.append((RECORDED, int(recorded)))
        else:
            try:
                count = int(label)
            except ValueError as exc:
                raise ValueError(f"thread setting {label!r}: expected default, recorded or a positive int") from exc
            if count < 1:
                raise ValueError(f"thread setting {label!r}: must be >= 1")
            out.append((str(count), count))
    return out


def _pools() -> list[dict]:
    from threadpoolctl import threadpool_info  # noqa: PLC0415 - ships with scikit-learn

    return [{k: pool.get(k) for k in POOL_KEYS} for pool in threadpool_info()]


def fit_with_threads(fn, *args, threads: int | None):
    """``(fn(*args), pools)`` with every BLAS/OpenMP pool capped at ``threads``; ``None`` = no cap.

    ``pools`` is what threadpoolctl saw INSIDE the context, so the default
    setting records the count it actually resolved to on this host.
    """
    if threads is None:
        return fn(*args), _pools()
    from threadpoolctl import threadpool_limits  # noqa: PLC0415

    with threadpool_limits(limits=threads):
        return fn(*args), _pools()


def compare_proba(refit, published, *, refit_macro_f1: float, published_macro_f1: float) -> dict:
    """Bit equality (values AND dtype), max |delta|, argmax flips and |delta macro-F1|."""
    refit, published = np.asarray(refit), np.asarray(published)
    if refit.shape != published.shape:
        raise ValueError(f"refit {refit.shape} and published {published.shape} are not the same cell")
    delta = np.abs(refit.astype(np.float64) - published.astype(np.float64))
    return {
        "bit_equal": bool(refit.dtype == published.dtype and np.array_equal(refit, published)),
        "max_abs_delta": float(delta.max()) if delta.size else 0.0,
        "argmax_flips": int((refit.argmax(axis=1) != published.argmax(axis=1)).sum()),
        "abs_delta_macro_f1": abs(float(refit_macro_f1) - float(published_macro_f1)),
        "refit_macro_f1": float(refit_macro_f1),
        "published_macro_f1": float(published_macro_f1),
    }


OUT_JSON = "c1_thread_repro.json"


def _gated(cells) -> list:
    return [c for c in cells if c["gate"]["passed"]]


def _setting_stats(rows) -> dict:
    return {
        "threads": rows[0]["threads"],
        "n_cells": len(rows),
        "n_bit_equal": sum(r["bit_equal"] for r in rows),
        "max_abs_delta": max(r["max_abs_delta"] for r in rows),
        "max_argmax_flips": max(r["argmax_flips"] for r in rows),
        "max_abs_delta_macro_f1": max(r["abs_delta_macro_f1"] for r in rows),
    }


def summarize(cells) -> dict:
    """Per setting over the gated cells, which settings reproduce EVERY cell, and the verdict.

    ``pin_to`` is the smallest explicit count that reproduces: ``default`` is
    never pinned, because what it resolves to depends on the host and the
    environment -- the very thing the pin removes.
    """
    gated = _gated(cells)
    by_setting: dict = {}
    for cell in gated:
        for row in cell["fits"]:
            by_setting.setdefault(row["setting"], []).append(row)
    per_setting = {label: _setting_stats(rows) for label, rows in by_setting.items()}
    reproducing = [label for label, s in per_setting.items() if s["n_bit_equal"] == s["n_cells"] == len(gated)]
    explicit = [per_setting[label]["threads"] for label in reproducing if label != DEFAULT]
    all_passed = bool(cells) and len(gated) == len(cells)
    pin_to = min(explicit) if explicit and all_passed else None
    verdict = "inconclusive" if not all_passed else ("pin" if pin_to is not None else "declare")
    return {"all_gates_passed": all_passed, "n_cells": len(cells), "n_gated": len(gated),
            "per_setting": per_setting, "reproducing_settings": reproducing if all_passed else [],
            "pin_to": pin_to, "verdict": verdict}


DECISION_RULE = (
    "Decision rule (Paula's, stated before the run): if some explicit N reproduces every published cell "
    "bit for bit, C1 gets pinned to that explicit N in a later change and its environment sidecar records "
    "it; if none reproduces, C1 is declared like v7.1, with the measured |delta| below. C1's conclusions "
    "(no improvement, CIs far from 0) do not depend on this; its bit-level reproducibility does."
)


def _verdict_line(s: dict) -> str:
    if s["verdict"] == "pin":
        return f"Verdict: **pin C1 to {s['pin_to']} BLAS threads** (reproduces every cell bit for bit)."
    if s["verdict"] == "declare":
        return ("Verdict: **no thread count reproduces every cell bit for bit**; C1 is declared with the "
                "measured |delta| per setting.")
    return "Verdict: **inconclusive**: at least one cell failed its gate, so no refit here is of the published fit."


def _fmt_threads(row) -> str:
    return "/".join(str(n) for n in dict.fromkeys(row["resolved_threads"])) or "?"


def _yes(flag) -> str:
    return "yes" if flag else "no"


def _control(flag) -> str:
    return "n/a" if flag is None else _yes(flag)


def render_readme(doc: dict) -> str:
    s = doc["summary"]
    rec = doc["recorded_threads"]
    lines = [
        "# Diagnostic: BLAS thread count vs the published C1 probas",
        "",
        f"Generated by `scripts/c1_thread_repro_check.py` from `{OUT_JSON}`; do not edit.",
        "Diagnostic only: no protocol changes and nothing published is refitted in place or replaced.",
        "",
        f"Code revision `{doc['code_revision']}`. Each cell's train and val features are re-extracted with the "
        "C1 grid's own code; a cell is refitted only if its val features hash to the published cell's "
        "`features_sha256` and its split, labels and masks match the cell manifest (the gate).",
        f"Recorded thread setting of the original run: {rec['value'] if rec['value'] is not None else 'none'} "
        f"({rec['source']}).",
        "",
        DECISION_RULE,
        "",
        _verdict_line(s),
        f"Settings reproducing every cell: {', '.join(s['reproducing_settings']) or 'none'}. "
        f"Cells gated: {s['n_gated']} of {s['n_cells']}.",
        "",
        "| setting | threads | cells bit-equal | max abs delta proba | max argmax flips | max abs delta macro-F1 |",
        "|---|---|---|---|---|---|",
    ]
    for label, st in s["per_setting"].items():
        lines.append(f"| {label} | {st['threads'] if st['threads'] is not None else 'no cap'} "
                     f"| {st['n_bit_equal']} of {st['n_cells']} | {st['max_abs_delta']:.3e} "
                     f"| {st['max_argmax_flips']} | {st['max_abs_delta_macro_f1']:.3e} |")
    lines += ["", "| cell | setting | resolved threads | bit-equal | max abs delta proba | argmax flips "
              "| abs delta macro-F1 |", "|---|---|---|---|---|---|---|"]
    for cell in doc["cells"]:
        for r in cell["fits"]:
            lines.append(f"| {cell['name']} | {r['setting']} | {_fmt_threads(r)} | {_yes(r['bit_equal'])} "
                         f"| {r['max_abs_delta']:.3e} | {r['argmax_flips']} | {r['abs_delta_macro_f1']:.3e} |")
    lines += ["", "Per cell (the same-count repeat is the control: if it is not bit-equal, no row above means "
              "anything):", ""]
    lines += [f"- {c['name']}: features from {c.get('features_from')}; cell code revision "
              f"`{c.get('cell_code_revision')}`; same-count repeat bit-equal: **{_control(c.get('repeat_bit_equal'))}**"
              for c in doc["cells"]]
    failed = [c for c in doc["cells"] if not c["gate"]["passed"]]
    if failed:
        lines += ["", "Cells that failed their gate (not refitted):", ""]
        lines += [f"- {c['name']}: {'; '.join(c['gate']['problems'])}" for c in failed]
    return "\n".join(lines) + "\n"
