"""Seed robustness of the pinned E3 F4-vs-B71 comparison (post-hoc, docs/33 errata).

Pure aggregation of the three ``e3_final_eval.json`` (B* seeds 42, 43, 44). Nothing is
recomputed: the paired deltas, their grouped-bootstrap CIs and ``excludes_zero`` are read
as the final stage wrote them. The ``reading`` is a deterministic rule fixed before the
seed 43 and 44 results existed: ``robust`` when the macro-F1 CI excludes 0 in 3 of 3
seeds, ``partial`` in 1 or 2, ``none`` in 0.
"""

from __future__ import annotations

from dermssl import e3_seed as seed_map

ARM, CONTROL, SUBSET = "F4", "B71", "all"
METRICS = ("macro_f1", "balanced_accuracy")
READINGS = {3: "robust", 2: "partial", 1: "partial", 0: "none"}


class RobustnessInputError(ValueError):
    """An input file is not what its seed slot, the pinned selection or the contrast layout requires."""


def reading(n_excluding_zero: int) -> str:
    return READINGS[n_excluding_zero]


def _seed_of(doc: dict) -> int:
    return doc.get("fingerprint", {}).get("b71_seed", seed_map.BASE_SEED)


def _contrast(doc: dict, seed: int, metric: str) -> dict:
    for record in doc["contrasts"]:
        if (record["arm"], record["control"], record["subset"], record["metric"]) == (ARM, CONTROL, SUBSET, metric):
            return {"delta": record["point_delta"], "ci95": [record["ci95_low"], record["ci95_high"]],
                    "excludes_zero": record["excludes_zero"], "direction": record["direction"]}
    raise RobustnessInputError(f"semilla {seed}: sin contraste {ARM}-{CONTROL} de {metric} en el subconjunto {SUBSET!r}")


def _seed_record(doc: dict, seed: int) -> dict:
    rows = doc["rows"]
    return {"weights_sha256": doc["fingerprint"]["weights_sha256"],
            CONTROL: {m: rows[CONTROL][m] for m in METRICS}, ARM: {m: rows[ARM][m] for m in METRICS},
            **{m: _contrast(doc, seed, m) for m in METRICS}}


def _validate(docs: dict, selection_sha256: str | None) -> None:
    if set(docs) != set(seed_map.SEEDS):
        raise RobustnessInputError(f"se esperaban las semillas {seed_map.SEEDS}, llegaron {sorted(docs)}")
    for seed, doc in docs.items():
        if _seed_of(doc) != seed:
            raise RobustnessInputError(f"el archivo de la ranura {seed} es de la semilla {_seed_of(doc)}")
    pins = {doc["selection_sha256"] for doc in docs.values()}
    if len(pins) != 1:
        raise RobustnessInputError(f"los tres archivos no salen de la misma selection: {sorted(pins)}")
    if selection_sha256 is not None and pins != {selection_sha256}:
        raise RobustnessInputError(f"la selection de los archivos no es la pinned ({selection_sha256[:12]})")
    if len({doc["fingerprint"]["weights_sha256"] for doc in docs.values()}) != len(docs):
        raise RobustnessInputError("dos semillas comparten el sha256 de weights: no son pesos distintos")


def summarize(docs: dict[int, dict], inputs_sha256: dict[int, str], *, selection_sha256: str | None = None) -> dict:
    """The robustness document; ``docs`` and ``inputs_sha256`` are keyed by seed."""
    _validate(docs, selection_sha256)
    seeds = {str(seed): _seed_record(docs[seed], seed) for seed in seed_map.SEEDS}
    deltas = [seeds[str(s)]["macro_f1"]["delta"] for s in seed_map.SEEDS]
    n = sum(seeds[str(s)]["macro_f1"]["excludes_zero"] for s in seed_map.SEEDS)
    return {"comparison": f"{ARM} vs {CONTROL}, paired, subset {SUBSET}", "selection_sha256": docs[seed_map.BASE_SEED]["selection_sha256"],
            "seeds": seeds, "delta_macro_f1": {"mean": sum(deltas) / len(deltas), "min": min(deltas), "max": max(deltas)},
            "n_excluding_zero": n, "reading": reading(n),
            "inputs_sha256": {str(s): inputs_sha256[s] for s in seed_map.SEEDS}}
