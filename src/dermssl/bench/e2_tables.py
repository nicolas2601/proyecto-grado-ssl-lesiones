"""E2 (docs/33): D0-D3 on PAD ``target_test``, read from ``pad_da/pad_e2_eval.json``; nothing is typed in.

The probe is fitted on HAM10000 (source labels) and applied FROZEN to PAD ``target_test`` (BCC/MEL/NV).
``target_test`` was read once (lock). D3 - D2 is the registered primary prediction.
"""

from __future__ import annotations

from pathlib import Path

from dermssl.bench.contrast import Contrast, compact_cells, COMPACT_HEADER, from_record, interval, num
from dermssl.bench.render import cell
from dermssl.bench.rows import Value, _load
from dermssl.bench.tables import Table

E2_JSON = "resultados/banco_completo/pad_da/pad_e2_eval.json"
LOCK = "resultados/banco_completo/pad_da/target_test_read.lock"
ARMS = (("D0", "DINOv2 público"), ("D1", "A* s42 limpio"),
        ("D2", "D1 continuado con el pool limpio (control de cómputo)"),
        ("D3", "D2 más las imágenes de `target_adapt` de PAD, sin etiquetas"))
METRICS = ("macro_f1", "balanced_accuracy", "accuracy", "auc", "ece", "mcc")
LABEL = "`target_test` de PAD leído una sola vez (lock)"


def _thousands(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _value(arm: dict, metric: str) -> Value:
    ci = arm.get("ci95", {}).get(metric)
    return Value(arm[metric], tuple(ci) if ci else None)


def exists(repo: Path) -> bool:
    return (repo / E2_JSON).is_file()


def method_note(doc: dict, repo: Path) -> str:
    boot, comp = doc["bootstrap"], doc["compute"]
    lock = "con lock presente" if (repo / LOCK).is_file() else "SIN lock en el repositorio"
    mel = doc["arms"]["D0"]["per_class"]["mel"]["support"]
    return (f"{LABEL} ({lock}). Probe `{doc['probe']}` ajustado en HAM10000 (n = {_thousands(doc['n_fit'])}) y aplicado "
            f"congelado; el backbone nunca ve etiquetas. Cómputo canónico: batch {comp['batch_size']} + autocast "
            f"{comp['autocast']}. Bootstrap pareado de {_thousands(boot['n_resamples'])} remuestras, semilla "
            f"{boot['rng_seed']}, agrupado por `{boot['unit']}`. Lectura «{doc['reading']}» "
            f"({'/'.join(c.upper() for c in doc['classes'])}): {_thousands(doc['n_images'])} imágenes de "
            f"{_thousands(doc['n_patients'])} pacientes (de {_thousands(doc['n_target_test_images'])} imágenes de "
            f"`target_test`), solo {mel} melanomas: los IC son anchos. «Fuera de rango» = proporción de imágenes "
            "cuyo argmax de 7 clases cae fuera de las clases de la lectura (descriptivo). Fuente: "
            f"`{E2_JSON}`.")


def arms_table(repo: Path) -> Table:
    doc = _load(repo, E2_JSON)
    rows = [[k, desc, *(cell(_value(doc["arms"][k], m)) for m in METRICS), num(doc["outside_rate"][k])]
            for k, desc in ARMS]
    header = ["Brazo", "Descripción", "F1 macro", "Exactitud balanceada", "Accuracy", "AUC", "ECE ↓", "MCC",
              "Fuera de rango"]
    return Table("D0-D3 en PAD `target_test` (E2)", header, rows, method_note(doc, repo), "e2-arms")


def contrasts(repo: Path) -> list[Contrast]:
    doc = _load(repo, E2_JSON)
    unit = doc["bootstrap"]["unit"]
    return [from_record(r, arm=r["arm"], control=r["control"], unit=unit, source=E2_JSON) for r in doc["paired"]]


def primary_line(repo: Path) -> str:
    doc = _load(repo, E2_JSON)
    p = doc["primary_contrast"]
    met = "sí" if doc["prediction_d3_minus_d2_met"] else "no"
    return (f"Predicción primaria registrada (D3 − D2, F1 macro > 0 con IC que excluye 0): cumplida = {met} "
            f"(`prediction_d3_minus_d2_met` del JSON); Δ {interval(from_record(p, arm='D3', control='D2', unit='', source=''))}.")


def contrasts_table(repo: Path) -> Table:
    items = contrasts(repo)
    order = [("D3", "D2"), ("D2", "D1"), ("D3", "D1"), ("D1", "D0"), ("D2", "D0"), ("D3", "D0")]
    ranked = [c for pair in order for c in items if (c.arm, c.control) == pair]
    note = ("D3 − D2 es la predicción primaria registrada; los demás son descriptivos. " + primary_line(repo))
    return Table("D0-D3 (E2): contrastes pareados", COMPACT_HEADER, [compact_cells(c) for c in ranked], note,
                 "e2-contrasts")
