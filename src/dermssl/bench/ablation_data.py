"""Tables of the ablation document (docs/33): every number read from a committed JSON.

v7 and v7.1 on HAM10000, clean vs original, F1-F5 (E3 holdout), F4 sensitivity S0/S1/S2, seed
robustness, and the D0-D3 cells that are still pending. F1-F5 and F4 are exploratory.
"""

from __future__ import annotations

from pathlib import Path

from dermssl.bench import pending
from dermssl.bench.contrast import num
from dermssl.bench.ham_rows import COMPUTES, E3_DIR
from dermssl.bench.contrasts_load import ROBUSTNESS
from dermssl.bench.render import cell
from dermssl.bench.rows import (DATASETS, FROZEN, OK, SENSITIVITY, V71_JSON, V71_NPZ, Value, _load,
                                dataset_rows, v71_derived)
from dermssl.bench.tables import Table

SELECTION = f"{E3_DIR}/e3_selection.json"
V71_KEYS = (("public_v7", "v7: DINOv2 público"), ("A_v7", "v7: A*"), ("B_v7", "v7: B*"),
            ("public71", "v7.1: DINOv2 público, mejor protocolo"), ("public_A71", "v7.1: DINOv2 público, protocolo de A71"),
            ("public_B71", "v7.1: DINOv2 público, protocolo de B71"), ("A71", "v7.1: A71"), ("B71", "v7.1: B71 (propuesto)"))
VARIANTS = (("B71", "Referencia: B71 sin segmentación"),
            ("F1", "concat(features de la imagen completa, features del recorte con margen 10 %)"),
            ("F2", "F1 con el recorte tomado de la resolución original"),
            ("F2_448", "F2 con el recorte a 448 px (`crop_orig448`)"),
            ("F3", "TTA guiada por máscara (imagen completa + 4 recortes)"),
            ("F4", "F1 + descriptores de forma y color de la máscara"),
            ("F5", "F1 + anillo perilesional"))
YES_NO = {True: "sí", False: "no"}
READING = {"none": "ninguna semilla (0 de 3): sin diferencia demostrable", "partial": "parcial (1 o 2 de 3 semillas)",
           "robust": "robusta (3 de 3 semillas)"}


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _ci(rec: dict, key: str) -> str:
    return cell(Value(rec[key], tuple(rec["ci95"][key]) if key in rec.get("ci95", {}) else None))


def version_table(repo: Path) -> Table:
    doc, rows = _load(repo, V71_JSON), []
    for key, label in V71_KEYS:
        values = v71_derived(repo, key)[1]
        rows.append([label, f"`{doc['results'][key]['config']}`", *(cell(values[m]) for m in
                     ("macro_f1", "balanced_accuracy", "auc", "ece", "mcc"))])
    note = (f"HAM10000, val n = {thousands(doc['n_eval'])} (partición v6/v7), probe logreg, fuente `{V71_JSON}` y "
            f"`{V71_NPZ}`. El ECE puntual puede caer fuera de su IC bootstrap; se publica tal cual lo trae la fuente.")
    return Table("v7 y v7.1 en HAM10000", ["Método", "Configuración", "F1 macro", "Exactitud balanceada", "AUC",
                                           "ECE ↓", "MCC"], rows, note, "ablation-v71")


def clean_table(repo: Path) -> Table:
    rows = []
    for ds in DATASETS.values():
        for r in dataset_rows(repo, ds):
            if not r.name.startswith(("B71", "A71")) or "F4" in r.name or r.block not in (FROZEN, SENSITIVITY):
                continue
            ok = r.status == OK
            rows.append([ds.title, r.name, YES_NO[r.transductive], cell(r.values["macro_f1"]) if ok else r.status,
                         num(r.values["mcc"].point) if ok else r.status, r.note])
    note = ("Limpio = pesos re-preentrenados sin las imágenes de val de ISIC en el pool (docs/33, sección 2). "
            f"Lectura de la predicción registrada (B71-limpio frente a B71 original): pendiente; "
            f"{pending.CLEAN_VS_ORIGINAL.note}.")
    return Table("Limpios (E1) frente a originales", ["Dataset", "Método", "Transductivo", "F1 macro [IC 95 %]",
                                                      "MCC", "Nota"], rows, note, "ablation-clean")


def _selection_row(doc: dict, key: str, label: str) -> list[str]:
    rec = doc["rows"][key]
    adopt = doc["adoption"]["variants"].get(key)
    cells = [key, label, _ci(rec, "macro_f1"), num(rec["dark_macro_f1"])]
    if adopt is None:
        return cells + [num(rec["mel_recall"]), "—", "—"]
    return cells + [num(rec["mel_recall"]), YES_NO[adopt["passes_macro_f1"] and adopt["passes_delta"]],
                    YES_NO[adopt["adopted"]]]


def selection_table(repo: Path) -> Table:
    doc = _load(repo, SELECTION)
    adopted = [doc["rows"][k]["macro_f1"] for k in doc["adoption"]["adopted"]]
    note = (f"Selección SOLO en el holdout de HAM. Regla registrada: `{doc['rule']}`. Ganadora por la regla: "
            f"{doc['winner']}; las {len(adopted)} variantes adoptables están dentro de {num(max(adopted) - min(adopted))} "
            "de F1 macro entre sí, con IC solapados: no hay ganadora clara. Exploratorio (docs/33, erratas).")
    rows = [_selection_row(doc, k, label) for k, label in VARIANTS]
    return Table("F1-F5 con segmentación (E3): holdout", ["Variante", "Descripción", "F1 macro holdout [IC 95 %]",
                 "F1 macro, tercil L* oscuro", "Recall de melanoma", "Cumple la regla", "Adoptable"], rows, note,
                 "ablation-e3-holdout")


def val_table(repo: Path) -> Table:
    doc, rows = _load(repo, COMPUTES[0][2]), []
    for key in doc["val_plan"]:
        rec = doc["rows"][key]
        rows.append([key, *(_ci_grouped(rec, m) for m in ("macro_f1", "balanced_accuracy", "mel_recall")),
                     num(rec["auc"]), num(rec["ece"])])
    note = ("Val leído una sola vez, solo para B71 y la ganadora F4 (cómputo S0). F1, F2, F2_448, F3 y F5 no se "
            "evalúan en val por diseño (selección solo en el holdout), no por falta de GPU.")
    return Table("F4 frente a B71 en val (E3, cómputo S0)", ["Variante", "F1 macro", "Exactitud balanceada",
                 "Recall de melanoma", "AUC", "ECE ↓"], rows, note, "ablation-e3-val")


def _ci_grouped(rec: dict, key: str) -> str:
    return cell(Value(rec[key], tuple(rec["ci95_grouped"][key])))


def _pair(repo: Path, rel: str, metric: str) -> dict:
    return next(c for c in _load(repo, rel)["contrasts"] if c["metric"] == metric and c["subset"] == "all")


def _delta(rec: dict) -> str:
    return f"{num(rec['point_delta'], True)} [{num(rec['ci95_low'], True)}; {num(rec['ci95_high'], True)}]"


def sensitivity_table(repo: Path) -> Table:
    rows, excluding, positive = [], 0, 0
    for tag, label, rel in COMPUTES:
        doc, f1 = _load(repo, rel), _pair(repo, rel, "macro_f1")
        excluding, positive = excluding + f1["excludes_zero"], positive + (f1["point_delta"] > 0)
        rows.append([f"{tag}: {label}", num(doc["rows"]["B71"]["macro_f1"]), num(doc["rows"]["F4"]["macro_f1"]),
                     _delta(f1), YES_NO[f1["excludes_zero"]], _delta(_pair(repo, rel, "balanced_accuracy")),
                     _delta(_pair(repo, rel, "mel_recall"))])
    robust = excluding == len(COMPUTES)
    reading = ("mejora marginal robusta al cómputo" if robust else
               "sin diferencia demostrable de forma robusta; la dirección favorece a F4" if positive == len(COMPUTES)
               else "sin diferencia demostrable de forma robusta")
    note = (f"Regla de lectura fijada de antemano: el IC pareado de F1 macro debe excluir 0 en S0, S1 y S2. Lo cumple "
            f"en {excluding} de {len(COMPUTES)}: {reading}. Cómputo canónico: S1.")
    return Table("F4 − B71: sensibilidad al cómputo de extracción (HAM10000, n = 2.004)",
                 ["Cómputo", "B71 F1 macro", "F4 F1 macro", "Δ F1 macro [IC 95 %]", "IC excluye 0",
                  "Δ exactitud balanceada [IC 95 %]", "Δ recall de melanoma [IC 95 %]"], rows, note, "ablation-f4-compute")


def robustness_table(repo: Path) -> Table:
    doc, rows = _load(repo, ROBUSTNESS), []
    for seed, rec in doc["seeds"].items():
        f1, ba = rec["macro_f1"], rec["balanced_accuracy"]
        rows.append([seed, num(rec["B71"]["macro_f1"]), num(rec["F4"]["macro_f1"]),
                     f"{num(f1['delta'], True)} [{num(f1['ci95'][0], True)}; {num(f1['ci95'][1], True)}]",
                     YES_NO[f1["excludes_zero"]], f"{num(ba['delta'], True)} [{num(ba['ci95'][0], True)}; "
                     f"{num(ba['ci95'][1], True)}]", YES_NO[ba["excludes_zero"]]])
    d = doc["delta_macro_f1"]
    favours = sum(r["macro_f1"]["delta"] > 0 for r in doc["seeds"].values())
    note = (f"Robustez de F4 − B71 a la semilla de B* (post-hoc, cómputo canónico). El IC excluye 0 en "
            f"{doc['n_excluding_zero']} de {len(doc['seeds'])} semillas: lectura «{doc['reading']}», "
            f"{READING[doc['reading']]}. Δ medio de F1 macro {num(d['mean'], True)} (mín. {num(d['min'], True)}, máx. "
            f"{num(d['max'], True)}); la dirección favorece a F4 en {favours} de {len(doc['seeds'])} semillas. "
            f"Fuente `{ROBUSTNESS}`.")
    return Table("Robustez de F4 − B71 a la semilla de B*", ["Semilla", "B71 F1 macro", "F4 F1 macro",
                 "Δ F1 macro [IC 95 %]", "IC excluye 0", "Δ exactitud balanceada [IC 95 %]", "IC excluye 0"],
                 rows, note, "ablation-f4-seeds")


def d_table(repo: Path) -> Table:
    split = _load(repo, "resultados/banco_completo/pad_da/target_split.json")
    arms = (("D0", "DINOv2 público"), ("D1", "A* s42 limpio"),
            ("D2", "D1 continuado con el pool limpio (control de cómputo)"),
            ("D3", "D2 más las imágenes de `target_adapt` de PAD, sin etiquetas"))
    rows = [[k, d, "pendiente", pending.E2.reason, pending.E2.eta] for k, d in arms]
    note = (f"E2: probe ajustado en HAM train y aplicado congelado a PAD `target_test` (BCC/MEL/NEV), bootstrap por "
            f"paciente. Primario: D3 − D2. Split de PAD ya generado: `target_adapt` {thousands(split['target_adapt']['n_images'])} "
            f"imágenes, `target_test` {thousands(split['target_test']['n_images'])} imágenes. {pending.ETA_SOURCE}.")
    return Table("D0-D3 (E2): pendientes", ["Brazo", "Descripción", "Estado", "Motivo", "ETA"], rows, note,
                 "ablation-d")


def tables(repo: Path) -> list[Table]:
    return [version_table(repo), clean_table(repo), selection_table(repo), val_table(repo),
            sensitivity_table(repo), robustness_table(repo), d_table(repo)]
