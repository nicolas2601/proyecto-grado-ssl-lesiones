"""C7: bundled A71/B71 protocol comparison and combined F4 ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thesis_style as style  # noqa: E402

A71_SOURCE = "resultados/clasificacion/v71/v71_final_eval.json"
E3_SOURCE = "resultados/banco_completo/e3/e3_final_eval.json"
E3B_SOURCE = "resultados/banco_completo/e3b_bs64/e3b_summary.json"
ROBUST_SOURCE = "resultados/banco_completo/e3/robustez/e3_robustness/e3_robustness.json"
SEED_SOURCES = {
    42: ("resultados/banco_completo/e3/robustez/e3_s1_bs64_autocast/e3_final_eval.json", "4009c31"),
    43: ("resultados/banco_completo/e3/robustez/e3_rob_s43/e3_final_eval.json", "5a27f44^"),
    44: ("resultados/banco_completo/e3/robustez/e3_rob_s44/e3_final_eval.json", "5a27f44^"),
}
A71_NOTE = ("B71−A71 es una comparación pareada de protocolos agrupados, no un efecto aislado del tono: "
           "A71 usa una sopa de pesos de semillas 42 y 43; B71 usa pesos de semilla 42. "
           "También difieren en extracción (cls_attn vs cls4) y sonda (la0.5 vs balanced); ambos usan TTA. "
           "Bootstrap pareado por imagen (fila), 1.000 remuestras, semilla 20260923.")
F4_NOTE = ("F4 combina recorte de bbox y descriptores de máscara; no equivale solo a segmentación. "
           "El contraste E3 registrado es positivo, pero el IC excluye cero solo en ese cómputo. "
           "La sensibilidad por semillas tiene 0/3 IC que excluyen cero (lectura registrada: none); "
           "por tanto no se afirma mejora robusta. Semilla 42 procede de S1 batch64+autocast; "
           "las diferencias de SHA posteriores corresponden a campos MCC añadidos, con F1/IC idénticos. "
           "No se aíslan recorte ni descriptores por separado.")


def _read_json(repo: Path, relative: str) -> dict:
    return json.loads((repo / relative).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stored_contrast(rows: list[dict], arm: str, control: str, metric: str) -> dict:
    matches = [row for row in rows if (row["arm"], row["control"], row["metric"]) == (arm, control, metric)
               and row.get("subset", "all") == "all"]
    if len(matches) != 1:
        raise ValueError(f"Expected one {arm}-{control} {metric} contrast, got {len(matches)}")
    row = matches[0]
    if row["point_delta"] != row["point_arm"] - row["point_control"]:
        raise ValueError(f"{row}: stored paired direction differs from point values")
    if not row["ci95_low"] <= row["point_delta"] <= row["ci95_high"]:
        raise ValueError(f"{row}: paired interval does not contain point estimate")
    return row


def a71_b71_data(repo: Path = REPO) -> dict:
    root = Path(repo)
    source_path = root / A71_SOURCE
    doc = _read_json(root, A71_SOURCE)
    arms = {}
    for key in ("A71", "B71"):
        row = doc["results"][key]
        low, high = row["ci95"]["macro_f1"]
        point = row["macro_f1"]
        if not low <= point <= high:
            raise ValueError(f"{A71_SOURCE}: {key} macro-F1 outside its CI")
        arms[key] = {"label": key, "config": row["config"], "macro_f1": point,
                     "ci95": (low, high), "n": doc["n_eval"]}
    contrast = _stored_contrast(doc["paired"], "B71", "A71", "macro_f1")
    bootstrap = doc["bootstrap"]
    if not bootstrap["paired"] or bootstrap["n_resamples"] != 1000:
        raise ValueError(f"{A71_SOURCE}: unexpected A71/B71 bootstrap protocol")
    return {"arms": arms, "contrast": {"delta": contrast["point_delta"],
            "ci95": (contrast["ci95_low"], contrast["ci95_high"]), "arm": "B71", "control": "A71"},
            "support": doc["n_eval"], "bootstrap": bootstrap, "resampling_unit": "image",
            "source_hashes": {A71_SOURCE: _sha256(source_path)}}


def _historical_sha256(repo: Path, revision: str, path: str) -> str:
    output = subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{path}"])
    return hashlib.sha256(output).hexdigest()


def _seed_row(repo: Path, seed: int, robustness: dict) -> dict:
    relative, revision = SEED_SOURCES[seed]
    doc = _read_json(repo, relative)
    historical_hash = _historical_sha256(repo, revision, relative)
    if historical_hash != robustness["inputs_sha256"][str(seed)]:
        raise ValueError(f"E3 robustness seed {seed}: historical source hash differs from summary")
    row = robustness["seeds"][str(seed)]
    contrast = _stored_contrast(doc["contrasts"], "F4", "B71", "macro_f1")
    for arm in ("B71", "F4"):
        if doc["rows"][arm]["macro_f1"] != row[arm]["macro_f1"]:
            raise ValueError(f"E3 robustness seed {seed}: {arm} macro-F1 differs from its summary")
    if (contrast["point_delta"] != row["macro_f1"]["delta"]
            or [contrast["ci95_low"], contrast["ci95_high"]] != row["macro_f1"]["ci95"]):
        raise ValueError(f"E3 robustness seed {seed}: F1 contrast differs from its summary")
    return {"seed": seed, "delta": contrast["point_delta"],
            "ci95": (contrast["ci95_low"], contrast["ci95_high"]),
            "b71": row["B71"]["macro_f1"], "f4": row["F4"]["macro_f1"],
            "source": relative, "source_revision": revision,
            "historical_source_sha256": historical_hash, "current_source_sha256": _sha256(repo / relative)}


def f4_robustness_data(repo: Path = REPO) -> dict:
    root = Path(repo)
    e3 = _read_json(root, E3_SOURCE)
    e3b = _read_json(root, E3B_SOURCE)
    robustness = _read_json(root, ROBUST_SOURCE)
    if e3["selection_sha256"] != robustness["selection_sha256"] or e3["selection_sha256"] != e3b["selection_sha256"]:
        raise ValueError("E3 and E3b selection fingerprints do not match robustness summary")
    contrast = _stored_contrast(e3["contrasts"], "F4", "B71", "macro_f1")
    if (contrast["point_arm"] != e3["rows"]["F4"]["macro_f1"]
            or contrast["point_control"] != e3["rows"]["B71"]["macro_f1"]):
        raise ValueError(f"{E3_SOURCE}: paired points differ from the source rows")
    seeds = [_seed_row(root, seed, robustness) for seed in sorted(SEED_SOURCES)]
    excluding_zero = sum(low > 0 or high < 0 for low, high in (row["ci95"] for row in seeds))
    if excluding_zero != robustness["n_excluding_zero"] or robustness["reading"] != "none":
        raise ValueError(f"{ROBUST_SOURCE}: registered seed robustness reading is inconsistent")
    support = e3b["n"]["val"]
    source_hashes = {path: _sha256(root / path) for path in (E3_SOURCE, E3B_SOURCE, ROBUST_SOURCE)}
    source_hashes.update({row["source"]: row["current_source_sha256"] for row in seeds})
    return {"support": support, "registered": {"delta": contrast["point_delta"],
            "ci95": (contrast["ci95_low"], contrast["ci95_high"]), "resampling_unit": e3["bootstrap"]["grouped_by"],
            "n_resamples": e3["bootstrap"]["n_resamples"], "seed": e3["bootstrap"]["seed"]},
            "seeds": seeds, "reading": robustness["reading"],
            "n_excluding_zero": excluding_zero,
            "f4_definition": e3b["protocol"]["variant"], "source_hashes": source_hashes,
            "robustness_source_hashes": robustness["inputs_sha256"]}


def _a71_figure(data: dict):
    arms = (data["arms"]["A71"], data["arms"]["B71"])
    delta = data["contrast"]
    with style.context():
        fig, (ax_f1, ax_delta) = plt.subplots(1, 2, figsize=(style.FIGURE_WIDTH_IN, 4.3))
        for y, arm in enumerate(arms):
            point, (low, high) = arm["macro_f1"], arm["ci95"]
            ax_f1.errorbar(point, y, xerr=[[point - low], [high - point]], marker="o" if y == 0 else "s",
                           markersize=6, color=style.COLORS["accent"], ecolor=style.COLORS["accent"],
                           markerfacecolor=style.COLORS["accent"] if y == 0 else style.COLORS["background"],
                           markeredgewidth=style.LINE_WIDTHS["bar"], elinewidth=style.LINE_WIDTHS["curve"], capsize=3)
        ax_f1.set_xlim(0.6, 0.82)
        ax_f1.set_xticks((0.6, 0.7, 0.8), ("0,6", "0,7", "0,8"))
        ax_f1.set_yticks((0, 1), ("A71\ncls_attn · LA0,5", "B71\nCLS4 · balanced"))
        ax_f1.set_ylim(-0.6, 1.6)
        ax_f1.set_xlabel("F1 macro")
        ax_f1.set_title("(a) F1 macro e IC 95 %", loc="left", fontsize=style.PANEL_TITLE_PT)
        ax_delta.axvline(0, color=style.COLORS["frame"], linewidth=style.LINE_WIDTHS["frame"])
        value = delta["delta"]
        low, high = delta["ci95"]
        ax_delta.errorbar(value, 0, xerr=[[value - low], [high - value]], marker="o", markersize=6,
                          color=style.COLORS["accent"], ecolor=style.COLORS["accent"],
                          elinewidth=style.LINE_WIDTHS["curve"], capsize=3)
        ax_delta.set_xlim(-0.12, 0.12)
        ax_delta.set_xticks((-0.1, 0, 0.1), ("−0,1", "0", "+0,1"))
        ax_delta.set_yticks([])
        ax_delta.set_ylim(-0.6, 0.6)
        ax_delta.set_xlabel("Δ F1 macro (B71 − A71)")
        ax_delta.set_title("(b) Δ F1 pareada", loc="left", fontsize=style.PANEL_TITLE_PT)
        for ax in (ax_f1, ax_delta):
            style.style_axes(ax, "x")
        fig.suptitle(f"A71 y B71 · validación HAM10000 (n = {style.fmt_es(data['support'], 0)})",
                     x=0.08, ha="left", fontsize=style.PANEL_TITLE_PT, fontweight="normal")
        fig.subplots_adjust(left=0.27, right=0.98, top=0.82, bottom=0.22, wspace=0.42)
    return fig


def _f4_figure(data: dict):
    rows = [{"label": "E3 registrado", **data["registered"]},
            *[{"label": f"Semilla {row['seed']}", **row} for row in data["seeds"]]]
    with style.context():
        fig, ax = plt.subplots(figsize=(style.FIGURE_WIDTH_IN, 4.8))
        ax.axvline(0, color=style.COLORS["frame"], linewidth=style.LINE_WIDTHS["frame"], zorder=1)
        for y, row in enumerate(rows):
            value = row["delta"]
            low, high = row["ci95"]
            seed_point = y > 0
            ax.errorbar(value, y, xerr=[[value - low], [high - value]],
                        marker="s" if seed_point else "o", markersize=6,
                        color=style.COLORS["frame"] if seed_point else style.COLORS["accent"],
                        markerfacecolor=style.COLORS["background"] if seed_point else style.COLORS["accent"],
                        markeredgewidth=style.LINE_WIDTHS["bar"],
                        ecolor=style.COLORS["frame"] if seed_point else style.COLORS["accent"],
                        elinewidth=style.LINE_WIDTHS["curve"], capsize=3)
            ax.annotate(style.fmt_es(value, 3, signed=True), (high, y), xytext=(5, 0),
                        textcoords="offset points", va="center", color=style.COLORS["text"])
        ax.set_yticks(range(len(rows)), [row["label"] for row in rows])
        ax.set_ylim(len(rows) - 0.4, -0.7)
        ax.set_xlim(-0.06, 0.12)
        ticks = (-0.04, 0, 0.04, 0.08)
        ax.set_xticks(ticks, [style.fmt_es(tick, 2, signed=True) if tick else "0,00" for tick in ticks])
        ax.set_xlabel("Δ F1 macro (B71+F4 − B71)")
        ax.set_title("E3 · F4 (recorte + descriptores)", loc="left", fontsize=style.PANEL_TITLE_PT)
        style.style_axes(ax, "x")
        fig.subplots_adjust(left=0.30, right=0.98, top=0.9, bottom=0.23)
    return fig


def _write_sidecars(repo: Path, target: Path, stem: str, figure_id: str, data: dict,
                    caption: str, note: str, script: str) -> list[Path]:
    fig = _a71_figure(data) if figure_id == "C7-A71-B71" else _f4_figure(data)
    try:
        paths = style.save_figure(fig, stem, target)
    finally:
        plt.close(fig)
    trace = {"figure_id": figure_id, "status": "Cálculo adicional desde contrastes por imagen",
             "support": data["support"], "evaluation_unit": "image",
             "source_hashes": data["source_hashes"], "script": script,
             "script_sha256": _sha256(repo / script)}
    if figure_id == "C7-A71-B71":
        trace.update({"metric": "macro_f1", "arms": data["arms"], "contrast": data["contrast"],
                      "resampling_unit": data["resampling_unit"], "bootstrap": data["bootstrap"],
                      "causal_scope": "bundled protocol contrast; extraction and probe also differ"})
    else:
        trace.update({"metric": "macro_f1", "registered_contrast": data["registered"],
                      "seed_contrasts": data["seeds"], "seed_reading": data["reading"],
                      "seeds_excluding_zero": data["n_excluding_zero"],
                      "f4_definition": data["f4_definition"],
                      "robustness_input_sha256": data["robustness_source_hashes"],
                      "seed_hash_reconciliation": "s42 matches commit 4009c31; s43/s44 match pre-MCC snapshot 5a27f44^; F1 values unchanged in current JSONs"})
    (target / f"{stem}.traceability.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / f"{stem}.caption.txt").write_text(caption + "\n", encoding="utf-8")
    (target / f"{stem}.note.txt").write_text(note + "\n", encoding="utf-8")
    return paths


def build(repo: Path = REPO, outdir: Path | None = None) -> dict[str, list[Path]]:
    root = Path(repo)
    target = Path(outdir) if outdir is not None else root / "resultados/figuras"
    specs = (("a71_vs_b71", "C7-A71-B71", a71_b71_data, "Comparación pareada A71 frente a B71 en HAM10000; extracción y sonda cambian junto con el protocolo.", A71_NOTE),
             ("b71_vs_b71_f4", "C7-B71-F4", f4_robustness_data, "Contraste pareado de F1 macro B71+F4 frente a B71 y sensibilidad del resultado por semilla.", F4_NOTE))
    suffixes = (*style.EXPORT_SUFFIXES, ".caption.txt", ".note.txt", ".traceability.json")
    collisions = [target / f"fig_c7_{name}{suffix}" for name, *_ in specs for suffix in suffixes
                  if (target / f"fig_c7_{name}{suffix}").exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite {collisions[0]}")
    target.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, figure_id, loader, caption, note in specs:
        data = loader(root)
        outputs[name] = _write_sidecars(root, target, f"fig_c7_{name}", figure_id, data,
                                        caption, note, "scripts/figures/fig_c7_ablations.py")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=REPO / "resultados/figuras")
    args = parser.parse_args()
    for name, paths in build(REPO, args.outdir).items():
        print(name, *(str(path) for path in paths))
