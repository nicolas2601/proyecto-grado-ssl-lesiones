"""C1: macro-F1 points for B71 and the frozen baselines, from committed JSON."""

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
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thesis_style as style  # noqa: E402
from dermssl.bench import rows as bench_rows  # noqa: E402

STEM = "fig_c1_f1_macro"
E3B_SOURCE = "resultados/banco_completo/e3b_bs64/e3b_summary.json"
DATASETS = ("ham10000", "isic2019", "isic2020")
PROTOCOLS = {
    "ham10000": "E3b canónico, batch 64 + autocast; B71 con TTA; bootstrap por lesion_id",
    "isic2019": "E1 canónico, batch 64 + autocast; StratifiedKFold, n=5.067, sin patient_id; B71 original transductivo",
    "isic2020": "E1 canónico, batch 64 + autocast; partición por patient_id, n=6.660; B71 original transductivo",
}
CAPTIONS = {
    "ham10000": "F1 macro de B71 con TTA canónico frente a ocho baselines congelados re-evaluados en el mismo split v6/v7 (2.004 imágenes).",
    "isic2019": "F1 macro del E1 canónico (batch 64 + autocast; n=5.067): B71 original transductivo y ocho baselines congelados con probe logreg; validación StratifiedKFold sin patient_id.",
    "isic2020": "F1 macro del E1 canónico (batch 64 + autocast; n=6.660): B71 original transductivo y ocho baselines congelados con probe logreg; validación agrupada por patient_id.",
}
NOTE = ("Las barras muestran F1 macro puntual y los bigotes, IC 95 % individuales publicados por cada fuente. "
        "No son intervalos de diferencias ni pruebas pareadas.")


def _read_json(repo: Path, relative: str) -> dict:
    return json.loads((repo / relative).read_text(encoding="utf-8"))


def _source_row(name: str, point: float, ci, support: int, sources: tuple[str, ...]) -> dict:
    return {"name": name, "f1": float(point), "ci": tuple(ci) if ci else None,
            "n": int(support), "sources": list(sources)}


def _ham10000(repo: Path) -> dict:
    doc = _read_json(repo, E3B_SOURCE)
    proposed = doc["proposed"]["rows"]["B71_tta"]
    rows = [_source_row("B71 (TTA, canónico)", proposed["macro_f1"],
                        proposed["ci95_grouped"]["macro_f1"], proposed["n"], (E3B_SOURCE,))]
    for model, name in zip(bench_rows.MODELS, bench_rows.MODEL_NAMES.values(), strict=True):
        record = doc["models"][model]["rows"]["full"]
        rows.append(_source_row(name, record["macro_f1"], record["ci95_grouped"]["macro_f1"],
                                record["n"], (E3B_SOURCE,)))
    return {"title": "HAM10000", "n_val": doc["n"]["val"], "rows": rows,
            "resampling_unit": doc["protocol"]["bootstrap"]["grouped_by"],
            "source_code_revision": doc["code_revision"], "protocol": PROTOCOLS["ham10000"]}


def _isic(repo: Path, dataset: str) -> dict:
    ds = bench_rows.DATASETS[dataset]
    measured = bench_rows.dataset_rows(repo, ds)
    rows_by_name = {row.name: row for row in measured
                    if row.block == bench_rows.FROZEN and row.status == bench_rows.OK}
    names = ["B71 original (canónico)", *bench_rows.MODEL_NAMES.values()]
    missing = [name for name in names if name not in rows_by_name]
    if missing:
        raise ValueError(f"Missing C1 rows for {dataset}: {missing}")
    selected = [rows_by_name[name] for name in names]
    supports = {row.n_val for row in selected}
    if len(supports) != 1 or None in supports:
        raise ValueError(f"C1 validation support differs across {dataset} rows: {supports}")
    rows = [_source_row(row.name, row.values["macro_f1"].point, row.values["macro_f1"].ci,
                        row.n_val, row.sources) for row in selected]
    return {"title": ds.title, "n_val": supports.pop(), "rows": rows,
            "resampling_unit": "image" if dataset == "isic2019" else "patient_id",
            "source_code_revision": None, "protocol": PROTOCOLS[dataset]}


def figure_data(repo: Path = REPO) -> dict:
    repo = Path(repo)
    datasets = {"ham10000": _ham10000(repo)}
    datasets.update({dataset: _isic(repo, dataset) for dataset in DATASETS[1:]})
    for dataset, panel in datasets.items():
        if len(panel["rows"]) != 9:
            raise ValueError(f"C1 requires B71 and eight frozen baselines for {dataset}")
        if any(not 0 <= row["f1"] <= 1 for row in panel["rows"]):
            raise ValueError(f"C1 F1 macro outside [0, 1]: {dataset}")
        if any(row["ci"] is None for row in panel["rows"]):
            raise ValueError(f"C1 source interval missing for {dataset}")
        if any(not row["ci"][0] <= row["f1"] <= row["ci"][1] for row in panel["rows"]):
            raise ValueError(f"C1 source interval does not contain point estimate: {dataset}")
    return {"datasets": datasets, "metric": "macro_f1", "baseline_models": list(bench_rows.MODEL_NAMES.values())}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision(repo: Path, relative: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%H", "--", relative],
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _traceability(repo: Path, dataset: str, panel: dict) -> dict:
    relative_sources = sorted({source for row in panel["rows"] for source in row["sources"]})
    return {
        "figure_id": "C1",
        "dataset": dataset,
        "metric": "macro_f1",
        "support": panel["n_val"],
        "evaluation_unit": "image",
        "resampling_unit": panel["resampling_unit"],
        "protocol": panel["protocol"],
        "source_code_revision": panel["source_code_revision"],
        "figure_script": {"path": "scripts/figures/fig_c1_macro_f1.py",
                          "sha256": _sha256(repo / "scripts/figures/fig_c1_macro_f1.py")},
        "sources": [{"path": relative, "sha256": _sha256(repo / relative),
                     "last_changed_commit": _git_revision(repo, relative)} for relative in relative_sources],
        "rows": panel["rows"],
        "interval_semantics": "individual model intervals; not paired difference intervals",
    }


def _draw(panel: dict):
    rows = panel["rows"]
    height = 2.4 + 0.43 * len(rows)
    with style.context():
        fig, ax = plt.subplots(figsize=(style.FIGURE_WIDTH_IN, height), facecolor=style.COLORS["background"])
        positions = list(range(len(rows)))
        ax.barh(positions, [row["f1"] for row in rows], height=0.62,
                color=[style.COLORS["accent"], *([style.COLORS["bar_light"]] * 8)],
                edgecolor=style.COLORS["accent"], linewidth=style.LINE_WIDTHS["bar"])
        for index, (y, row) in enumerate(zip(positions, rows, strict=True)):
            low, high = row["ci"]
            ax.errorbar(row["f1"], y, xerr=[[row["f1"] - low], [high - row["f1"]]],
                        fmt="none", ecolor=style.COLORS["text"] if index == 0 else style.COLORS["accent"],
                        elinewidth=style.LINE_WIDTHS["curve"],
                        capsize=2, zorder=3)
            ax.annotate(style.fmt_es(row["f1"], 3), (high, y), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=style.LABEL_PT,
                        color=style.COLORS["text"])
        ax.set_yticks(positions, [row["name"] for row in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xticks((0, 0.25, 0.5, 0.75, 1.0), ("0", "0,25", "0,50", "0,75", "1,00"))
        ax.set_xlabel("F1 macro")
        ax.set_title(f"{panel['title']} · n = {style.fmt_es(panel['n_val'], 0)}",
                     loc="left", color=style.COLORS["text"], fontsize=style.PANEL_TITLE_PT, pad=9)
        style.style_axes(ax, "x")
        fig.subplots_adjust(left=0.35, right=0.95, top=0.90, bottom=0.14)
    return fig


def _write_sidecars(repo: Path, target: Path, dataset: str, panel: dict) -> None:
    stem = f"{STEM}_{dataset}"
    trace = _traceability(repo, dataset, panel)
    note = NOTE
    if dataset == "ham10000":
        note += (" Fila VGG19: ΔF1 +0,3331 se conserva según E3b/CONTRASTES_SSL.md; "
                 "pendiente confirmación con Karen por posible medición anterior a 6bd96b7.")
    (target / f"{stem}.caption.txt").write_text(CAPTIONS[dataset] + "\n", encoding="utf-8")
    (target / f"{stem}.note.txt").write_text(note + "\n", encoding="utf-8")
    (target / f"{stem}.traceability.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
                                                         encoding="utf-8")


def build_all(repo: Path = REPO, outdir: Path | None = None) -> dict[str, list[Path]]:
    repo = Path(repo)
    target = Path(outdir) if outdir is not None else repo / "resultados/figuras"
    datasets = figure_data(repo)["datasets"]
    stems = [f"{STEM}_{dataset}" for dataset in DATASETS]
    suffixes = (*style.EXPORT_SUFFIXES, ".caption.txt", ".note.txt", ".traceability.json")
    collisions = [target / f"{stem}{suffix}" for stem in stems for suffix in suffixes
                  if (target / f"{stem}{suffix}").exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite {collisions[0]}")
    target.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for dataset, panel in datasets.items():
        fig = _draw(panel)
        try:
            outputs[dataset] = style.save_figure(fig, f"{STEM}_{dataset}", target)
        finally:
            plt.close(fig)
        _write_sidecars(repo, target, dataset, panel)
    return outputs


if __name__ == "__main__":
    for dataset, paths in build_all().items():
        print(dataset, *(str(path) for path in paths))
