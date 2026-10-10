"""C2: paired macro-F1 differences from committed contrast artifacts only."""

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
from dermssl.bench import contrasts_load, rows as bench_rows  # noqa: E402
from dermssl.isic_eval import scoring as isic_scoring  # noqa: E402

STEM = "fig_c2_paired_delta_f1"
E3B_SOURCE = contrasts_load.E3B_DIRS["canónico (batch 64)"]
ISIC_VARIANT = "canónico (batch 64 + autocast)"
CAPTION = ("C2: diferencias pareadas de F1 macro entre B71 y los controles con predicciones sobre las mismas imágenes. "
           "Panel (a), HAM10000, compara ocho baselines congelados; panel (b), ISIC, compara el control público B71.")
NOTE = ("N: HAM10000 2.004; ISIC 2019 5.067; ISIC 2020 6.660. Bootstrap pareado, 1.000 remuestras, semilla 20260923. "
        "Unidades: HAM lesion_id; ISIC 2019 imagen; ISIC 2020 patient_id. Sin ajuste por multiplicidad. "
        "VGG19: ΔF1 +0,3331 se conserva según E3b/CONTRASTES_SSL.md; pendiente confirmar con Karen si la medición es anterior a 6bd96b7.")


def _contrast_row(contrast, dataset: str, support: int, n_resamples: int, seed: int) -> dict:
    if contrast.unavailable:
        raise ValueError(f"C2 contrast unavailable: {contrast.source}: {contrast.unavailable}")
    if not contrast.low <= contrast.delta <= contrast.high:
        raise ValueError(f"C2 interval does not contain its point estimate: {contrast.source}")
    point_difference = contrast.point_arm - contrast.point_control
    if abs(point_difference - contrast.delta) > 1e-12:
        raise ValueError(f"C2 delta direction mismatch in {contrast.source}")
    return {
        "dataset": dataset,
        "arm": contrast.arm,
        "control": contrast.control,
        "label": contrast.control,
        "delta": contrast.delta,
        "low": contrast.low,
        "high": contrast.high,
        "n_val": support,
        "unit": contrast.unit,
        "n_resamples": n_resamples,
        "seed": seed,
        "source": contrast.source,
    }


def _ham_group(repo: Path) -> dict:
    summary = json.loads((repo / E3B_SOURCE).read_text(encoding="utf-8"))
    above, _ = contrasts_load.e3b_contrasts(repo, E3B_SOURCE)
    arm_name = contrasts_load.B71_VARIANTS["B71_tta"]
    contrasts = [item for item in above if item.arm == arm_name and item.metric == "macro_f1"]
    labels = set(bench_rows.MODEL_NAMES.values())
    if len(contrasts) != len(labels) or {item.control for item in contrasts} != labels:
        raise ValueError("E3b C2 must contain exactly one paired F1 contrast per frozen baseline")
    order = {name: index for index, name in enumerate(bench_rows.MODEL_NAMES.values())}
    contrasts.sort(key=lambda item: order[item.control])
    protocol = summary["protocol"]["bootstrap"]
    comparisons = [_contrast_row(item, "ham10000", summary["n"]["val"], protocol["n_resamples"],
                                 protocol["seed"]) for item in contrasts]
    return {"dataset": "ham10000", "title": "HAM10000 · B71-TTA frente a ocho baselines congelados",
            "n_val": summary["n"]["val"], "unit": protocol["grouped_by"],
            "n_resamples": protocol["n_resamples"], "seed": protocol["seed"],
            "source_code_revision": summary["code_revision"], "comparisons": comparisons}


def _isic_group(repo: Path, dataset: str) -> dict:
    contrasts = contrasts_load.isic_all(repo, dataset)[ISIC_VARIANT]
    selected = [item for item in contrasts if item.metric == "macro_f1" and item.arm == "B71 original"
                and item.control == "DINOv2 público (prot. B71)"]
    if len(selected) != 1:
        raise ValueError(f"Expected one paired B71-original/public-B71 C2 contrast for {dataset}")
    contrast = selected[0]
    source_doc = json.loads((repo / contrast.source).read_text(encoding="utf-8"))
    source_revision = bench_rows.DATASETS[dataset]
    b71_row = next(row for row in bench_rows.dataset_rows(repo, source_revision)
                   if row.name == "B71 original (canónico)" and row.status == bench_rows.OK)
    comparisons = [_contrast_row(contrast, dataset, b71_row.n_val,
                                 source_doc["key"]["n_resamples"], isic_scoring.SEED)]
    return {"dataset": dataset, "title": f"{source_revision.title} · B71 original frente al control público",
            "n_val": b71_row.n_val, "unit": contrast.unit,
            "n_resamples": source_doc["key"]["n_resamples"], "seed": isic_scoring.SEED,
            "source_code_revision": None, "comparisons": comparisons}


def figure_data(repo: Path = REPO) -> dict:
    repo = Path(repo)
    groups = [_ham_group(repo), _isic_group(repo, "isic2019"), _isic_group(repo, "isic2020")]
    if sum(len(group["comparisons"]) for group in groups) != 10:
        raise ValueError("C2 expects eight HAM and one paired public-control contrast per ISIC dataset")
    return {"groups": groups, "metric": "macro_f1", "multiplicity_adjustment": "none"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision(repo: Path, relative: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%H", "--", relative],
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _traceability(repo: Path, data: dict) -> dict:
    comparisons = [item for group in data["groups"] for item in group["comparisons"]]
    sources = sorted({item["source"] for item in comparisons})
    e3b_summary = json.loads((repo / E3B_SOURCE).read_text(encoding="utf-8"))
    if E3B_SOURCE not in sources:
        sources.append(E3B_SOURCE)
    return {
        "figure_id": "C2",
        "metric": "macro_f1",
        "multiplicity_adjustment": "none",
        "figure_script": {"path": "scripts/figures/fig_c2_paired_delta_f1.py",
                          "sha256": _sha256(repo / "scripts/figures/fig_c2_paired_delta_f1.py")},
        "e3b_source_code_revision": e3b_summary["code_revision"],
        "groups": data["groups"],
        "comparison_count": len(comparisons),
        "sources": [{"path": path, "sha256": _sha256(repo / path),
                     "last_changed_commit": _git_revision(repo, path)} for path in sources],
    }


def _delta_tick(value: float, _position: int) -> str:
    if abs(value) < 1e-12:
        return style.fmt_es(0, 2)
    return style.fmt_es(value, 2, signed=True)


def _draw_forest_axis(ax, comparisons: list[dict], labels: list[str], limits: tuple[float, float]) -> None:
    positions = list(range(len(comparisons)))
    ax.axvline(0, color=style.COLORS["frame"], linewidth=style.LINE_WIDTHS["frame"], zorder=1)
    for y, row in zip(positions, comparisons, strict=True):
        ax.hlines(y, row["low"], row["high"], color=style.COLORS["accent"],
                  linewidth=style.LINE_WIDTHS["curve"], zorder=2)
        ax.plot(row["delta"], y, "o", markersize=5, markerfacecolor=style.COLORS["accent"],
                markeredgecolor=style.COLORS["accent"], zorder=3)
        ax.annotate(style.fmt_es(row["delta"], 3, signed=True), (row["high"], y), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=style.LABEL_PT,
                    color=style.COLORS["text"])
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(*limits)
    ax.xaxis.set_major_formatter(_delta_tick)
    style.style_axes(ax, "x")


def _draw(data: dict):
    ham, *isic = data["groups"]
    comparisons = [*ham["comparisons"], *(group["comparisons"][0] for group in isic)]
    low = min(0.0, *(row["low"] for row in comparisons))
    high = max(0.0, *(row["high"] for row in comparisons))
    span = high - low
    limits = low - span * 0.08, high + span * 0.48

    with style.context():
        fig, (ax_ham, ax_isic) = plt.subplots(
            2, 1, sharex=True, figsize=(style.FIGURE_WIDTH_IN, 8.8),
            gridspec_kw={"height_ratios": (8, 2)}, facecolor=style.COLORS["background"])
        _draw_forest_axis(ax_ham, ham["comparisons"], [row["control"] for row in ham["comparisons"]], limits)
        isic_labels = [f"ISIC 2019 · n = {style.fmt_es(isic[0]['n_val'], 0)}",
                       f"ISIC 2020 · n = {style.fmt_es(isic[1]['n_val'], 0)}"]
        _draw_forest_axis(ax_isic, [group["comparisons"][0] for group in isic], isic_labels, limits)
        ax_ham.tick_params(axis="x", labelbottom=False)
        ax_ham.set_xlabel("")
        ax_isic.set_xlabel("Δ F1 macro · B71 − control")
        fig.subplots_adjust(left=0.46, right=0.96, top=0.88, bottom=0.12, hspace=0.48)
        fig.text(0.06, ax_ham.get_position().y1 + 0.025,
                 "(a) HAM10000 · B71-TTA − ocho baselines congelados",
                 color=style.COLORS["text"], fontsize=style.PANEL_TITLE_PT)
        fig.text(0.06, ax_isic.get_position().y1 + 0.025,
                 "(b) ISIC 2019/2020 · B71 original − público B71",
                 color=style.COLORS["text"], fontsize=style.PANEL_TITLE_PT)
    return fig



def build(repo: Path = REPO, outdir: Path | None = None) -> list[Path]:
    repo = Path(repo)
    target = Path(outdir) if outdir is not None else repo / "resultados/figuras"
    stem = STEM
    suffixes = (*style.EXPORT_SUFFIXES, ".caption.txt", ".note.txt", ".traceability.json")
    collisions = [target / f"{stem}{suffix}" for suffix in suffixes if (target / f"{stem}{suffix}").exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite {collisions[0]}")
    target.mkdir(parents=True, exist_ok=True)

    data = figure_data(repo)
    fig = _draw(data)
    try:
        paths = style.save_figure(fig, stem, target)
    finally:
        plt.close(fig)
    (target / f"{stem}.caption.txt").write_text(CAPTION + "\n", encoding="utf-8")
    (target / f"{stem}.note.txt").write_text(NOTE + "\n", encoding="utf-8")
    (target / f"{stem}.traceability.json").write_text(
        json.dumps(_traceability(repo, data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths


if __name__ == "__main__":
    for path in build():
        print(path)
