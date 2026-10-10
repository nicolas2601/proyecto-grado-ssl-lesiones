"""C6: confidence reliability diagrams from per-image class probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fig_c5_pr_roc as c5  # noqa: E402
import thesis_style as style  # noqa: E402

N_BINS = 15
ECE_TOLERANCE = 1e-6
SOURCES = {
    "ham10000": {
        "title": "HAM10000 · E3b canónico", "kind": "e3b",
        "archive": "resultados/banco_completo/e3b_bs64/e3b_probas.npz",
        "metadata": ("resultados/banco_completo/e3b_bs64/e3b_probas.json",
                     "resultados/banco_completo/e3b_bs64/e3b_summary.json"),
        "arms": (("B71_tta", "B71 (TTA)", "proposed"),
                 ("dinov2_vitb14", "DINOv2 ViT-B/14", "baseline")),
    },
    "ham10000_v71": {
        "title": "HAM10000 · validación v7.1", "kind": "v71",
        "archive": "resultados/clasificacion/v71/v71_final_eval.probas.npz",
        "metadata": ("resultados/clasificacion/v71/v71_final_eval.json",),
        "arms": (("B71", "B71", "results"), ("public_B71", "B71 público", "results")),
    },
    "isic2019": {
        "title": "ISIC 2019 · E1 canónico", "kind": "isic",
        "archive_dir": "resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019",
        "metadata": (),
        "arms": (("B71__original", "B71 original", "result"),
                 ("B71__clean", "B71 limpio", "result"),
                 ("PUBLIC_B71", "B71 público", "result")),
    },
    "isic2020": {
        "title": "ISIC 2020 · E1 canónico", "kind": "isic",
        "archive_dir": "resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020",
        "metadata": (),
        "arms": (("B71__original", "B71 original", "result"),
                 ("B71__clean", "B71 limpio", "result"),
                 ("PUBLIC_B71", "B71 público", "result")),
    },
    "isic2020_ft": {
        "title": "ISIC 2020 · ajuste supervisado", "kind": "ft",
        "archive_dir": "resultados/banco_completo/ft_isic2020/probas",
        "metadata": ("resultados/banco_completo/ft_isic2020/probas/ft_isic2020_probas.sidecar.json",
                     "resultados/banco_completo/ft_isic2020/json/results_dinov2_vitb14_ft_isic2020.json"),
        "arms": (("dinov2_vitb14", "DINOv2-B/14", "finetune"),),
    },
}
MARKERS = ("o", "s", "^")
NOTE = ("Confianza = probabilidad máxima entre clases; acierto = argmax coincide con etiqueta real. "
        "Se usan 15 bins equidistantes: (lo, hi], salvo el primero [0, hi]. La línea diagonal indica calibración perfecta. "
        "El soporte (n) exacto por bin y el ECE recalculado constan en el sidecar de trazabilidad.")


def reliability_data(labels: np.ndarray, probabilities: np.ndarray, published_ece: float,
                     bins: int = N_BINS) -> dict:
    y = np.asarray(labels, dtype=int)
    proba = np.asarray(probabilities, dtype=float)
    if (y.ndim != 1 or proba.ndim != 2 or len(y) != len(proba) or bins < 1
            or proba.shape[1] < 2 or not np.isfinite(proba).all()
            or np.any(proba < 0) or np.any(proba > 1)
            or not np.allclose(proba.sum(axis=1), 1, atol=1e-6)
            or np.any(y < 0) or np.any(y >= proba.shape[1])):
        raise ValueError("C6 requires aligned labels and valid class probabilities")
    confidence = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (confidence > low) & (confidence <= high) if index else (confidence >= low) & (confidence <= high)
        support = int(mask.sum())
        mean_confidence = float(confidence[mask].mean()) if support else None
        accuracy = float(correct[mask].mean()) if support else None
        if support:
            ece += support / len(y) * abs(mean_confidence - accuracy)
        rows.append({"index": index, "low": float(low), "high": float(high),
                     "confidence": mean_confidence, "accuracy": accuracy, "n": support})
    matches = abs(ece - float(published_ece)) <= ECE_TOLERANCE
    return {"confidence_definition": "maximum class probability", "correctness": "argmax equals true class",
            "bin_count": bins, "bins": rows, "ece_recomputed": ece,
            "published_ece": float(published_ece), "ece_matches_published": matches,
            "ece": ece if matches else None}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _e3b_panel(repo: Path, config: dict, archive_path: Path) -> tuple[np.ndarray, dict, list[Path]]:
    c5.load_panel(repo, "ham10000")
    manifest_path = archive_path.with_suffix(".json")
    manifest = _read_json(manifest_path)
    summary_path = repo / config["metadata"][1]
    summary = _read_json(summary_path)
    if manifest["sha256"] != hashlib.sha256(archive_path.read_bytes()).hexdigest():
        raise ValueError(f"{archive_path}: probabilities differ from archive manifest")
    with np.load(archive_path, allow_pickle=False) as archive:
        labels = np.asarray(archive["y_val"], dtype=int)
        classes = summary["classes"]
        arms = {key: (np.asarray(archive[key], dtype=float),
                      summary["proposed"]["rows"][key]["ece"] if kind == "proposed"
                      else summary["models"][key]["rows"]["full"]["ece"], label, archive_path)
                for key, label, kind in config["arms"]}
    if any(probabilities.shape != (len(labels), len(classes)) for probabilities, *_ in arms.values()):
        raise ValueError(f"{archive_path}: class probability shape differs from source classes")
    return labels, arms, [archive_path, manifest_path, summary_path]


def _v71_panel(repo: Path, config: dict, archive_path: Path) -> tuple[np.ndarray, dict, list[Path]]:
    metadata_path = repo / config["metadata"][0]
    classes_path = repo / c5.SOURCES["ham10000"]["json"]
    record = _read_json(metadata_path)
    class_count = len(_read_json(classes_path)["classes"])
    with np.load(archive_path, allow_pickle=False) as archive:
        labels = np.asarray(archive["y_true"], dtype=int)
        arms = {key: (np.asarray(archive[key], dtype=float), record["results"][key]["ece"], label, archive_path)
                for key, label, _ in config["arms"]}
    if len(labels) != record["n_eval"] or any(probabilities.shape != (len(labels), class_count)
                                                 for probabilities, *_ in arms.values()):
        raise ValueError(f"{archive_path}: V7.1 support or class dimensions differ from evaluation record")
    return labels, arms, [archive_path, metadata_path, classes_path]


def _isic_panel(repo: Path, dataset: str, config: dict) -> tuple[np.ndarray, dict, list[Path]]:
    c5.load_panel(repo, dataset)
    folder = repo / config["archive_dir"]
    labels = None
    arms, sources = {}, []
    for key, label, _ in config["arms"]:
        archive_path = folder / f"{key}.probas.npz"
        metadata_path = folder / f"{key}.json"
        metadata = _read_json(metadata_path)["result"]
        with np.load(archive_path, allow_pickle=False) as archive:
            current_labels = np.asarray(archive["y"], dtype=int)
            probabilities = np.asarray(archive["proba"], dtype=float)
        if labels is None:
            labels = current_labels
        elif not np.array_equal(labels, current_labels):
            raise ValueError(f"{archive_path}: labels differ across ISIC arms")
        arms[key] = (probabilities, metadata["metrics"]["ece"], label, archive_path)
        sources.extend((archive_path, metadata_path))
    if labels is None:
        raise ValueError(f"{folder}: no ISIC probability arms configured")
    return labels, arms, sources


def _ft_panel(repo: Path, config: dict) -> tuple[np.ndarray, dict, list[Path]]:
    c5.load_panel(repo, "isic2020_ft")
    folder = repo / config["archive_dir"]
    archive_path = folder / "dinov2_vitb14.probas.npz"
    sidecar_path = repo / config["metadata"][0]
    result_path = repo / config["metadata"][1]
    sidecar = _read_json(sidecar_path)
    result = _read_json(result_path)["probes"]["finetune"]
    record = next(item for item in sidecar["models"] if item["model"] == "dinov2_vitb14")
    with np.load(archive_path, allow_pickle=False) as archive:
        labels = np.asarray(archive["y"], dtype=int)
        probabilities = np.asarray(archive["proba"], dtype=float)
    if record["probas_sha256"] != hashlib.sha256(archive_path.read_bytes()).hexdigest():
        raise ValueError(f"{archive_path}: probabilities differ from fine-tuning sidecar")
    return labels, {"dinov2_vitb14": (probabilities, result["ece"], "DINOv2-B/14", archive_path)}, \
        [archive_path, sidecar_path, result_path]


def load_panel(repo: Path, dataset: str) -> dict:
    if dataset not in SOURCES:
        raise ValueError(f"Unknown C6 dataset: {dataset}")
    root = Path(repo)
    config = SOURCES[dataset]
    archive_path = root / config["archive"] if config.get("archive") else None
    if config["kind"] == "e3b":
        labels, raw_arms, source_paths = _e3b_panel(root, config, archive_path)
        title = config["title"]
    elif config["kind"] == "v71":
        labels, raw_arms, source_paths = _v71_panel(root, config, archive_path)
        title = config["title"]
    elif config["kind"] == "isic":
        labels, raw_arms, source_paths = _isic_panel(root, dataset, config)
        title = config["title"]
    else:
        labels, raw_arms, source_paths = _ft_panel(root, config)
        title = config["title"]
    arms = {}
    for key, (probabilities, published_ece, label, path) in raw_arms.items():
        reliability = reliability_data(labels, probabilities, published_ece)
        arms[key] = {"label": label, "source": str(path.relative_to(root)), **reliability}
    return {"dataset": dataset, "title": title, "n": len(labels), "arms": arms,
            "source_hashes": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(set(source_paths))}}


def _draw(panel: dict):
    with style.context():
        fig, ax = plt.subplots(figsize=(style.FIGURE_WIDTH_IN, 5.0))
        ax.plot([0, 1], [0, 1], color=style.COLORS["frame"], linestyle=":",
                linewidth=style.LINE_WIDTHS["bar"], label="Calibración perfecta")
        for index, arm in enumerate(panel["arms"].values()):
            bins = [row for row in arm["bins"] if row["n"]]
            label = f"{arm['label']} · ECE {style.fmt_es(arm['ece'], 3)}" if arm["ece"] is not None else f"{arm['label']} · ECE n/d"
            ax.plot([row["confidence"] for row in bins], [row["accuracy"] for row in bins],
                    color=style.COLORS["accent"] if index == 0 else style.COLORS["frame"],
                    linestyle="-" if index == 0 else "--" if index == 1 else ":",
                    marker=("o", "s", "^")[index], markersize=5, linewidth=style.LINE_WIDTHS["curve"], label=label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks((0, 0.5, 1), ("0", "0,5", "1"))
        ax.set_yticks((0, 0.5, 1), ("0", "0,5", "1"))
        ax.set_xlabel("Confianza media por bin")
        ax.set_ylabel("Exactitud por bin")
        handles, labels = ax.get_legend_handles_labels()
        style.style_axes(ax, "both")
        fig.suptitle(f"{panel['title']} · n = {style.fmt_es(panel['n'], 0)}; bins = {N_BINS}",
                     x=0.08, ha="left", fontsize=style.PANEL_TITLE_PT, fontweight="normal")
        fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.55, 0.05), handlelength=1.6, columnspacing=1.5)
        fig.subplots_adjust(left=0.13, right=0.97, top=0.84, bottom=0.27)
    return fig


def _write_panel(repo: Path, target: Path, panel: dict) -> list[Path]:
    stem = f"fig_c6_calibration_{panel['dataset']}"
    fig = _draw(panel)
    try:
        paths = style.save_figure(fig, stem, target)
    finally:
        plt.close(fig)
    trace = {"figure_id": "C6", "dataset": panel["dataset"], "status": "Cálculo adicional desde probabilidades por imagen",
             "evaluation_unit": "image", "support": panel["n"], "bins": N_BINS,
             "bin_edges": "15 equal-width bins: (lo, hi], first bin [0, hi]",
             "ece_tolerance": ECE_TOLERANCE,
             "confidence_definition": "maximum class probability", "correctness": "argmax equals true class",
             "arms": {key: {k: v for k, v in arm.items() if k != "source"} for key, arm in panel["arms"].items()},
             "sources": panel["source_hashes"], "script": "scripts/figures/fig_c6_calibration.py"}
    if panel["dataset"] == "ham10000_v71":
        trace["missing"] = ["Per-image IDs are absent from the V7.1 NPZ; archive rows cannot be independently matched to eval_ids_sha256."]
    (target / f"{stem}.traceability.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / f"{stem}.caption.txt").write_text(f"Confiabilidad de las probabilidades por clase predicha en {panel['title']}.\n", encoding="utf-8")
    note = NOTE
    if any(not arm["ece_matches_published"] for arm in panel["arms"].values()):
        note += " ECE no se muestra para brazos cuyo recálculo no coincide con el valor publicado."
    if panel["dataset"] == "ham10000_v71":
        note += " El NPZ V7.1 no contiene IDs por imagen; la correspondencia exacta con el hash de partición no puede verificarse independientemente."
    (target / f"{stem}.note.txt").write_text(note + "\n", encoding="utf-8")
    return paths


def build(repo: Path = REPO, outdir: Path | None = None) -> dict[str, list[Path]]:
    root = Path(repo)
    target = Path(outdir) if outdir is not None else root / "resultados/figuras"
    datasets = tuple(SOURCES)
    suffixes = (*style.EXPORT_SUFFIXES, ".caption.txt", ".note.txt", ".traceability.json")
    collisions = [target / f"fig_c6_calibration_{dataset}{suffix}" for dataset in datasets
                  for suffix in suffixes if (target / f"fig_c6_calibration_{dataset}{suffix}").exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite {collisions[0]}")
    target.mkdir(parents=True, exist_ok=True)
    return {dataset: _write_panel(root, target, load_panel(root, dataset)) for dataset in datasets}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=REPO / "resultados/figuras")
    args = parser.parse_args()
    for dataset, paths in build(REPO, args.outdir).items():
        print(dataset, *(str(path) for path in paths))
