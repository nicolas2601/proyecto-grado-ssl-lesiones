"""C5: melanoma-versus-rest PR and ROC curves from committed per-image probabilities."""

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

from dermssl.isic_eval import splits as isic_splits  # noqa: E402
import thesis_style as style  # noqa: E402

SOURCES = {
    "ham10000": {
        "title": "HAM10000 · E3b canónico",
        "relative": "resultados/banco_completo/e3b_bs64/e3b_probas.npz",
        "truth": "y_val", "mel_index": 4,
        "arms": (("B71_tta", "B71 (TTA)"), ("dinov2_vitb14", "DINOv2 ViT-B/14")),
        "json": "resultados/banco_completo/e3b_bs64/e3b_summary.json",
    },
    "isic2019": {
        "title": "ISIC 2019 · E1 canónico",
        "relative": "resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019",
        "json": "resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019",
        "mel_index": 0, "arms": (("B71__original", "B71 original"), ("B71__clean", "B71 limpio"),
                                      ("PUBLIC_B71", "B71 público")),
    },
    "isic2020": {
        "title": "ISIC 2020 · E1 canónico",
        "relative": "resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020",
        "json": "resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020",
        "mel_index": 1, "arms": (("B71__original", "B71 original"), ("B71__clean", "B71 limpio"),
                                      ("PUBLIC_B71", "B71 público")),
    },
    "isic2020_ft": {
        "title": "ISIC 2020 · ajuste supervisado",
        "relative": "resultados/banco_completo/ft_isic2020/probas",
        "json": "resultados/banco_completo/ft_isic2020/probas/ft_isic2020_probas.sidecar.json",
        "mel_index": 1, "arms": (("dinov2_vitb14", "DINOv2-B/14"),),
    },
}

NOTE = ("Positivo: melanoma; negativo: resto. Precision–recall: prevalencia de melanoma de la partición como baseline; "
        "AP se calcula como suma de incrementos de sensibilidad por precisión en cada umbral único. ROC: diagonal "
        "de azar y AUC como área trapezoidal de los puntos empíricos. No se seleccionan umbrales en esta partición.")


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    y = np.asarray(labels, dtype=int)
    score = np.asarray(scores, dtype=float)
    if y.ndim != 1 or score.shape != y.shape or not np.isfinite(score).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("C5 requires aligned finite binary labels and scores")
    positives = int(y.sum())
    if positives == 0 or positives == len(y):
        raise ValueError("C5 requires both positive and negative labels")
    order = np.argsort(-score, kind="stable")
    y_sorted, score_sorted = y[order], score[order]
    ends = np.r_[np.flatnonzero(np.diff(score_sorted)), len(y) - 1]
    true_positive = np.cumsum(y_sorted)[ends].astype(float)
    false_positive = (ends + 1) - true_positive
    recall = true_positive / positives
    precision = true_positive / (true_positive + false_positive)
    average_precision = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

    negatives = len(y) - positives
    false_positive_rate = false_positive / negatives
    true_positive_rate = recall
    fpr = np.r_[0.0, false_positive_rate]
    tpr = np.r_[0.0, true_positive_rate]
    roc_auc = float(np.trapezoid(tpr, fpr))
    return {"prevalence": positives / len(y), "average_precision": average_precision,
            "roc_auc": roc_auc, "precision": np.r_[precision[::-1], 1.0], "recall": np.r_[recall[::-1], 0.0],
            "fpr": fpr, "tpr": tpr}


def _mel_probability(scores: np.ndarray, classes: list[str], path: Path) -> np.ndarray:
    if (scores.ndim != 2 or scores.shape[1] != len(classes)
            or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1)
            or not np.allclose(scores.sum(axis=1), 1, atol=1e-6)
            or not any(name.casefold() == "mel" for name in classes)):
        raise ValueError(f"{path}: probability columns do not match verified class list")
    return scores[:, next(index for index, name in enumerate(classes) if name.casefold() == "mel")]


def load_panel(repo: Path, dataset: str) -> dict:
    if dataset not in SOURCES:
        raise ValueError(f"Unknown C5 dataset: {dataset}")
    config = SOURCES[dataset]
    root = Path(repo)
    arms: dict[str, dict] = {}
    source_paths: list[Path] = []
    if dataset == "ham10000":
        path = root / config["relative"]
        manifest_path = path.with_suffix(".json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"{path}: probability archive hash differs from manifest")
        with np.load(path, allow_pickle=False) as archive:
            truth = np.asarray(archive[config["truth"]], dtype=int)
            ids = np.asarray(archive["val_ids"])
            classes = json.loads((root / config["json"]).read_text(encoding="utf-8"))["classes"]
            for key, label in config["arms"]:
                scores = _mel_probability(np.asarray(archive[key], dtype=float), classes, path)
                arms[key] = {"label": label, "scores": scores, "source": path}
        source_paths.extend((path, manifest_path, root / config["json"]))
    elif dataset in ("isic2019", "isic2020"):
        folder = root / config["relative"]
        truth = ids = None
        for index, (key, label) in enumerate(config["arms"]):
            path = folder / f"{key}.probas.npz"
            with np.load(path, allow_pickle=False) as archive:
                labels = np.asarray(archive["y"], dtype=int)
                current_ids = np.asarray(archive["val_ids"])
                scores = np.asarray(archive["proba"], dtype=float)
            source_json = folder / f"{key}.json"
            record = json.loads(source_json.read_text(encoding="utf-8"))["result"]
            if (record["probas_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
                    or record["n_val"] != len(labels)
                    or record["split_identity"]["val_ids_sha256"] != isic_splits.sorted_ids_sha256(current_ids)):
                raise ValueError(f"{path}: probabilities or IDs differ from committed evaluation metadata")
            if truth is None:
                truth, ids = labels, current_ids
            elif not np.array_equal(labels, truth) or not np.array_equal(current_ids, ids):
                raise ValueError(f"{path}: image labels or IDs differ from first arm")
            if dataset == "isic2019":
                classes = isic_splits.ISIC2019_CLASSES
                melanoma = _mel_probability(scores, list(classes), path)
            else:
                if (scores.shape != (len(labels), len(isic_splits.ISIC2020_CLASSES))
                        or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1)
                        or not np.allclose(scores.sum(axis=1), 1, atol=1e-6)):
                    raise ValueError(f"{path}: probability columns do not match ISIC 2020 class order")
                melanoma = scores[:, 1]
            arms[key] = {"label": label, "scores": melanoma, "source": path}
            source_paths.extend((path, source_json))
        assert truth is not None
    else:
        sidecar_path = root / config["json"]
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        truth = ids = None
        for key, label in config["arms"]:
            path = root / config["relative"] / f"{key}.probas.npz"
            with np.load(path, allow_pickle=False) as archive:
                labels = np.asarray(archive["y"], dtype=int)
                current_ids = np.asarray(archive["val_ids"])
                scores = np.asarray(archive["proba"], dtype=float)
            record = next(row for row in sidecar["models"] if row["model"] == key)
            if record["status"] != "pareado" or not record["ids_order_equal_reference"] or record["probas_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f"{path}: fine-tuning archive is not verified against sidecar")
            if truth is None:
                truth, ids = labels, current_ids
            elif not np.array_equal(labels, truth) or not np.array_equal(current_ids, ids):
                raise ValueError(f"{path}: image labels or IDs differ from first arm")
            arms[key] = {"label": label, "scores": scores[:, 1], "source": path}
            source_paths.append(path)
        assert truth is not None
        source_paths.append(sidecar_path)
    positive = truth == config["mel_index"]
    labels = positive.astype(int)
    results = {key: {**arm, **binary_metrics(labels, arm["scores"])} for key, arm in arms.items()}
    for arm in results.values():
        arm["source"] = str(arm["source"].relative_to(root))
    return {"dataset": dataset, "title": config["title"], "n": len(labels), "positive_count": int(labels.sum()),
            "prevalence": float(labels.mean()), "arms": results,
            "source_hashes": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(set(source_paths))}}


def _draw(panel: dict):
    with style.context():
        fig, axes = plt.subplots(1, 2, figsize=(style.FIGURE_WIDTH_IN, 4.8))
        pr_ax, roc_ax = axes
        for index, arm in enumerate(panel["arms"].values()):
            color = style.COLORS["accent"] if index == 0 else style.COLORS["frame"]
            linestyle = "-" if index == 0 else "--" if index == 1 else ":"
            pr_ax.plot(arm["recall"], arm["precision"], color=color, linestyle=linestyle,
                       label=f"{arm['label']} · AP {style.fmt_es(arm['average_precision'], 3)}")
            roc_ax.plot(arm["fpr"], arm["tpr"], color=color, linestyle=linestyle,
                        label=f"{arm['label']} · AUC {style.fmt_es(arm['roc_auc'], 3)}")
        pr_ax.axhline(panel["prevalence"], color=style.COLORS["frame"], linestyle=":",
                      linewidth=style.LINE_WIDTHS["bar"])
        roc_ax.plot([0, 1], [0, 1], color=style.COLORS["frame"], linestyle=":",
                    linewidth=style.LINE_WIDTHS["bar"])
        for ax, title, xlabel, ylabel in ((pr_ax, "(a) Precisión–sensibilidad", "Sensibilidad", "Precisión"),
                                          (roc_ax, "(b) ROC", "Tasa de falsos positivos", "Sensibilidad")):
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks((0, 0.5, 1), ("0", "0,5", "1"))
            ax.set_yticks((0, 0.5, 1), ("0", "0,5", "1"))
            ax.set_title(title, loc="left", fontsize=style.PANEL_TITLE_PT)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            style.style_axes(ax, "both")
        pr_handles = [plt.Line2D([], [], color=style.COLORS["frame"], linestyle=":",
                                 linewidth=style.LINE_WIDTHS["bar"],
                                 label=f"Prevalencia · {style.fmt_es(panel['prevalence'], 3)}")]
        roc_handles = [plt.Line2D([], [], color=style.COLORS["frame"], linestyle=":",
                                  linewidth=style.LINE_WIDTHS["bar"], label="Azar")]
        pr_handles.extend(plt.Line2D([], [], color=style.COLORS["accent"] if i == 0 else style.COLORS["frame"],
                                     linestyle="-" if i == 0 else "--" if i == 1 else ":",
                                     linewidth=style.LINE_WIDTHS["curve"], label=f"{arm['label']} · AP {style.fmt_es(arm['average_precision'], 3)}")
                          for i, arm in enumerate(panel["arms"].values()))
        roc_handles.extend(plt.Line2D([], [], color=style.COLORS["accent"] if i == 0 else style.COLORS["frame"],
                                      linestyle="-" if i == 0 else "--" if i == 1 else ":",
                                      linewidth=style.LINE_WIDTHS["curve"], label=f"{arm['label']} · AUC {style.fmt_es(arm['roc_auc'], 3)}")
                           for i, arm in enumerate(panel["arms"].values()))
        pr_ax.legend(handles=pr_handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25),
                     fontsize=style.LABEL_PT - 1, handlelength=1.5, labelspacing=0.4, borderaxespad=0)
        roc_ax.legend(handles=roc_handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25),
                      fontsize=style.LABEL_PT - 1, handlelength=1.5, labelspacing=0.4, borderaxespad=0)
        fig.suptitle(f"{panel['title']} · n = {style.fmt_es(panel['n'], 0)}; MEL = {style.fmt_es(panel['positive_count'], 0)}",
                     x=0.08, ha="left", fontsize=style.PANEL_TITLE_PT, fontweight="normal")
        fig.subplots_adjust(left=0.13, right=0.97, top=0.84, bottom=0.34, wspace=0.32)
    return fig


def _write_sidecars(repo: Path, target: Path, panel: dict) -> list[Path]:
    stem = f"fig_c5_pr_roc_{panel['dataset']}"
    fig = _draw(panel)
    try:
        paths = style.save_figure(fig, stem, target)
    finally:
        plt.close(fig)
    trace = {"figure_id": "C5", "dataset": panel["dataset"], "status": "Cálculo adicional desde probabilidades por imagen",
             "positive_class": "melanoma", "positive_count": panel["positive_count"], "support": panel["n"],
             "evaluation_unit": "image", "arms": {key: {k: v for k, v in arm.items() if k not in
                          ("precision", "recall", "fpr", "tpr", "scores")} for key, arm in panel["arms"].items()},
             "sources": panel["source_hashes"], "metrics": {
                 "precision_recall": "average precision = sum(delta recall * precision at each distinct score)",
                 "roc": "trapezoidal area under empirical ROC from distinct scores"},
             "threshold_selection_on_target_test": False,
             "script": "scripts/figures/fig_c5_pr_roc.py"}
    (target / f"{stem}.traceability.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / f"{stem}.caption.txt").write_text(f"Curvas de precisión–sensibilidad y ROC de melanoma frente al resto en {panel['title']}.\n",
                                                   encoding="utf-8")
    note = NOTE + (" La prevalencia es la fracción de positivos en esta partición, no una prevalencia poblacional."
                   if panel["dataset"] == "pad_ufes" else "")
    (target / f"{stem}.note.txt").write_text(note + "\n", encoding="utf-8")
    return paths


def build(repo: Path = REPO, outdir: Path | None = None) -> dict[str, list[Path]]:
    target = Path(outdir) if outdir is not None else Path(repo) / "resultados/figuras"
    datasets = ("ham10000", "isic2019", "isic2020", "isic2020_ft")
    stems = [f"fig_c5_pr_roc_{dataset}" for dataset in datasets]
    suffixes = (*style.EXPORT_SUFFIXES, ".caption.txt", ".note.txt", ".traceability.json")
    collisions = [target / f"{stem}{suffix}" for stem in stems for suffix in suffixes if (target / f"{stem}{suffix}").exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite {collisions[0]}")
    target.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for dataset in datasets:
        panel = load_panel(Path(repo), dataset)
        outputs[dataset] = _write_sidecars(Path(repo), target, panel)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=REPO / "resultados/figuras")
    args = parser.parse_args()
    for dataset, paths in build(REPO, args.outdir).items():
        print(dataset, *(str(path) for path in paths))
