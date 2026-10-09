"""Objective 1: separate dataset class frequencies and registered PAD phototypes.

Each dataset keeps its original label taxonomy. ISIC 2020 has only a binary
benign/malignant label; registered Fitzpatrick values come only from PAD.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))

import beautiful as bf  # noqa: E402
import fig1_ita as ita  # noqa: E402
import fig_obj1_diagnosis_tone as tone  # noqa: E402
from dermssl import pad_ufes as pad  # noqa: E402
from dermssl import pad_ufes_files as pad_files  # noqa: E402

DATASETS = (*ita.DATASETS, "pad_ufes_20")
CLASSES = {
    "ham10000": ("NV", "MEL", "BKL", "BCC", "AKIEC", "VASC", "DF"),
    "isic2019": ("NV", "MEL", "BCC", "BKL", "AK", "SCC", "VASC", "DF"),
    "isic2020": ("BENIGN", "MALIGNANT"),
    "pad_ufes_20": tuple(pad.EXPECTED_DIAGNOSTIC_COUNTS),
}
TOTALS = {"ham10000": 10015, "isic2019": 25331, "isic2020": 33126,
          "pad_ufes_20": pad.EXPECTED_IMAGES}
FST_BUCKETS = (*ita.BUCKETS, "missing")
ORIENTATIONS = ("vertical", "horizontal")
PAD_MISSING_FST = 804
FST_STEM = "fig_obj1_pad_fst_registered"
GRID_GROUPS = (("top", DATASETS[:2]), ("bottom", DATASETS[2:]))
BACKGROUND, PANEL, INK, MUTED = "#FFFFFF", "#FFFFFF", "#302C29", "#645D58"
BORDER, GRID = "#AAA39E", "#E5DED8"
LABELS = {"BENIGN": "Benigno", "MALIGNANT": "Maligno"}


def count_e5(repo: Path, dataset: str) -> dict[str, int]:
    if dataset not in ita.DATASETS:
        raise ValueError(f"Unknown E5 dataset: {dataset}")
    source = Path(repo) / ita.E5_DIR / f"e5_fitzpatrick_{dataset}.json"
    _, rows = ita._load(source)
    if not rows or any(not row.get("image_id") or not row.get("label") for row in rows):
        raise ValueError(f"Invalid E5 image rows: {dataset}")
    identifiers = [row["image_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Duplicate E5 image IDs: {dataset}")
    return dict(Counter(row["label"].upper() for row in rows))


def count_pad(records: list[pad.PadRecord]) -> tuple[dict[str, int], dict[str, int]]:
    diagnoses = Counter()
    phototypes = dict.fromkeys(FST_BUCKETS, 0)
    for record in records:
        diagnoses[record.diagnostic] += 1
        if record.fst_status == "missing" and record.fst is None:
            phototypes["missing"] += 1
            continue
        if record.fst_status != "ok" or type(record.fst) is not int or record.fst not in pad.FST_VALID:
            raise ValueError(f"Invalid PAD FST: {record.img_id if hasattr(record, 'img_id') else record.diagnostic}")
        phototypes[ita.BUCKETS[record.fst - 1]] += 1
    return dict(diagnoses), phototypes


def load_pad(metadata: Path) -> tuple[dict[str, int], dict[str, int]]:
    official = next(item for item in pad_files.OFFICIAL_FILES if item.name == pad_files.METADATA_NAME)
    pad_files.verify(Path(metadata), official)
    records = pad.parse_metadata(metadata)
    pad.check_published_counts(records)
    return count_pad(records)


def validate_diagnoses(dataset: str, counts: dict[str, int]) -> None:
    if dataset not in CLASSES or set(counts) != set(CLASSES[dataset]):
        raise ValueError(f"Unexpected classes for {dataset}: {sorted(counts)}")
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError(f"Invalid class count: {dataset}")
    if sum(counts.values()) != TOTALS[dataset]:
        raise ValueError(f"Unexpected total for {dataset}: {sum(counts.values())}")
    if dataset == "pad_ufes_20" and counts != pad.EXPECTED_DIAGNOSTIC_COUNTS:
        raise ValueError("PAD diagnosis counts differ from published counts")


def validate_fst(counts: dict[str, int], total: int) -> None:
    if sum(counts.values()) != total:
        raise ValueError(f"FST total mismatch: {sum(counts.values())} != {total}")
    if set(counts) != set(FST_BUCKETS) or any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError(f"Invalid FST bucket counts: {counts}")


def _style_axis(ax, grid_axis: str) -> None:
    ax.set_facecolor(PANEL)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, length=0, pad=9)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.75)


def _draw_horizontal(ax, labels: list[str], values: list[int], colors: list[tuple], n_images: int,
                     font_size: float = 11) -> None:
    positions = list(range(len(labels)))
    ax.barh(positions, values, height=0.7, color=[fill for fill, _ in colors],
            edgecolor=[edge for _, edge in colors], linewidth=1)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.28)
    for y, value in zip(positions, values, strict=True):
        caption = f"{bf.fmt_es(value, 0)}  ({bf.fmt_es(100 * value / n_images, 1)} %)"
        ax.text(value + max(values) * 0.015, y, caption, va="center", fontsize=font_size, color=INK)
    ax.set_xlabel("Número de imágenes")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.xaxis.set_major_formatter(lambda value, _: bf.fmt_es(value, 0))
    _style_axis(ax, "x")


def _draw_vertical(ax, labels: list[str], values: list[int], colors: list[tuple], n_images: int) -> None:
    positions = list(range(len(labels)))
    bars = ax.bar(positions, values, width=0.72, color=[fill for fill, _ in colors],
                  edgecolor=[edge for _, edge in colors], linewidth=1)
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(bf.fmt_es(value, 0), (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=10, color=INK)
    ax.set_xticks(positions, labels)
    ax.set_xlabel("Clase del dataset")
    ax.set_ylabel("Número de imágenes · escala logarítmica")
    ax.set_yscale("log")
    ax.set_ylim(0.8, max(values) * 3)
    ax.yaxis.set_major_formatter(lambda value, _: bf.fmt_es(value, 0))
    _style_axis(ax, "y")


def _diagnosis_chart(dataset: str, counts: dict[str, int], orientation: str):
    ordered = sorted(counts, key=lambda label: (-counts[label], CLASSES[dataset].index(label)))
    n_images = sum(counts.values())
    size = (12, 6.3) if orientation == "vertical" else (11, max(5.2, 2.8 + 0.65 * len(ordered)))
    fig, ax = plt.subplots(figsize=size, facecolor=BACKGROUND)
    colors = [tone.diagnosis_style(label)[1:] for label in ordered]
    labels = [LABELS.get(label, label) for label in ordered]
    values = [counts[label] for label in ordered]
    if orientation == "vertical":
        _draw_vertical(ax, labels, values, colors, n_images)
    else:
        _draw_horizontal(ax, labels, values, colors, n_images)
    name = "PAD-UFES-20" if dataset == "pad_ufes_20" else ita.DATASET_LABELS[dataset]
    kind = "clases binarias" if dataset == "isic2020" else "diagnósticos"
    ax.set_title(f"{name} · distribución de {kind}", loc="left", pad=18, fontsize=17, color=INK)
    note = f"N = {bf.fmt_es(n_images, 0)} imágenes; etiquetas originales del dataset."
    if dataset == "isic2020":
        note += " Benigno/maligno es clasificación binaria, no diagnóstico específico."
    fig.text(0.08, 0.035, note, color=MUTED, fontsize=10)
    fig.subplots_adjust(left=0.12 if orientation == "vertical" else 0.16,
                        right=0.96 if orientation == "vertical" else 0.94, top=0.85,
                        bottom=0.17 if orientation == "vertical" else 0.22)
    return fig


def _fst_chart(counts: dict[str, int], orientation: str):
    registered = sum(counts[bucket] for bucket in ita.BUCKETS)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BACKGROUND)
    values = [counts[bucket] for bucket in ita.BUCKETS]
    colors = [(color, ita._mix(color, "black", 0.7)) for color in tone.SKIN_FILLS]
    if orientation == "vertical":
        _draw_vertical(ax, list(ita.BUCKETS), values, colors, registered)
        ax.set_xlabel("Fototipo Fitzpatrick registrado")
    else:
        _draw_horizontal(ax, list(ita.BUCKETS), values, colors, registered)
        ax.set_ylabel("Fototipo Fitzpatrick registrado")
    ax.set_title("PAD-UFES-20 · fototipo registrado", loc="left", pad=17, fontsize=17, color=INK)
    note = (f"N = {bf.fmt_es(registered + counts['missing'], 0)}; con FST = "
            f"{bf.fmt_es(registered, 0)}; sin FST = {bf.fmt_es(counts['missing'], 0)}.\n")
    if orientation == "horizontal":
        note += (f"Porcentajes de barras: base {bf.fmt_es(registered, 0)} con FST; "
                 f"{bf.fmt_es(counts['missing'], 0)} sin FST excluidos. ")
    else:
        note += f"{bf.fmt_es(counts['missing'], 0)} sin FST excluidos de las barras. "
    note += "V–VI: conteos descriptivos, sin inferencia."
    fig.text(0.09, 0.045, note, color=MUTED, fontsize=10)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.86, bottom=0.25)
    return fig


def render(diagnoses: dict[str, dict[str, int]], fst: dict[str, int], outdir: Path) -> list[Path]:
    target = Path(outdir)
    stems = [*(f"fig_obj1_diagnosis_{dataset}_{orientation}" for dataset in DATASETS
               for orientation in ORIENTATIONS), *(f"{FST_STEM}_{orientation}" for orientation in ORIENTATIONS)]
    collisions = [target / f"{stem}{suffix}" for stem in stems for suffix in bf.EXPORT_SUFFIXES
                  if (target / f"{stem}{suffix}").exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite {collisions[0]}")
    bf.apply_style(width=11)
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "svg.fonttype": "none"})
    paths = []
    for dataset in DATASETS:
        for orientation in ORIENTATIONS:
            fig = _diagnosis_chart(dataset, diagnoses[dataset], orientation)
            paths.extend(bf.save_all(fig, f"fig_obj1_diagnosis_{dataset}_{orientation}", target))
            plt.close(fig)
    for orientation in ORIENTATIONS:
        fig = _fst_chart(fst, orientation)
        paths.extend(bf.save_all(fig, f"{FST_STEM}_{orientation}", target))
        plt.close(fig)
    return paths


def render_diagnosis_grid(diagnoses: dict[str, dict[str, int]], outdir: Path) -> list[Path]:
    target = Path(outdir)
    stems = [f"fig_obj1_diagnosis_{group}" for group, _ in GRID_GROUPS]
    collisions = [target / f"{stem}{suffix}" for stem in stems for suffix in bf.EXPORT_SUFFIXES
                  if (target / f"{stem}{suffix}").exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite {collisions[0]}")
    for dataset in DATASETS:
        validate_diagnoses(dataset, diagnoses[dataset])

    bf.apply_style(width=10)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 10,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none"})
    paths = []
    for group, datasets in GRID_GROUPS:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), facecolor=BACKGROUND)
        for ax, dataset in zip(axes, datasets, strict=True):
            counts = diagnoses[dataset]
            ordered = sorted(counts, key=lambda label: (-counts[label], CLASSES[dataset].index(label)))
            labels = [LABELS.get(label, label) for label in ordered]
            values = [counts[label] for label in ordered]
            colors = [tone.diagnosis_style(label)[1:] for label in ordered]
            _draw_horizontal(ax, labels, values, colors, sum(values), font_size=6.5)
            name = "PAD-UFES-20" if dataset == "pad_ufes_20" else ita.DATASET_LABELS[dataset]
            ax.set_title(f"{name} · N = {bf.fmt_es(sum(values), 0)}",
                         loc="left", fontsize=9.5, color=INK, pad=7)

        fig.suptitle("Distribución de diagnósticos · " + ("fila superior" if group == "top" else "fila inferior"),
                     x=0.07, ha="left", fontsize=11, color=INK)
        fig.text(0.07, 0.025,
                 "Etiquetas originales. ISIC 2020: clases binarias. Conteos descriptivos; "
                 "no representan prevalencia poblacional.", color=MUTED, fontsize=6.5)
        fig.subplots_adjust(left=0.12, right=0.97, top=0.83, bottom=0.16, wspace=0.28)
        stem = f"fig_obj1_diagnosis_{group}"
        paths.extend(bf.save_all(fig, stem, target))
        plt.close(fig)
    return paths


def build(repo: Path, metadata: Path, outdir: Path | None = None) -> list[Path]:
    diagnoses = {dataset: count_e5(repo, dataset) for dataset in ita.DATASETS}
    diagnoses["pad_ufes_20"], fst = load_pad(metadata)
    for dataset, counts in diagnoses.items():
        validate_diagnoses(dataset, counts)
    validate_fst(fst, pad.EXPECTED_IMAGES)
    if fst["missing"] != PAD_MISSING_FST:
        raise ValueError(f"PAD FST missing differs from validated source: {fst['missing']}")
    return render(diagnoses, fst, Path(outdir) if outdir else Path(repo) / "resultados/figuras")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pad-metadata", type=Path, required=True)
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()
    for path in build(REPO, args.pad_metadata, args.outdir):
        print(path)
