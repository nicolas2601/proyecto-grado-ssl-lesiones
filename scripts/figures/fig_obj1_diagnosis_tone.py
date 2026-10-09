"""Objective 1: diagnosis-by-tone composition across four real datasets.

HAM10000/ISIC2019/ISIC2020 show unvalidated E5 ROI-ITA buckets; PAD-UFES-20
shows Fitzpatrick values recorded in its published metadata. They are not the
same measurement. Missing values remain visible and every bar has its own n.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))

import beautiful as bf  # noqa: E402
import fig1_ita as ita  # noqa: E402
from dermssl import pad_ufes as pad  # noqa: E402
from dermssl import pad_ufes_files as pad_files  # noqa: E402

DATASETS = ("ham10000", "isic2019", "isic2020", "pad_ufes_20")
BUCKETS = (*ita.BUCKETS, "missing")
SKIN_FILLS = ("#FEF3E0", "#F2C598", "#CE8B51", "#B6723F", "#91552D", "#53311B")
MISSING_FILL, MISSING_EDGE = "#FAF9F7", "#AAA39E"
STEM = "fig_obj1_diagnosis_by_tone"
OUTDIR = Path("resultados") / "figuras"
SURFACE, PANEL, BORDER, GRID = "#FBFBFA", "#FAF9F7", "#AAA39E", "#E5DED8"
INK, MUTED = "#302C29", "#645D58"
DIAGNOSES = {
    "nv": ("NV", "#A58B7B", "#725043"), "nev": ("NV", "#A58B7B", "#725043"),
    "mel": ("MEL", "#70696A", "#4A4041"), "bkl": ("BKL", "#BCA390", "#8E7768"),
    "bcc": ("BCC", "#BE8277", "#9F5D50"), "akiec": ("AKIEC", "#BDA1A1", "#967478"),
    "ak": ("AK/ACK", "#C5ACB7", "#9A7785"), "ack": ("AK/ACK", "#C5ACB7", "#9A7785"),
    "vasc": ("VASC", "#AC7079", "#855660"), "df": ("DF", "#A8A59B", "#77796F"),
    "scc": ("SCC", "#D3A7A3", "#AD7974"), "sek": ("SEK", "#9D8C81", "#766559"),
    "benign": ("Benigno", "#E5D9CF", "#8A6C58"),
    "malignant": ("Maligno", "#E5D9CF", "#8A6C58"),
}
DIAGNOSIS_ORDER = ("NV", "MEL", "BKL", "BCC", "AKIEC", "AK/ACK", "VASC", "DF", "SCC", "SEK", "Benigno", "Maligno")


def diagnosis_style(label: str) -> tuple[str, str, str]:
    try:
        return DIAGNOSES[label.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown diagnosis: {label!r}") from exc


def empty_counts() -> dict[str, int]:
    return dict.fromkeys(BUCKETS, 0)


def load_e5_counts(repo: Path, dataset: str) -> dict[str, dict[str, int]]:
    if dataset not in ita.DATASETS:
        raise ValueError(f"Unknown E5 dataset: {dataset}")
    raw, rows = ita._load(Path(repo) / ita.E5_DIR / f"e5_fitzpatrick_{dataset}.json")
    if not raw["per_image"]["rows"]:
        raise ValueError(f"E5 dataset has no images: {dataset}")
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        diagnosis = diagnosis_style(row["label"])[0]
        bucket = row["roi_bucket"] if row["roi_status"] == "ok" else "missing"
        if bucket not in BUCKETS or bucket is None:
            raise ValueError(f"Invalid ROI-ITA bucket: {dataset} {row['image_id']}")
        counts.setdefault(diagnosis, empty_counts())[bucket] += 1
    return counts


def count_pad(records: list[pad.PadRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        diagnosis = diagnosis_style(record.diagnostic)[0]
        if record.fst is not None and record.fst not in pad.FST_VALID:
            raise ValueError(f"Invalid PAD FST: {record.fst}")
        bucket = ita.BUCKETS[record.fst - 1] if record.fst else "missing"
        counts.setdefault(diagnosis, empty_counts())[bucket] += 1
    return counts


def load_pad_counts(metadata: Path) -> dict[str, dict[str, int]]:
    official = next(item for item in pad_files.OFFICIAL_FILES if item.name == pad_files.METADATA_NAME)
    pad_files.verify(metadata, official)
    records = pad.parse_metadata(metadata)
    pad.check_published_counts(records)
    return count_pad(records)


def figure_counts(repo: Path, metadata: Path) -> dict[str, dict[str, dict[str, int]]]:
    counts = {dataset: load_e5_counts(repo, dataset) for dataset in ita.DATASETS}
    counts["pad_ufes_20"] = load_pad_counts(metadata)
    return counts


def _diagnoses(rows: dict[str, dict[str, int]]) -> list[str]:
    return [diagnosis for diagnosis in DIAGNOSIS_ORDER if diagnosis in rows]


def _style_axis(ax, title: str, subtitle: str, rows: dict[str, dict[str, int]], slots: int) -> list[float]:
    ax.set_facecolor(PANEL)
    ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=11, color=INK, pad=12, linespacing=1.6)
    labels = _diagnoses(rows)
    positions = [(slots - len(labels)) / 2 + index for index in range(len(labels))]
    ax.set_yticks(positions, labels=labels, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xticks((0, 25, 50, 75, 100))
    ax.set_xticklabels(("0", "25", "50", "75", "100 %"), color=MUTED, fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=15)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.8)
    for tick, label in zip(ax.get_yticklabels(), labels, strict=True):
        _, fill, edge = next(style for style in DIAGNOSES.values() if style[0] == label)
        tick.set_color(ita._text_color(fill))
        tick.set_bbox({"facecolor": fill, "edgecolor": edge,
                       "boxstyle": "square,pad=0.22"})
    return positions


def _plot_panel(ax, dataset: str, rows: dict[str, dict[str, int]], slots: int) -> None:
    total = sum(sum(counts.values()) for counts in rows.values())
    valid = sum(sum(counts[bucket] for bucket in ita.BUCKETS) for counts in rows.values())
    title = "PAD-UFES-20 · FST registrado" if dataset == "pad_ufes_20" else f"{ita.DATASET_LABELS[dataset]} · grupos ITA estimados"
    category = "FST registrado" if dataset == "pad_ufes_20" else "ITA estimado"
    subtitle = (f"N = {bf.fmt_es(total, 0)} · con {category} = {bf.fmt_es(valid, 0)} "
                f"· sin dato = {bf.fmt_es(total - valid, 0)}")
    positions = _style_axis(ax, title, subtitle, rows, slots)
    labels = _diagnoses(rows)
    for y, diagnosis in zip(positions, labels, strict=True):
        counts = rows[diagnosis]
        n = sum(counts.values())
        left = 0.0
        for index, bucket in enumerate(BUCKETS):
            width = 100 * counts[bucket] / n
            if not width:
                continue
            color = SKIN_FILLS[index] if bucket != "missing" else MISSING_FILL
            edge = ita._mix(color, "black", 0.7) if bucket != "missing" else MISSING_EDGE
            ax.barh(y, width, left=left, height=0.69, color=color, edgecolor=edge,
                    linewidth=0.65, hatch="///" if bucket == "missing" else None, zorder=3)
            left += width
        ax.text(101.5, y, f"n={bf.fmt_es(n, 0)}", va="center", fontsize=8.5, color=INK)
    ax.set_ylim(slots - 0.4, -0.6)


def _write_counts(path: Path, all_counts: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dataset", "diagnosis", "n_total", *ita.BUCKETS, "missing"))
        for dataset in DATASETS:
            for diagnosis in _diagnoses(all_counts[dataset]):
                counts = all_counts[dataset][diagnosis]
                writer.writerow((dataset, diagnosis, sum(counts.values()), *(counts[b] for b in BUCKETS)))


def build(repo: Path, pad_metadata: Path, outdir: Path | None = None) -> list[Path]:
    repo = Path(repo)
    target = Path(outdir) if outdir else repo / OUTDIR
    target.mkdir(parents=True, exist_ok=True)
    all_counts = figure_counts(repo, Path(pad_metadata))
    bf.apply_style(width=16)
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor=SURFACE)
    slots = max(len(_diagnoses(rows)) for rows in all_counts.values())
    for ax, dataset in zip(axes.flat, DATASETS, strict=True):
        _plot_panel(ax, dataset, all_counts[dataset], slots)
    handles = [patches.Patch(facecolor=color, edgecolor=ita._mix(color, "black", 0.7), label=bucket)
               for bucket, color in zip(ita.BUCKETS, SKIN_FILLS, strict=True)]
    handles.append(patches.Patch(facecolor=MISSING_FILL, edgecolor=MISSING_EDGE,
                                 hatch="///", label="Sin ITA / sin FST"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.075), ncol=7,
               frameon=False, labelcolor=INK, fontsize=11)
    fig.text(0.06, 0.014, "HAM/ISIC: ITA de ROI no validado como fototipo; PAD: FST registrado en metadatos. "
             "Variables NO equivalentes. PAD V–VI: solo descriptivo.", color=MUTED, fontsize=10)
    fig.subplots_adjust(left=0.13, right=0.95, top=0.95, bottom=0.17, wspace=0.32, hspace=0.52)
    paths = bf.save_all(fig, STEM, target)
    plt.close(fig)
    _write_counts(target / f"{STEM}.csv", all_counts)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pad-metadata", type=Path, required=True)
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()
    for path in build(REPO, args.pad_metadata, args.outdir):
        print(path)
