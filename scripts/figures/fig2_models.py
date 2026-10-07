"""Thesis figure 2: the clean A71/B71 weights against the public DINOv2 controls and the 8 baselines.

Three rows of panels (macro-F1 with CI95, MCC, paired deltas clean - public), one column per dataset. Every number
comes from the bench loaders (``dermssl.bench.rows`` and ``dermssl.bench.contrasts_doc``); nothing is parsed from the
markdown tables and no interval is invented.

Regenerate: PYTHONPATH=src:$HOME/work/tmp/mpl ~/.cache/tvenv/bin/python scripts/figures/fig2_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import beautiful as bf  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator, NullLocator  # noqa: E402

from dermssl.bench import contrasts_doc, rows as R  # noqa: E402

STEM = "fig2_models"
FIG_WIDTH, FIG_HEIGHT, FONT_WIDTH = 18.0, 14.0, 7.0  # FONT_WIDTH 7 -> 14 pt text, readable with 12 rows per panel
ARMS = ("A71", "B71")
ARM_COLOR = {"A71": 0, "B71": 2}  # index into bf.FILLS / bf.EDGES; everything else is gray (index 1)
GRAY = 1
METRICS = ("macro_f1", "mcc")
METRIC_LABEL = {"macro_f1": "F1 macro", "mcc": "MCC"}
FT_OFFSET, GROUP_GAP = 0.17, 0.6
PUBLIC_LABEL = {"A71": "DINOv2 público · prot. A71", "B71": "DINOv2 público · prot. B71"}
MCC_NOTE = ("Δ con IC 95 % pareado, mismo protocolo. MCC: intervalo solo donde el banco lo publica (A71, B71 y DINOv2 público en ISIC 2019 e ISIC 2020); en "
            "HAM10000 y en los baselines se dibuja el punto sin intervalo.")


def _point(value: R.Value) -> tuple[float, float | None, float | None]:
    return (value.point, *(value.ci if value.ci else (None, None)))


def _pick(rows: list[R.Row], block: str, prefix: str, exclude: str = "mejor protocolo") -> R.Row:
    hits = [r for r in rows if r.block == block and r.status == R.OK and r.name.startswith(prefix)
            and exclude not in r.name]
    assert len(hits) == 1, (block, prefix, [r.name for r in hits])
    return hits[0]


def _dataset_models(rows: list[R.Row], key: str) -> dict[str, R.Row]:
    own_block = R.CLEAN_HAM if key == "ham10000" else R.FROZEN
    out = {f"{arm}-limpio": _pick(rows, own_block, f"{arm}-limpio") for arm in ARMS}
    out.update({PUBLIC_LABEL[arm]: _pick(rows, R.FROZEN, f"DINOv2 público, protocolo de {arm}") for arm in ARMS})
    for name in R.MODEL_NAMES.values():
        out[name] = next(r for r in rows if r.block == R.FROZEN and r.status == R.OK and r.name == name)
    return out


def _deltas(items: list, key: str) -> dict[tuple[str, str], tuple[float, float, float]]:
    out = {}
    for arm in ARMS:
        for metric in METRICS:
            hits = [c for c in items if c.arm == f"{arm}-limpio" and c.control == f"DINOv2 público (prot. {arm})"
                    and c.metric == metric and ("ham_clean" in c.source if key == "ham10000" else f"/{key}/" in c.source)]
            assert len(hits) == 1, (key, arm, metric, len(hits))
            out[(arm, metric)] = (hits[0].delta, hits[0].low, hits[0].high)
    return out


def _baseline_order(per_ds: dict[str, dict[str, R.Row]]) -> list[str]:
    mean = {n: sum(m[n].values["macro_f1"].point for m in per_ds.values()) / len(per_ds) for n in R.MODEL_NAMES.values()}
    return sorted(mean, key=lambda n: -mean[n])


def figure_data(repo: Path) -> dict:
    """Everything the figure draws, read through the bench loaders."""
    repo = Path(repo)
    _, items = contrasts_doc.build(repo)
    rows = {k: R.dataset_rows(repo, ds) for k, ds in R.DATASETS.items()}
    models = {k: _dataset_models(rows[k], k) for k in rows}
    order = [f"{a}-limpio" for a in ARMS] + list(PUBLIC_LABEL.values()) + _baseline_order(models)
    data = {"order": order, "datasets": {}, "sources": ["dermssl.bench.rows", "dermssl.bench.contrasts_doc"]}
    for key, ds in R.DATASETS.items():
        entry = {"title": ds.title, "deltas": _deltas(items, key), "finetune": {}}
        for metric in METRICS:
            entry[metric] = {n: _point(models[key][n].values[metric]) for n in order}
        if key == "isic2020":
            ft = {r.name: r for r in rows[key] if r.block == R.FINETUNE and r.status == R.OK}
            entry["finetune"] = {n: {m: _point(r.values[m]) for m in METRICS} for n, r in ft.items()}
        data["datasets"][key] = entry
    return data


def _ypos(n_models: int) -> list[float]:
    """Slots for the model list: A71/B71, the two public controls and the baselines, with a gap between groups."""
    return [0, 1, 2 + GROUP_GAP, 3 + GROUP_GAP] + [4 + 2 * GROUP_GAP + i for i in range(n_models - 4)]


def _color(name: str) -> tuple[str, str]:
    idx = ARM_COLOR.get(name[:3], GRAY) if name.endswith("-limpio") else GRAY
    return bf.FILLS[idx], bf.EDGES[idx]


def _draw(ax, y: float, triple, fill: str, edge: str, hollow: bool = False, marker: str = "o") -> None:
    point, low, high = triple
    if low is not None:
        ax.hlines(y, low, high, color=edge, linewidth=1.6, zorder=2)
    ax.plot(point, y, marker, markersize=8, markerfacecolor="white" if hollow else fill, markeredgecolor=edge,
            markeredgewidth=1.6 if hollow else 1.0, zorder=3)


def _model_panel(ax, entry: dict, order: list[str], metric: str) -> None:
    ys = _ypos(len(order))
    ft = entry["finetune"]
    for y, name in zip(ys, order):
        fill, edge = _color(name)
        shift = FT_OFFSET if ft else 0.0
        _draw(ax, y - shift, entry[metric][name], fill, edge)
        if name in ft:
            _draw(ax, y + shift, ft[name][metric], fill, edge, hollow=True)
        if name.endswith("-limpio"):
            point = entry[metric][name][0]
            ax.annotate(bf.fmt_es(point, 2), (entry[metric][name][2] or point, y - shift), xytext=(6, 0),
                        textcoords="offset points", va="center", color=edge, fontsize=11)


def _delta_panel(ax, entry: dict) -> None:
    ax.axvline(0, color=bf.EDGES[GRAY], linewidth=1.0, linestyle="--", zorder=1)
    slots = [(arm, metric) for arm in ARMS for metric in METRICS]
    for y, (arm, metric) in zip((0, 1, 2 + GROUP_GAP, 3 + GROUP_GAP), slots):
        idx = ARM_COLOR[arm]
        triple = entry["deltas"][(arm, metric)]
        _draw(ax, y, triple, bf.FILLS[idx], bf.EDGES[idx], marker="o" if metric == "macro_f1" else "D")
        ax.annotate("+" + bf.fmt_es(triple[0], 2) if triple[0] >= 0 else bf.fmt_es(triple[0], 2), (triple[2], y),
                    xytext=(6, 0), textcoords="offset points", va="center", color=bf.EDGES[idx], fontsize=11)


def _delta_labels() -> list[tuple[float, str]]:
    return [(y, f"{arm}-limpio · {METRIC_LABEL[m]}") for y, (arm, m) in
            zip((0, 1, 2 + GROUP_GAP, 3 + GROUP_GAP), [(a, m) for a in ARMS for m in METRICS])]


def _legend_handles() -> list[Line2D]:
    def dot(fill, edge, hollow=False):
        return Line2D([], [], marker="o", linestyle="", markersize=9, markeredgecolor=edge,
                      markerfacecolor="white" if hollow else fill)
    return [dot(bf.FILLS[0], bf.EDGES[0]), dot(bf.FILLS[2], bf.EDGES[2]), dot(bf.FILLS[1], bf.EDGES[1]),
            dot(bf.FILLS[1], bf.EDGES[1], hollow=True)]


LEGEND_LABELS = ["A71 limpio", "B71 limpio", "congelado (controles y baselines)", "con fine-tuning (solo ISIC 2020)"]


def _format_axis(ax) -> None:
    bf.style_axes(ax)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: bf.fmt_es(v, 2)))
    ax.tick_params(axis="x", labelsize=12)


def build(repo: Path, outdir: Path) -> list[Path]:
    """Draw the figure and write ``fig2_models`` as pdf, svg and png."""
    data = figure_data(repo)
    bf.apply_style(FONT_WIDTH)
    order, keys = data["order"], list(data["datasets"])
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    grid = GridSpec(4, 3, figure=fig, height_ratios=[0.01, 12.4, 12.4, 4.9], hspace=0.38, wspace=0.07)
    top = fig.add_subplot(grid[0, :])
    top.axis("off")
    bf.top_legend(top, _legend_handles(), LEGEND_LABELS, handletextpad=0.3, columnspacing=1.6)
    axes = [[fig.add_subplot(grid[r, c]) for c in range(3)] for r in (1, 2, 3)]
    for r, row in enumerate(axes):
        for c, ax in enumerate(row):
            if c:
                ax.sharey(row[0])
            entry = data["datasets"][keys[c]]
            _delta_panel(ax, entry) if r == 2 else _model_panel(ax, entry, order, METRICS[r])
            _format_axis(ax)
            if c:
                ax.tick_params(axis="y", labelleft=False)
        ypos = _ypos(len(order)) if r < 2 else [y for y, _ in _delta_labels()]
        row[0].set_yticks(ypos, order if r < 2 else [t for _, t in _delta_labels()])
        row[0].set_ylim((ypos[-1] + 0.7, -0.7))
        row[0].yaxis.set_minor_locator(NullLocator())
        for ax in row:
            ax.set_xlabel(METRIC_LABEL[METRICS[r]] if r < 2 else "Δ limpio − público")
            ax.set_xlim(*(_shared_xlim(data, METRICS[r]) if r < 2 else _delta_xlim(data["datasets"][keys[ax.get_subplotspec().colspan.start]])))
            if r == 2:
                ax.xaxis.set_major_locator(MultipleLocator(0.05))
    for c, key in enumerate(keys):
        axes[0][c].set_title(data["datasets"][key]["title"], fontweight="normal", pad=8)
    fig.text(0.01, 0.065, MCC_NOTE, ha="left", va="top", fontsize=11)
    paths = bf.save_all(fig, STEM, outdir)
    bf.close(fig)
    return paths


def _delta_xlim(entry: dict) -> tuple[float, float]:
    lows = [t[1] for t in entry["deltas"].values()] + [0.0]
    highs = [t[2] for t in entry["deltas"].values()]
    span = max(highs) - min(lows)
    return min(lows) - 0.05 * span, max(highs) + 0.22 * span


def _shared_xlim(data: dict, metric: str) -> tuple[float, float]:
    vals = []
    for entry in data["datasets"].values():
        triples = list(entry[metric].values()) + [t[metric] for t in entry["finetune"].values()]
        vals += [v for p, lo, hi in triples for v in (p, lo, hi) if v is not None]
    pad = 0.08 * (max(vals) - min(vals))
    return min(vals) - pad, max(vals) + pad + 0.06


if __name__ == "__main__":
    for path in build(REPO, REPO / "resultados" / "figuras"):
        print(path)
