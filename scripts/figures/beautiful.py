"""Shared style of the thesis figures, after AndreyChurkin/BeautifulFigures (Purple Teal palette).

Principles taken from its reference script: filled marks with a darker edge, a major and a minor grid drawn
below the data, a horizontal frameless legend on top, a monospace font sized to the figure width, and a
vector export (PDF + SVG) next to a 300 dpi PNG. Numbers never live here: each figure script reads them from
the committed result files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

FILLS = ("#9671bd", "#7e7e7e", "#77b5b6")
EDGES = ("#6a408d", "#4e4e4e", "#378d94")
PREFERRED_FONTS = ("Courier New", "DejaVu Sans Mono")
FONT_PT_PER_INCH_OF_WIDTH = 2.0  # 20 pt at 10 in wide, as in the reference script
DPI_PNG = 300
MAJOR_GRID = {"linewidth": 0.75, "alpha": 0.25}
MINOR_GRID = {"linewidth": 0.25, "alpha": 0.15}
MINUS = "−"
EXPORT_SUFFIXES = (".pdf", ".svg", ".png")


def _first_installed_font() -> str:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((name for name in PREFERRED_FONTS if name in installed), PREFERRED_FONTS[-1])


FONT_FAMILY = _first_installed_font()


def apply_style(width: float = 10.0) -> None:
    """Set the global rcParams for a figure ``width`` inches wide (font size scales with the width)."""
    size = FONT_PT_PER_INCH_OF_WIDTH * width
    matplotlib.rcParams.update({
        "font.family": FONT_FAMILY, "font.size": size, "axes.titlesize": size, "axes.labelsize": size,
        "xtick.labelsize": size, "ytick.labelsize": size, "legend.fontsize": size,
        "svg.fonttype": "none", "pdf.fonttype": 42, "axes.unicode_minus": True,
    })


def style_axes(ax) -> None:
    """Major and minor grids below the data, minor ticks on."""
    ax.set_axisbelow(True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="major", **MAJOR_GRID)
    ax.grid(True, which="minor", **MINOR_GRID)
    ax.tick_params(which="minor", length=0)


def top_legend(ax, handles=None, labels=None, ncol: int | None = None, **kwargs):
    """Horizontal legend without frame, centred above the axes."""
    handles, labels = (handles, labels) if handles is not None else ax.get_legend_handles_labels()
    return ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol or len(labels),
                     frameon=False, **kwargs)


def save_all(fig, stem: str, outdir: Path) -> list[Path]:
    """Write ``stem`` as PDF, SVG and PNG (300 dpi) into ``outdir`` and return the three paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / f"{stem}{suffix}" for suffix in EXPORT_SUFFIXES]
    for path in paths:
        fig.savefig(path, dpi=DPI_PNG, bbox_inches="tight")
    return paths


def fmt_es(value: float, decimals: int = 2) -> str:
    """Spanish number format: decimal comma, dot thousands, a real minus sign."""
    text = f"{abs(value):,.{decimals}f}".replace(",", "\0").replace(".", ",").replace("\0", ".")
    return f"{MINUS if value < 0 else ''}{text}"


def close(fig) -> None:
    plt.close(fig)
