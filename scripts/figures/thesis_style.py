"""Shared, fixed-size style for thesis C1/C2 and later figure exports."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
from typing import Iterator

import matplotlib
from matplotlib import font_manager

FIGURE_WIDTH_CM = 17.3
CM_PER_INCH = 2.54
FIGURE_WIDTH_IN = FIGURE_WIDTH_CM / CM_PER_INCH
DPI = 300
LABEL_PT = 11
PANEL_TITLE_PT = 12
FONT_CANDIDATES = ("Courier New", "Nimbus Mono PS")

COLORS = {
    "background": "#FFFFFF",
    "bar_light": "#F4EDE6",
    "accent": "#8A6C58",
    "text": "#222222",
    "frame": "#AAA39E",
    "abstention": "#FAF9F7",
}
ITA_COLORS = ("#FEF3E0", "#F2C598", "#CE8B51", "#B5723F", "#91552D", "#53311B")
LINE_WIDTHS = {"frame": 0.6, "bar": 0.7, "curve": 1.1}
EXPORT_SUFFIXES = (".pdf", ".svg", ".png")


def _fontconfig_path(family: str) -> str | None:
    executable = shutil.which("fc-match")
    if executable is None:
        return None
    result = subprocess.run([executable, "-f", "%{file}", family], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    path = result.stdout.strip()
    if not path or not Path(path).is_file():
        return None
    if font_manager.FontProperties(fname=path).get_name().casefold() != family.casefold():
        return None
    font_manager.fontManager.addfont(path)
    return path


def _resolve_font() -> str:
    for family in FONT_CANDIDATES:
        try:
            font_path = font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            font_path = _fontconfig_path(family)
            if font_path is None:
                continue
        installed_family = font_manager.FontProperties(fname=font_path).get_name()
        if installed_family.casefold() == family.casefold():
            return family
    choices = " or ".join(FONT_CANDIDATES)
    raise RuntimeError(f"Install {choices} before exporting thesis figures")


FONT_FAMILY = _resolve_font()


def rc_params() -> dict:
    """Return explicit rcParams; font sizes never scale with figure width."""
    return {
        "font.family": FONT_FAMILY,
        "font.size": LABEL_PT,
        "font.weight": "normal",
        "axes.titlesize": PANEL_TITLE_PT,
        "axes.titleweight": "normal",
        "axes.labelsize": LABEL_PT,
        "xtick.labelsize": LABEL_PT,
        "ytick.labelsize": LABEL_PT,
        "legend.fontsize": LABEL_PT,
        "text.color": COLORS["text"],
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "figure.facecolor": COLORS["background"],
        "axes.facecolor": COLORS["background"],
        "savefig.facecolor": COLORS["background"],
        "savefig.edgecolor": COLORS["background"],
        "savefig.bbox": None,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "pdf.use14corefonts": False,
        "axes.linewidth": LINE_WIDTHS["frame"],
        "grid.linewidth": LINE_WIDTHS["frame"],
        "lines.linewidth": LINE_WIDTHS["curve"],
        "axes.unicode_minus": True,
    }


@contextmanager
def context() -> Iterator[None]:
    """Apply thesis style locally without altering figures using BeautifulFigures."""
    with matplotlib.rc_context(rc=rc_params()):
        yield


def style_axes(ax, grid_axis: str = "x") -> None:
    ax.set_facecolor(COLORS["background"])
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=COLORS["frame"], linewidth=LINE_WIDTHS["frame"], alpha=0.28)
    ax.tick_params(colors=COLORS["text"], length=0, pad=5)
    for spine in ax.spines.values():
        spine.set_color(COLORS["frame"])
        spine.set_linewidth(LINE_WIDTHS["frame"])


def fmt_es(value: float, decimals: int = 2, *, signed: bool = False) -> str:
    prefix = "+" if signed and value >= 0 else ""
    rendered = f"{abs(value):,.{decimals}f}".replace(",", "\0").replace(".", ",").replace("\0", ".")
    sign = "−" if value < 0 else prefix
    return f"{sign}{rendered}"


def save_figure(fig, stem: str, outdir: Path) -> list[Path]:
    """Export one figure as editable SVG, embedded-font PDF, and 300 dpi PNG."""
    if not stem or Path(stem).name != stem:
        raise ValueError(f"Figure stem must be a filename: {stem!r}")
    target = Path(outdir)
    paths = [target / f"{stem}{suffix}" for suffix in (".pdf", ".svg", ".png")]
    collision = next((path for path in paths if path.exists()), None)
    if collision is not None:
        raise FileExistsError(f"Refusing to overwrite {collision}")

    target.mkdir(parents=True, exist_ok=True)
    height = float(fig.get_size_inches()[1])
    fig.set_size_inches(FIGURE_WIDTH_IN, height, forward=True)
    fig.set_facecolor(COLORS["background"])
    with context():
        for path in paths:
            fig.savefig(path, format=path.suffix[1:], dpi=DPI,
                        facecolor=COLORS["background"], edgecolor=COLORS["background"],
                        bbox_inches=None)
    svg_path = target / f"{stem}.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
                        encoding="utf-8")
    return paths
