"""T1: conceptual ITA geometry in CIELAB coordinates."""

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
from matplotlib.patches import Arc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thesis_style as style  # noqa: E402
from dermssl.fitzpatrick import MIN_B_FOR_ITA  # noqa: E402

STEM = "fig_t1_ita_geometry"
SOURCE = "src/dermssl/fitzpatrick.py"
ITA_FORMULA = "ITA = arctan((L* − 50) / b*) × 180 / π"
CAPTION = "Esquema conceptual del ángulo ITA a partir de las coordenadas CIELAB L* y b*."
NOTE = (
    "ITA = arctan((L*−50)/b*) expresado en grados. El estimador reporta ITA cuando la mediana b* ≥ 5; "
    "de lo contrario, se abstiene. Las flechas y ángulos son esquemáticos, no corresponden a observaciones; "
    "el esquema no constituye validación clínica."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision(repo: Path, relative: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%H", "--", relative],
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def draw():
    with style.context():
        fig = plt.figure(figsize=(style.FIGURE_WIDTH_IN, 5.8), facecolor=style.COLORS["background"])
        axis = fig.add_axes((0.20, 0.13, 0.62, 0.68))
        axis.set_xlim(-0.35, 3.65)
        axis.set_ylim(-2.05, 2.05)
        axis.set_aspect("equal", adjustable="box")
        axis.annotate("", xy=(3.55, 0), xytext=(-0.25, 0),
                      arrowprops={"arrowstyle": "-|>", "color": style.COLORS["frame"],
                                  "linewidth": style.LINE_WIDTHS["frame"]})
        axis.annotate("", xy=(0, 1.95), xytext=(0, -1.95),
                      arrowprops={"arrowstyle": "-|>", "color": style.COLORS["frame"],
                                  "linewidth": style.LINE_WIDTHS["frame"]})
        for endpoint in ((2.95, 1.55), (2.95, -1.55)):
            axis.annotate("", xy=endpoint, xytext=(0, 0),
                          arrowprops={"arrowstyle": "-|>", "color": style.COLORS["accent"],
                                      "linewidth": style.LINE_WIDTHS["curve"]})
        axis.add_patch(Arc((0, 0), 1.3, 1.3, theta1=0, theta2=27.7,
                           color=style.COLORS["text"], linewidth=style.LINE_WIDTHS["frame"]))
        axis.add_patch(Arc((0, 0), 1.3, 1.3, theta1=-27.7, theta2=0,
                           color=style.COLORS["text"], linewidth=style.LINE_WIDTHS["frame"]))
        axis.plot(0, 0, marker="o", markersize=3, color=style.COLORS["text"], zorder=4)
        axis.text(0.9, 0.27, "θ+", ha="center", color=style.COLORS["text"])
        axis.text(0.9, -0.42, "θ−", ha="center", color=style.COLORS["text"])
        axis.text(1.65, 1.55, "ITA > 0", color=style.COLORS["text"])
        axis.text(1.65, -1.78, "ITA < 0", color=style.COLORS["text"])
        axis.set_xlabel("b* (CIELAB)", labelpad=8)
        axis.set_ylabel("L* − 50", labelpad=8)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        fig.text(0.5, 0.97, "Geometría del ángulo ITA", ha="center", va="top",
                 fontsize=style.PANEL_TITLE_PT, color=style.COLORS["text"])
        fig.text(0.5, 0.90, ITA_FORMULA, ha="center", va="center",
                 fontsize=style.LABEL_PT, color=style.COLORS["text"])
        fig.text(0.5, 0.85, f"Se reporta si la mediana b* ≥ {style.fmt_es(MIN_B_FOR_ITA, 0)}",
                 ha="center", va="center", fontsize=style.LABEL_PT, color=style.COLORS["text"])
        fig.text(0.5, 0.04, "Esquema conceptual; flechas sin datos de pacientes.", ha="center",
                 fontsize=style.LABEL_PT, color=style.COLORS["text"])
    return fig, axis


def _traceability(repo: Path) -> dict:
    source = repo / SOURCE
    script = repo / "scripts/figures/fig_t1_ita_geometry.py"
    return {
        "figure_id": "T1",
        "kind": "conceptual_schematic",
        "formula": "arctan((L*-50)/b*) * 180/pi",
        "reporting_guard": {"median_b_star_minimum": MIN_B_FOR_ITA, "comparison": ">="},
        "source": {"path": SOURCE, "sha256": _sha256(source),
                   "last_changed_commit": _git_revision(repo, SOURCE)},
        "figure_script": {"path": "scripts/figures/fig_t1_ita_geometry.py", "sha256": _sha256(script)},
        "status": "conceptual; no patient observations; not clinical validation",
    }


def build(repo: Path = REPO, outdir: Path | None = None) -> list[Path]:
    repo = Path(repo)
    target = Path(outdir) if outdir is not None else (
        repo / "resultados/figuras/thesis_c1_c2_20261009_final/T1"
    )
    suffixes = (*style.EXPORT_SUFFIXES, ".caption.txt", ".note.txt", ".traceability.json")
    collisions = [target / f"{STEM}{suffix}" for suffix in suffixes if (target / f"{STEM}{suffix}").exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite {collisions[0]}")
    target.mkdir(parents=True, exist_ok=True)
    fig, _ = draw()
    try:
        paths = style.save_figure(fig, STEM, target)
    finally:
        plt.close(fig)
    (target / f"{STEM}.caption.txt").write_text(CAPTION + "\n", encoding="utf-8")
    (target / f"{STEM}.note.txt").write_text(NOTE + "\n", encoding="utf-8")
    (target / f"{STEM}.traceability.json").write_text(
        json.dumps(_traceability(repo), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()
    for path in build(outdir=args.outdir):
        print(path)
