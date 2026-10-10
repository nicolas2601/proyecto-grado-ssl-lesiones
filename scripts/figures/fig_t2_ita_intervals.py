"""T2: LaTeX table of ITA estimator intervals from verified source thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from dermssl.fitzpatrick import FST_THRESHOLDS, MIN_B_FOR_ITA, fst_bucket

STEM = "tab_t2_ita_intervals"
SOURCE = "src/dermssl/fitzpatrick.py"
GROUPS = ("I", "II", "III", "IV", "V", "VI")
CAPTION = "Reglas de pertenencia para grupos del estimador ITA, según los cortes de Groh."
NOTE = (
    "Los grupos I–VI son categorías del estimador ITA; no equivalen a fototipos clínicos validados ni a fototipos Fitzpatrick. "
    "Cada corte exacto se asigna al grupo más oscuro. La estimación requiere mediana b* ≥ 5; los casos en abstención no reciben grupo."
)


def _degree(value: float) -> str:
    return f"{value:g}".replace("-", "−")


def _latex_degree(value: float) -> str:
    return f"{value:g}"


def intervals() -> list[dict]:
    thresholds = tuple(float(value) for value in FST_THRESHOLDS)
    if len(thresholds) + 1 != len(GROUPS) or any(
        upper <= lower for upper, lower in zip(thresholds, thresholds[1:])
    ):
        raise ValueError(f"ITA thresholds must be descending and define six groups: {thresholds}")
    rows = []
    for index, threshold in enumerate(thresholds):
        group = GROUPS[index]
        if fst_bucket(threshold) != index + 2:
            raise ValueError(f"ITA boundary {threshold} does not map to the darker group")
        if index == 0:
            condition = f"ITA > {_degree(threshold)}°"
            latex = rf"$\mathrm{{ITA}} > {_latex_degree(threshold)}^\circ$"
        else:
            upper = thresholds[index - 1]
            condition = f"{_degree(threshold)}° < ITA ≤ {_degree(upper)}°"
            latex = rf"${_latex_degree(threshold)}^\circ < \mathrm{{ITA}} \le {_latex_degree(upper)}^\circ$"
        rows.append({"group": group, "condition": condition, "latex": latex})
    last_threshold = thresholds[-1]
    if fst_bucket(last_threshold) != len(GROUPS):
        raise ValueError(f"ITA boundary {last_threshold} does not map to the darkest group")
    rows.append({
        "group": GROUPS[-1],
        "condition": f"ITA ≤ {_degree(last_threshold)}°",
        "latex": rf"$\mathrm{{ITA}} \le {_latex_degree(last_threshold)}^\circ$",
    })
    return rows


def render_table() -> str:
    body = "\n".join(f"Grupo {row['group']} & {row['latex']} " + r"\\" for row in intervals())
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{CAPTION}}}\n"
        "\\label{tab:ita-estimator-intervals}\n"
        "\\begin{tabular}{ll}\n"
        "\\hline\n"
        "Grupo del estimador & Regla de pertenencia (grados) \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\par\\smallskip\n"
        "\\begin{minipage}{0.94\\linewidth}\n"
        f"\\footnotesize \\textit{{Nota.}} {NOTE}\n"
        "\\end{minipage}\n"
        "\\end{table}\n"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision(repo: Path, relative: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%H", "--", relative],
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _traceability(repo: Path) -> dict:
    source = repo / SOURCE
    script = repo / "scripts/figures/fig_t2_ita_intervals.py"
    return {
        "figure_id": "T2",
        "artifact_type": "latex_table",
        "source": {"path": SOURCE, "sha256": _sha256(source),
                   "last_changed_commit": _git_revision(repo, SOURCE)},
        "thresholds": list(FST_THRESHOLDS),
        "reporting_guard": {"median_b_star_minimum": MIN_B_FOR_ITA, "comparison": ">="},
        "boundary_rule": "ITA equals a cut point maps to the darker group, per fst_bucket's strict > comparisons",
        "groups": intervals(),
        "script": {"path": "scripts/figures/fig_t2_ita_intervals.py", "sha256": _sha256(script)},
        "status": "ITA estimator categories; not validated clinical Fitzpatrick phototypes",
    }


def build(repo: Path = REPO, outdir: Path | None = None) -> Path:
    repo = Path(repo)
    target = Path(outdir) if outdir is not None else (
        repo / "resultados/figuras/thesis_c1_c2_20261009_final/T2"
    )
    suffixes = (".tex", ".caption.txt", ".note.txt", ".traceability.json")
    collisions = [target / f"{STEM}{suffix}" for suffix in suffixes if (target / f"{STEM}{suffix}").exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite {collisions[0]}")
    target.mkdir(parents=True, exist_ok=True)
    table_path = target / f"{STEM}.tex"
    table_path.write_text(render_table(), encoding="utf-8")
    (target / f"{STEM}.caption.txt").write_text(CAPTION + "\n", encoding="utf-8")
    (target / f"{STEM}.note.txt").write_text(NOTE + "\n", encoding="utf-8")
    (target / f"{STEM}.traceability.json").write_text(
        json.dumps(_traceability(repo), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return table_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()
    print(build(outdir=args.outdir))
