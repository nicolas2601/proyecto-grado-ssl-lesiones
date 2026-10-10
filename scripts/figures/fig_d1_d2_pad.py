"""D1 and D2: PAD-UFES-20 domain-adaptation readings D0-D3, from the committed E2 evaluation."""

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thesis_style as style  # noqa: E402

SOURCE = "resultados/banco_completo/pad_da/pad_e2_eval.json"
ARMS = ("D0", "D1", "D2", "D3")
ARM_LABELS = {"D0": "D0 público", "D1": "D1 A* limpio", "D2": "D2 control", "D3": "D3 adaptado"}
MARKERS = {"D0": ("o", False), "D1": ("s", False), "D2": ("^", False), "D3": ("^", True)}
CONTRASTS = (("D3", "D2"), ("D3", "D1"), ("D3", "D0"))
CLASS_ORDER = ("bcc", "mel", "nv")
NOTE_D1 = ("Evaluación única en PAD-UFES-20 target_test, lectura primaria BCC/MEL/NV: 567 imágenes de 363 pacientes "
           "(25 melanomas). Barras del panel (a): IC 95 % bootstrap pareado por patient_id, 1.000 remuestras. "
           "El panel (b) muestra sensibilidad puntual por diagnóstico, sin IC (el artefacto no los publica). "
           "D2 y D3 provienen de adaptaciones de 2.000 pasos y no equivalen a B71.")
NOTE_D2 = ("Δ de F1 macro con IC 95 % bootstrap pareado por patient_id (1.000 remuestras, semilla 20260923), sin ajuste por "
           "multiplicidad. D3 − D2 es la lectura registrada (mejora frente al control adaptado); D3 − D0 compara con el "
           "DINOv2 público: el IC queda por debajo de cero, de modo que el público sigue siendo mejor.")


def load(repo: Path) -> dict:
    path = repo / SOURCE
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc["n_images"] != 567 or doc["n_patients"] != 363 or list(doc["classes"]) != list(CLASS_ORDER):
        raise ValueError(f"{path}: unexpected PAD primary reading")
    for arm in ARMS:
        low, high = doc["arms"][arm]["ci95"]["macro_f1"]
        if not low <= doc["arms"][arm]["macro_f1"] <= high:
            raise ValueError(f"{path}: {arm} macro-F1 outside its CI")
    return {"doc": doc, "path": path}


def _marker_kwargs(arm: str) -> dict:
    marker, filled = MARKERS[arm]
    return {"marker": marker, "markeredgecolor": style.COLORS["accent"], "markeredgewidth": style.LINE_WIDTHS["bar"],
            "markerfacecolor": style.COLORS["accent"] if filled else style.COLORS["background"]}


def contrast(doc: dict, arm: str, control: str) -> dict:
    rows = [p for p in doc["paired"] if (p["arm"], p["control"], p["metric"], p["fraction"]) == (arm, control, "macro_f1", 1.0)]
    if len(rows) != 1:
        raise ValueError(f"expected one paired macro_f1 row for {arm}-{control}, got {len(rows)}")
    return rows[0]


def build_d1(repo: Path, outdir: Path) -> list[Path]:
    src = load(repo)
    doc = src["doc"]
    with style.context():
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(style.FIGURE_WIDTH_IN, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})
        for y, arm in enumerate(ARMS):
            f1 = doc["arms"][arm]["macro_f1"]
            lo, hi = doc["arms"][arm]["ci95"]["macro_f1"]
            ax_a.errorbar(f1, y, xerr=[[f1 - lo], [hi - f1]], linestyle="none", markersize=7, color=style.COLORS["accent"],
                          elinewidth=style.LINE_WIDTHS["curve"], capsize=3, capthick=style.LINE_WIDTHS["curve"], **_marker_kwargs(arm))
            ax_a.annotate(style.fmt_es(f1, 3), (f1, y), xytext=(0, 9), textcoords="offset points", ha="center", va="bottom")
        ax_a.set_ylim(len(ARMS) - 0.4, -0.7)
        ax_a.set_yticks(range(len(ARMS)), [ARM_LABELS[a] for a in ARMS])
        ax_a.set_xlim(0, 1)
        ax_a.set_xticks((0, 0.5, 1), [style.fmt_es(v, 1) for v in (0, 0.5, 1)])
        ax_a.set_xlabel("F1 macro")
        ax_a.set_title("(a) F1 macro", loc="left")
        style.style_axes(ax_a, grid_axis="x")
        for y, cls in enumerate(CLASS_ORDER):
            for arm in ARMS:
                recall = doc["arms"][arm]["per_class"][cls]["recall"]
                ax_b.plot([recall], [y + {"D0": -0.21, "D1": -0.07, "D2": 0.07, "D3": 0.21}[arm]], linestyle="none",
                          markersize=7, color=style.COLORS["accent"], **_marker_kwargs(arm))
            support = doc["arms"]["D0"]["per_class"][cls]["support"]
            ax_b.text(1.03, y, f"n = {style.fmt_es(support, 0)}", va="center", ha="left")
        ax_b.set_ylim(len(CLASS_ORDER) - 0.5, -0.5)
        ax_b.set_yticks(range(len(CLASS_ORDER)), [c.upper() for c in CLASS_ORDER])
        ax_b.set_xlim(0, 1)
        ax_b.set_xticks((0, 0.5, 1), [style.fmt_es(v, 1) for v in (0, 0.5, 1)])
        ax_b.set_xlabel("Sensibilidad por diagnóstico")
        ax_b.set_title("(b) Sensibilidad", loc="left")
        ax_b.text(1.03, -0.62, "soporte", ha="left", va="bottom", transform=ax_b.get_yaxis_transform())
        style.style_axes(ax_b, grid_axis="x")
        handles = [plt.Line2D([], [], linestyle="none", markersize=7, color=style.COLORS["accent"], label=ARM_LABELS[a],
                              **_marker_kwargs(a)) for a in ARMS]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, handletextpad=0.2, columnspacing=1.2)
        fig.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.27, wspace=0.6)
        stem = "fig_d1_pad_d0_d3"
        paths = style.save_figure(fig, stem, outdir)
        plt.close(fig)
    _sidecars(repo, outdir, stem, src, "Cómo se comportan D0–D3 en PAD-UFES-20 (fotos clínicas): F1 macro y sensibilidad por diagnóstico.",
              "F1 macro con IC 95 % y sensibilidad por diagnóstico de D0 a D3 en PAD-UFES-20 (target_test, BCC/MEL/NV).", NOTE_D1)
    return paths


def build_d2(repo: Path, outdir: Path) -> list[Path]:
    src = load(repo)
    doc = src["doc"]
    rows = [(arm, ctrl, contrast(doc, arm, ctrl)) for arm, ctrl in CONTRASTS]
    with style.context():
        fig, ax = plt.subplots(figsize=(style.FIGURE_WIDTH_IN, 2.9))
        for y, (arm, ctrl, row) in enumerate(rows):
            delta, lo, hi = row["point_delta"], row["ci95_low"], row["ci95_high"]
            ax.errorbar(delta, y, xerr=[[delta - lo], [hi - delta]], linestyle="none", markersize=7, color=style.COLORS["accent"],
                        elinewidth=style.LINE_WIDTHS["curve"], capsize=3, capthick=style.LINE_WIDTHS["curve"],
                        marker="o" if (arm, ctrl) == ("D3", "D2") else "s", markeredgecolor=style.COLORS["accent"],
                        markeredgewidth=style.LINE_WIDTHS["bar"],
                        markerfacecolor=style.COLORS["accent"] if (arm, ctrl) == ("D3", "D2") else style.COLORS["background"])
            ax.annotate(style.fmt_es(delta, 3, signed=True), (delta, y), xytext=(0, 9), textcoords="offset points",
                        ha="center", va="bottom")
        ax.axvline(0, color=style.COLORS["text"], linewidth=style.LINE_WIDTHS["bar"], zorder=1)
        ax.set_ylim(len(rows) - 0.4, -0.7)
        ax.set_yticks(range(len(rows)), [f"{a} − {c}" for a, c, _ in rows])
        ax.set_xlim(-0.19, 0.19)
        ticks = np.linspace(-0.15, 0.15, 7)
        ax.set_xticks(ticks, [style.fmt_es(v, 2, signed=True) if abs(v) > 1e-9 else "0,00" for v in ticks])
        ax.set_xlabel("Δ F1 macro (brazo − control)")
        style.style_axes(ax, grid_axis="x")
        fig.subplots_adjust(left=0.14, right=0.97, top=0.95, bottom=0.25)
        stem = "fig_d2_pad_contrasts"
        paths = style.save_figure(fig, stem, outdir)
        plt.close(fig)
    _sidecars(repo, outdir, stem, src, "Si la adaptación mejora frente a su control (D3 − D2) y si supera al público (D3 − D0).",
              "Contrastes pareados de F1 macro en PAD-UFES-20 con IC 95 %: D3 − D2, D3 − D1 y D3 − D0.", NOTE_D2,
              extra={"contrasts": [{"arm": a, "control": c, "delta": r["point_delta"], "ci95": [r["ci95_low"], r["ci95_high"]],
                                    "excludes_zero": r["excludes_zero"]} for a, c, r in rows]})
    return paths


def _sidecars(repo, outdir, stem, src, question, caption, note, extra=None) -> None:
    out = Path(outdir)
    doc = src["doc"]
    trace = {"figure": stem, "question": question, "status": "Artefactos verificados", "source": SOURCE,
             "source_sha256": hashlib.sha256(src["path"].read_bytes()).hexdigest(), "code_revision": doc["code_revision"],
             "dataset": "PAD-UFES-20 target_test", "reading": doc["reading"], "n_images": doc["n_images"],
             "n_patients": doc["n_patients"], "bootstrap": doc["bootstrap"], "classes": doc["classes"],
             "missing": ["fracción de predicciones con argmax fuera del mapeo de la lectura: no figura en este artefacto",
                         "IC por diagnóstico: no publicados"], "script": "scripts/figures/fig_d1_d2_pad.py"}
    if extra:
        trace.update(extra)
    (out / f"{stem}.traceability.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / f"{stem}.caption.txt").write_text(caption + "\n", encoding="utf-8")
    (out / f"{stem}.note.txt").write_text(note + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=REPO / "resultados" / "figuras")
    args = parser.parse_args()
    for path in (*build_d1(REPO, args.outdir), *build_d2(REPO, args.outdir)):
        print(path)
