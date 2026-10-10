"""Build the figure delivery package: ordered folders, LaTeX captions, index with sources and hashes, SHA256SUMS and an SVG zip."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIGURES = "resultados/figuras"
FINAL = f"{FIGURES}/thesis_c1_c2_20261009_final"
FORMATS = (".svg", ".pdf", ".png")
DATASET_SETS = {"c1": ("ham10000", "isic2019", "isic2020")}

# (group, stems) in the order of the brief; every entry is (stem, folder, extensions).
PLAN = (
    ("1_caracterizacion", (("fig_char_diagnosis_distribution", FIGURES), ("fig_char_ita_distribution", FIGURES))),
    ("2_segmentacion", (("fig_s2_dice_ci", FIGURES), ("fig_s3_dice_distribution", FIGURES))),
    ("3_tono", (("fig_t1_ita_geometry", f"{FINAL}/T1"), ("tab_t2_ita_intervals", f"{FINAL}/T2"),
                ("fig_t3_ita_vs_fitzpatrick", FIGURES), ("fig_t4_ita_violins", f"{FINAL}/T4"))),
    ("4_clasificacion", tuple([(f"fig_c1_f1_macro_{d}", f"{FINAL}/C1") for d in DATASET_SETS["c1"]]
                              + [("fig_c2_paired_delta_f1", f"{FINAL}/C2")]
                              + [(f"fig_c3_confusion_{d}", FIGURES) for d in DATASET_SETS["c1"]]
                              + [(f"fig_c4_per_diagnosis_{d}", f"{FINAL}/C4") for d in DATASET_SETS["c1"]]
                              + [(f"fig_c5_pr_roc_{d}", FIGURES) for d in ("ham10000", "isic2019", "isic2020", "isic2020_ft")]
                              + [(f"fig_c6_calibration_{d}", FIGURES) for d in ("ham10000", "ham10000_v71", "isic2019", "isic2020", "isic2020_ft")]
                              + [("fig_c7_a71_vs_b71", FIGURES), ("fig_c7_b71_vs_b71_f4", FIGURES)])),
    ("5_dominio", (("fig_d1_pad_d0_d3", FIGURES), ("fig_d2_pad_contrasts", FIGURES))),
)
# Pending-from-pod figures join the package automatically when their exports exist.
OPTIONAL = (("3_tono", "fig_a_cielab_ring", FIGURES), ("3_tono", "fig_t5_tone_pairs", f"{FIGURES}/thesis_s1_t5_qualitative_20261009/T5"),
            ("2_segmentacion", "fig_s1_qualitative_masks", FIGURES))
PENDING_TEXT = {
    "fig_a_cielab_ring": "Figura A (CIELAB del anillo perilesional): cálculo en el servidor de la UNAB (HAM10000, ISIC 2019, ISIC 2020).",
    "fig_t5_tone_pairs": "T5 antes/después de la transformación de tono: faltan las imágenes originales/transformadas del servidor.",
    "fig_s1_qualitative_masks": "S1 comparación cualitativa de máscaras: faltan imágenes y máscaras por caso del servidor.",
}
NEW_EXPERIMENTS = ("Ablaciones aisladas de F4 (recorte, descriptores, combinación): requieren ejecutar esas condiciones.",
                   "Eficiencia de etiquetas (F1 frente a fracción etiquetada, con muestreos repetidos).",
                   "Robustez a iluminación (perturbaciones controladas sobre las mismas imágenes).")
LIMITATIONS = ("ITA no equivale al fototipo Fitzpatrick; no está validado como estimador de fototipo.",
               "La fila de VGG19 queda pendiente de confirmar con la Dra. Karen (posible medición anterior a 6bd96b7).",
               "C5/C6 no incluyen PAD-UFES-20: sus archivos de probabilidades no traen etiquetas por imagen.",
               "La transformación de tono no está demostrada como mejora de clasificación.",
               "ISIC 2018 es parcialmente transductivo; B71 original de ISIC 2019/2020 también (ver notas).")
STATUS_OVERRIDES = {"fig_t1_ita_geometry": "Esquema conceptual (no es un resultado experimental)",
                    "tab_t2_ita_intervals": "Tabla derivada del código (cortes de Groh)"}
KNOWN_STATUSES = ("Cifras documentadas", "Artefactos verificados", "Cálculo adicional", "Nuevo experimento", "Pendiente")
LATEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                 "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def normalise_status(raw: str | None) -> str:
    """Keep a status already written with the agreed vocabulary; every other figure was reviewed against its artifacts."""
    for known in KNOWN_STATUSES:
        if raw and raw.startswith(known):
            return raw
    return "Artefactos verificados"


def latex(text: str) -> str:
    return "".join(LATEX_ESCAPES.get(ch, ch) for ch in text)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(stem: str, folder: Path) -> list[Path]:
    exts = (".tex",) if stem.startswith("tab_") else FORMATS
    needed = [folder / f"{stem}{ext}" for ext in exts] + [folder / f"{stem}.{kind}.txt" for kind in ("caption", "note")]
    needed.append(folder / f"{stem}.traceability.json")
    missing = [p for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{stem}: missing {[p.name for p in missing]}")
    return needed


def collect(repo: Path) -> tuple[list[dict], list[str]]:
    entries, pending = [], []
    for group, items in PLAN:
        for stem, folder in items:
            entries.append({"group": group, "stem": stem, "files": _files(stem, repo / folder)})
    for group, stem, folder in OPTIONAL:
        if (repo / folder / f"{stem}.svg").is_file():
            entries.append({"group": group, "stem": stem, "files": _files(stem, repo / folder)})
        else:
            pending.append(PENDING_TEXT[stem])
    return entries, pending


def build(repo: Path, outdir: Path) -> dict:
    entries, pending = collect(repo)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    md, tex, index = ["# Leyendas de las figuras", ""], ["% Leyendas generadas; sin numeración de figura.", ""], []
    svgs: list[Path] = []
    for entry in entries:
        group_dir = outdir / entry["group"]
        group_dir.mkdir(exist_ok=True)
        for source in entry["files"]:
            shutil.copy2(source, group_dir / source.name)
            if source.suffix == ".svg":
                svgs.append(group_dir / source.name)
        stem = entry["stem"]
        caption = (group_dir / f"{stem}.caption.txt").read_text(encoding="utf-8").strip()
        note = (group_dir / f"{stem}.note.txt").read_text(encoding="utf-8").strip()
        trace = json.loads((group_dir / f"{stem}.traceability.json").read_text(encoding="utf-8"))
        md += [f"## {stem}", "", caption, "", f"Nota. {note}", ""]
        tex += [f"% {stem}", f"\\caption{{{latex(caption)}}}", f"\\textit{{Nota.}} {latex(note)}", ""]
        status = STATUS_OVERRIDES.get(stem) or normalise_status(trace.get("status") or trace.get("estado"))
        question = trace.get("question") or trace.get("pregunta") or caption
        index.append(f"| {entry['group']} | {stem} | {question} | {status} | {'; '.join(p.name for p in entry['files'] if p.suffix in ('.svg', '.tex'))} |")
    (outdir / "leyendas_figuras.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (outdir / "leyendas_figuras.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    lines = ["# Índice de la entrega", "", "| Grupo | Figura | Pregunta | Estado | Archivo principal |", "|---|---|---|---|---|", *index, "",
             "## Pendientes (dependen del servidor de la UNAB)", "", *([f"- {item}" for item in pending] or ["- Ninguno"]), "",
             "## Requieren experimentos nuevos (sin gráfica)", "", *[f"- {item}" for item in NEW_EXPERIMENTS], "",
             "## Limitaciones relevantes", "", *[f"- {item}" for item in LIMITATIONS], ""]
    (outdir / "INDICE.md").write_text("\n".join(lines), encoding="utf-8")
    archive = outdir / "entrega_pau_svg.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for svg in sorted(svgs):
            zf.write(svg, svg.relative_to(outdir))
    sums = [f"{_sha(p)}  {p.relative_to(outdir)}" for p in sorted(outdir.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    (outdir / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return {"figures": len(entries), "pending": pending, "svg": len(svgs), "zip": archive}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=REPO / "resultados" / "entrega_pau")
    args = parser.parse_args()
    summary = build(REPO, args.outdir)
    print(f"{summary['figures']} figures, {summary['svg']} SVG, {len(summary['pending'])} pending -> {args.outdir}")
    sys.exit(0)
