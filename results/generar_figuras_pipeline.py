"""Publication-quality pipeline figures (fig7-fig10).

Replaces the hand-drawn SVG versions with matplotlib diagrams styled after
scientific-paper block diagrams (flat colored blocks, numbered stages,
consistent typography, 300 DPI).

The system Python has no matplotlib; run inside any venv:

    python -m venv /tmp/figvenv && /tmp/figvenv/bin/pip install matplotlib
    /tmp/figvenv/bin/python results/generar_figuras_pipeline.py

Outputs PNG (300 DPI) into assets/figuras/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "assets" / "figuras"

# Scientific-paper palette: soft fills, saturated borders.
C = {
    "data":  ("#DBEAFE", "#1E40AF"),   # blue
    "sup":   ("#E0E7FF", "#4338CA"),   # indigo
    "ssl":   ("#DCFCE7", "#15803D"),   # green
    "feat":  ("#F1F5F9", "#475569"),   # slate
    "probe": ("#FEF3C7", "#B45309"),   # amber
    "metric": ("#FEE2E2", "#B91C1C"),  # red
    "note":  ("#FFFFFF", "#94A3B8"),
}
INK = "#0F172A"
ARROW = "#334155"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "svg.fonttype": "none",
})


def box(ax, x, y, w, h, title, lines=(), key="note", title_size=9.5,
        body_size=8.0, lw=1.4):
    fc, ec = C[key]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
    cy = y + h - 0.32 if lines else y + h / 2
    ax.text(x + w / 2, cy, title, ha="center",
            va="center" if not lines else "top",
            fontsize=title_size, fontweight="bold", color=ec, zorder=4)
    if lines:
        body_top = y + h - 0.62
        step = (body_top - y - 0.18) / max(len(lines), 1)
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, body_top - step * (i + 0.5), ln, ha="center",
                    va="center", fontsize=body_size, color=INK, zorder=4)
    return x, y, w, h


def group(ax, x, y, w, h, label, ec="#94A3B8"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15",
        facecolor="#FFFFFF", edgecolor=ec, linewidth=1.1,
        linestyle=(0, (4, 2)), zorder=1))
    ax.text(x + 0.15, y + h - 0.12, label, ha="left", va="top",
            fontsize=9.5, fontweight="bold", color="#475569", zorder=2)


def arrow(ax, p1, p2, style="-|>", color=ARROW, lw=1.5, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=14, color=color,
        linewidth=lw, linestyle=ls, zorder=5,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def canvas(w_in, h_in, xmax, ymax, title, subtitle):
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    ax.text(xmax / 2, ymax - 0.25, title, ha="center", va="top",
            fontsize=14, fontweight="bold")
    ax.text(xmax / 2, ymax - 0.78, subtitle, ha="center", va="top",
            fontsize=9, color="#64748B", style="italic")
    return fig, ax


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("OK", OUT / f"{name}.png")


# ----------------------------------------------------------------- fig 7
def fig7():
    fig, ax = canvas(15, 7.8, 20, 10.4,
                     "Banco de baselines — pipeline de evaluación (v5)",
                     "7 backbones congelados, protocolo idéntico por "
                     "construcción; solo cambia el extractor de features")

    group(ax, 0.3, 1.6, 3.6, 7.3, "1. Datos")
    box(ax, 0.6, 4.6, 3.0, 3.3, "HAM10000",
        ["10 015 imágenes", "7 clases diagnósticas",
         "split train / val", "StratifiedGroupKFold",
         "por lesion_id", "assert: 0 lesiones", "compartidas"],
        key="data")

    group(ax, 4.5, 1.6, 5.0, 7.3, "2. Backbones congelados")
    box(ax, 4.8, 5.4, 4.4, 2.9, "Supervisados (comparadores)",
        ["AlexNet 2012 · 256d", "VGG19 2014 · 512d",
         "EfficientNet-B0 2019 · 1280d", "ResNet-50 2015 · 2048d",
         "ViT-B/16 2021 · 768d"], key="sup")
    box(ax, 4.8, 3.2, 4.4, 2.0, "SSL (sin etiquetas)",
        ["SimSiam ResNet-50 2021 · 2048d",
         "DINOv2 ViT-S/14 2023 · 384d"], key="ssl")
    ax.text(7.0, 2.55, "requires_grad=False + model.eval()\n"
            "assert trainable == 0  (cero fine-tuning)",
            ha="center", va="center", fontsize=7.8, color="#B91C1C",
            fontweight="bold")

    group(ax, 10.1, 1.6, 3.3, 7.3, "3. Features")
    box(ax, 10.4, 4.7, 2.7, 3.1, "Extracción",
        ["GAP (CNN)", "token CLS (ViT)", "L2-norm coseno",
         "1 vector / imagen", "cache en disco"], key="feat")

    group(ax, 14.0, 1.6, 3.0, 7.3, "4. Probes")
    box(ax, 14.25, 6.8, 2.5, 1.5, "kNN",
        ["coseno · k=20", "cero entrenamiento"], key="probe")
    box(ax, 14.25, 5.0, 2.5, 1.5, "Linear",
        ["regresión logística", "class_weight balanced"], key="probe")
    box(ax, 14.25, 3.2, 2.5, 1.5, "Prototipos",
        ["centroide por clase", "cercanía coseno"], key="probe")

    group(ax, 17.55, 1.6, 2.3, 7.3, "5. Métricas")
    box(ax, 17.75, 3.9, 1.9, 4.3, "Reporte",
        ["balanced acc", "F1 macro", "precision macro", "recall macro",
         "AUC OvR", "accuracy", "ECE"], key="metric", body_size=7.6)

    arrow(ax, (3.6, 6.2), (4.8, 7.0), rad=0.15)
    arrow(ax, (3.6, 6.2), (4.8, 4.2), rad=-0.15)
    arrow(ax, (9.2, 7.1), (10.4, 6.4), rad=0.12)
    arrow(ax, (9.2, 4.2), (10.4, 5.9), rad=-0.12)
    for yy in (7.55, 5.75, 3.95):
        arrow(ax, (13.1, 6.2), (14.25, yy), rad=0.08 * (6.2 - yy) / 2)
    for yy in (7.55, 5.75, 3.95):
        arrow(ax, (16.75, yy), (17.75, 6.05), rad=0.06 * (yy - 6.05) / 2)

    ax.text(10.0, 0.75, "Mismo split, mismos probes y mismas métricas para "
            "los 7 modelos → cualquier diferencia es atribuible al backbone.",
            ha="center", fontsize=8.6, color="#475569")
    save(fig, "fig7_pipeline_banco")


# ----------------------------------------------------------------- fig 8
def fig8():
    fig, ax = canvas(15, 7.0, 20, 9.2,
                     "Protocolo de test: baselines vs. propuesta (full SSL)",
                     "arriba: cómo se evalúa cada modelo de la tabla — "
                     "abajo: nuestra propuesta, 100 % sin etiquetas")

    group(ax, 0.3, 5.0, 19.4, 3.1, "BASELINE — por cada modelo de la tabla",
          ec="#4338CA")
    box(ax, 0.8, 5.5, 3.4, 1.9, "1 · Input",
        ["HAM10000 / ISIC", "imágenes dermatoscópicas"], key="data")
    box(ax, 5.3, 5.5, 4.6, 1.9, "2 · Backbone congelado",
        ["AlexNet · VGG19 · EfficientNet-B0", "ResNet-50 · ViT-B/16 · "
         "SimSiam · DINOv2"], key="sup")
    box(ax, 11.0, 5.5, 3.4, 1.9, "3 · Probes",
        ["kNN · linear · prototipos"], key="probe")
    box(ax, 15.5, 5.5, 3.7, 1.9, "4 · Métricas",
        ["balanced acc · F1 · precision", "recall · AUC · ECE"],
        key="metric")
    arrow(ax, (4.2, 6.45), (5.3, 6.45))
    arrow(ax, (9.9, 6.45), (11.0, 6.45))
    arrow(ax, (14.4, 6.45), (15.5, 6.45))

    group(ax, 0.3, 1.3, 19.4, 3.1,
          "PROPUESTA — nuestro modelo (full SSL, nada supervisado)",
          ec="#15803D")
    box(ax, 0.8, 1.8, 3.4, 1.9, "1 · Input",
        ["pool dermatoscópico", "~48-70k imágenes SIN etiquetas"],
        key="data")
    box(ax, 5.3, 1.8, 4.6, 1.9, "2 · Transformación",
        ["skin tone (Fitzpatrick)", "constancia de color · crops por lesión"],
        key="ssl")
    box(ax, 10.6, 1.8, 4.2, 1.9, "3 · Pre-entrenamiento SSL",
        ["DINOv2 continual", "SimSiam desde cero"], key="ssl",
        title_size=9.0)
    box(ax, 15.5, 1.8, 3.7, 1.9, "4 · Evaluación",
        ["MISMOS probes y métricas", "que la fila baseline"], key="probe")
    arrow(ax, (4.2, 2.75), (5.3, 2.75))
    arrow(ax, (9.9, 2.75), (10.6, 2.75))
    arrow(ax, (14.8, 2.75), (15.5, 2.75))

    ax.text(10.0, 0.55, "Las etiquetas se usan ÚNICAMENTE en la evaluación "
            "(probes y métricas), nunca en el pre-entrenamiento.  Hipótesis: "
            "ver piel sin etiquetas produce mejores features.",
            ha="center", fontsize=8.6, color="#475569", fontweight="bold")
    save(fig, "fig8_baseline_vs_proposal")


# ----------------------------------------------------------------- fig 9
def fig9():
    fig, ax = canvas(15, 7.6, 20, 10.0,
                     "v6 — brazo principal: DINO continual "
                     "(estudiante / profesor)",
                     "parte de DINOv2 ViT-S/14 público y continúa el "
                     "aprendizaje sobre piel, 100 % sin etiquetas")

    box(ax, 0.5, 4.1, 3.2, 2.6, "Imagen del pool",
        ["2 crops globales 224", "6 crops locales 96",
         "augmentation fuerte"], key="data")

    group(ax, 4.6, 5.2, 8.6, 3.6, "ESTUDIANTE — recibe gradientes",
          ec="#15803D")
    box(ax, 4.9, 5.6, 3.9, 2.5, "Backbone ViT-S/14",
        ["init: DINOv2 público", "ve TODOS los crops",
         "(globales + locales)"], key="ssl")
    box(ax, 9.2, 5.6, 3.7, 2.5, "Cabeza DINO",
        ["MLP 3 capas", "prototipos 65 536", "salida softmax"], key="ssl")
    arrow(ax, (8.8, 6.85), (9.2, 6.85))

    group(ax, 4.6, 1.0, 8.6, 3.6, "PROFESOR — sin gradientes (EMA)",
          ec="#4338CA")
    box(ax, 4.9, 1.4, 3.9, 2.5, "Backbone ViT-S/14",
        ["copia EMA del estudiante", "solo crops GLOBALES"], key="sup")
    box(ax, 9.2, 1.4, 3.7, 2.5, "Cabeza DINO",
        ["centering + sharpening", "temperatura 0.04-0.07"], key="sup")
    arrow(ax, (8.8, 2.65), (9.2, 2.65))

    box(ax, 14.6, 3.7, 4.3, 2.8, "Loss DINO",
        ["cross-entropy", "estudiante → profesor",
         "gradiente SOLO al estudiante", "(stop-grad en el profesor)"],
        key="metric")

    arrow(ax, (3.7, 5.9), (4.9, 6.85), rad=0.12)
    arrow(ax, (3.7, 4.9), (4.9, 2.65), rad=-0.12)
    arrow(ax, (12.9, 6.85), (14.9, 6.5), rad=0.1)
    arrow(ax, (12.9, 2.65), (14.9, 3.7), rad=-0.1)
    arrow(ax, (8.9, 5.05), (8.9, 4.75), style="-|>", ls=(0, (4, 2)),
          color="#7C3AED", lw=1.6)
    ax.text(9.15, 4.9, "actualización EMA (sin backprop)", ha="left",
            va="center", fontsize=7.8, color="#7C3AED", fontweight="bold")

    ax.text(10.0, 0.45, "ENTREGABLE: el backbone del estudiante — se congela "
            "y se evalúa en v5 como un baseline más, a igual presupuesto de "
            "GPU-horas que SimSiam.", ha="center", fontsize=8.6,
            color="#475569")
    save(fig, "fig9_v6_dino_continual")


# ---------------------------------------------------------------- fig 10
def fig10():
    fig, ax = canvas(15, 6.6, 20, 8.6,
                     "v6 — brazo purista: SimSiam desde inicialización "
                     "aleatoria",
                     "cero pesos externos: responde qué logra el SSL solo "
                     "con nuestro pool de piel")

    box(ax, 0.5, 3.4, 3.0, 2.4, "Imagen del pool",
        ["2 vistas aumentadas", "(augmentation aleatoria", "independiente)"],
        key="data")

    box(ax, 4.6, 3.4, 3.6, 2.4, "Backbone ResNet-50",
        ["init ALEATORIA", "compartido entre vistas"], key="ssl")
    box(ax, 9.2, 3.4, 3.3, 2.4, "Projector",
        ["MLP 3 capas", "BN + ReLU · 2048d"], key="ssl")
    box(ax, 13.5, 5.3, 3.3, 2.0, "Predictor",
        ["MLP 2 capas", "solo rama activa"], key="ssl")

    box(ax, 13.5, 1.4, 3.3, 2.0, "stop-gradient",
        ["rama objetivo sin", "backprop (clave anti-colapso)"], key="sup")

    box(ax, 17.3, 3.4, 2.4, 2.4, "Loss",
        ["cosine similarity", "negativa, simétrica"], key="metric",
        body_size=7.6)

    arrow(ax, (3.5, 4.6), (4.6, 4.6))
    arrow(ax, (8.2, 4.6), (9.2, 4.6))
    arrow(ax, (12.5, 5.1), (13.5, 6.1), rad=0.12)
    arrow(ax, (12.5, 4.1), (13.5, 2.6), rad=-0.12)
    arrow(ax, (16.8, 6.1), (18.1, 5.8), rad=0.12)
    arrow(ax, (16.8, 2.6), (18.1, 3.4), rad=-0.12)

    ax.text(10.0, 0.55, "Detector de colapso: std de los embeddings "
            "normalizados ≈ 1/√d → colapso; se monitorea cada época con "
            "MLflow.  ENTREGABLE: el backbone ResNet-50, congelado y "
            "evaluado con el mismo protocolo v5.", ha="center",
            fontsize=8.6, color="#475569")
    save(fig, "fig10_v6_simsiam")


if __name__ == "__main__":
    fig7()
    fig8()
    fig9()
    fig10()
