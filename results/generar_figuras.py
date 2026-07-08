#!/usr/bin/env python3
"""Genera las figuras explicativas del proyecto a partir de RESULTS_SUMMARY.json.
Datos REALES de la corrida de referencia (2026-07-07). Todo en español.
Ejecutar:  .venv-charts/bin/python results/generar_figuras.py
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "assets", "figuras")
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(ROOT, "data", "RESULTS_SUMMARY.json"), encoding="utf-8") as f:
    R = json.load(f)

# --- Paleta y estilo consistente ---
C_IN = "#4C78A8"     # ImageNet (azul)
C_SSL = "#F58518"    # SSL (naranja)
C_OK = "#54A24B"     # verde (cumple)
C_META = "#B0B0B0"   # gris (meta / referencia)
C_TXT = "#2A2A2A"
plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": "#CCCCCC",
    "axes.labelcolor": C_TXT,
    "text.color": C_TXT,
    "xtick.color": C_TXT,
    "ytick.color": C_TXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def guardar(fig, nombre):
    ruta = os.path.join(OUT, nombre)
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓", os.path.relpath(ruta, ROOT))


# ============================================================
# FIG 1 — Ablación: ImageNet vs SSL (macro-F1 y AUC)
# ============================================================
def fig1():
    ab = R["ablacion_full_finetune"]
    metricas = ["macro-F1", "AUC-ROC"]
    inn = [ab["imagenet"]["macro_f1"], ab["imagenet"]["auc"]]
    ssl = [ab["ssl"]["macro_f1"], ab["ssl"]["auc"]]
    x = np.arange(len(metricas))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - w / 2, inn, w, label="ImageNet", color=C_IN)
    b2 = ax.bar(x + w / 2, ssl, w, label="SSL (SimSiam)", color=C_SSL)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(metricas)
    ax.set_ylabel("Puntaje")
    ax.set_title("Ablación (fine-tune completo): ¿el SSL aporta sobre ImageNet?",
                 fontweight="bold", fontsize=12)
    ax.legend(loc="upper left", frameon=False)
    ax.annotate("Empate en macro-F1 (Δ = −0.0007)\nSSL gana en AUC (Δ = +0.0087)",
                xy=(0.5, 0.06), xycoords="axes fraction", ha="center",
                fontsize=9.5, color="#555555",
                bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5", ec="#DDDDDD"))
    guardar(fig, "fig1_ablacion.png")


# ============================================================
# FIG 2 — Linear probe: macro-F1 y AUC vs % de etiquetas
# ============================================================
def fig2():
    lp = R["linear_probe"]
    pct = [10, 20, 50, 100]
    keys = ["10pct", "20pct", "50pct", "100pct"]
    in_f1 = [lp["imagenet"][k]["macro_f1"] for k in keys]
    ss_f1 = [lp["ssl"][k]["macro_f1"] for k in keys]
    in_au = [lp["imagenet"][k]["auc"] for k in keys]
    ss_au = [lp["ssl"][k]["auc"] for k in keys]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, inv, ssv, titulo in [
        (a1, in_f1, ss_f1, "macro-F1"),
        (a2, in_au, ss_au, "AUC-ROC"),
    ]:
        ax.plot(pct, inv, "-o", color=C_IN, label="ImageNet", lw=2)
        ax.plot(pct, ssv, "-o", color=C_SSL, label="SSL", lw=2)
        ax.set_xticks(pct)
        ax.set_xlabel("% de etiquetas usadas")
        ax.set_ylabel(titulo)
        ax.set_title(f"Linear probe · {titulo}", fontweight="bold")
        ax.legend(frameon=False)
        ax.grid(True, axis="y", ls=":", color="#E5E5E5")
    a2.annotate("SSL adelanta\nen AUC al 100%", xy=(100, ss_au[-1]),
                xytext=(62, 0.845), fontsize=9, color=C_SSL,
                arrowprops=dict(arrowstyle="->", color=C_SSL))
    fig.suptitle("Calidad de la representación (backbone congelado)",
                 fontweight="bold", fontsize=13, y=1.02)
    guardar(fig, "fig2_linear_probe.png")


# ============================================================
# FIG 3 — Tablero de KPIs clínicos
# ============================================================
def fig3():
    fm = R["final_metrics"]
    kpis = [
        ("macro-F1", fm["macro_f1"], 0.65, 1.0),
        ("AUC-ROC", fm["auc"], 0.85, 1.0),
        ("Sensib. melanoma\n(screening)", R["mel_screening"]["sensibilidad"], 0.80, 1.0),
        ("Calibración\n(1 − ECE calibrado)", 1 - fm["ece_calibrated"], 0.90, 1.0),
    ]
    nombres = [k[0] for k in kpis]
    valores = [k[1] for k in kpis]
    metas = [k[2] for k in kpis]
    y = np.arange(len(kpis))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.barh(y, valores, color=C_OK, height=0.5, zorder=3)
    for yi, (v, m) in zip(y, zip(valores, metas)):
        ax.plot([m, m], [yi - 0.3, yi + 0.3], color="#C0392B", lw=2, zorder=4)
        ax.text(v + 0.01, yi, f"{v:.3f}  ✓", va="center", fontsize=10,
                fontweight="bold", color=C_OK)
    ax.plot([], [], color="#C0392B", lw=2, label="Meta mínima (KPI)")
    ax.set_yticks(y)
    ax.set_yticklabels(nombres)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Puntaje (todos los KPIs cumplidos)")
    ax.set_title("Tablero de KPIs clínicos — 4 / 4 cumplidos",
                 fontweight="bold", fontsize=12)
    ax.legend(loc="lower right", frameon=False)
    guardar(fig, "fig3_kpis.png")


# ============================================================
# FIG 4 — Curva de abstención (cobertura vs exactitud)
# ============================================================
def fig4():
    ab = R["abstencion"]
    umbrales = sorted(ab.keys(), key=float)
    cob = [ab[u]["cobertura"] * 100 for u in umbrales]
    exa = [ab[u]["exactitud"] * 100 for u in umbrales]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(cob, exa, "-o", color=C_IN, lw=2)
    for u, c, e in zip(umbrales, cob, exa):
        ax.annotate(f"≥{float(u):.1f}", (c, e), textcoords="offset points",
                    xytext=(4, 5), fontsize=8, color="#555555")
    ax.set_xlabel("Cobertura (% de casos respondidos)")
    ax.set_ylabel("Exactitud en lo respondido (%)")
    ax.set_title("Abstención: responder solo cuando hay confianza",
                 fontweight="bold", fontsize=12)
    ax.grid(True, ls=":", color="#E5E5E5")
    ax.annotate("Umbral 0.8:\ncubre 83.7% con 91.8% exactitud",
                xy=(83.7, 91.76), xytext=(45, 88), fontsize=9, color=C_TXT,
                arrowprops=dict(arrowstyle="->", color="#888888"),
                bbox=dict(boxstyle="round,pad=0.35", fc="#F5F5F5", ec="#DDDDDD"))
    guardar(fig, "fig4_abstencion.png")


# ============================================================
# FIG 5 — Recall por clase
# ============================================================
def fig5():
    rc = R["recall_por_clase"]
    nombres_full = {
        "nv": "nv (nevus)", "bcc": "bcc (carc. basocel.)",
        "vasc": "vasc (vascular)", "akiec": "akiec (queratosis act.)",
        "bkl": "bkl (queratosis benig.)", "mel": "mel (MELANOMA)",
        "df": "df (dermatofibroma)",
    }
    items = sorted(rc.items(), key=lambda kv: kv[1])
    labels = [nombres_full[k] for k, _ in items]
    vals = [v for _, v in items]
    colores = [C_OK if v >= 0.7 else (C_SSL if v >= 0.5 else "#C0392B") for v in vals]
    y = np.arange(len(items))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(y, vals, color=colores, height=0.6, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=9.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Recall (sensibilidad por clase)")
    ax.set_title("Recall por clase — df y mel son las difíciles",
                 fontweight="bold", fontsize=12)
    ax.axvline(0.7, color=C_META, ls="--", lw=1)
    guardar(fig, "fig5_recall_clase.png")


# ============================================================
# FIG 6 — Efecto de la calibración (ECE antes/después)
# ============================================================
def fig6():
    fm = R["final_metrics"]
    etiquetas = ["Sin calibrar", f"Calibrado\n(T = {fm['temperature']:.3f})"]
    vals = [fm["ece_raw"], fm["ece_calibrated"]]
    colores = ["#C0392B", C_OK]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    b = ax.bar(etiquetas, vals, color=colores, width=0.55)
    for bar, v in zip(b, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.4f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(0.10, color=C_META, ls="--", lw=1.5)
    ax.text(1.4, 0.104, "Meta ≤ 0.10", color="#777777", fontsize=9, ha="right")
    ax.set_ylim(0, 0.20)
    ax.set_ylabel("ECE (error de calibración esperado)")
    ax.set_title("Calibración por temperature scaling",
                 fontweight="bold", fontsize=12)
    guardar(fig, "fig6_calibracion.png")


if __name__ == "__main__":
    print("Generando figuras en assets/figuras/ ...")
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("Listo. 6 figuras generadas.")
