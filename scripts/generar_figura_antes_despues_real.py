"""Genera la figura DEFINITIVA de validación visual de la Transformación Tonal:
ANTES y DESPUÉS con imágenes 100% REALES y máscaras de Teacher 026.
Diseño pedagógico y editorial de alto impacto:
1. Imagen Original Real.
2. Máscara Anatómica Real de Teacher 026 (overlay translúcido sobre la lesión).
3. Efecto Sin Acople (m=0: 'efecto parche' evidente).
4. Transformación Real V10_C1M_LUM (Luminancia pura con dosis visible).
5. Mapa de Calor Residual (|dif| amplificado) que demuestra objetivamente la acción del operador.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "dermssl"))

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from skimage.color import rgb2lab, lab2rgb

# Cargar casos reales con máscaras de Teacher 026
NPZ_PATH = Path(
    r"C:\Users\mpaul\Documents\ChatGPT\Proyecto de grado 2\research\tono-literatura-20261105\campana-v10-tono-20261108\figuras_insumos\insumos_ddi_bosque_masks026_v10.npz"
)
data = np.load(NPZ_PATH)

OUT_DIR = REPO / "docs" / "figuras_analisis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_DIR = Path("C:/Users/mpaul/.gemini/antigravity/brain/1580e03d-ac50-4f3c-ae85-a13b89cdf691")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "figure.titlesize": 12,
})

def aplicar_transformacion_lum_visual(rgb_in, mask_in, delta_L=15.0, m=0.8609):
    """Aplica la modulación exacta en CIELAB Luminancia pura L* con acople m.
    
    rgb_in: float32 [0, 1]
    mask_in: boolean [H, W] (True = lesión, False = piel)
    """
    lab = rgb2lab(rgb_in)
    L = lab[:, :, 0]
    
    # Suavizado de borde perilesional (feathering)
    mask_soft = ndi.gaussian_filter(mask_in.astype(np.float32), sigma=2.0)
    
    # Ponderador espacial: 1 en piel, m en lesión
    w_espacial = (1.0 - mask_soft) + m * mask_soft
    
    # Modulación pura en L* (a* y b* intactos)
    L_trans = L + delta_L * w_espacial
    L_trans = np.clip(L_trans, 0.0, 100.0)
    
    lab_trans = lab.copy()
    lab_trans[:, :, 0] = L_trans
    
    rgb_out = lab2rgb(lab_trans)
    return np.clip(rgb_out, 0.0, 1.0)

# Selección de 3 casos reales emblemáticos:
# Caso 1: FST II (Piel clara, oscurecer hacia piel oscura)
# Caso 2: FST VI con pelos densos (Piel oscura, aclarar hacia piel clara)
# Caso 3: DDI FST V-VI (Caso clínico real)

casos = [
    {
        "rgb": data["bosque_rgb_II"].astype(np.float32) / 255.0,
        "mask": data["bosque_mask_II"] > 0,
        "delta_L": -22.0, # Oscurecer claramente
        "nombre": "Caso 1: FST II (HAM10000)\nSimulación hacia Fototipo Oscuro",
        "accion": "Oscurecimiento $\Delta L^* = -22$",
    },
    {
        "rgb": data["bosque_rgb_VI"].astype(np.float32) / 255.0,
        "mask": data["bosque_mask_VI"] > 0,
        "delta_L": +22.0, # Aclarar claramente
        "nombre": "Caso 2: FST VI (Pelos Densos)\nSimulación hacia Fototipo Claro",
        "accion": "Aclaramiento $\Delta L^* = +22$",
    },
    {
        "rgb": data["ddi_rgb_56"].astype(np.float32) / 255.0,
        "mask": data["ddi_mask_56"] > 0,
        "delta_L": +20.0, # Aclarar en DDI
        "nombre": "Caso 3: FST V–VI (Dataset DDI Real)\nEvaluación Externa de Equidad",
        "accion": "Aclaramiento $\Delta L^* = +20$",
    },
]

fig, axes = plt.subplots(3, 5, figsize=(14.0, 8.5), facecolor="white")

for i, c in enumerate(casos):
    rgb = c["rgb"]
    mask = c["mask"]
    dL = c["delta_L"]
    
    # 1. Original
    axes[i, 0].imshow(rgb)
    axes[i, 0].set_title(f"1. Original Real ($x$)\n{c['accion']}", pad=5)
    axes[i, 0].set_ylabel(c["nombre"], fontsize=9.5, fontweight="bold", labelpad=10)
    
    # 2. Máscara Teacher 026 (Overlay visual translúcido rojo + contorno)
    overlay = rgb.copy()
    borde = mask ^ ndi.binary_erosion(mask, iterations=2)
    overlay[mask] = overlay[mask] * 0.5 + np.array([0.85, 0.1, 0.1]) * 0.5 # Tinte rojo suave
    overlay[borde] = [1.0, 1.0, 0.0] # Borde amarillo nítido
    axes[i, 1].imshow(np.clip(overlay, 0, 1))
    axes[i, 1].set_title("2. Máscara Teacher 026\n(Segmentación anatómica $M$)", pad=5)
    
    # 3. Sin Acople (m = 0, lesión congelada)
    trans_c0 = aplicar_transformacion_lum_visual(rgb, mask, delta_L=dL, m=0.0)
    axes[i, 2].imshow(trans_c0)
    axes[i, 2].set_title("3. Sin Acople ($m=0$)\n(Fallo: efecto parche / sticker)", pad=5)
    
    # 4. Con Acople V10_C1M_LUM (m = 0.8609, Luminancia pura)
    trans_lum = aplicar_transformacion_lum_visual(rgb, mask, delta_L=dL, m=0.8609)
    axes[i, 3].imshow(trans_lum)
    axes[i, 3].set_title("4. V10_C1M_LUM ($m=0.8609$)\n(Acople natural en Luminancia)", pad=5)
    
    # 5. Mapa Residual de Diferencia |trans_lum - rgb|
    dif = np.abs(trans_lum - rgb).mean(axis=2)
    im_dif = axes[i, 4].imshow(dif, cmap="inferno", vmin=0.0, vmax=0.15)
    axes[i, 4].set_title("5. Mapa Residual ($|\Delta x|$)\n(Piel se mueve, lesión preservada)", pad=5)
    
    # Anotar valores medios en piel vs lesion
    val_piel = dif[~mask].mean()
    val_lesion = dif[mask].mean()
    axes[i, 4].text(
        0.05, 0.08,
        f"Piel: {val_piel:.3f}\nLesión: {val_lesion:.3f}\nRazón: {val_lesion/val_piel:.2f}",
        transform=axes[i, 4].transAxes, fontsize=8, color="white",
        bbox=dict(facecolor="black", alpha=0.7, pad=2, edgecolor="none")
    )

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(
    "Evidencia Visual y Objetiva de la Transformación Tonal Acoplada (V10_C1M_LUM)\n"
    "Casos Reales con Máscaras de Teacher 026: Demostración del Acople Físico ($m=0.8609$) frente al Fallo de Parche ($m=0$)",
    fontsize=12, fontweight="bold", y=0.98, color="#302B27"
)

plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.94])

out_png = OUT_DIR / "figura_antes_despues_transformacion_real.png"
out_pdf = OUT_DIR / "figura_antes_despues_transformacion_real.pdf"
out_svg = OUT_DIR / "figura_antes_despues_transformacion_real.svg"

fig.savefig(out_png, dpi=300, facecolor="white")
fig.savefig(out_pdf, facecolor="white")
fig.savefig(out_svg, facecolor="white")

if BRAIN_DIR.exists():
    import shutil
    shutil.copy(out_png, BRAIN_DIR / "figura_antes_despues_transformacion_real.png")

print(f"[EXITO] Figura visual fidedigna generada en {out_png}")
