"""Genera la figura de auditoría y validación de segmentación:
Casos con pelos densos y estratificación por tamaño con MÁSCARAS Y PROBABILIDADES REALES de Teacher 026.
Diseño limpio editorial: sin textos redundantes, sin explicaciones que van en el pie,
espaciado calibrado para CERO superposición de textos y escala 1:1 (11 pt).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Configuración de rutas
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from dermssl.segmentation_methods.common.hair import hair_mask, remove_hair

INSUMOS_NPZ = Path(
    r"C:\Users\mpaul\Documents\ChatGPT\Proyecto de grado 2\research\tono-literatura-20261105\campana-v10-tono-20261108\figuras_insumos\insumos_ddi_bosque_masks026_v10.npz"
)
OUT_DIR = REPO / "docs" / "figuras_analisis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_DIR = Path("C:/Users/mpaul/.gemini/antigravity/brain/1580e03d-ac50-4f3c-ae85-a13b89cdf691")

# Paleta institucional Cappuccino
PALETTE = {
    "cappdark": "#4B3832",
    "cappwine": "#854442",
    "capptan": "#BE9B7B",
    "cappcream": "#FFF4E6",
    "cappnearblack": "#3C2F2F",
    "white": "#FFFFFF",
}

# Tipografía idéntica a la plantilla de tesis (11 pt base, peso normal)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "figure.titlesize": 12,
})

CASOS = [
    {
        "rgb_key": "bosque_rgb_VI",
        "mask_key": "bosque_mask_VI",
        "prob_key": "bosque_prob_VI",
        "meta_key": "bosque_meta_VI",
        "label": "Pelos Extremos\n(FST VI · ID 308161)",
    },
    {
        "rgb_key": "bosque_rgb_II",
        "mask_key": "bosque_mask_II",
        "prob_key": "bosque_prob_II",
        "meta_key": "bosque_meta_II",
        "label": "Pelos Densos\n(FST II · ID 100521)",
    },
    {
        "rgb_key": "bosque_rgb_IV",
        "mask_key": "bosque_mask_IV",
        "prob_key": "bosque_prob_IV",
        "meta_key": "bosque_meta_IV",
        "label": "Lesión Mediana\n(FST IV · ID 100141)",
    },
    {
        "rgb_key": "bosque_rgb_III",
        "mask_key": "bosque_mask_III",
        "prob_key": "bosque_prob_III",
        "meta_key": "bosque_meta_III",
        "label": "Lesión Grande\n(FST III · ID 100231)",
    },
]


def obtener_contorno(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def main() -> None:
    data = np.load(INSUMOS_NPZ)

    n_rows = len(CASOS)
    n_cols = 6

    # Canvas calibrado: 10 pulgadas de ancho por 6.8 de alto para dar amplio margen
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10.0, 6.8), dpi=300)
    fig.patch.set_facecolor(PALETTE["white"])

    # Títulos concisos y directos, sin texto de más
    col_headers = [
        "1. Original",
        "2. Máscara de Pelos",
        "3. Inpainting Telea",
        "4. Probabilidad SSL",
        "5. Máscara Teacher 026",
        "6. Contorno Superpuesto",
    ]

    for row_idx, caso in enumerate(CASOS):
        img_rgb = data[caso["rgb_key"]]
        mask_026 = data[caso["mask_key"]]
        prob_026 = data[caso["prob_key"]]

        # Detección y remoción real
        hmask = hair_mask(img_rgb)
        clean_img, _ = remove_hair(img_rgb)

        # Col 1: Original
        axes[row_idx, 0].imshow(img_rgb)

        # Col 2: Pelos
        hair_disp = np.zeros((*hmask.shape, 3), dtype=np.uint8)
        hair_disp[hmask == 1] = [255, 255, 255]
        axes[row_idx, 1].imshow(hair_disp)

        # Col 3: Inpaint
        axes[row_idx, 2].imshow(clean_img)

        # Col 4: Probabilidad Teacher 026
        axes[row_idx, 3].imshow(prob_026, cmap="magma", vmin=0.0, vmax=1.0)

        # Col 5: Máscara binaria Teacher 026 (crema cálido sobre negro)
        mask_disp = np.zeros((*mask_026.shape, 3), dtype=np.uint8)
        mask_disp[mask_026 == 1] = [245, 240, 230]
        axes[row_idx, 4].imshow(mask_disp)

        # Col 6: Contorno sobre imagen original
        axes[row_idx, 5].imshow(img_rgb)
        for cnt in obtener_contorno(mask_026):
            cnt = cnt.squeeze()
            if cnt.ndim == 2:
                # Línea base blanca fina para contraste
                axes[row_idx, 5].plot(cnt[:, 0], cnt[:, 1], color=PALETTE["white"], linewidth=1.8)
                # Contorno vino institucional
                axes[row_idx, 5].plot(cnt[:, 0], cnt[:, 1], color=PALETTE["cappwine"], linewidth=1.2)

        # Limpieza rigurosa de ticks y marcos
        for c in range(n_cols):
            ax = axes[row_idx, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(PALETTE["cappdark"])
                spine.set_linewidth(0.45)

        # Etiqueta de fila con margen generoso (labelpad=14) para evitar colisión
        axes[row_idx, 0].set_ylabel(
            caso["label"],
            fontsize=10.5,
            color=PALETTE["cappnearblack"],
            fontweight="normal",
            labelpad=14,
        )

    # Cabeceras limpias con pad=10
    for col_idx, header in enumerate(col_headers):
        axes[0, col_idx].set_title(header, fontsize=11, pad=10, color=PALETTE["cappnearblack"])

    # Leyenda compacta inferior bien separada
    fig.legend(
        handles=[
            plt.Line2D([0], [0], color=PALETTE["cappwine"], linestyle="-", linewidth=1.6, label="Contorno Teacher 026 real"),
            plt.Line2D([0], [0], marker="s", markersize=6, markerfacecolor="white", markeredgecolor=PALETTE["cappdark"], linestyle="none", label="Máscara de pelos (Black-hat)"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        fontsize=10.5,
    )

    # Márgenes calibrados: left=0.18 da espacio sobrado a las etiquetas de fila
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06, left=0.17, top=0.92, wspace=0.08, hspace=0.10)

    out_png = OUT_DIR / "figura_segmentacion_pelos_densos_teacher026.png"
    out_pdf = OUT_DIR / "figura_segmentacion_pelos_densos_teacher026.pdf"
    out_svg = OUT_DIR / "figura_segmentacion_pelos_densos_teacher026.svg"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close()

    for ext in ["png", "pdf", "svg"]:
        src_f = OUT_DIR / f"figura_segmentacion_pelos_densos_teacher026.{ext}"
        dst_f = BRAIN_DIR / f"figura_segmentacion_pelos_densos_teacher026.{ext}"
        dst_f.write_bytes(src_f.read_bytes())

    print("Figura de segmentación re-generada con CERO solapamiento.")


if __name__ == "__main__":
    main()
