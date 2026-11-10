"""Generador de Diagramas de Arquitectura Científicos para la Tesis de Grado.

Genera cuatro figuras de calidad de publicación (SVG, PDF, PNG 300 DPI):
1. Bloque 1: Segmentación Sin Etiquetas (V8 / Teacher 026 / CCAM / UM).
2. Bloque 2: Transformación Tonal Acoplada Espacialmente (V10 / Conexión Máscara -> L*).
3. Bloque 3: Clasificación Multiclase SSL (V7 / ViT-B/14 / ISIC Pool / Probabilidades Calibradas).
4. Pipeline Completo Integrado por Bloques (End-to-End).

Estilo editorial: Perfil institucional Cappuccino, tipografía Times New Roman 11pt,
capas 3D en perspectiva isométrica, imágenes y máscaras reales de PH2 y DDI,
fórmulas matemáticas en KaTeX/LaTeX, mínimo texto y cero saturación visual.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, FancyBboxPatch
import numpy as np

# Rutas de salida
OUT_DIR = REPO / "docs" / "figuras_analisis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_DIR = Path(r"C:\Users\mpaul\.gemini\antigravity\brain\1580e03d-ac50-4f3c-ae85-a13b89cdf691")

# Paleta Institucional Cappuccino & Acentos Científicos Suaves
PALETTE = {
    "cappdark": "#4B3832",        # Texto oscuro y contornos primarios
    "cappwine": "#854442",        # Acento lesional / melanoma / contrastes
    "capptan": "#BE9B7B",         # Conectores y cajas secundarias
    "cappcream": "#FFF4E6",       # Fondos de contenedores y bloques
    "cappnearblack": "#302B27",   # Texto de etiquetas y fórmulas
    "connector": "#65564B",       # Flechas y flujos
    "contour": "#A38A77",         # Bordes suaves
    "white": "#FFFFFF",           # Fondo general
    # Acentos funcionales de capas
    "ssl_blue": "#8FA8BF",        # Capas de extracción de características SSL (SimCLR, MoCo)
    "ssl_blue_light": "#B8C9D9",  # Capa superior 3D
    "ssl_blue_dark": "#6A8B9C",   # Sombra lateral 3D
    "dec_peach": "#E8A87C",       # Capas del Decoder / Segmentador
    "dec_peach_light": "#F2C4A2",
    "dec_peach_dark": "#C88656",
    "ccam_sage": "#9BC4A5",       # Módulo CCAM y UM
    "ccam_sage_light": "#BFE0C7",
    "ccam_sage_dark": "#7CA586",
    "vit_wine": "#A35C5A",        # Bloques Vision Transformer
    "vit_wine_light": "#C98583",
    "vit_wine_dark": "#7D3D3C",
    "gray_box": "#F5EFE6",        # Cajas de contenedor suave
}

# Configuración de tipografía estándar (11 pt base, peso normal)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "figure.titlesize": 12,
    "mathtext.fontset": "stix",
})


def draw_3d_slab(ax, x0, y0, w, h, depth=0.4, color="#8FA8BF", light_color="#B8C9D9", dark_color="#6A8B9C", edgecolor="#4B3832", alpha=0.9):
    """Dibuja una lámina de tensor 3D en perspectiva isométrica (feature map slab)."""
    dx = depth * 0.7
    dy = depth * 0.7
    # Cara frontal
    front = Polygon([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]],
                    closed=True, facecolor=color, edgecolor=edgecolor, lw=1.0, alpha=alpha, zorder=4)
    # Cara superior
    top = Polygon([[x0, y0 + h], [x0 + w, y0 + h], [x0 + w + dx, y0 + h + dy], [x0 + dx, y0 + h + dy]],
                  closed=True, facecolor=light_color, edgecolor=edgecolor, lw=1.0, alpha=alpha, zorder=4)
    # Cara lateral derecha
    right = Polygon([[x0 + w, y0], [x0 + w + dx, y0 + dy], [x0 + w + dx, y0 + h + dy], [x0 + w, y0 + h]],
                    closed=True, facecolor=dark_color, edgecolor=edgecolor, lw=1.0, alpha=alpha, zorder=4)
    ax.add_patch(front)
    ax.add_patch(top)
    ax.add_patch(right)


def draw_feature_tensor_stack(ax, x0, y0, w, h, n_slabs=3, spacing=0.25, depth=0.35, color="#8FA8BF", light="#B8C9D9", dark="#6A8B9C"):
    """Dibuja un stack de múltiples láminas 3D para representar un tensor C x H x W."""
    for i in range(n_slabs):
        xi = x0 + i * spacing
        yi = y0 + i * spacing * 0.5
        draw_3d_slab(ax, xi, yi, w, h, depth=depth, color=color, light_color=light, dark_color=dark)


def draw_arrow(ax, p1, p2, text=None, text_offset=(0, 0.25), color="#65564B", lw=1.5):
    """Dibuja una flecha limpia de conexión entre componentes."""
    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45", color=color, lw=lw),
        zorder=5
    )
    if text:
        mx = (p1[0] + p2[0]) / 2 + text_offset[0]
        my = (p1[1] + p2[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, color="#302B27", fontsize=10, ha="center", va="center", zorder=6)


def embed_image(ax, img_array, x, y, w, h, title=None, border_color="#A38A77"):
    """Inserta una imagen o máscara real dentro de un recuadro con borde fino."""
    ins = ax.inset_axes([x, y, w, h], transform=ax.transData)
    if len(img_array.shape) == 2:
        ins.imshow(img_array, cmap="gray", vmin=0, vmax=255 if img_array.max() > 1 else 1)
    else:
        ins.imshow(img_array)
    ins.axis("off")
    # Marco exterior
    rect = patches.Rectangle((x, y), w, h, fill=False, edgecolor=border_color, lw=1.2, zorder=6)
    ax.add_patch(rect)
    if title:
        ax.text(x + w / 2, y - 0.25, title, fontsize=10, ha="center", va="top", color="#302B27")


# ==============================================================================
# 1. DIAGRAMA DE ARQUITECTURA DE SEGMENTACIÓN SIN ETIQUETAS (V8 / TEACHER 026)
# ==============================================================================
def generar_diagrama_segmentacion():
    """Genera el diagrama del Bloque 1: Segmentación autosupervisada modular."""
    fig, ax = plt.subplots(figsize=(15, 6.2), dpi=300)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    fig.patch.set_facecolor(PALETTE["white"])

    # Cargar imagen y máscara real de PH2
    img_raw = cv2.imread(str(REPO / "data" / "ph2" / "images" / "IMD002.png"))
    mask_raw = cv2.imread(str(REPO / "data" / "ph2" / "masks" / "IMD002_segmentation.png"), cv2.IMREAD_GRAYSCALE)
    if img_raw is not None:
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = np.full((256, 256, 3), 200, dtype=np.uint8)
    if mask_raw is None:
        mask_raw = np.zeros((256, 256), dtype=np.uint8)

    # Simular mapa CAM y pseudo-máscara tri-tono para coherencia visual real
    cam_sim = cv2.GaussianBlur(mask_raw, (45, 45), 0)
    cam_norm = cv2.normalize(cam_sim, None, 0, 255, cv2.NORM_MINMAX)
    # Tri-tono: Fondo 0, Incertidumbre 128, Lesión 255
    pseudo_tri = np.zeros_like(mask_raw)
    pseudo_tri[cam_norm > 80] = 128
    pseudo_tri[cam_norm > 160] = 255

    # Título general
    ax.text(10, 8.1, "Bloque 1: Arquitectura de Segmentación Sin Etiquetas (V8 / Teacher 026)",
            fontsize=12, fontweight="bold", ha="center", color=PALETTE["cappdark"])

    # --- Contenedor A: Extracción de Saliencia Contrastiva ---
    box_a = FancyBboxPatch((0.5, 0.8), 8.5, 6.8, boxstyle="round,pad=0.2,rounding_size=0.3",
                           facecolor="#FAF0E6", edgecolor="#BE9B7B", lw=1.2, zorder=1)
    ax.add_patch(box_a)
    ax.text(4.75, 7.3, r"A. Generación de Saliencia Contrastiva ($f_{CL} \to \mathrm{CCAM}$)",
            fontsize=11, fontweight="bold", ha="center", color=PALETTE["cappdark"])

    # 1. Entrada real
    embed_image(ax, img_rgb, 1.0, 4.2, 2.0, 2.0, title=r"$x \in \mathbb{R}^{224 \times 224 \times 3}$")
    
    # Flecha a Hair Removal
    draw_arrow(ax, (3.1, 5.2), (3.8, 5.2), text=r"$S_2'$")
    
    # 2. Imagen limpia (después de Black-hat)
    embed_image(ax, img_rgb, 4.0, 4.2, 2.0, 2.0, title=r"$x_{\mathrm{clean}}$")

    # Flecha a Backbone SSL
    draw_arrow(ax, (6.1, 5.2), (6.9, 5.2))

    # 3. Stack 3D de capas de características SSL
    draw_feature_tensor_stack(ax, 7.1, 4.3, 1.1, 1.8, n_slabs=3, spacing=0.25,
                              color=PALETTE["ssl_blue"], light=PALETTE["ssl_blue_light"], dark=PALETTE["ssl_blue_dark"])
    ax.text(7.65, 3.8, r"$f_{CL} \in \mathbb{R}^{C \times 14 \times 14}$" + "\n" + r"(SimCLR + MoCo)",
            fontsize=9.5, ha="center", color=PALETTE["cappnearblack"])

    # Flecha a CCAM
    draw_arrow(ax, (8.1, 3.2), (8.1, 2.3))

    # 4. Módulo CCAM (Conv 1x1 + Upsample)
    ccam_box = FancyBboxPatch((5.8, 1.2), 3.2, 1.0, boxstyle="round,pad=0.1",
                              facecolor=PALETTE["ccam_sage_light"], edgecolor=PALETTE["ccam_sage_dark"], lw=1.2, zorder=3)
    ax.add_patch(ccam_box)
    ax.text(7.4, 1.7, "CCAM ($f_{CC}$)\n" + r"$\mathrm{Conv}_{1\times 1} \to \mathrm{Upsample} \to \sim$",
            fontsize=9.5, ha="center", va="center", color=PALETTE["cappdark"], fontweight="bold")

    # Flecha hacia el CAM resultante
    draw_arrow(ax, (5.8, 1.7), (4.5, 1.7))

    # 5. Mapa térmico CAM resultante
    embed_image(ax, cam_norm, 2.2, 1.2, 2.0, 2.0, title=r"$\mathrm{CAM} \in [0, 1]^{224 \times 224}$")

    # --- Contenedor B: Módulo de Incertidumbre y Pseudo-etiquetas ---
    box_b = FancyBboxPatch((9.5, 0.8), 9.8, 6.8, boxstyle="round,pad=0.2,rounding_size=0.3",
                           facecolor="#FFF5EE", edgecolor="#BE9B7B", lw=1.2, zorder=1)
    ax.add_patch(box_b)
    ax.text(14.4, 7.3, "B. Módulo de Incertidumbre y Refinamiento Iterativo (Teacher 026)",
            fontsize=11, fontweight="bold", ha="center", color=PALETTE["cappdark"])

    # Flecha de CAM a Uncertainty Module
    draw_arrow(ax, (3.2, 3.3), (10.2, 4.5), text=r"$\tau_l, \tau_h$")

    # 6. Uncertainty Module (UM) Box
    um_box = FancyBboxPatch((10.0, 4.0), 2.8, 2.2, boxstyle="round,pad=0.1",
                            facecolor=PALETTE["cappcream"], edgecolor=PALETTE["cappdark"], lw=1.2, zorder=3)
    ax.add_patch(um_box)
    ax.text(11.4, 5.7, "Uncertainty Module ($f_{UM}$)", fontsize=10, ha="center", fontweight="bold")
    ax.text(11.4, 5.0, r"$\mathrm{Low} < \tau_l \to \mathrm{Fondo (0)}$", fontsize=9, ha="center")
    ax.text(11.4, 4.5, r"$\mathrm{High} > \tau_h \to \mathrm{Centralidad}$", fontsize=9, ha="center")
    ax.text(11.4, 4.1, r"$\mathrm{Mid} \to \mathrm{Incertidumbre (0.5)}$", fontsize=9, ha="center")

    # Flecha a Pseudo-label tri-tono
    draw_arrow(ax, (12.9, 5.1), (13.7, 5.1))

    # 7. Pseudo-etiqueta tri-tono
    embed_image(ax, pseudo_tri, 13.9, 4.1, 2.0, 2.0, title=r"$\tilde{y} \in \{0, 0.5, 1\}$")

    # Flecha a Red Segmentadora
    draw_arrow(ax, (16.0, 5.1), (16.8, 5.1))

    # 8. Red Segmentadora (Encoder-Decoder)
    draw_feature_tensor_stack(ax, 17.0, 4.3, 0.9, 1.6, n_slabs=2, spacing=0.2,
                              color=PALETTE["dec_peach"], light=PALETTE["dec_peach_light"], dark=PALETTE["dec_peach_dark"])
    ax.text(17.4, 3.8, r"$\mathrm{U\text{-}Net} \, (f_E, f_D)$" + "\n" + r"$\mathcal{L}_{\mathrm{unc}}(\tilde{y}, p)$",
            fontsize=9.5, ha="center", color=PALETTE["cappnearblack"])

    # Flecha al resultado final
    draw_arrow(ax, (17.5, 3.2), (17.5, 2.3))

    # 9. Máscara binaria anatómica final Teacher 026
    embed_image(ax, mask_raw, 16.5, 1.2, 2.0, 2.0, title=r"Máscara $M \in \{0, 1\}$")

    # Bucle iterativo de autoaprendizaje (Feedback arrow)
    ax.annotate(
        "", xy=(11.4, 3.8), xytext=(16.3, 2.2),
        arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45",
                        color=PALETTE["cappwine"], lw=1.5, connectionstyle="arc3,rad=-0.25"),
        zorder=5
    )
    ax.text(13.8, 2.3, "Iterative Refinement (Teacher 026)",
            fontsize=9.5, ha="center", color=PALETTE["cappwine"], style="italic")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.04)
    # Guardar en formatos vectoriales y alta definición
    for ext in ("png", "pdf", "svg"):
        p = OUT_DIR / f"diagrama_bloque1_segmentacion_v8.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300)
        if BRAIN_DIR.exists():
            fig.savefig(BRAIN_DIR / f"diagrama_bloque1_segmentacion_v8.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Bloque 1 exportado con éxito.")


# ==============================================================================
# 2. DIAGRAMA DE TRANSFORMACIÓN TONAL ACOPLADA ESPACIALMENTE (V10)
# ==============================================================================
def generar_diagrama_transformacion_acople():
    """Genera el diagrama del Bloque 2: Cómo se conecta la máscara con la transformación tonal."""
    fig, ax = plt.subplots(figsize=(15, 6.2), dpi=300)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    fig.patch.set_facecolor(PALETTE["white"])

    # Cargar imagen y máscara real
    img_raw = cv2.imread(str(REPO / "data" / "ph2" / "images" / "IMD002.png"))
    mask_raw = cv2.imread(str(REPO / "data" / "ph2" / "masks" / "IMD002_segmentation.png"), cv2.IMREAD_GRAYSCALE)
    if img_raw is not None:
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = np.full((256, 256, 3), 200, dtype=np.uint8)
    if mask_raw is None:
        mask_raw = np.zeros((256, 256), dtype=np.uint8)

    # Simular la transformación tonal real en espacio CIELAB
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(img_lab)
    # Acople espacial m = 0.8609
    m = 0.8609
    delta_L = 30.0  # Desplazamiento de luminancia
    M_norm = (mask_raw > 127).astype(np.float32)
    mod_factor = (1.0 - M_norm) + m * M_norm
    L_trans = np.clip(L.astype(np.float32) + delta_L * mod_factor, 0, 255).astype(np.uint8)
    img_lab_trans = cv2.merge([L_trans, a, b])
    img_trans_rgb = cv2.cvtColor(img_lab_trans, cv2.COLOR_LAB2RGB)

    # Título general
    ax.text(10, 8.1, "Bloque 2: Módulo de Conexión y Transformación Tonal Acoplada Espacialmente (Línea V10)",
            fontsize=12, fontweight="bold", ha="center", color=PALETTE["cappdark"])

    # Contenedor principal
    box_trans = FancyBboxPatch((0.5, 0.8), 19.0, 6.8, boxstyle="round,pad=0.2,rounding_size=0.3",
                               facecolor="#FFF9F5", edgecolor="#BE9B7B", lw=1.2, zorder=1)
    ax.add_patch(box_trans)

    # 1. Entradas duales: Imagen Original y Máscara Binaria
    embed_image(ax, img_rgb, 1.2, 4.4, 2.2, 2.2, title=r"Imagen Original $x(u, v)$")
    embed_image(ax, mask_raw, 1.2, 1.4, 2.2, 2.2, title=r"Máscara $M(u, v) \in \{0, 1\}$" + "\n" + r"(desde Bloque 1)")

    # Flechas convergentes hacia Conversión CIELAB y Partición
    draw_arrow(ax, (3.5, 5.5), (4.8, 4.8), text=r"$\text{RGB} \to \text{CIELAB}$")
    draw_arrow(ax, (3.5, 2.5), (4.8, 3.8), text=r"$\Omega_{\text{skin}} \cup \Omega_{\text{lesion}}$")

    # 2. Caja de Partición de Dominios
    part_box = FancyBboxPatch((5.0, 2.8), 3.4, 2.8, boxstyle="round,pad=0.1",
                              facecolor=PALETTE["cappcream"], edgecolor=PALETTE["cappdark"], lw=1.2, zorder=3)
    ax.add_patch(part_box)
    ax.text(6.7, 5.1, "Partición Anatómica", fontsize=10, ha="center", fontweight="bold")
    ax.text(6.7, 4.3, r"$\Omega_{\mathrm{skin}} = \{u, v \mid M=0\}$", fontsize=9.5, ha="center")
    ax.text(6.7, 3.8, "(Piel perilesional sana)", fontsize=8.5, ha="center", color="#65564B")
    ax.text(6.7, 3.3, r"$\Omega_{\mathrm{lesion}} = \{u, v \mid M=1\}$", fontsize=9.5, ha="center")
    ax.text(6.7, 2.9, "(Tumor / lesión cutánea)", fontsize=8.5, ha="center", color="#65564B")

    # Flecha a la Ecuación Central de Acople
    draw_arrow(ax, (8.5, 4.2), (9.8, 4.2))

    # 3. Caja de la Ecuación Matemática Central
    eq_box = FancyBboxPatch((10.0, 2.3), 5.4, 3.8, boxstyle="round,pad=0.15",
                            facecolor="#FFF4E6", edgecolor=PALETTE["cappwine"], lw=1.5, zorder=3)
    ax.add_patch(eq_box)
    ax.text(12.7, 5.5, "Ecuación de Acople Espacial ($L^*$ y $m$)",
            fontsize=10.5, ha="center", fontweight="bold", color=PALETTE["cappwine"])
    
    # Ecuación matemática resaltada
    eq_text = r"$L^*_{\mathrm{trans}}(u, v) = L^*(u, v) + \Delta L^* \cdot [ (1 - M) + m \cdot M ]$"
    ax.text(12.7, 4.4, eq_text, fontsize=11, ha="center", va="center", color=PALETTE["cappdark"])
    
    ax.text(12.7, 3.5, r"• Factor de Acople: $m = 0.8609 \to \mathrm{Preserva } \, \Delta b^*$",
            fontsize=9.5, ha="center", color=PALETTE["cappnearblack"])
    ax.text(12.7, 2.9, r"• Margen C1M: Freno suave si $L^* < 28$ (evita saturación)",
            fontsize=9.5, ha="center", color=PALETTE["cappnearblack"])

    # Flecha a la Imagen Transmitida
    draw_arrow(ax, (15.5, 4.2), (16.6, 4.2), text=r"$\mathrm{LAB} \to \mathrm{RGB}$")

    # 4. Imagen Transformada resultante (Piel clara aumentada sin alterar tumor)
    embed_image(ax, img_trans_rgb, 16.8, 3.0, 2.4, 2.4,
                title="Imagen Acoplada $\\tilde{x}$\n(hacia Bloque 3: Encoder)")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.04)
    for ext in ("png", "pdf", "svg"):
        p = OUT_DIR / f"diagrama_bloque2_transformacion_acople.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300)
        if BRAIN_DIR.exists():
            fig.savefig(BRAIN_DIR / f"diagrama_bloque2_transformacion_acople.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Bloque 2 exportado con éxito.")


# ==============================================================================
# 3. DIAGRAMA DE ARQUITECTURA DE CLASIFICACIÓN MULTICLASE SSL (V7 / VIT-B/14)
# ==============================================================================
def generar_diagrama_clasificacion():
    """Genera el diagrama del Bloque 3: Clasificador Vision Transformer SSL con cabeza calibrada."""
    fig, ax = plt.subplots(figsize=(15, 6.2), dpi=300)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    fig.patch.set_facecolor(PALETTE["white"])

    # Cargar imagen
    img_raw = cv2.imread(str(REPO / "data" / "ph2" / "images" / "IMD002.png"))
    if img_raw is not None:
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = np.full((256, 256, 3), 200, dtype=np.uint8)

    # Título general
    ax.text(10, 8.1, "Bloque 3: Arquitectura de Clasificación Multiclase SSL (Línea V7 / DINOv2 ViT-B/14)",
            fontsize=12, fontweight="bold", ha="center", color=PALETTE["cappdark"])

    box_cls = FancyBboxPatch((0.5, 0.8), 19.0, 6.8, boxstyle="round,pad=0.2,rounding_size=0.3",
                            facecolor="#FAF5F0", edgecolor="#BE9B7B", lw=1.2, zorder=1)
    ax.add_patch(box_cls)

    # 1. Entrada: Imagen modulada x_tilde
    embed_image(ax, img_rgb, 1.0, 3.8, 2.2, 2.2, title=r"$\tilde{x} \in \mathbb{R}^{224 \times 224 \times 3}$" + "\n" + r"(desde Bloque 2)")

    # Flecha a Patch Embedding
    draw_arrow(ax, (3.3, 4.9), (4.2, 4.9))

    # 2. Patch Embedding (Grid de Parches 14x14)
    patch_box = FancyBboxPatch((4.4, 3.2), 2.8, 3.4, boxstyle="round,pad=0.1",
                               facecolor=PALETTE["cappcream"], edgecolor=PALETTE["cappdark"], lw=1.2, zorder=3)
    ax.add_patch(patch_box)
    ax.text(5.8, 6.1, "Patch Embedding", fontsize=10, ha="center", fontweight="bold")
    ax.text(5.8, 5.5, r"$16 \times 16 = 256 \text{ parches}$", fontsize=9.5, ha="center")
    ax.text(5.8, 5.0, r"$\mathbf{x}_p \in \mathbb{R}^{14 \times 14 \times 3}$", fontsize=9.5, ha="center")
    ax.text(5.8, 4.4, "+ Positional Embed", fontsize=9, ha="center", color="#65564B")
    ax.text(5.8, 3.9, r"+ Token $[\mathrm{CLS}]$", fontsize=9, ha="center", color=PALETTE["cappwine"], fontweight="bold")
    ax.text(5.8, 3.4, r"$\mathbf{z}_0 \in \mathbb{R}^{257 \times 768}$", fontsize=9.5, ha="center")

    # Flecha a Transformer Encoder
    draw_arrow(ax, (7.3, 4.9), (8.2, 4.9))

    # 3. Transformer Encoder Stack 3D (L=12 bloques ViT-B/14)
    draw_feature_tensor_stack(ax, 8.4, 3.6, 1.5, 2.6, n_slabs=4, spacing=0.25,
                              color=PALETTE["vit_wine"], light=PALETTE["vit_wine_light"], dark=PALETTE["vit_wine_dark"])
    ax.text(9.2, 2.8, "ViT-B/14 ($L=12$ bloques)\nMulti-Head Attention (MSA)\nPreentrenado en ISIC 2019 SSL",
            fontsize=9.5, ha="center", color=PALETTE["cappdark"], fontweight="bold")

    # Flecha a Latent Representation
    draw_arrow(ax, (10.9, 4.9), (11.8, 4.9))

    # 4. Latent Vector z_[CLS]
    z_box = FancyBboxPatch((12.0, 4.1), 1.6, 1.6, boxstyle="round,pad=0.1",
                           facecolor="#EAD7C5", edgecolor=PALETTE["cappdark"], lw=1.2, zorder=3)
    ax.add_patch(z_box)
    ax.text(12.8, 5.1, r"$\mathbf{z}_{[\mathrm{CLS}]}$", fontsize=11, ha="center", fontweight="bold", color=PALETTE["cappdark"])
    ax.text(12.8, 4.5, r"$\in \mathbb{R}^{768}$", fontsize=9.5, ha="center")

    # Flecha a Calibrated Linear Head
    draw_arrow(ax, (13.7, 4.9), (14.6, 4.9))

    # 5. Linear Head & Calibración de Temperatura
    head_box = FancyBboxPatch((14.8, 3.2), 3.0, 3.4, boxstyle="round,pad=0.1",
                              facecolor=PALETTE["cappcream"], edgecolor=PALETTE["cappwine"], lw=1.2, zorder=3)
    ax.add_patch(head_box)
    ax.text(16.3, 6.1, "Cabeza Calibrada", fontsize=10, ha="center", fontweight="bold", color=PALETTE["cappwine"])
    ax.text(16.3, 5.4, r"Logits: $\mathbf{s} = \mathbf{W}\mathbf{z} + \mathbf{b}$", fontsize=9.5, ha="center")
    ax.text(16.3, 4.7, "Escalamiento con $T$:", fontsize=9, ha="center", color="#65564B")
    ax.text(16.3, 4.0, r"$P(y=c) = \frac{e^{s_c / T}}{\sum_j e^{s_j / T}}$", fontsize=10, ha="center")
    ax.text(16.3, 3.4, r"$\mathrm{ECE} \downarrow = 1.9\,\% \mathrm{ (Confiable)}$", fontsize=9, ha="center", fontweight="bold")

    # Flecha a Probabilidades
    draw_arrow(ax, (17.9, 4.9), (18.6, 4.9))

    # 6. Gráfica de barras de salida con las 7 clases diagnósticas
    classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC"]
    probs = [0.82, 0.08, 0.04, 0.02, 0.02, 0.01, 0.01]
    colors = [PALETTE["cappwine"] if c == "MEL" else PALETTE["capptan"] for c in classes]

    ins_bar = ax.inset_axes([18.7, 3.0, 1.1, 3.8], transform=ax.transData)
    ins_bar.barh(range(len(classes)), probs, color=colors, height=0.6)
    ins_bar.set_yticks(range(len(classes)))
    ins_bar.set_yticklabels(classes, fontsize=8)
    ins_bar.invert_yaxis()
    ins_bar.set_xlim(0, 1)
    ins_bar.set_xticks([0, 1])
    ins_bar.set_xticklabels(["0", "1"], fontsize=7)
    ins_bar.set_title(r"$P(y \mid \tilde{x})$", fontsize=9)
    ins_bar.spines["top"].set_visible(False)
    ins_bar.spines["right"].set_visible(False)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.04)
    for ext in ("png", "pdf", "svg"):
        p = OUT_DIR / f"diagrama_bloque3_clasificacion_vit_v7.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300)
        if BRAIN_DIR.exists():
            fig.savefig(BRAIN_DIR / f"diagrama_bloque3_clasificacion_vit_v7.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Bloque 3 exportado con éxito.")


# ==============================================================================
# 4. DIAGRAMA DEL PIPELINE COMPLETO INTEGRADO POR BLOQUES (END-TO-END)
# ==============================================================================
def generar_diagrama_pipeline_completo():
    """Genera la figura general que integra los 3 bloques metodológicos."""
    fig, ax = plt.subplots(figsize=(16, 7.5), dpi=300)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 9.5)
    ax.axis("off")
    fig.patch.set_facecolor(PALETTE["white"])

    # Cargar imágenes
    img_raw = cv2.imread(str(REPO / "data" / "ph2" / "images" / "IMD002.png"))
    mask_raw = cv2.imread(str(REPO / "data" / "ph2" / "masks" / "IMD002_segmentation.png"), cv2.IMREAD_GRAYSCALE)
    if img_raw is not None:
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = np.full((256, 256, 3), 200, dtype=np.uint8)
    if mask_raw is None:
        mask_raw = np.zeros((256, 256), dtype=np.uint8)

    # Imagen transformada
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(img_lab)
    m = 0.8609
    delta_L = 30.0
    M_norm = (mask_raw > 127).astype(np.float32)
    mod_factor = (1.0 - M_norm) + m * M_norm
    L_trans = np.clip(L.astype(np.float32) + delta_L * mod_factor, 0, 255).astype(np.uint8)
    img_lab_trans = cv2.merge([L_trans, a, b])
    img_trans_rgb = cv2.cvtColor(img_lab_trans, cv2.COLOR_LAB2RGB)

    # Título general
    ax.text(11, 9.1, "Pipeline Metodológico Integrado por Bloques Funcionales",
            fontsize=13, fontweight="bold", ha="center", color=PALETTE["cappdark"])

    # ------------------ BLOQUE 1 ------------------
    b1 = FancyBboxPatch((0.4, 0.8), 6.5, 7.8, boxstyle="round,pad=0.2,rounding_size=0.3",
                        facecolor="#FAF0E6", edgecolor="#BE9B7B", lw=1.2, zorder=1)
    ax.add_patch(b1)
    ax.text(3.65, 8.2, "Bloque 1: Segmentación SSL (V8)", fontsize=11, ha="center", color=PALETTE["cappdark"], fontweight="bold")

    embed_image(ax, img_rgb, 0.8, 5.0, 2.0, 2.0, title=r"$x \in \mathbb{R}^{224 \times 224 \times 3}$")
    draw_arrow(ax, (2.9, 6.0), (3.6, 6.0), text=r"$S_2'$")
    
    # Capas SSL
    draw_feature_tensor_stack(ax, 3.8, 5.2, 0.9, 1.6, n_slabs=2, spacing=0.2,
                              color=PALETTE["ssl_blue"], light=PALETTE["ssl_blue_light"], dark=PALETTE["ssl_blue_dark"])
    ax.text(4.25, 4.7, r"$f_{CL} \to \mathrm{CCAM}$", fontsize=9, ha="center")

    draw_arrow(ax, (4.25, 4.3), (4.25, 3.4))

    # Teacher 026
    t_box = FancyBboxPatch((2.2, 2.3), 4.1, 1.0, boxstyle="round,pad=0.1",
                           facecolor=PALETTE["cappcream"], edgecolor=PALETTE["cappdark"], lw=1.0, zorder=3)
    ax.add_patch(t_box)
    ax.text(4.25, 2.8, "Teacher 026 Refinement\n" + r"Pérdida con Incertidumbre $\mathcal{L}_{\mathrm{unc}}$",
            fontsize=8.5, ha="center", va="center")

    draw_arrow(ax, (2.2, 2.8), (1.8, 2.8))

    # Máscara de Salida del Bloque 1
    embed_image(ax, mask_raw, 0.8, 1.4, 2.0, 2.0, title=r"Máscara $M \in \{0, 1\}$")

    # ------------------ CONEXIÓN BLOQUE 1 -> BLOQUE 2 ------------------
    # Flecha destacada de la máscara hacia el Bloque 2
    ax.annotate(
        "", xy=(7.6, 3.2), xytext=(2.9, 2.4),
        arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.5",
                        color=PALETTE["cappwine"], lw=2.0, connectionstyle="arc3,rad=-0.15"),
        zorder=5
    )
    ax.text(5.2, 2.0, r"Conexión: $M(u, v) \to \mathrm{Acople}$",
            fontsize=9.5, ha="center", color=PALETTE["cappwine"], fontweight="bold")

    # Flecha de la imagen original hacia el Bloque 2
    ax.annotate(
        "", xy=(7.6, 6.2), xytext=(2.9, 6.0),
        arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45",
                        color=PALETTE["connector"], lw=1.5, connectionstyle="arc3,rad=0.1"),
        zorder=5
    )
    ax.text(5.2, 6.8, "Imagen original $x$", fontsize=9, ha="center", color=PALETTE["connector"])

    # ------------------ BLOQUE 2 ------------------
    b2 = FancyBboxPatch((7.6, 0.8), 6.5, 7.8, boxstyle="round,pad=0.2,rounding_size=0.3",
                        facecolor="#FFF9F5", edgecolor="#BE9B7B", lw=1.2, zorder=1)
    ax.add_patch(b2)
    ax.text(10.85, 8.2, "Bloque 2: Transformación Tonal (V10)", fontsize=11, ha="center", color=PALETTE["cappdark"], fontweight="bold")

    # Caja de Conversión y Acople
    c2_box = FancyBboxPatch((8.1, 4.0), 5.5, 3.6, boxstyle="round,pad=0.15",
                            facecolor=PALETTE["cappcream"], edgecolor=PALETTE["cappwine"], lw=1.2, zorder=3)
    ax.add_patch(c2_box)
    ax.text(10.85, 7.1, r"$\mathrm{RGB} \to \mathrm{CIELAB} \, (L^*, a^*, b^*)$", fontsize=10, ha="center", fontweight="bold")
    ax.text(10.85, 6.3, r"$L^*_{\mathrm{trans}} = L^* + \Delta L^* \cdot [(1 - M) + m M]$", fontsize=9.5, ha="center")
    ax.text(10.85, 5.3, r"$m = 0.8609 \to \mathrm{Preserva } \, \Delta b^*$", fontsize=9, ha="center", color=PALETTE["cappwine"])
    ax.text(10.85, 4.6, r"Margen C1M: $\sigma(L^*)$ para $L^* < 28$", fontsize=9, ha="center")

    draw_arrow(ax, (10.85, 4.0), (10.85, 3.5))

    # Imagen Transmitida
    embed_image(ax, img_trans_rgb, 9.85, 1.4, 2.0, 2.0, title="Imagen Acoplada $\\tilde{x}$")

    # ------------------ CONEXIÓN BLOQUE 2 -> BLOQUE 3 ------------------
    draw_arrow(ax, (11.95, 2.4), (14.8, 4.8), text=r"$\tilde{x} \to \mathrm{ViT}$", color=PALETTE["cappwine"], lw=2.0)

    # ------------------ BLOQUE 3 ------------------
    b3 = FancyBboxPatch((14.8, 0.8), 6.8, 7.8, boxstyle="round,pad=0.2,rounding_size=0.3",
                        facecolor="#FAF5F0", edgecolor="#BE9B7B", lw=1.2, zorder=1)
    ax.add_patch(b3)
    ax.text(18.2, 8.2, "Bloque 3: Clasificador SSL (V7)", fontsize=11, ha="center", color=PALETTE["cappdark"], fontweight="bold")

    # Patch Embedding y ViT
    draw_feature_tensor_stack(ax, 15.3, 5.0, 1.2, 2.0, n_slabs=3, spacing=0.2,
                              color=PALETTE["vit_wine"], light=PALETTE["vit_wine_light"], dark=PALETTE["vit_wine_dark"])
    ax.text(15.9, 4.5, "ViT-B/14\n$256$ parches", fontsize=9, ha="center", fontweight="bold")

    draw_arrow(ax, (16.9, 6.0), (17.6, 6.0))

    # Token CLS
    ax.text(18.2, 6.0, r"$\mathbf{z}_{[\mathrm{CLS}]} \in \mathbb{R}^{768}$", fontsize=9.5, ha="center", fontweight="bold")
    draw_arrow(ax, (18.2, 5.5), (18.2, 4.8))

    # Cabeza Calibrada
    cal_box = FancyBboxPatch((17.0, 3.8), 2.4, 0.9, boxstyle="round,pad=0.1",
                             facecolor=PALETTE["cappcream"], edgecolor=PALETTE["cappwine"], lw=1.0, zorder=3)
    ax.add_patch(cal_box)
    ax.text(18.2, 4.25, r"$\mathrm{Softmax}((\mathbf{W}\mathbf{z}+\mathbf{b})/T)$", fontsize=8.5, ha="center", va="center")

    draw_arrow(ax, (18.2, 3.8), (18.2, 3.2))

    # Barras de salida
    ins_bar = ax.inset_axes([17.2, 1.4, 2.0, 1.6], transform=ax.transData)
    classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VAS"]
    probs = [0.82, 0.08, 0.04, 0.02, 0.02, 0.01, 0.01]
    colors = [PALETTE["cappwine"] if c == "MEL" else PALETTE["capptan"] for c in classes]
    ins_bar.bar(range(len(classes)), probs, color=colors, width=0.6)
    ins_bar.set_xticks(range(len(classes)))
    ins_bar.set_xticklabels(classes, fontsize=7)
    ins_bar.set_ylim(0, 1)
    ins_bar.set_title(r"$P(y \mid \tilde{x})$ (7 patologías)", fontsize=8)
    ins_bar.spines["top"].set_visible(False)
    ins_bar.spines["right"].set_visible(False)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.04)
    for ext in ("png", "pdf", "svg"):
        p = OUT_DIR / f"diagrama_pipeline_completo_bloques.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300)
        if BRAIN_DIR.exists():
            fig.savefig(BRAIN_DIR / f"diagrama_pipeline_completo_bloques.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Pipeline completo exportado con éxito.")


def main():
    print("Generando suite completa de diagramas de arquitectura...")
    generar_diagrama_segmentacion()
    generar_diagrama_transformacion_acople()
    generar_diagrama_clasificacion()
    generar_diagrama_pipeline_completo()
    print("Todos los diagramas fueron generados en docs/figuras_analisis/ (PNG, PDF, SVG).")


if __name__ == "__main__":
    main()
