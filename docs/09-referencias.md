# 09 — Referencias

## Métodos SSL

- **SimSiam** — Chen & He, "Exploring Simple Siamese Representation Learning", CVPR 2021.
  (El método que usamos hoy: negative-free, corre en 1 GPU.)
- **DINO / DINOv2** — Caron et al. 2021 / Oquab et al. 2023. (Núcleo del anteproyecto; objetivo con
  cómputo real.)
- **Barlow Twins** — Zbontar et al. 2021. (Alternativa considerada.)
- **SimCLR** — Chen et al. 2020. (Probado en Práctica 1; pierde desde cero con cómputo de estudiante.)
- **MAE / SimMIM** — He et al. 2022 / Xie et al. 2022. (Masked Image Modeling en 2D; alternativa
  reconstructiva viable si el director quiere explorar MIM — ver evaluación de MambaMIM abajo.)

## Datasets y dominio clínico

- **HAM10000** — Tschandl et al. 2018, "The HAM10000 dataset", Scientific Data.
- **BCN20000**, **ISIC 2019**, **CO2Wounds-V2**.
- **Fitzpatrick** — escala de fototipos de piel (Fitzpatrick 1988).
- **ABCDE** — regla clínica de melanoma (Friedman/Kopf/Rigel 1985).

## Fiabilidad, sesgo y XAI

- **BOSQUE** — arXiv 2507.14176: test set de Bogotá con fototipo Fitzpatrick para medir sesgo racial.
- **Fairness melanoma** — arXiv 2411.12846: recomendaciones de equidad.
- **Grad-CAM + experto** — arXiv 2407.00104.
- **Artefactos (pelos)** — ScienceDirect S0010482524014203.
- **XAI en HAM10000** — doi 10.3390/bioengineering13060677.
- **Validación externa ISIC2019** — PMC11098764.

## Metodología

- **CRISP-ML(Q)** — Studer et al. 2021, arXiv:2003.05155.
- **CRISP-DM** (legacy) — Wirth & Hipp 2000.

---

## Evaluación: MambaMIM (sugerido por el director)

**Repo:** https://github.com/FengheTan9/MambaMIM · **Paper:** Medical Image Analysis 2025,
arXiv 2408.08070 · Apache 2.0.

**Qué es:** framework de pre-entrenamiento SSL por **Masked Image Modeling (MIM)** con backbone
**Mamba** (state-space models) para imagen médica.

**Veredicto para nuestro proyecto: NO usarlo como motor directo.** 4 desajustes duros:

| MambaMIM | Nuestro proyecto |
|---|---|
| **3D** volumétrico (96³) | **2D** dermatoscopia RGB (224px) |
| **Segmentación** (nnUNet, máscaras) | **Clasificación** 7 clases |
| **CT** (Hounsfield) | Fotografía de piel (RGB) |
| **Mamba** + kernels CUDA `mamba-ssm` | ResNet-50/ViT, T4 de Kaggle |

Adaptarlo = reescribir el paper (3D→2D, seg→clasif, CT→dermoscopia) + pelear con `mamba-ssm` en
Kaggle. **No es defendible para un pregrado con T4.**

**Lo que SÍ vale:**
1. **Referencia de estado del arte** (SSL-MIM médico, 2025, revista buena) para el marco teórico.
2. Señal de que **MIM (masked image modeling)** es una dirección válida. Si el director quiere
   explorar MIM de verdad, la versión viable en 2D es **MAE / SimMIM** sobre ViT/ResNet (corre en
   Kaggle), como brazo alternativo al SimSiam contrastivo actual.

> Pregunta abierta para el director: ¿MambaMIM como referencia, o quiere que probemos el enfoque MIM
> (vía MAE 2D) frente al contrastivo (SimSiam)?
