# 14 — Mejoras del pipeline (para que el SSL gane)

Investigación del estado del arte (2024-2026) + cambios concretos aplicados al notebook para que el
**SSL supere** al baseline ImageNet, no solo lo iguale.

## Diagnóstico: ¿por qué el SSL empataba?

En la corrida de referencia el SSL empató a ImageNet en macro-F1 (0.7447 vs 0.7454) y solo ganó en
AUC. La causa **no** era un SSL débil — era **cómo lo afinábamos**:

> **Kumar et al., 2022 — "Fine-tuning can distort pretrained features and underperform
> out-of-distribution"** (arXiv:2202.10054): el *full fine-tune* ingenuo **distorsiona** las
> features pre-entrenadas. Cuando la representación de partida es buena (como la del SSL
> domain-adaptive), re-entrenar todo el backbone con un LR uniforme **borra esa ventaja** y converge
> al mismo punto que ImageNet. El efecto es peor **fuera de distribución** — justo el escenario
> clínico real.

Nuestro linear probe ya mostraba que la representación SSL es competitiva (gana AUC). El problema era
que el fine-tune la tapaba. **Ese es el fix.**

---

## Mejoras aplicadas (en el notebook)

### ✅ Mejora 1 — LP-FT (Linear Probe → Fine-Tune)
En vez de fine-tune directo, dos fases:
1. **LP:** backbone **congelado**, se entrena solo la cabeza lineal hasta alinearla.
2. **FT:** se descongela todo con **LR chico**, partiendo de una cabeza ya buena.

Así el fine-tune **no necesita mover mucho el backbone** → preserva la representación SSL.
Kumar 2022 reporta **+1 % en-distribución y +7-10 % fuera-de-distribución** vs fine-tune ingenuo.
Se aplica a **ambos brazos** (comparación justa); el que más se beneficia es el SSL, porque su
representación de partida es la que había que preservar.

`train_arm` reescrito (celda nueva tras la 4): `LP_EPOCHS=10`, `FT_EPOCHS=15`, `LP_LR=1e-3`, `FT_LR=1e-4`.

### ✅ Mejora 2 — Layer-wise LR Decay (LLRD)
En la fase FT, cada bloque de la ResNet recibe un LR distinto: **alto en la cabeza, decreciente
hacia las capas tempranas** (`fc` → `layer4` → … → `conv1`, factor 0.75 por bloque). Las capas
tempranas (bordes, texturas — genéricas) casi no se tocan; las profundas (semánticas) se adaptan.
Estándar en fine-tune moderno; refuerza la preservación de features.

### ✅ Mejora 3 — Más datos NO etiquetados para el SSL
El SSL **escala con datos sin etiqueta** — es su superpoder, y algo que la supervisada **no puede
aprovechar**. Los papers de dermatología usan **~200 k** imágenes ISIC; nosotros usábamos ~33 k. El
notebook ahora **detecta y suma automáticamente cualquier dataset ISIC** agregado como Input
(ISIC 2019 + 2020 + archive…), no solo el de 2019. **Acción para Nico:** agregar ISIC 2020 y/o más
archivos ISIC como Input antes de correr → el pool SSL crece y el SSL despega.

---

## Mejoras futuras (requieren más cómputo — GPUs UNAB)

| Mejora | Evidencia | Costo |
|---|---|---|
| **DINO / DINOv2 (ViT)** en vez de SimSiam | DINO en dermatología **supera ImageNet +5-8 % AUC**; DINOv2-B llega a **93 %** en HAM10000 | ViT + mucho cómputo |
| **Multi-crop** (vistas locales+globales) | Ingrediente clave de DINO/SwAV; mejora mucho la representación | +memoria/tiempo |
| **MAE / SimMIM** (masked image modeling) | Masking espacialmente correlacionado mejora bordes de lesión | ViT |
| **Color constancy** (shades-of-gray) en augmentations | Normalización estándar en dermatoscopía | bajo (a validar) |
| **SSL más largo** (100+ épocas) sobre el pool grande | La representación mejora aunque la loss se aplane | tiempo |

---

## Expectativa honesta

Con **LP-FT + LLRD + más datos SSL**, lo esperable es que el **SSL pase a ganar** (o al menos amplíe
su ventaja en AUC y empiece a ganar en macro-F1), sobre todo en el **régimen de pocas etiquetas** y
en **validación externa** (OOD), que es donde LP-FT más ayuda. No es garantía —es investigación— pero
son palancas con respaldo empírico, no corazonadas. El salto grande (superar claramente a la
supervisada) llega con **DINO/ViT + 200 k imágenes**, que es el plan con las GPUs de la UNAB.

## Referencias

- Kumar, A. et al. (2022). *Fine-tuning can distort pretrained features and underperform
  out-of-distribution.* ICLR. arXiv:2202.10054.
- Chen, X. & He, K. (2021). *Exploring Simple Siamese Representation Learning (SimSiam).* CVPR.
- Caron, M. et al. (2021). *Emerging Properties in Self-Supervised Vision Transformers (DINO).* ICCV.
- Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.*
- Estudios 2024-2025 SSL en dermatoscopía (HAM10000 / ISIC): DINO/DINOv2 > ImageNet en AUC;
  SSL con ~200 k imágenes ISIC sin etiquetar.
