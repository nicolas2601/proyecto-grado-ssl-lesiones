# 14 — Mejoras del pipeline (para que el SSL gane)

> **Documento histórico.** Describe una etapa anterior del proyecto y fue superado. Los resultados vigentes están en el [README](../README.md) y en los documentos 27 a 33. Se conserva para mostrar cómo se llegó al diseño actual.

Investigación del estado del arte (2024-2026) + cambios concretos aplicados al notebook para que el
**SSL supere** al baseline ImageNet, no solo lo iguale.

> ✅ **RESULTADO (v4, 2026-07-12): estas mejoras FUNCIONARON.** El SSL pasó de empatar (v2) a
> **superar a ImageNet** en macro-F1 (0.6907 vs 0.6614, Δ +0.0294, fuera del ruido) y en AUC. Abajo
> queda el diagnóstico y las palancas que lo lograron.

## Diagnóstico: ¿por qué el SSL empataba (antes)?

En la corrida de referencia previa el SSL empataba a ImageNet en macro-F1 y solo ganaba en AUC. La
causa **no** era un SSL débil — era **cómo lo afinábamos** y **cuántos datos** le dábamos:

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

### ✅ Mejora 3 — Más datos NO etiquetados para el SSL (LA CLAVE DE LA VICTORIA)
El SSL **escala con datos sin etiqueta** — es su superpoder, y algo que la supervisada **no puede
aprovechar**. Los papers de dermatología usan **~200 k** imágenes ISIC; antes usábamos ~33 k. En v4
se **sumó ISIC 2020** → el pool SSL creció a **~48 k** imágenes. El notebook detecta y suma
automáticamente cualquier dataset ISIC agregado como Input. **Este fue el factor decisivo**: pasar de
~33 k a ~48 k imágenes sin etiqueta es lo que llevó el empate a una victoria clara del SSL.

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

## Resultado (confirmado en v4)

Con **LP-FT + LLRD + más datos SSL**, el **SSL pasó a ganar**: macro-F1 0.6907 vs 0.6614 (Δ +0.0294,
~5× la std → fuera del ruido) y ventaja también en AUC y en el linear probe (representación más
separable). Las tres palancas tenían respaldo empírico y se cumplieron. El siguiente salto (superar
por márgenes grandes, y confirmar en validación externa/OOD) llega con **DINO/ViT + ~200 k imágenes**,
que es el plan con las GPUs de la UNAB.

## Referencias

- Kumar, A. et al. (2022). *Fine-tuning can distort pretrained features and underperform
  out-of-distribution.* ICLR. arXiv:2202.10054.
- Chen, X. & He, K. (2021). *Exploring Simple Siamese Representation Learning (SimSiam).* CVPR.
- Caron, M. et al. (2021). *Emerging Properties in Self-Supervised Vision Transformers (DINO).* ICCV.
- Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.*
- Estudios 2024-2025 SSL en dermatoscopía (HAM10000 / ISIC): DINO/DINOv2 > ImageNet en AUC;
  SSL con ~200 k imágenes ISIC sin etiquetar.
