# 20. Validación externa del estimador FST y calibración del umbral ITA

Cierra la Fase 1 (FST) del cambio `openspec/changes/fst-domain-adaptation-circle`: la
"calibración provisional con Fitzpatrick17k" que el propio `proposal.md` prometía pero
que nunca se había ejecutado. Este documento reporta esa validación, el bug de
calibración que apareció en el camino, y qué se puede y no se puede concluir de los
números.

## 1. Qué se corrió

Todo en `scripts/diagnostico_fst_comparativa.py` (CPU, sin GPU, pura numpy/PIL) y
`scripts/validar_fitzpatrick17k.py`, sobre el servidor propio de la UNAB:

- Diagnóstico A/B (`estimate_fst_whole_image` vs `estimate_fst` con fallback Otsu, el
  U-Net real de la Fase 2 todavía no corrió sobre datos reales) sobre los tres datasets
  del pool: HAM10000 (10.015), ISIC2019 redimensionado (25.331) e ISIC2020 (33.126) —
  68.472 imágenes en total.
- La misma comparación sobre ISIC2019 **full-res** (25.331), para aislar si el resize
  explicaba la divergencia entre las dos estimaciones.
- Validación externa contra **Fitzpatrick17k-C**, la versión limpia y con splits
  estandarizados publicada por Abhishek, Jain & Hamarneh ("Investigating the Quality of
  DermaMNIST and Fitzpatrick17k Dermatological Image Datasets", *Scientific Data* 2025,
  Zenodo 10.5281/zenodo.11101337) — 11.098 imágenes con FST de experto válido (1-6) de
  las 11.394 totales, 296 excluidas por FST desconocido (`fst=0`). Se usó la versión
  **-C** (deduplicada, sin conflictos de etiqueta) en vez de un mirror crudo de Kaggle,
  porque el propio paper reporta 803 pares de imágenes casi idénticas con FST distinto
  en la versión sin depurar — no era un ground truth confiable tal cual.

## 2. El bug de calibración que apareció, y cómo se detectó

`fitzpatrick.py` declaraba `FST_THRESHOLDS = (55.0, 41.0, 28.0, 19.0, 10.0)` con un
comentario que decía *"son los cortes empíricos de Groh et al."*. Es falso: esos son los
cortes **clásicos** de Chardon et al. 1991 — la Tabla 1 de Corbin & Marques 2023 (el
paper del que sale CIRCLe en este proyecto) los reproduce exactos. Los cortes empíricos
reales de Groh et al. son otros: `(40.0, 23.0, 12.0, 0.0, -25.0)` (Tabla 2 del mismo
paper).

Se detectó comparando ambas tablas contra las citadas en Corbin & Marques al investigar
por qué la validación contra Fitzpatrick17k-C daba tan mal (ver sección 3). El propio
paper reporta que, en su experimento, cambiar de la tabla clásica a la empírica sube el
accuracy off-by-one de 45.9%→60.3% (imagen completa) y 53.3%→70.4% (con máscara YCbCr) —
una mejora demasiado grande para no revisar cuál tabla estábamos usando de verdad.

Corregido en `src/dermssl/fitzpatrick.py` (commit `b651efe`), con el test
`test_fst_bucket_at_groh_thresholds_and_boundaries` actualizado a los umbrales reales y
el notebook v5 regenerado. 559/559 tests pasan tras el cambio.

## 3. Qué se sostiene

**Antes de corregir el umbral** (Chardon 1991, mal etiquetado como Groh):

| Método | Exacta | Off-by-one |
|---|---|---|
| Sin máscara | 22.8% | 52.4% |
| Con máscara (Otsu) | 29.8% | 65.5% (39 imágenes sin resolver) |

**Después de corregir el umbral** (Groh empírico real):

| Método | Exacta | Off-by-one |
|---|---|---|
| Sin máscara | **30.8%** | **66.4%** |
| Con máscara (Otsu) | 28.5% | 65.6% |

Línea base trivial (adivinar siempre el bucket más común, bucket 2, 32.9% de las 11.098
imágenes): **32.9% exacta, 73.5% off-by-one**.

- **La corrección de umbral funciona, pero de forma asimétrica**: mejora sustancialmente
  la versión sin máscara (+8 puntos exacta, +14 off-by-one) y deja la versión con
  máscara prácticamente sin cambios. El efecto neto es que las dos versiones, que antes
  mostraban una brecha grande a favor de enmascarar, ahora rinden casi igual. Buena
  parte de "enmascarar ayuda mucho" que se reportó en la sesión de diagnóstico
  (18.2%-27.5% de cambio de bucket entre las dos corridas, ver `docs/*` de esa sesión en
  Engram) era en parte un artefacto de estar comparando contra el umbral equivocado, no
  únicamente un efecto real del enmascarado.
- **El resize de ISIC2019 no explica la divergencia observada.** ISIC2019 resized dio
  27.5% de cambio de bucket entre las dos corridas; ISIC2019 full-res dio 24.8% — una
  diferencia de ~2.7 puntos, no el salto grande (20.1%→34.0%) que había quedado anotado
  de una sesión anterior de Kaggle. Con n=25.331 en las dos corridas, el resize no es el
  sospechoso principal.
- **El resultado replica un hallazgo publicado, casi al punto porcentual.** Alipour,
  Burke & Courtney ("Limitations of ITA for Skin Type Estimation Under Uncontrolled
  Imaging Conditions", *Skin Research and Technology*, 2026) reportan 22.9% de accuracy
  exacta con umbral fijo bajo iluminación no controlada — nuestro 22.8% (antes de
  corregir el umbral) es prácticamente idéntico. Ellos también reportan que FST VI (el
  tono más oscuro) recibe cero clasificaciones correctas en su matriz de confusión; no
  se revisó todavía si a nosotros nos pasa lo mismo con el bucket 6 (pendiente).

## 4. Qué NO se puede afirmar

- **Que el estimador sea clínicamente válido.** Nunca fue el objetivo; el propio
  `design.md` lo llama "calibración provisional", y esta validación lo confirma: ni
  antes ni después del fix se supera de forma consistente la línea base trivial.
- **Que este resultado represente el comportamiento del estimador en su dominio real de
  uso.** Fitzpatrick17k (fotos clínicas de dos atlas web, DermaAmin y Atlas
  Dermatologico) y el dataset de Alipour et al. (PAD-UFES-20, igual de "no controlado")
  son fotos clínicas variadas — no dermatoscópicas. HAM10000/ISIC, el dominio real donde
  se va a usar el estimador, tiene iluminación estandarizada por el dermatoscopio. Esta
  validación estresa el escenario más adverso posible para ITA, no necesariamente el
  escenario real de uso.
- **Que la recalibración de umbral sea suficiente por sí sola.** Alipour et al. 2026 lo
  dicen explícito sobre su propio experimento: *"current results do not demonstrate that
  such modifications [threshold recalibration] would be sufficient"* para lograr una
  clasificación de FST confiable. La mejora que medimos es real y vale la pena haberla
  hecho, pero no cierra la pregunta de fondo.
- **Que la Fase 2 (máscaras) o la Fase 3 (adaptación SSL) estén realmente completas.**
  Ambas están marcadas `[x]` en `tasks.md`, pero durante esta sesión se encontró que:
  (a) el gap de confianza del U-Net en `mask_manifest.py` nunca afectaba la elegibilidad
  de una máscara (corregido, commit `3617ace`, sin impacto retroactivo porque ese camino
  nunca corrió sobre datos reales); y (b) el pre-entrenamiento continual de DINOv2 (v6,
  la adaptación de dominio real) nunca se ejecutó en ninguna rama ni infraestructura —
  confirmado revisando `origin/feature/v6-gpu-ssl-pretraining`, que solo tiene el plan
  (`docs/16`) y diagramas de arquitectura, cero checkpoints ni resultados.

## 5. Una hipótesis que el dato descartó: constancia de color antes del ITA

Alipour et al. 2026 sugieren en su trabajo futuro que algoritmos de constancia de color
podrían mitigar la sensibilidad de ITA a iluminación no controlada. El repo ya tenía
`gray_world` y `shades_of_gray` implementados en `pretrain/color_constancy.py` (pensados
como augmentation SSL, no para esto) y se agregó un flag `--color-constancy` a
`validar_fitzpatrick17k.py` para probarlo directo (commit `008fcf1`).

Un smoke de 20 imágenes bastó para descartarlo: el accuracy se derrumbó a 5-10% con
las dos variantes (contra ~23-30% sin corrección), consistente en ambos métodos y
ambos algoritmos — no es ruido de muestra chica, es un fallo sistemático. La hipótesis
más probable: `gray_world` asume que la escena promedia a gris, un supuesto que una
foto dominada por piel (que es rojiza/anaranjada, no gris) viola por diseño. En vez de
corregir el sesgo de iluminación, la corrección borra la información de tono de piel
que se está intentando medir. No se corrió la validación completa sobre esta vía — el
smoke ya fue suficientemente concluyente para no gastar el cómputo.

**Aprendizaje transferible**: la constancia de color de tipo gray-world no es un
preprocesamiento neutral para tareas donde el color del sujeto ES la señal (como
estimar tono de piel) — solo tiene sentido cuando el color es ruido a remover, no la
variable de interés.

## 6. El fallback de Otsu confunde piel oscura con piel clara, no solo con tonos vecinos

Alipour et al. 2026 reportan cero aciertos en FST VI (el tono más oscuro) con su método
de umbral fijo. Revisamos la fila del bucket 6 en nuestra matriz de confusión
(`validacion_fitzpatrick17k_v2.json`, 373 imágenes con FST 6 real):

| Método | Aciertos (bucket 6 → 6) | Recall |
|---|---|---|
| Sin máscara | 199/373 | 53.4% |
| Con máscara (Otsu) | 111/371 | 29.9% |

No repetimos su hallazgo más extremo — ninguna de las dos versiones da cero. Pero
apareció un problema propio, más preocupante que un simple recall bajo: **con máscara,
145 de 371 imágenes de piel realmente oscura (39%) se clasifican como bucket 1, la piel
MÁS CLARA de la escala** — no un error de un tono, el error más grande posible. Hipótesis
sin confirmar: `BORDER_LUMINANCE_THRESHOLD` en `fitzpatrick.py`, usado para separar
"borde/fondo" de "piel" en el fallback de Otsu, probablemente está calibrado pensando en
piel más clara; sobre piel oscura real podría estar excluyendo piel válida y quedándose
con parches de brillo o artefacto, produciendo una lectura de ITA falsamente clara.

Esto es exactamente el tipo de sesgo contra piel oscura que motiva todo este proyecto
(CIRCLe), encontrado en la **herramienta de etiquetado** antes de llegar siquiera a
CIRCLe. Es la razón principal para priorizar una máscara de lesión real (DoubleU-Net,
sección 7) sobre el fallback de Otsu para HAM10000/ISIC — el problema aparece en los
propios datos de esta validación, no es exclusivo de Fitzpatrick17k.

## 7. Consecuencia para el proyecto

La calibración de umbral era gratis y valía la pena — ya está corregida y commiteada. No
hay evidencia de que seguir ajustando el umbral por sí solo vaya a cerrar la brecha
contra la línea base trivial; la literatura (Alipour et al. 2026) sugiere que el techo
de un ITA de umbral fijo es más o menos el que estamos viendo. Antes de darle este FST a
CIRCLe como pseudo-label, dos caminos siguen abiertos y sin resolver:

1. Correr el U-Net real de la Fase 2 sobre datos reales (todavía no pasó) y repetir esta
   misma validación — el fallback de Otsu puede no ser representativo de lo que el
   pipeline final va a usar.
2. Si el resultado con el U-Net real tampoco supera la línea base, ponderar por
   confianza en vez de tratar cada bucket de FST como verdad absoluta al entrenar
   CIRCLe — el gap de `mask_confidence` corregido en este documento (sección 4) es
   justamente la pieza de infraestructura que hace falta para eso.

La Fase 3 (adaptación de dominio vía pre-entrenamiento continual de DINOv2) sigue
pendiente de ejecución real, en cualquier infraestructura, y es una tarea separada de
varias horas de GPU — no bloquea ni depende de lo anterior.
