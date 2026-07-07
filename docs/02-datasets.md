# 02 — Datasets

## Resumen

| Dataset | Uso en el proyecto | Notas |
|---|---|---|
| **HAM10000** | Núcleo: entrenamiento + evaluación supervisada (7 clases) | 10.015 imgs, dermatoscopia, piel clara |
| **ISIC 2019** | Pool SSL (imágenes **sin etiqueta** para domain-adaptive) | ~25k imgs, resized 224×224 |
| **BCN20000** | Referencia / validación externa potencial | Dermatoscopia |
| **CO2Wounds-V2** | Referencia (heridas), usado en EDA | Contexto regional |

## HAM10000 (referencia principal)

- 10.015 imágenes dermatoscópicas, **7 clases** (ver [01](01-contexto-y-objetivos.md)).
- Fuente Kaggle: `kmader/skin-cancer-mnist-ham10000`.
- **Punto crítico — `lesion_id`:** HAM contiene **varias fotos de la misma lesión**. Si se parte el
  dataset por `image_id`, la misma lesión cae en train y val → **data leakage** que infla las
  métricas. Se debe partir por `lesion_id` (ver [06 — Problemáticas](06-problematicas.md)).
- Desbalance fuerte (58×) → macro-F1 como métrica principal.
- Artefactos conocidos: pelos, reglas, marcadores → riesgo de *shortcut learning*
  (el modelo "hace trampa" mirando el artefacto en vez de la lesión).

## ISIC 2019 (pool SSL)

- Se usa **sin etiquetas** para el pre-entrenamiento SSL domain-adaptive.
- Fuente Kaggle: dataset "ISIC 2019 JPG 224x224 RESIZED" (`nischaydnk`).
- En la corrida actual el pool SSL = ISIC (~25k) + solo las lesiones de **train** de HAM
  (≈ 33.332 imágenes en total). El split de val de HAM **queda fuera** del SSL para no contaminar.
- Si además existe un `*GroundTruth*.csv` de ISIC adjunto, el pipeline lo usa para **validación
  externa** (mapeo ISIC → 7 clases HAM).

## BCN20000 y CO2Wounds-V2

- Usados en el EDA del Objetivo 1 (análisis exploratorio cruzado).
- BCN20000 es candidato para validación externa cross-dataset en el Objetivo 3.

## Higiene de datos (reglas)

1. Split **lesión-disjunto** (por `lesion_id`), verificado con `assert` de cero solapamiento.
2. Pesos de clase = raíz del inverso de frecuencia (suaviza el desbalance sin sobre-predecir raras).
3. Carga robusta de imágenes: una imagen truncada **no** debe tumbar el run
   (`LOAD_TRUNCATED_IMAGES = True` + placeholder ante error).
