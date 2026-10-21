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

## Cruces de fuentes para el pool semisupervisado v8 (2026-10-21)

Esta auditoría corresponde al candidato de segmentación semisupervisada v8. No reemplaza ni
redefine los splits históricos de clasificación descritos arriba.

- **HAM10000 no se concatena completo a ISIC 2019.** 9.929 de sus 10.015 IDs normalizados ya
  aparecen en el pool ISIC 2019 candidato. La comparación SHA exacta dio cero, pero el cruce por ID
  confirma que SHA exacto no basta como único criterio de deduplicación. Las 86 imágenes HAM
  restantes por ID se dejaron fuera del candidato.
  HAM aporta `lesion_id`, pero no `patient_id`.
- **ISIC2020 amplía el pool.** De 33.126 imágenes se eliminaron 433 filas excedentes por SHA exacto.
  Los 425 pares de la lista oficial coinciden con SHA locales; se observaron otros 8 pares exactos
  no listados por la fuente, cuyo origen sigue sin determinarse. El filtro pHash conservador
  (DCT 8×8, distancia de Hamming ≤4) excluyó una unión de 1.511 IDs ISIC2020 frente al pool
  ISIC2019, las 2.594 imágenes oficiales de evaluación ISIC2018, HAM residual y PH2. El pHash
  identifica candidatos similares, no confirma por sí solo que sean la misma imagen. La asignación
  exclusiva de esas exclusiones fue: 944 frente al pool ISIC2019, 424 adicionales frente a ISIC2018,
  143 adicionales frente a HAM residual y cero frente a PH2.
- El manifiesto candidato resultante contiene **53.756 imágenes**: 22.574 de ISIC2019 y 31.182 de
  ISIC2020. No incorpora `target`, diagnósticos ni `lesion_id`; conserva `patient_id` de ISIC2020.
  `patient_independence_accredited`, `historical_ssl_exposure_independence_verified`,
  `training_consumable` y `launch_ready` permanecen en `false`.
- Manifiesto en UNABIA:
  `~/work/dermlip_slic_v8/preflight/isic2019-isic2020-pool-candidate-20261021T162138Z/manifest.json`.
  SHA-256: `60775b56e0f636ebc2d861a20b429c2f01bc55e74232582f56c3ebdff5e480ee`.
  El recibo está en la misma carpeta; su SHA-256 es
  `178809f412552d367b1bcfea475f213b40da4d99dbec758225ff8b0d8894f2cf`. El registro de
  exclusiones `exclusions.jsonl` tiene SHA-256
  `002af9db83ce16e567a2b339f65a4ff06eea71e91dde155d492a4b9ba37e5cf4`.
- Para la clasificación, deduplicar HAM, ISIC2019 e ISIC2020 antes de combinarlos. Mantener los
  IDs de paciente/lesión disponibles para separar los conjuntos de entrenamiento y evaluación; el
  manifiesto v8 todavía no acredita independencia cruzada por paciente. Las 1.511 exclusiones
  pHash son específicas del pool de segmentación v8; no se deben copiar como filtro de clasificación
  sin volver a cruzarlas contra los splits y holdouts de esa tarea.

### Derm1M como posible expansión futura

El artículo de Derm1M describe 1.029.761 pares imagen-texto de 403.563 imágenes únicas, reunidos
desde videos, PubMed, foros, material educativo y conjuntos públicos. El repositorio informa un
paquete de 34,5 GB, acceso con aceptación de condiciones y licencia CC BY-NC 4.0. Su estructura y
experimentos corresponden a un corpus de visión-lenguaje usado para preentrenar DermLIP; no es un
benchmark de segmentación con mascarillas de lesión. Como el pipeline v8 ya usa un checkpoint
DermLIP preentrenado en Derm1M, agregar ese corpus ahora podría repetir exposición a imágenes ya
vistas por el backbone; esa posible exposición no está cuantificada.

**Recomendación vigente:** conservar ISIC2019 + el subconjunto depurado de ISIC2020 para la primera
evaluación semisupervisada. Considerar Derm1M en una ablation posterior solo si se define un filtro
de modalidad dermatoscópica, se comprueba la licencia/acceso y se auditan duplicados y exposición
contra los conjuntos de evaluación.

Fuentes: [ISIC Challenge Data](https://challenge.isic-archive.com/data/), [Derm1M paper](https://arxiv.org/abs/2503.14911), [Derm1M dataset card](https://huggingface.co/datasets/redlessone/Derm1M).

## BCN20000 y CO2Wounds-V2

- Usados en el EDA del Objetivo 1 (análisis exploratorio cruzado).
- BCN20000 es candidato para validación externa cross-dataset en el Objetivo 3.

## Higiene de datos (reglas)

1. Split **lesión-disjunto** (por `lesion_id`), verificado con `assert` de cero solapamiento.
2. Pesos de clase = raíz del inverso de frecuencia (suaviza el desbalance sin sobre-predecir raras).
3. Carga robusta de imágenes: una imagen truncada **no** debe tumbar el run
   (`LOAD_TRUNCATED_IMAGES = True` + placeholder ante error).
