# 20. Esquema de fst_manifest.csv (Fase 3, costura GPU/CPU)

Documenta el contrato de datos entre el U-Net que corre en Kaggle
(`src/dermssl/torch_unet.py`, GPU) y todo lo que se evalua en CPU
(`metrics.stratified_summary`, `gates.fairness_gate`,
`scripts/notebook_code_v5.py`). Es la "costura" D4 de `design.md`: nada
fuera de `scripts/kaggle_fst_manifest.py` importa torch, y nada dentro de
ese script decide fairness - solo escribe filas.

Este documento fija el ESQUEMA, no trae datos reales: `fst_manifest.csv`
nunca se commitea (son estimaciones sobre imagenes de pacientes, generadas
en una corrida Kaggle real con GPU).

## 1. Columnas

| Columna | Tipo | Rango / valores | Descripcion |
|---|---|---|---|
| `image_id` | string | - | Stem del archivo de imagen (sin extension), la misma clave que usan `data_resolver`/`va_ids` en el notebook v5 para unir por `image_id`. |
| `ita` | float | `[-90, 90]` o `NaN` | Individual Typology Angle en grados, mediana sobre pixeles de piel (`fitzpatrick.ita_degrees`). `NaN` cuando `status != "ok"`. |
| `fst_bucket` | int (1-6) o string | `1..6` o `"unavailable"` | Bucket de Fitzpatrick (Groh et al.) o marcador explicito cuando no hay estimacion. |
| `skin_fraction` | float | `[0, 1]` | Fraccion de pixeles clasificados como piel (no-lesion, no-borde) sobre el total de la imagen. |
| `mask_source` | string | `"unet"` \| `"otsu_fallback"` | Cual mascara uso realmente `estimate_fst`: la del U-Net (primaria, D3) o el fallback Otsu de `pretrain.lesion_guided` (cuando la del U-Net es degenerada: area fuera de `[MIN_MASK_AREA_FRACTION, MAX_MASK_AREA_FRACTION]`). |
| `mask_confidence` | float | `[0, 1]` | Certeza media de la salida sigmoide del U-Net respecto al umbral 0.5. |
| `fallback_used` | bool | `True` \| `False` | Indica que la mascara primaria falto, fue degenerada o quedo bajo el umbral de confianza y se uso Otsu. |
| `confidence` | float | `[0, 1]` | Confianza heuristica FST. Baja con fallback o artefactos; vale 0 si el bucket no esta disponible. No es probabilidad calibrada. |
| `artifact_detected` | bool | `True` \| `False` | Bandera conservadora para lineas oscuras de alto contraste o borde oscuro compatibles con pelo, reglas o tinta. No elimina la fila. |
| `status` | string | `"ok"` \| `"no_skin"` \| `"degenerate_mask"` \| `"ita_out_of_range"` | Copia literal de `FSTEstimate.status` (`fitzpatrick.py`). Solo `"ok"` trae un `fst_bucket` valido. |

## 2. Ejemplo (datos inventados, solo para ilustrar el formato)

```csv
image_id,ita,fst_bucket,skin_fraction,mask_source,mask_confidence,fallback_used,confidence,artifact_detected,status
ISIC_0024306,47.32,2,0.61,unet,0.91,False,0.91,False,ok
ISIC_0024307,nan,unavailable,0.02,otsu_fallback,0.22,True,0.0,False,no_skin
ISIC_0024308,12.05,4,0.55,otsu_fallback,0.31,True,0.42,True,ok
```

## 3. Como se genera (paso manual, no corre en CI)

`scripts/kaggle_fst_manifest.py` entrena el U-Net una vez sobre ISIC 2018
Task 1 y despues corre inferencia sobre el pool objetivo, escribiendo una
fila por imagen. Separa train/validacion de forma determinista, reporta BCE,
Dice e IoU por epoca y restaura el checkpoint con mejor Dice:

```bash
python scripts/kaggle_fst_manifest.py \
    --unet-train-images /kaggle/input/isic2018.../train/images \
    --unet-train-masks  /kaggle/input/isic2018.../train/masks \
    --target-image-root /kaggle/input/skin-cancer-mnist-ham10000/images \
    --output-csv fst_manifest.csv --checkpoint-out unet_best.pt \
    --epochs 20 --validation-fraction 0.20 --seed 42
```

Dataset de entrenamiento del U-Net (tarea 0.1, resuelta): Kaggle handle
`tschandl/isic2018-challenge-task1-data-segmentation` (mirror de
tschandl/HAM10000, 2594 pares JPEG+PNG, mismo Task 1 del challenge oficial
ISIC 2018). Fallback si el mirror se retira: archivo oficial de ISIC.

Requiere GPU real (Kaggle T4 o equivalente) y no corre en este entorno de
CI - ver `tests/test_torch_unet_static.py` (solo `py_compile`/`ast`) y la
fila "Deferred (GPU)" de la Testing Strategy en `design.md`.

## 4. Que hace el resto del pipeline con el archivo

`scripts/notebook_code_v5.py` busca `WORK / "fst_manifest.csv"`; si no
existe, el bloque de fairness no toca `gate_list` y la corrida v5 queda
byte-identica a como era antes de la Fase 2 (`design.md`, tabla "Error
Handling": "Manifest missing -> ... v5 behavior unchanged"). Si existe, se
une por `image_id`; buckets ausentes, invalidos o marcados `unavailable` se
conservan en el conteo separado. El macro-F1 estratificado se calcula con el
probe lineal balanceado `logreg_balanced`, no con k-NN, y alimenta
`metrics.stratified_summary` + `gates.fairness_gate`.
