# 20. Esquema de fst_manifest.csv (Fase 3, costura GPU/CPU)

Documenta el contrato de datos entre el U-Net que corre en Kaggle
(`src/dermssl/torch_unet.py`, GPU) y todo lo que se evalua en CPU
(`metrics.stratified_summary`, `gates.fairness_gate`,
`scripts/notebook_code_v5.py`). Es la costura entre GPU y CPU: nada
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
| `fst_bucket` | int (1-6) o vacio | `1..6` | Bucket de Fitzpatrick (Groh et al.), o celda vacia cuando no hay estimacion (`status != "ok"`). Vacio se lee como `NaN` con `pandas.read_csv` - asi lo filtra `notebook_code_v5.py` (`pd.isna(...)`). |
| `skin_fraction` | float | `[0, 1]` | Fraccion de pixeles clasificados como piel (no-lesion, no-borde) sobre el total de la imagen. |
| `mask_source` | string | `"unet"` \| `"otsu_fallback"` | Cual mascara uso realmente `estimate_fst`: la del U-Net (primaria) o el fallback Otsu de `pretrain.lesion_guided` (cuando la del U-Net es degenerada: area fuera de `[MIN_MASK_AREA_FRACTION, MAX_MASK_AREA_FRACTION]`). |
| `status` | string | `"ok"` \| `"no_skin"` \| `"degenerate_mask"` \| `"ita_out_of_range"` | Copia literal de `FSTEstimate.status` (`fitzpatrick.py`). Solo `"ok"` trae un `fst_bucket` valido. |

## 2. Ejemplo (datos inventados, solo para ilustrar el formato)

```csv
image_id,ita,fst_bucket,skin_fraction,mask_source,status
ISIC_0024306,47.32,2,0.61,unet,ok
ISIC_0024307,,,0.02,otsu_fallback,no_skin
ISIC_0024308,12.05,4,0.55,otsu_fallback,ok
ISIC_0024309,,,,otsu_fallback,degenerate_mask
```

## 3. Como se genera (paso manual, no corre en CI)

`scripts/kaggle_fst_manifest.py` entrena el U-Net una vez sobre ISIC 2018
Task 1 y despues corre inferencia sobre el pool objetivo, escribiendo una
fila por imagen:

```bash
python scripts/kaggle_fst_manifest.py \
    --unet-train-images /kaggle/input/isic2018.../train/images \
    --unet-train-masks  /kaggle/input/isic2018.../train/masks \
    --target-image-root /kaggle/input/skin-cancer-mnist-ham10000/images \
    --output-csv fst_manifest.csv --epochs 20
```

Dataset de entrenamiento del U-Net: Kaggle handle
`tschandl/isic2018-challenge-task1-data-segmentation` (mirror de
tschandl/HAM10000, 2594 pares JPEG+PNG, mismo Task 1 del challenge oficial
ISIC 2018). Fallback si el mirror se retira: archivo oficial de ISIC.

Requiere GPU real (Kaggle T4 o equivalente) y no corre en este entorno de
CI - ver `tests/test_torch_unet_static.py`, que solo verifica que el
codigo compila y su AST (sin entrenar ni hacer inferencia real).

## 4. Que hace el resto del pipeline con el archivo

`scripts/notebook_code_v5.py` busca `WORK / "fst_manifest.csv"`; si no
existe, el bloque de fairness no toca `gate_list` y la corrida v5 queda
byte-identica a como era antes de agregar la equidad por fototipo. Si existe, se
une por `image_id`, se descartan las filas con `fst_bucket` vacio
(`pd.isna`), y el resto alimenta `metrics.stratified_summary` +
`gates.fairness_gate` ("G3 equidad por fototipo").
