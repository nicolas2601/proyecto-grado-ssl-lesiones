# Clasificador: v6, C1 y v7 en una tabla

Generado por `scripts/build_classifier_table.py` desde JSON y NPZ commiteados. No editar a mano: `--check` lo compara en CI.

## Resumen por versión

ECE: menor es mejor; en Macro-F1, Bal-acc y AUC, mayor es mejor. Entre corchetes, IC 95 % bootstrap; «—» = la fuente no trae IC por brazo.

Los protocolos no son idénticos y las filas no se comparan entre versiones sin leerlos: v6 = banco con probe `logreg`, fracción 1,0, semilla 0, val n = 2004; C1 = condición `none`, n = 2004; v7 = probe `logreg`, val n = 2004 (s42) y 2004 (s43).

La accuracy sola engaña con este desbalance: `nv` es 1341 de 2004 imágenes de val, así que predecir siempre `nv` ya da 0,6692. La métrica primaria sigue siendo macro-F1. Accuracy de v7: argmax de las probabilidades, con el mismo bootstrap que la sensibilidad de melanoma (abajo).

| Versión | Backbone/brazo | Macro-F1 | Bal-acc | AUC | ECE | Accuracy | Fuente |
|---|---|---|---|---|---|---|---|
| v6 | dinov2_public_vits14 | 0,5829 — | 0,5603 — | 0,9322 — | 0,0811 — | 0,7909 — | `resultados/clasificacion/bank.json` |
| v6 | dinov2_public_vitb14 | 0,6158 — | 0,5924 — | 0,9257 — | 0,1307 — | 0,7819 — | `resultados/clasificacion/bank.json` |
| v6 | ssl_plain_vits14 | 0,5307 — | 0,4766 — | 0,9306 — | 0,0207 — | 0,7834 — | `resultados/clasificacion/bank.json` |
| v6 | ssl_plain_vitb14 | 0,5856 — | 0,5429 — | 0,9447 — | 0,0161 — | 0,7989 — | `resultados/clasificacion/bank.json` |
| v6 | ssl_proposed_vits14 | 0,4840 — | 0,4362 — | 0,9212 — | 0,0382 — | 0,7754 — | `resultados/clasificacion/bank.json` |
| v6 | ssl_proposed_vitb14 | 0,5237 — | 0,4774 — | 0,9434 — | 0,0298 — | 0,7914 — | `resultados/clasificacion/bank.json` |
| C1 | dinov2_public_vits14 | 0,5813 — | 0,5601 — | 0,9324 — | 0,0822 — | 0,7899 — | `resultados/clasificacion/c1/c1_cells.json` |
| C1 | dinov2_public_vitb14 | 0,6137 — | 0,5902 — | 0,9259 — | 0,1306 — | 0,7819 — | `resultados/clasificacion/c1/c1_cells.json` |
| C1 | ssl_plain_vits14 | 0,5333 — | 0,4794 — | 0,9307 — | 0,0213 — | 0,7844 — | `resultados/clasificacion/c1/c1_cells.json` |
| C1 | ssl_plain_vitb14 | 0,5784 — | 0,5345 — | 0,9454 — | 0,0159 — | 0,7974 — | `resultados/clasificacion/c1/c1_cells.json` |
| C1 | ssl_proposed_vits14 | 0,4823 — | 0,4342 — | 0,9209 — | 0,0382 — | 0,7750 — | `resultados/clasificacion/c1/c1_cells.json` |
| C1 | ssl_proposed_vitb14 | 0,5261 — | 0,4803 — | 0,9436 — | 0,0268 — | 0,7934 — | `resultados/clasificacion/c1/c1_cells.json` |
| v7 s42 | A* | 0,6464 [0,5980; 0,6894] | 0,6021 [0,5565; 0,6488] | 0,9619 [0,9563; 0,9671] | 0,0135 [0,0138; 0,0336] | 0,8318 [0,8154; 0,8468] | `resultados/clasificacion/v7/final_eval.json` |
| v7 s43 | A* | 0,6850 [0,6353; 0,7258] | 0,6457 [0,5968; 0,6942] | 0,9662 [0,9603; 0,9711] | 0,0202 [0,0169; 0,0377] | 0,8443 [0,8293; 0,8603] | `resultados/clasificacion/v7/s43/final_eval.json` |
| media s42/s43 | A* | 0,6657 — | 0,6239 — | 0,9641 — | 0,0169 — | 0,8381 — | `resultados/clasificacion/v7/final_eval.json`, `resultados/clasificacion/v7/s43/final_eval.json` |
| v7 s42 | B* | 0,6606 [0,6157; 0,7040] | 0,6327 [0,5871; 0,6838] | 0,9629 [0,9563; 0,9690] | 0,0325 [0,0253; 0,0497] | 0,8313 [0,8149; 0,8478] | `resultados/clasificacion/v7/final_eval.json` |
| v7 s43 | B* | 0,6638 [0,6152; 0,7052] | 0,6119 [0,5657; 0,6607] | 0,9623 [0,9563; 0,9677] | 0,0178 [0,0166; 0,0377] | 0,8308 [0,8149; 0,8468] | `resultados/clasificacion/v7/s43/final_eval.json` |
| media s42/s43 | B* | 0,6622 — | 0,6223 — | 0,9626 — | 0,0252 — | 0,8311 — | `resultados/clasificacion/v7/final_eval.json`, `resultados/clasificacion/v7/s43/final_eval.json` |
| v7 s42 = s43 | DINOv2 público | 0,6158 [0,5690; 0,6550] | 0,5924 [0,5459; 0,6376] | 0,9257 [0,9123; 0,9370] | 0,1307 [0,1159; 0,1476] | 0,7819 [0,7655; 0,7994] | `resultados/clasificacion/v7/final_eval.json`, `resultados/clasificacion/v7/s43/final_eval.json` |
| referencia | siempre nv | — | — | — | — | 0,6692 — | `y_true` de `resultados/clasificacion/v7/final_eval.probas.npz` |

## Por clase (v7, desde las probabilidades guardadas)

Predicción = argmax de `final_eval.probas.npz`. Orden de clases de `bank.json` (`split.classes`). El macro-F1 recalculado desde las probabilidades coincide con `final_eval.json` (tolerancia 1e-09); si no, este script falla.

IC 95 % de la sensibilidad y del AUC de melanoma: bootstrap percentil sobre imágenes, 1000 remuestras con reemplazo, semilla 20260923 (mismo esquema de remuestreo que `src/dermssl/paired_bootstrap.py`), por brazo y semilla, desde las probabilidades commiteadas.

| Brazo | Sensibilidad melanoma [IC 95 %] | AUC melanoma, uno contra el resto [IC 95 %] | Macro-F1 |
|---|---|---|---|
| A* s42 | 0,4865 [0,4235; 0,5490] | 0,8985 [0,8784; 0,9168] | 0,6464 |
| A* s43 | 0,4775 [0,4087; 0,5354] | 0,9111 [0,8925; 0,9279] | 0,6850 |
| B* s42 | 0,5045 [0,4422; 0,5683] | 0,9100 [0,8917; 0,9261] | 0,6606 |
| B* s43 | 0,4505 [0,3871; 0,5089] | 0,8985 [0,8779; 0,9162] | 0,6638 |
| DINOv2 público s42 = s43 | 0,4279 [0,3632; 0,4978] | 0,8331 [0,8054; 0,8603] | 0,6158 |

### F1 por clase

| Brazo | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| A* s42 | 0,6290 | 0,7071 | 0,6727 | 0,4444 | 0,5524 | 0,9235 | 0,5957 |
| A* s43 | 0,6116 | 0,7488 | 0,7239 | 0,5854 | 0,5594 | 0,9263 | 0,6400 |
| B* s42 | 0,5625 | 0,7327 | 0,6848 | 0,5116 | 0,5558 | 0,9234 | 0,6531 |
| B* s43 | 0,5833 | 0,7236 | 0,6975 | 0,6111 | 0,5141 | 0,9210 | 0,5957 |
| DINOv2 público s42 = s43 | 0,4853 | 0,6377 | 0,5919 | 0,5854 | 0,4471 | 0,8969 | 0,6667 |

### Recall por clase

| Brazo | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| A* s42 | 0,5909 | 0,6796 | 0,6727 | 0,3478 | 0,4865 | 0,9545 | 0,4828 |
| A* s43 | 0,5606 | 0,7379 | 0,7091 | 0,5217 | 0,4775 | 0,9612 | 0,5517 |
| B* s42 | 0,5455 | 0,7184 | 0,6864 | 0,4783 | 0,5045 | 0,9441 | 0,5517 |
| B* s43 | 0,5303 | 0,6990 | 0,6864 | 0,4783 | 0,4505 | 0,9560 | 0,4828 |
| DINOv2 público s42 = s43 | 0,5000 | 0,6408 | 0,6000 | 0,5217 | 0,4279 | 0,9045 | 0,5517 |

### Matrices de confusión (filas: clase real; columnas: predicha)

#### A* s42

| real \ predicha | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---:|---:|---:|---:|---:|---:|---:|
| akiec | 39 | 7 | 12 | 0 | 3 | 5 | 0 |
| bcc | 7 | 70 | 8 | 1 | 5 | 12 | 0 |
| bkl | 8 | 7 | 148 | 0 | 20 | 37 | 0 |
| df | 1 | 0 | 6 | 8 | 3 | 4 | 1 |
| mel | 2 | 4 | 21 | 0 | 108 | 86 | 1 |
| nv | 1 | 4 | 25 | 2 | 27 | 1280 | 2 |
| vasc | 0 | 3 | 0 | 2 | 3 | 7 | 14 |

#### A* s43

| real \ predicha | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---:|---:|---:|---:|---:|---:|---:|
| akiec | 37 | 8 | 8 | 0 | 6 | 7 | 0 |
| bcc | 8 | 76 | 3 | 2 | 4 | 10 | 0 |
| bkl | 7 | 7 | 156 | 1 | 9 | 40 | 0 |
| df | 0 | 1 | 5 | 12 | 3 | 2 | 0 |
| mel | 1 | 4 | 21 | 0 | 106 | 90 | 0 |
| nv | 2 | 1 | 17 | 2 | 25 | 1289 | 5 |
| vasc | 0 | 3 | 1 | 1 | 4 | 4 | 16 |

#### B* s42

| real \ predicha | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---:|---:|---:|---:|---:|---:|---:|
| akiec | 36 | 9 | 11 | 1 | 3 | 6 | 0 |
| bcc | 7 | 74 | 4 | 4 | 5 | 9 | 0 |
| bkl | 13 | 3 | 151 | 0 | 18 | 35 | 0 |
| df | 1 | 0 | 3 | 11 | 3 | 5 | 0 |
| mel | 3 | 6 | 25 | 0 | 112 | 75 | 1 |
| nv | 2 | 4 | 27 | 2 | 37 | 1266 | 3 |
| vasc | 0 | 3 | 0 | 2 | 3 | 5 | 16 |

#### B* s43

| real \ predicha | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---:|---:|---:|---:|---:|---:|---:|
| akiec | 35 | 8 | 9 | 0 | 8 | 6 | 0 |
| bcc | 7 | 72 | 7 | 1 | 7 | 9 | 0 |
| bkl | 9 | 6 | 151 | 0 | 18 | 36 | 0 |
| df | 1 | 0 | 4 | 11 | 3 | 4 | 0 |
| mel | 1 | 3 | 20 | 0 | 100 | 98 | 0 |
| nv | 1 | 4 | 21 | 0 | 29 | 1282 | 4 |
| vasc | 0 | 3 | 1 | 1 | 2 | 8 | 14 |

#### DINOv2 público s42 = s43

| real \ predicha | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---:|---:|---:|---:|---:|---:|---:|
| akiec | 33 | 4 | 20 | 1 | 3 | 5 | 0 |
| bcc | 9 | 66 | 7 | 2 | 3 | 14 | 2 |
| bkl | 16 | 14 | 132 | 0 | 23 | 35 | 0 |
| df | 0 | 1 | 4 | 12 | 2 | 4 | 0 |
| mel | 3 | 8 | 29 | 0 | 95 | 87 | 0 |
| nv | 9 | 8 | 33 | 3 | 74 | 1213 | 1 |
| vasc | 0 | 3 | 1 | 0 | 3 | 6 | 16 |
