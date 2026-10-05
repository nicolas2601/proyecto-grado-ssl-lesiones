# Banco completo: ablación de nuestros métodos

Todos los números son de la partición de val de cada dataset y salen de un JSON del repositorio. F1-F5 y F4 son exploratorios: no cambian el propuesto (B71 sin segmentación). D0-D3 esperan a E2.

Generado por `scripts/build_bench_ablation.py` desde los JSON del repositorio. No editar a mano. Plan y reglas: `docs/33-banco-completo-multidataset.md`.

## Glosario

- **Accuracy (exactitud):** porcentaje total de aciertos. Engaña con desbalance: si una clase domina, decir siempre esa clase ya da un número alto (ver abajo).
- **Recall (sensibilidad):** de las lesiones que de verdad son de una clase, qué fracción detecta el modelo. **Recall macro** = el promedio de las clases (= exactitud balanceada).
- **Precisión:** de las veces que el modelo dice «esta clase», qué fracción acierta. **Precisión macro** = el promedio de las clases.
- **F1-score:** media armónica de precisión y recall; solo es alto si los dos lo son. **F1 macro** = el F1 de cada clase promediado sin pesar por tamaño. Es la **métrica principal** y el criterio de orden.
- **AUC (OvR macro):** probabilidad de que el modelo le dé más puntaje de una clase a una imagen de esa clase que a una que no lo es (promedio de las clases). 0,5 = azar, 1 = perfecto. No depende de umbral.
- **ECE (↓):** cuánto se aleja la confianza del modelo de su acierto real. Menor es mejor.
- **MCC (coeficiente de correlación de Matthews):** resume toda la matriz de confusión en un número de -1 a 1 (0 = azar, 1 = perfecto). Para los baselines se calcula exacto desde `matriz_confusion`.
- **IC 95 %:** intervalo de confianza por bootstrap, tal como lo publica cada fuente. Si los IC de dos modelos se solapan, no se puede afirmar que uno gane: es empate.
- **Congelado (sin fine-tuning):** la red no se toca; solo se entrena el clasificador lineal de encima (regresión logística).
- **Fine-tuning:** se reentrena toda la red con las etiquetas del dataset.
- **Segmentación:** uso de máscaras de lesión como entrada. La columna dice «no» en todas las filas salvo en el bloque exploratorio de HAM10000 (variante F4 del bloque E3 de docs/33), que no cambia el propuesto.
- **Transductivo:** el pool de preentrenamiento auto-supervisado contenía las imágenes de val (sin etiquetas). Solo aplica a B71/A71 originales en ISIC 2019 e ISIC 2020.
- **Cómputo canónico:** extracción de features con batch 64 y autocast CUDA (el de v7.1). El batch 16 mueve las cifras unas milésimas (orden de reducción en fp16) y se reporta solo como **sensibilidad**.
- **Pesos limpios (sensibilidad):** B71/A71 re-preentrenados con un pool SSL sin las imágenes de val de ISIC (docs/33, sección 2). En HAM10000 son un análisis posterior que se lee al lado de la fila registrada de **pesos originales (selección)**; no es selección y no reemplaza al propuesto.
- **Contraste pareado:** diferencia entre dos modelos sobre las mismas imágenes, con IC bootstrap de la diferencia. «Mejor» solo si el IC excluye 0; si lo incluye, «sin diferencia demostrable».
- **Exploratorio:** variantes post-hoc o con selección en otro conjunto; se reportan pero no cambian el propuesto ni se leen como resultado confirmado.

## Cómo leer este documento

Particiones de val de este banco (no son la misma partición):

- HAM10000, baselines (Kaggle): val n = 2.014; StratifiedGroupKFold por lesion_id (0 lesiones compartidas).
- HAM10000, propuesto (v6/v7): val n = 2.004; partición agrupada por `lesion_id`.
- ISIC 2019, baselines (Kaggle): val n = 5.067; StratifiedKFold estratificado simple SIN grupo de paciente (el release no trae patient_id) - riesgo de fuga entre imagenes del mismo paciente, metricas con sesgo optimista.
- ISIC 2020, baselines (Kaggle): val n = 6.660; StratifiedGroupKFold por patient_id (0 pacientes compartidos).

- **Entre particiones distintas la comparación es solo de intervalos (IC 95 %)**, no un test pareado: HAM10000 usa la partición v6/v7 (n = 2.004) para el propuesto y la de Kaggle (n = 2.014) para los baselines; ISIC 2019 e ISIC 2020 tienen la suya. Los contrastes pareados solo existen donde dos modelos comparten partición e imágenes y hay probabilidades por imagen (docs/33, sección 5); están al final de cada tabla y, todos juntos, en `CONTRASTES_SSL.md` (sin corrección por multiplicidad).
- **Optimista: época elegida en val.** Todas las filas de fine-tuning (HAM10000, ISIC 2019 e ISIC 2020) eligen su mejor época por balanced accuracy sobre val; el plan decía holdout interno y se mantuvo la receta existente por consistencia entre datasets (errata de docs/33). Las de ISIC 2020 corrieron en nuestro pod (GPU RTX PRO 4000), no en el banco de Karen. El propuesto no tiene fine-tuning y no se ve afectado.
- **Modelo y protocolo no son lo mismo.** El propuesto usa el protocolo de B71 (features cls4, TTA, probe balanceado); la fila «DINOv2 público, protocolo de B71» es el control con el mismo protocolo y pesos públicos. Los 8 baselines siguen el protocolo de su propio banco (probe logreg).
- Los IC solo existen donde la fuente los publica: los baselines, para exactitud balanceada y F1 macro; B71/A71, para accuracy, exactitud balanceada, F1 macro, AUC y ECE. Precisión macro, recall macro y MCC de los baselines salen exactos de la matriz de confusión y van sin IC; en B71/A71 salen de las probabilidades commiteadas, también sin IC.
- **Pesos limpios en HAM10000: sensibilidad posterior, no es selección.** La fila registrada del propuesto (pesos originales, selección) no se reemplaza. Las filas del bloque «pesos limpios (sensibilidad)» usan la configuración y los umbrales de operación del holdout original, ajustados con los pesos originales y aplicados sin cambios a los limpios (supuesto declarado), con cómputo canónico; sus contrastes pareados (por `lesion_id`, con MCC) están en `CONTRASTES_SSL.md` y `TABLA_ABLACION.md`.
- Las filas «pendiente» no tienen número: se listan con su motivo técnico (sin ETA inventada), para que nada parezca olvidado. No se inventa ninguna cifra.

### v7 y v7.1 en HAM10000

HAM10000, val n = 2.004 (partición v6/v7), probe logreg, fuente `resultados/clasificacion/v71/v71_final_eval.json` y `resultados/clasificacion/v71/v71_final_eval.probas.npz`. El ECE puntual puede caer fuera de su IC bootstrap; se publica tal cual lo trae la fuente.

| Método | Configuración | F1 macro | Exactitud balanceada | AUC | ECE ↓ | MCC |
|---|---|---|---|---|---|---|
| v7: DINOv2 público | `public@224\|cls\|none\|id` | 0,6158 [0,5690; 0,6550] | 0,5924 [0,5459; 0,6376] | 0,9257 [0,9123; 0,9370] | 0,1307 [0,1159; 0,1476] | 0,5783 |
| v7: A* | `A_s42_a1.0@224\|cls\|none\|id` | 0,6464 [0,5980; 0,6894] | 0,6021 [0,5565; 0,6488] | 0,9619 [0,9563; 0,9671] | 0,0135 [0,0138; 0,0336] | 0,6647 |
| v7: B* | `B_s42_a1.0@224\|cls\|none\|id` | 0,6606 [0,6157; 0,7040] | 0,6327 [0,5871; 0,6838] | 0,9629 [0,9563; 0,9690] | 0,0325 [0,0253; 0,0497] | 0,6682 |
| v7.1: DINOv2 público, mejor protocolo | `public@224\|cls4\|la0.5\|tta` | 0,6857 [0,6412; 0,7208] | 0,6622 [0,6160; 0,7058] | 0,9598 [0,9516; 0,9665] | 0,0702 [0,0605; 0,0871] | 0,6683 |
| v7.1: DINOv2 público, protocolo de A71 | `public@224\|cls_attn\|la0.5\|tta` | 0,6320 [0,5875; 0,6710] | 0,6174 [0,5714; 0,6632] | 0,9432 [0,9339; 0,9519] | 0,0806 [0,0684; 0,0987] | 0,6110 |
| v7.1: DINOv2 público, protocolo de B71 | `public@224\|cls4\|balanced\|tta` | 0,6837 [0,6381; 0,7202] | 0,6659 [0,6176; 0,7102] | 0,9619 [0,9549; 0,9680] | 0,0688 [0,0573; 0,0846] | 0,6712 |
| v7.1: A71 | `A_soup_a1.0@224\|cls_attn\|la0.5\|tta` | 0,7229 [0,6795; 0,7592] | 0,7305 [0,6844; 0,7719] | 0,9644 [0,9558; 0,9718] | 0,0154 [0,0135; 0,0339] | 0,6996 |
| v7.1: B71 (propuesto) | `B_s42_a1.0@224\|cls4\|balanced\|tta` | 0,7141 [0,6703; 0,7547] | 0,6964 [0,6504; 0,7408] | 0,9602 [0,9516; 0,9680] | 0,0529 [0,0428; 0,0695] | 0,6901 |

### Limpios (E1) frente a originales

Limpio = pesos re-preentrenados sin las imágenes de val de ISIC en el pool (docs/33, sección 2). Predicción registrada (B71-limpio frente a B71 original, mismo cómputo canónico): ISIC 2019: Δ F1 macro -0,0055 [-0,0232; +0,0155], el IC incluye 0: se cumple la predicción (igual dentro del IC); ISIC 2020: Δ F1 macro +0,0093 [-0,0224; +0,0409], el IC incluye 0: se cumple la predicción (igual dentro del IC).

| Dataset | Método | Transductivo | F1 macro [IC 95 %] | MCC | Nota |
|---|---|---|---|---|---|
| HAM10000 | A71 (ablación sin término de tono) | no | 0,7229 [0,6795; 0,7592] | 0,6996 | pesos originales (selección); probe logreg C1; precisión, recall y MCC desde las probabilidades |
| HAM10000 | B71 (propuesto) | no | 0,7141 [0,6703; 0,7547] | 0,6901 | pesos originales (selección); probe logreg C1; precisión, recall y MCC desde las probabilidades; cómputo canónico = S1 (batch 64 + autocast, el de v7.1). F1 macro de B71 en S0/S1/S2: 0,7106 / 0,7141 / 0,7089 (el cómputo de extracción mueve la cifra) |
| HAM10000 | B71-limpio (canónico) | no | 0,7104 [0,6672; 0,7497] | 0,6790 | pesos limpios (pool SSL sin las imágenes de val de ISIC); análisis de sensibilidad posterior, no es selección: configuración y umbrales del holdout original (ajustados con los pesos originales y aplicados sin cambios); cómputo canónico (batch 64 + autocast); se lee al lado de la fila registrada de pesos originales, que no se reemplaza |
| HAM10000 | A71-limpio (canónico) | no | 0,7025 [0,6606; 0,7432] | 0,7047 | pesos limpios (pool SSL sin las imágenes de val de ISIC); análisis de sensibilidad posterior, no es selección: configuración y umbrales del holdout original (ajustados con los pesos originales y aplicados sin cambios); cómputo canónico (batch 64 + autocast); se lee al lado de la fila registrada de pesos originales, que no se reemplaza |
| ISIC 2019 | B71 original (canónico) | sí | 0,6717 [0,6436; 0,6951] | 0,6652 | cómputo canónico (batch 64 + autocast) |
| ISIC 2019 | B71-limpio (canónico) | no | 0,6662 [0,6401; 0,6911] | 0,6628 | cómputo canónico (batch 64 + autocast) |
| ISIC 2019 | A71-limpio (canónico) | no | 0,6549 [0,6278; 0,6771] | 0,6698 | cómputo canónico (batch 64 + autocast) |
| ISIC 2019 | A71 original (canónico) | sí | 0,6481 [0,6218; 0,6719] | 0,6649 | cómputo canónico (batch 64 + autocast) |
| ISIC 2019 | B71 original (batch 16) | sí | 0,6756 [0,6482; 0,6977] | 0,6677 | sensibilidad: cómputo no canónico (batch 16); transductivo |
| ISIC 2019 | A71 original (batch 16) | sí | 0,6522 [0,6256; 0,6759] | 0,6669 | sensibilidad: cómputo no canónico (batch 16); transductivo |
| ISIC 2020 | A71 original (canónico) | sí | 0,6543 [0,6164; 0,6896] | 0,3166 | cómputo canónico (batch 64 + autocast) |
| ISIC 2020 | A71-limpio (canónico) | no | 0,6467 [0,6095; 0,6837] | 0,3040 | cómputo canónico (batch 64 + autocast) |
| ISIC 2020 | B71-limpio (canónico) | no | 0,6229 [0,5865; 0,6602] | 0,2506 | cómputo canónico (batch 64 + autocast) |
| ISIC 2020 | B71 original (canónico) | sí | 0,6136 [0,5793; 0,6501] | 0,2336 | cómputo canónico (batch 64 + autocast) |
| ISIC 2020 | A71 original (batch 16) | sí | 0,6554 [0,6197; 0,6910] | 0,3196 | sensibilidad: cómputo no canónico (batch 16); transductivo |
| ISIC 2020 | B71 original (batch 16) | sí | 0,6123 [0,5781; 0,6486] | 0,2315 | sensibilidad: cómputo no canónico (batch 16); transductivo |

### HAM10000: pesos limpios (sensibilidad) frente al original y al público

Análisis de sensibilidad posterior, no es selección: configuración y umbrales del holdout original (los umbrales se ajustaron con los pesos originales y se aplican sin cambios a los limpios). Cómputo canónico, HAM10000 val n = 2.004, bootstrap pareado por `lesion_id`, 1.000 remuestras, semilla 20260923. Fuente: `resultados/banco_completo/ham_clean/ham_clean_val.json`.

| Brazo | Control | Métrica | Δ (brazo − control) [IC 95 %] | El IC excluye 0 | Lectura |
|---|---|---|---|---|---|
| B71-limpio | DINOv2 público (prot. B71) | F1 macro | +0,0267 [-0,0122; +0,0686] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | DINOv2 público (prot. B71) | Exactitud balanceada | +0,0276 [-0,0114; +0,0690] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | DINOv2 público (prot. B71) | AUC OvR macro | -0,0007 [-0,0071; +0,0059] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | DINOv2 público (prot. B71) | ECE (menor es mejor) | -0,0107 [-0,0260; +0,0045] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | DINOv2 público (prot. B71) | Recall de melanoma | -0,0045 [-0,0773; +0,0672] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | DINOv2 público (prot. B71) | AUC de melanoma | +0,0055 [-0,0112; +0,0226] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | DINOv2 público (prot. B71) | MCC | +0,0078 [-0,0227; +0,0368] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | B71 original | F1 macro | -0,0037 [-0,0360; +0,0270] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | B71 original | Exactitud balanceada | -0,0028 [-0,0339; +0,0282] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | B71 original | AUC OvR macro | +0,0010 [-0,0068; +0,0101] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | B71 original | ECE (menor es mejor) | +0,0052 [-0,0072; +0,0166] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | B71 original | Recall de melanoma | -0,0225 [-0,0820; +0,0294] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | B71 original | AUC de melanoma | -0,0035 [-0,0171; +0,0110] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | B71 original | MCC | -0,0111 [-0,0359; +0,0116] | no | sin diferencia demostrable (el IC incluye 0) |
| A71-limpio | DINOv2 público (prot. A71) | F1 macro | +0,0705 [+0,0255; +0,1132] | sí | A71-limpio mejor (el IC excluye 0) |
| A71-limpio | DINOv2 público (prot. A71) | Exactitud balanceada | +0,0929 [+0,0502; +0,1356] | sí | A71-limpio mejor (el IC excluye 0) |
| A71-limpio | DINOv2 público (prot. A71) | AUC OvR macro | +0,0198 [+0,0101; +0,0304] | sí | A71-limpio mejor (el IC excluye 0) |
| A71-limpio | DINOv2 público (prot. A71) | ECE (menor es mejor) | -0,0649 [-0,0766; -0,0415] | sí | A71-limpio mejor (el IC excluye 0) |
| A71-limpio | DINOv2 público (prot. A71) | Recall de melanoma | +0,1892 [+0,1048; +0,2715] | sí | A71-limpio mejor (el IC excluye 0) |
| A71-limpio | DINOv2 público (prot. A71) | AUC de melanoma | +0,0674 [+0,0390; +0,0949] | sí | A71-limpio mejor (el IC excluye 0) |
| A71-limpio | DINOv2 público (prot. A71) | MCC | +0,0937 [+0,0569; +0,1283] | sí | A71-limpio mejor (el IC excluye 0) |
| A71-limpio | A71 original | F1 macro | -0,0204 [-0,0578; +0,0140] | no | sin diferencia demostrable (el IC incluye 0) |
| A71-limpio | A71 original | Exactitud balanceada | -0,0202 [-0,0574; +0,0167] | no | sin diferencia demostrable (el IC incluye 0) |
| A71-limpio | A71 original | AUC OvR macro | -0,0014 [-0,0063; +0,0040] | no | sin diferencia demostrable (el IC incluye 0) |
| A71-limpio | A71 original | ECE (menor es mejor) | +0,0002 [-0,0162; +0,0145] | no | sin diferencia demostrable (el IC incluye 0) |
| A71-limpio | A71 original | Recall de melanoma | +0,0631 [+0,0084; +0,1174] | sí | A71-limpio mejor (el IC excluye 0) |
| A71-limpio | A71 original | AUC de melanoma | -0,0034 [-0,0189; +0,0104] | no | sin diferencia demostrable (el IC incluye 0) |
| A71-limpio | A71 original | MCC | +0,0051 [-0,0210; +0,0289] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | A71-limpio | F1 macro | +0,0079 [-0,0247; +0,0445] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | A71-limpio | Exactitud balanceada | -0,0167 [-0,0500; +0,0175] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | A71-limpio | AUC OvR macro | -0,0018 [-0,0083; +0,0051] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | A71-limpio | ECE (menor es mejor) | +0,0424 [+0,0204; +0,0511] | sí | A71-limpio mejor (el IC excluye 0) |
| B71-limpio | A71-limpio | Recall de melanoma | -0,1396 [-0,2146; -0,0686] | sí | A71-limpio mejor (el IC excluye 0) |
| B71-limpio | A71-limpio | AUC de melanoma | -0,0147 [-0,0318; +0,0024] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | A71-limpio | MCC | -0,0257 [-0,0551; +0,0028] | no | sin diferencia demostrable (el IC incluye 0) |

### F1-F5 con segmentación (E3): holdout

Selección SOLO en el holdout de HAM. Regla registrada: `macro_f1 >= best no-mask macro_f1 - 0.01 on the holdout AND delta > 0 vs the reference in melanoma recall OR in the macro-F1 of the darkest L* tercile (docs/33 E3)`. Ganadora por la regla: F4; las 5 variantes adoptables están dentro de 0,0055 de F1 macro entre sí, con IC solapados: no hay ganadora clara. Exploratorio (docs/33, erratas).

| Variante | Descripción | F1 macro holdout [IC 95 %] | F1 macro, tercil L* oscuro | Recall de melanoma | Cumple la regla | Adoptable |
|---|---|---|---|---|---|---|
| B71 | Referencia: B71 sin segmentación | 0,7725 [0,7328; 0,8062] | 0,7533 | 0,5955 | — | — |
| F1 | concat(features de la imagen completa, features del recorte con margen 10 %) | 0,7859 [0,7486; 0,8165] | 0,7754 | 0,6180 | sí | sí |
| F2 | F1 con el recorte tomado de la resolución original | 0,7819 [0,7464; 0,8117] | 0,7868 | 0,5899 | sí | sí |
| F2_448 | F2 con el recorte a 448 px (`crop_orig448`) | 0,7858 [0,7460; 0,8168] | 0,7643 | 0,6236 | sí | sí |
| F3 | TTA guiada por máscara (imagen completa + 4 recortes) | 0,7456 [0,7024; 0,7804] | 0,7479 | 0,5787 | no | no |
| F4 | F1 + descriptores de forma y color de la máscara | 0,7868 [0,7496; 0,8148] | 0,7951 | 0,6067 | sí | sí |
| F5 | F1 + anillo perilesional | 0,7812 [0,7440; 0,8126] | 0,7849 | 0,5955 | sí | sí |

### F4 frente a B71 en val (E3, cómputo S0)

Val leído una sola vez, solo para B71 y la ganadora F4 (cómputo S0). F1, F2, F2_448, F3 y F5 no se evalúan en val por diseño (selección solo en el holdout), no por falta de GPU.

| Variante | F1 macro | Exactitud balanceada | Recall de melanoma | AUC | ECE ↓ |
|---|---|---|---|---|---|
| B71 | 0,7106 [0,6591; 0,7548] | 0,6937 [0,6437; 0,7509] | 0,5135 [0,4350; 0,5945] | 0,9600 | 0,0532 |
| F4 | 0,7466 [0,6942; 0,7901] | 0,7213 [0,6679; 0,7810] | 0,5495 [0,4661; 0,6272] | 0,9656 | 0,0491 |

### F4 − B71: sensibilidad al cómputo de extracción (HAM10000, n = 2.004)

Regla de lectura fijada de antemano: el IC pareado de F1 macro debe excluir 0 en S0, S1 y S2. Lo cumple en 2 de 3: sin diferencia demostrable de forma robusta; la dirección favorece a F4. Cómputo canónico: S1.

| Cómputo | B71 F1 macro | F4 F1 macro | Δ F1 macro [IC 95 %] | IC excluye 0 | Δ exactitud balanceada [IC 95 %] | Δ recall de melanoma [IC 95 %] |
|---|---|---|---|---|---|---|
| S0: batch 16 + autocast | 0,7106 | 0,7466 | +0,0360 [+0,0020; +0,0778] | sí | +0,0276 [-0,0102; +0,0688] | +0,0360 [-0,0154; +0,0876] |
| S1: batch 64 + autocast (canónico, el de v7.1) | 0,7141 | 0,7441 | +0,0300 [-0,0046; +0,0715] | no | +0,0222 [-0,0159; +0,0624] | +0,0135 [-0,0446; +0,0714] |
| S2: batch 64, fp32 | 0,7089 | 0,7483 | +0,0393 [+0,0039; +0,0824] | sí | +0,0318 [-0,0069; +0,0749] | +0,0360 [-0,0154; +0,0876] |

### Robustez de F4 − B71 a la semilla de B*

Robustez de F4 − B71 a la semilla de B* (post-hoc, cómputo canónico). El IC excluye 0 en 0 de 3 semillas: lectura «none», ninguna semilla (0 de 3): sin diferencia demostrable. Δ medio de F1 macro +0,0217 (mín. +0,0092, máx. +0,0300); la dirección favorece a F4 en 3 de 3 semillas. Fuente `resultados/banco_completo/e3/robustez/e3_robustness/e3_robustness.json`.

| Semilla | B71 F1 macro | F4 F1 macro | Δ F1 macro [IC 95 %] | IC excluye 0 | Δ exactitud balanceada [IC 95 %] | IC excluye 0 |
|---|---|---|---|---|---|---|
| 42 | 0,7141 | 0,7441 | +0,0300 [-0,0046; +0,0715] | no | +0,0222 [-0,0159; +0,0624] | no |
| 43 | 0,7026 | 0,7118 | +0,0092 [-0,0339; +0,0499] | no | -0,0041 [-0,0473; +0,0354] | no |
| 44 | 0,7041 | 0,7301 | +0,0261 [-0,0039; +0,0577] | no | +0,0128 [-0,0168; +0,0440] | no |

### D0-D3 en PAD `target_test` (E2)

`target_test` de PAD leído una sola vez (lock) (con lock presente). Probe `logreg` ajustado en HAM10000 (n = 8.011) y aplicado congelado; el backbone nunca ve etiquetas. Cómputo canónico: batch 64 + autocast cuda. Bootstrap pareado de 1.000 remuestras, semilla 20260923, agrupado por `patient_id`. Lectura «primary» (BCC/MEL/NV): 567 imágenes de 363 pacientes (de 1.142 imágenes de `target_test`), solo 25 melanomas: los IC son anchos. «Fuera de rango» = proporción de imágenes cuyo argmax de 7 clases cae fuera de las clases de la lectura (descriptivo). Fuente: `resultados/banco_completo/pad_da/pad_e2_eval.json`.

| Brazo | Descripción | F1 macro | Exactitud balanceada | Accuracy | AUC | ECE ↓ | MCC | Fuera de rango |
|---|---|---|---|---|---|---|---|---|
| D0 | DINOv2 público | 0,4410 [0,3970; 0,4827] | 0,5284 [0,4750; 0,5853] | 0,5802 [0,5352; 0,6259] | 0,7599 [0,7061; 0,8146] | 0,3413 [0,3017; 0,3890] | 0,3740 [0,3144; 0,4306] | 0,3210 |
| D1 | A* s42 limpio | 0,2443 [0,2053; 0,2824] | 0,4244 [0,3519; 0,5012] | 0,2698 [0,2230; 0,3170] | 0,7257 [0,6749; 0,7783] | 0,5592 [0,5125; 0,6034] | 0,1871 [0,1407; 0,2341] | 0,4039 |
| D2 | D1 continuado con el pool limpio (control de cómputo) | 0,2877 [0,2458; 0,3326] | 0,4897 [0,4132; 0,5674] | 0,3333 [0,2850; 0,3811] | 0,7577 [0,7026; 0,8136] | 0,4753 [0,4316; 0,5236] | 0,2309 [0,1894; 0,2731] | 0,3422 |
| D3 | D2 más las imágenes de `target_adapt` de PAD, sin etiquetas | 0,3693 [0,3240; 0,4173] | 0,5271 [0,4529; 0,6087] | 0,4444 [0,3941; 0,4949] | 0,7740 [0,7136; 0,8370] | 0,3183 [0,2788; 0,3670] | 0,3039 [0,2564; 0,3521] | 0,4780 |

### D0-D3 (E2): contrastes pareados

D3 − D2 es la predicción primaria registrada; los demás son descriptivos. Predicción primaria registrada (D3 − D2, F1 macro > 0 con IC que excluye 0): cumplida = sí (`prediction_d3_minus_d2_met` del JSON); Δ +0,0816 [+0,0525; +0,1085].

| Brazo | Control | Métrica | Δ (brazo − control) [IC 95 %] | El IC excluye 0 | Lectura |
|---|---|---|---|---|---|
| D3 | D2 | F1 macro | +0,0816 [+0,0525; +0,1085] | sí | D3 mejor (el IC excluye 0) |
| D3 | D2 | Exactitud balanceada | +0,0375 [-0,0161; +0,0875] | no | sin diferencia demostrable (el IC incluye 0) |
| D3 | D2 | Accuracy | +0,1111 [+0,0755; +0,1468] | sí | D3 mejor (el IC excluye 0) |
| D3 | D2 | AUC OvR macro | +0,0162 [-0,0065; +0,0393] | no | sin diferencia demostrable (el IC incluye 0) |
| D3 | D2 | ECE (menor es mejor) | -0,1570 [-0,1943; -0,1199] | sí | D3 mejor (el IC excluye 0) |
| D3 | D2 | MCC | +0,0730 [+0,0419; +0,1034] | sí | D3 mejor (el IC excluye 0) |
| D2 | D1 | F1 macro | +0,0434 [+0,0196; +0,0682] | sí | D2 mejor (el IC excluye 0) |
| D2 | D1 | Exactitud balanceada | +0,0653 [+0,0297; +0,1112] | sí | D2 mejor (el IC excluye 0) |
| D2 | D1 | Accuracy | +0,0635 [+0,0366; +0,0914] | sí | D2 mejor (el IC excluye 0) |
| D2 | D1 | AUC OvR macro | +0,0321 [+0,0068; +0,0593] | sí | D2 mejor (el IC excluye 0) |
| D2 | D1 | ECE (menor es mejor) | -0,0838 [-0,1114; -0,0510] | sí | D2 mejor (el IC excluye 0) |
| D2 | D1 | MCC | +0,0438 [+0,0127; +0,0752] | sí | D2 mejor (el IC excluye 0) |
| D3 | D1 | F1 macro | +0,1250 [+0,0970; +0,1561] | sí | D3 mejor (el IC excluye 0) |
| D3 | D1 | Exactitud balanceada | +0,1027 [+0,0427; +0,1740] | sí | D3 mejor (el IC excluye 0) |
| D3 | D1 | Accuracy | +0,1746 [+0,1413; +0,2131] | sí | D3 mejor (el IC excluye 0) |
| D3 | D1 | AUC OvR macro | +0,0483 [+0,0212; +0,0828] | sí | D3 mejor (el IC excluye 0) |
| D3 | D1 | ECE (menor es mejor) | -0,2408 [-0,2812; -0,1993] | sí | D3 mejor (el IC excluye 0) |
| D3 | D1 | MCC | +0,1168 [+0,0788; +0,1617] | sí | D3 mejor (el IC excluye 0) |
| D1 | D0 | F1 macro | -0,1967 [-0,2304; -0,1638] | sí | D0 mejor (el IC excluye 0) |
| D1 | D0 | Exactitud balanceada | -0,1040 [-0,1670; -0,0421] | sí | D0 mejor (el IC excluye 0) |
| D1 | D0 | Accuracy | -0,3104 [-0,3556; -0,2643] | sí | D0 mejor (el IC excluye 0) |
| D1 | D0 | AUC OvR macro | -0,0343 [-0,0834; +0,0051] | no | sin diferencia demostrable (el IC incluye 0) |
| D1 | D0 | ECE (menor es mejor) | +0,2179 [+0,1684; +0,2611] | sí | D0 mejor (el IC excluye 0) |
| D1 | D0 | MCC | -0,1869 [-0,2366; -0,1380] | sí | D0 mejor (el IC excluye 0) |
| D2 | D0 | F1 macro | -0,1533 [-0,1893; -0,1150] | sí | D0 mejor (el IC excluye 0) |
| D2 | D0 | Exactitud balanceada | -0,0387 [-0,1074; +0,0342] | no | sin diferencia demostrable (el IC incluye 0) |
| D2 | D0 | Accuracy | -0,2469 [-0,2931; -0,1996] | sí | D0 mejor (el IC excluye 0) |
| D2 | D0 | AUC OvR macro | -0,0022 [-0,0565; +0,0447] | no | sin diferencia demostrable (el IC incluye 0) |
| D2 | D0 | ECE (menor es mejor) | +0,1341 [+0,0851; +0,1782] | sí | D0 mejor (el IC excluye 0) |
| D2 | D0 | MCC | -0,1431 [-0,1876; -0,0941] | sí | D0 mejor (el IC excluye 0) |
| D3 | D0 | F1 macro | -0,0717 [-0,1100; -0,0332] | sí | D0 mejor (el IC excluye 0) |
| D3 | D0 | Exactitud balanceada | -0,0013 [-0,0671; +0,0666] | no | sin diferencia demostrable (el IC incluye 0) |
| D3 | D0 | Accuracy | -0,1358 [-0,1838; -0,0874] | sí | D0 mejor (el IC excluye 0) |
| D3 | D0 | AUC OvR macro | +0,0140 [-0,0397; +0,0579] | no | sin diferencia demostrable (el IC incluye 0) |
| D3 | D0 | ECE (menor es mejor) | -0,0229 [-0,0727; +0,0265] | no | sin diferencia demostrable (el IC incluye 0) |
| D3 | D0 | MCC | -0,0701 [-0,1188; -0,0200] | sí | D0 mejor (el IC excluye 0) |
