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
- **Contraste pareado:** diferencia entre dos modelos sobre las mismas imágenes, con IC bootstrap de la diferencia. «Mejor» solo si el IC excluye 0; si lo incluye, «sin diferencia demostrable».
- **Exploratorio:** variantes post-hoc o con selección en otro conjunto; se reportan pero no cambian el propuesto ni se leen como resultado confirmado.

## Cómo leer este documento

Particiones de val de este banco (no son la misma partición):

- HAM10000, baselines (Kaggle): val n = 2.014; StratifiedGroupKFold por lesion_id (0 lesiones compartidas).
- HAM10000, propuesto (v6/v7): val n = 2.004; partición agrupada por `lesion_id`.
- ISIC 2019, baselines (Kaggle): val n = 5.067; StratifiedKFold estratificado simple SIN grupo de paciente (el release no trae patient_id) - riesgo de fuga entre imagenes del mismo paciente, metricas con sesgo optimista.
- ISIC 2020, baselines (Kaggle): val n = 6.660; StratifiedGroupKFold por patient_id (0 pacientes compartidos).

- **Entre particiones distintas la comparación es solo de intervalos (IC 95 %)**, no un test pareado: HAM10000 usa la partición v6/v7 (n = 2.004) para el propuesto y la de Kaggle (n = 2.014) para los baselines; ISIC 2019 e ISIC 2020 tienen la suya. Los contrastes pareados solo existen donde dos modelos comparten partición e imágenes y hay probabilidades por imagen (docs/33, sección 5); están al final de cada tabla y, todos juntos, en `CONTRASTES_SSL.md` (sin corrección por multiplicidad).
- **Optimista: época elegida en val.** Todas las filas de fine-tuning (HAM10000, ISIC 2019 y la corrida pendiente de ISIC 2020) eligen su mejor época por balanced accuracy sobre val; el plan decía holdout interno y se mantuvo la receta existente por consistencia entre datasets (errata de docs/33). El propuesto no tiene fine-tuning y no se ve afectado.
- **Modelo y protocolo no son lo mismo.** El propuesto usa el protocolo de B71 (features cls4, TTA, probe balanceado); la fila «DINOv2 público, protocolo de B71» es el control con el mismo protocolo y pesos públicos. Los 8 baselines siguen el protocolo de su propio banco (probe logreg).
- Los IC solo existen donde la fuente los publica: los baselines, para exactitud balanceada y F1 macro; B71/A71, para accuracy, exactitud balanceada, F1 macro, AUC y ECE. Precisión macro, recall macro y MCC de los baselines salen exactos de la matriz de confusión y van sin IC; en B71/A71 salen de las probabilidades commiteadas, también sin IC.
- Las filas «pendiente» no tienen número: se listan con su motivo técnico y la ETA declarada por el operador, para que nada parezca olvidado. No se inventa ninguna cifra.

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

Limpio = pesos re-preentrenados sin las imágenes de val de ISIC en el pool (docs/33, sección 2). Lectura de la predicción registrada (B71-limpio frente a B71 original): pendiente; el contraste B71-limpio contra B71 original exige el original con el mismo cómputo canónico; la corrida de batch 16 no es comparable (cambia el cómputo, no solo los datos); ETA ~15:00 UTC.

| Dataset | Método | Transductivo | F1 macro [IC 95 %] | MCC | Nota |
|---|---|---|---|---|---|
| HAM10000 | A71 (ablación sin término de tono) | no | 0,7229 [0,6795; 0,7592] | 0,6996 | probe logreg C1; precisión, recall y MCC desde las probabilidades |
| HAM10000 | B71 (propuesto) | no | 0,7141 [0,6703; 0,7547] | 0,6901 | probe logreg C1; precisión, recall y MCC desde las probabilidades; cómputo canónico = S1 (batch 64 + autocast, el de v7.1). F1 macro de B71 en S0/S1/S2: 0,7106 / 0,7141 / 0,7089 (el cómputo de extracción mueve la cifra) |
| HAM10000 | B71-limpio (canónico) | no | pendiente | pendiente | E1 de B71-limpio en HAM10000 todavía no se corrió (no hay JSON en el repositorio); ETA no declarada por el operador |
| HAM10000 | A71-limpio (canónico) | no | pendiente | pendiente | falta A* s43 limpio (GPU0 en curso); A71-limpio es la sopa de A* s42 + s43 limpios; ETA ~15:00 UTC |
| ISIC 2019 | B71-limpio (canónico) | no | 0,6662 [0,6401; 0,6911] | 0,6628 | cómputo canónico (batch 64 + autocast) |
| ISIC 2019 | A71-limpio (canónico) | no | pendiente | pendiente | falta A* s43 limpio (GPU0 en curso); A71-limpio es la sopa de A* s42 + s43 limpios; ETA ~15:00 UTC |
| ISIC 2019 | B71 original (canónico) | sí | pendiente | pendiente | re-extracción de B71/A71 originales con el cómputo canónico (batch 64 + autocast) en GPU0 después de A* s43; hoy solo existe la corrida de batch 16; ETA ~15:00 UTC |
| ISIC 2019 | A71 original (canónico) | sí | pendiente | pendiente | re-extracción de B71/A71 originales con el cómputo canónico (batch 64 + autocast) en GPU0 después de A* s43; hoy solo existe la corrida de batch 16; ETA ~15:00 UTC |
| ISIC 2019 | B71 original (batch 16) | sí | 0,6756 [0,6482; 0,6977] | 0,6677 | sensibilidad: cómputo no canónico (batch 16); transductivo |
| ISIC 2019 | A71 original (batch 16) | sí | 0,6522 [0,6256; 0,6759] | 0,6669 | sensibilidad: cómputo no canónico (batch 16); transductivo |
| ISIC 2020 | B71-limpio (canónico) | no | 0,6229 [0,5865; 0,6602] | 0,2506 | cómputo canónico (batch 64 + autocast) |
| ISIC 2020 | A71-limpio (canónico) | no | pendiente | pendiente | falta A* s43 limpio (GPU0 en curso); A71-limpio es la sopa de A* s42 + s43 limpios; ETA ~15:00 UTC |
| ISIC 2020 | B71 original (canónico) | sí | pendiente | pendiente | re-extracción de B71/A71 originales con el cómputo canónico (batch 64 + autocast) en GPU0 después de A* s43; hoy solo existe la corrida de batch 16; ETA ~15:00 UTC |
| ISIC 2020 | A71 original (canónico) | sí | pendiente | pendiente | re-extracción de B71/A71 originales con el cómputo canónico (batch 64 + autocast) en GPU0 después de A* s43; hoy solo existe la corrida de batch 16; ETA ~15:00 UTC |
| ISIC 2020 | A71 original (batch 16) | sí | 0,6554 [0,6197; 0,6910] | 0,3196 | sensibilidad: cómputo no canónico (batch 16); transductivo |
| ISIC 2020 | B71 original (batch 16) | sí | 0,6123 [0,5781; 0,6486] | 0,2315 | sensibilidad: cómputo no canónico (batch 16); transductivo |

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

### D0-D3 (E2): pendientes

E2: probe ajustado en HAM train y aplicado congelado a PAD `target_test` (BCC/MEL/NEV), bootstrap por paciente. Primario: D3 − D2. Split de PAD ya generado: `target_adapt` 1.141 imágenes, `target_test` 1.142 imágenes. ETA declarada por el operador el 2026-10-05; no se mide desde el repositorio.

| Brazo | Descripción | Estado | Motivo | ETA |
|---|---|---|---|---|
| D0 | DINOv2 público | pendiente | los pesos de D2 y D3 (GPU) y la única lectura de `target_test` de PAD no han terminado; el split de PAD ya existe (`resultados/banco_completo/pad_da/target_split.json`) | ~10:30 UTC |
| D1 | A* s42 limpio | pendiente | los pesos de D2 y D3 (GPU) y la única lectura de `target_test` de PAD no han terminado; el split de PAD ya existe (`resultados/banco_completo/pad_da/target_split.json`) | ~10:30 UTC |
| D2 | D1 continuado con el pool limpio (control de cómputo) | pendiente | los pesos de D2 y D3 (GPU) y la única lectura de `target_test` de PAD no han terminado; el split de PAD ya existe (`resultados/banco_completo/pad_da/target_split.json`) | ~10:30 UTC |
| D3 | D2 más las imágenes de `target_adapt` de PAD, sin etiquetas | pendiente | los pesos de D2 y D3 (GPU) y la única lectura de `target_test` de PAD no han terminado; el split de PAD ya existe (`resultados/banco_completo/pad_da/target_split.json`) | ~10:30 UTC |
