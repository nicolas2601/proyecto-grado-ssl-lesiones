# Banco completo: ISIC 2020, propuesto y 8 baselines

Generado por `scripts/build_bench_tables.py` desde los JSON del repositorio. No editar a mano. Plan y reglas: `docs/33-banco-completo-multidataset.md`.

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

En este val la clase «benigno» es el 98,2 % de las imágenes (2 clases): predecir siempre «benigno» da una accuracy de 0,9817 y una exactitud balanceada de 0,5000.

## Cómo leer esta tabla

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

## Congelados (sin fine-tuning)

Orden: F1 macro de mayor a menor; las filas sin número van al final.

### Condición

| # | Modelo | Preentrenamiento | Fine-tuning | Segmentación | Transductivo | Val n | Estado | Nota | Fuente |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A71 original (canónico) | auto-supervisado propio (DINOv2 + SimSiam anclado sin término de tono, A*) | no | no | sí | 6.660 | ok | cómputo canónico (batch 64 + autocast) | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/A71__original.json` · `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/A71__original.probas.npz` |
| 2 | A71-limpio (canónico) | auto-supervisado propio (DINOv2 + SimSiam anclado sin término de tono, A*) | no | no | no | 6.660 | ok | cómputo canónico (batch 64 + autocast) | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/A71__clean.json` · `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/A71__clean.probas.npz` |
| 3 | B71-limpio (canónico) | auto-supervisado propio (DINOv2 + SimSiam anclado + término de tono, B*) | no | no | no | 6.660 | ok | cómputo canónico (batch 64 + autocast) | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/B71__clean.json` · `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/B71__clean.probas.npz` |
| 4 | B71 original (canónico) | auto-supervisado propio (DINOv2 + SimSiam anclado + término de tono, B*) | no | no | sí | 6.660 | ok | cómputo canónico (batch 64 + autocast) | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/B71__original.json` · `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/B71__original.probas.npz` |
| 5 | DINOv2 público, protocolo de A71 (canónico) | auto-supervisado (DINOv2 público) | no | no | no | 6.660 | ok | cómputo canónico (batch 64 + autocast) | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/PUBLIC_A71.json` · `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/PUBLIC_A71.probas.npz` |
| 6 | DINOv2 público, protocolo de B71 (canónico) | auto-supervisado (DINOv2 público) | no | no | no | 6.660 | ok | cómputo canónico (batch 64 + autocast) | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/PUBLIC_B71.json` · `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/PUBLIC_B71.probas.npz` |
| 7 | ResNet-50 | supervisado (ImageNet) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_resnet50_isic2020.json` |
| 8 | ViT-B/16 | supervisado (ImageNet) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_vit_b16_isic2020.json` |
| 9 | EfficientNet-B0 | supervisado (ImageNet) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_efficientnet_b0_isic2020.json` |
| 10 | DINOv2 ViT-B/14 | auto-supervisado (DINOv2 público) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_dinov2_vitb14_isic2020.json` |
| 11 | DINOv2 ViT-S/14 | auto-supervisado (DINOv2 público) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_dinov2_vits14_isic2020.json` |
| 12 | VGG19 | supervisado (ImageNet) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_vgg19_isic2020.json` |
| 13 | AlexNet | supervisado (ImageNet) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_alexnet_isic2020.json` |
| 14 | SimSiam ResNet-50 | auto-supervisado (SimSiam, ImageNet) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_simsiam_r50_isic2020.json` |

### Métricas [IC 95 % donde la fuente lo publica]

| # | Modelo | Accuracy | Exactitud balanceada | Precisión macro | Recall macro | F1 macro | AUC OvR macro | ECE ↓ | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A71 original (canónico) | 0,9688 [0,9624; 0,9741] | 0,6945 [0,6472; 0,7459] | 0,6288 | 0,6945 | 0,6543 [0,6164; 0,6896] | 0,9023 [0,8728; 0,9292] | 0,0332 [0,0282; 0,0389] | 0,3166 [0,2408; 0,3886] |
| 2 | A71-limpio (canónico) | 0,9667 [0,9599; 0,9724] | 0,6934 [0,6423; 0,7481] | 0,6194 | 0,6934 | 0,6467 [0,6095; 0,6837] | 0,9085 [0,8783; 0,9339] | 0,0339 [0,0294; 0,0400] | 0,3040 [0,2265; 0,3781] |
| 3 | B71-limpio (canónico) | 0,9671 [0,9600; 0,9731] | 0,6494 [0,6052; 0,6926] | 0,6050 | 0,6494 | 0,6229 [0,5865; 0,6602] | 0,8758 [0,8397; 0,9049] | 0,0111 [0,0087; 0,0164] | 0,2506 [0,1766; 0,3269] |
| 4 | B71 original (canónico) | 0,9649 [0,9576; 0,9712] | 0,6443 [0,5974; 0,6928] | 0,5946 | 0,6443 | 0,6136 [0,5793; 0,6501] | 0,8529 [0,8088; 0,8945] | 0,0122 [0,0094; 0,0174] | 0,2336 [0,1638; 0,3083] |
| 5 | DINOv2 público, protocolo de A71 (canónico) | 0,9634 [0,9563; 0,9694] | 0,6234 [0,5829; 0,6631] | 0,5809 | 0,6234 | 0,5971 [0,5636; 0,6272] | 0,8323 [0,7904; 0,8719] | 0,0119 [0,0086; 0,0170] | 0,1998 [0,1323; 0,2622] |
| 6 | DINOv2 público, protocolo de B71 (canónico) | 0,9706 [0,9643; 0,9759] | 0,5828 [0,5485; 0,6185] | 0,5856 | 0,5828 | 0,5842 [0,5507; 0,6181] | 0,8790 [0,8469; 0,9073] | 0,0125 [0,0100; 0,0177] | 0,1684 [0,1015; 0,2382] |
| 7 | ResNet-50 | 0,9161 | 0,6878 [0,6425; 0,7321] | 0,5448 | 0,6878 | 0,5601 [0,5396; 0,5790] | 0,8297 | 0,0275 | 0,1834 |
| 8 | ViT-B/16 | 0,8883 | 0,7340 [0,6903; 0,7777] | 0,5415 | 0,7340 | 0,5493 [0,5319; 0,5668] | 0,8275 | 0,0598 | 0,1971 |
| 9 | EfficientNet-B0 | 0,9186 | 0,6368 [0,5954; 0,6818] | 0,5352 | 0,6368 | 0,5457 [0,5264; 0,5640] | 0,7607 | 0,0378 | 0,1387 |
| 10 | DINOv2 ViT-B/14 | 0,8994 | 0,6753 [0,6284; 0,7186] | 0,5355 | 0,6753 | 0,5427 [0,5252; 0,5577] | 0,7999 | 0,0559 | 0,1578 |
| 11 | DINOv2 ViT-S/14 | 0,8470 | 0,7572 [0,7132; 0,7981] | 0,5346 | 0,7572 | 0,5266 [0,5119; 0,5393] | 0,8576 | 0,0561 | 0,1886 |
| 12 | VGG19 | 0,8233 | 0,7370 [0,6951; 0,7789] | 0,5286 | 0,7370 | 0,5101 [0,4978; 0,5234] | 0,8128 | 0,0333 | 0,1647 |
| 13 | AlexNet | 0,7940 | 0,7744 [0,7354; 0,8125] | 0,5292 | 0,7744 | 0,5008 [0,4894; 0,5122] | 0,8574 | 0,0442 | 0,1791 |
| 14 | SimSiam ResNet-50 | 0,7518 | 0,7650 [0,7301; 0,8000] | 0,5249 | 0,7650 | 0,4795 [0,4693; 0,4906] | 0,8468 | 0,0851 | 0,1624 |

## Con fine-tuning

Orden: F1 macro de mayor a menor; las filas sin número van al final.

### Condición

| # | Modelo | Preentrenamiento | Fine-tuning | Segmentación | Transductivo | Val n | Estado | Nota | Fuente |
|---|---|---|---|---|---|---|---|---|---|
| 1 | VGG19 | supervisado (ImageNet) | sí | no | no | 6.660 | ok | época elegida en val (sesgo optimista); mejor época 8 | `resultados/banco_completo/ft_isic2020/json/results_vgg19_ft_isic2020.json` |
| 2 | AlexNet | supervisado (ImageNet) | sí | no | no | 6.660 | ok | época elegida en val (sesgo optimista); mejor época 7 | `resultados/banco_completo/ft_isic2020/json/results_alexnet_ft_isic2020.json` |
| 3 | ViT-B/16 | supervisado (ImageNet) | sí | no | no | 6.660 | ok | época elegida en val (sesgo optimista); mejor época 3 | `resultados/banco_completo/ft_isic2020/json/results_vit_b16_ft_isic2020.json` |
| 4 | ResNet-50 | supervisado (ImageNet) | sí | no | no | 6.660 | ok | época elegida en val (sesgo optimista); mejor época 3 | `resultados/banco_completo/ft_isic2020/json/results_resnet50_ft_isic2020.json` |
| 5 | EfficientNet-B0 | supervisado (ImageNet) | sí | no | no | 6.660 | ok | época elegida en val (sesgo optimista); mejor época 2 | `resultados/banco_completo/ft_isic2020/json/results_efficientnet_b0_ft_isic2020.json` |
| 6 | SimSiam ResNet-50 | auto-supervisado (SimSiam, ImageNet) | sí | no | no | 6.660 | ok | época elegida en val (sesgo optimista); mejor época 8 | `resultados/banco_completo/ft_isic2020/json/results_simsiam_r50_ft_isic2020.json` |
| 7 | DINOv2 ViT-B/14 | auto-supervisado (DINOv2 público) | sí | no | no | 6.660 | ok | época elegida en val (sesgo optimista); mejor época 3 | `resultados/banco_completo/ft_isic2020/json/results_dinov2_vitb14_ft_isic2020.json` |
| 8 | DINOv2 ViT-S/14 | auto-supervisado (DINOv2 público) | sí | no | no | 6.660 | ok | época elegida en val (sesgo optimista); mejor época 5 | `resultados/banco_completo/ft_isic2020/json/results_dinov2_vits14_ft_isic2020.json` |

### Métricas [IC 95 % donde la fuente lo publica]

| # | Modelo | Accuracy | Exactitud balanceada | Precisión macro | Recall macro | F1 macro | AUC OvR macro | ECE ↓ | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | VGG19 | 0,9270 | 0,7416 [0,6977; 0,7826] | 0,5628 | 0,7416 | 0,5889 [0,5667; 0,6093] | 0,8929 | 0,0463 | 0,2464 |
| 2 | AlexNet | 0,9044 | 0,8065 [0,7661; 0,8455] | 0,5596 | 0,8065 | 0,5808 [0,5618; 0,5995] | 0,9089 | 0,0295 | 0,2703 |
| 3 | ViT-B/16 | 0,9059 | 0,7791 [0,7343; 0,8174] | 0,5560 | 0,7791 | 0,5756 [0,5553; 0,5938] | 0,9022 | 0,0231 | 0,2500 |
| 4 | ResNet-50 | 0,8941 | 0,7812 [0,7378; 0,8229] | 0,5509 | 0,7812 | 0,5651 [0,5471; 0,5824] | 0,9014 | 0,0192 | 0,2393 |
| 5 | EfficientNet-B0 | 0,8683 | 0,8123 [0,7727; 0,8474] | 0,5464 | 0,8123 | 0,5509 [0,5348; 0,5664] | 0,8997 | 0,0280 | 0,2407 |
| 6 | SimSiam ResNet-50 | 0,8691 | 0,8046 [0,7637; 0,8401] | 0,5456 | 0,8046 | 0,5500 [0,5343; 0,5649] | 0,9012 | 0,0279 | 0,2357 |
| 7 | DINOv2 ViT-B/14 | 0,9242 | 0,6115 [0,5725; 0,6528] | 0,5316 | 0,6115 | 0,5411 [0,5228; 0,5591] | 0,7820 | 0,2046 | 0,1187 |
| 8 | DINOv2 ViT-S/14 | 0,8117 | 0,7392 [0,6980; 0,7804] | 0,5275 | 0,7392 | 0,5045 [0,4928; 0,5169] | 0,8301 | 0,0918 | 0,1621 |

## Sensibilidad (cómputo no canónico: batch 16, transductivo)

Orden: F1 macro de mayor a menor; las filas sin número van al final.

### Condición

| # | Modelo | Preentrenamiento | Fine-tuning | Segmentación | Transductivo | Val n | Estado | Nota | Fuente |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A71 original (batch 16) | auto-supervisado propio (DINOv2 + SimSiam anclado sin término de tono, A*) | no | no | sí | 6.660 | ok | sensibilidad: cómputo no canónico (batch 16); transductivo | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/A71__original.json` · `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/A71__original.probas.npz` |
| 2 | B71 original (batch 16) | auto-supervisado propio (DINOv2 + SimSiam anclado + término de tono, B*) | no | no | sí | 6.660 | ok | sensibilidad: cómputo no canónico (batch 16); transductivo | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/B71__original.json` · `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/B71__original.probas.npz` |
| 3 | DINOv2 público, protocolo de A71 (batch 16) | auto-supervisado (DINOv2 público) | no | no | no | 6.660 | ok | sensibilidad: cómputo no canónico (batch 16) | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/PUBLIC_A71.json` · `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/PUBLIC_A71.probas.npz` |
| 4 | DINOv2 público, protocolo de B71 (batch 16) | auto-supervisado (DINOv2 público) | no | no | no | 6.660 | ok | sensibilidad: cómputo no canónico (batch 16) | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/PUBLIC_B71.json` · `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/PUBLIC_B71.probas.npz` |

### Métricas [IC 95 % donde la fuente lo publica]

| # | Modelo | Accuracy | Exactitud balanceada | Precisión macro | Recall macro | F1 macro | AUC OvR macro | ECE ↓ | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A71 original (batch 16) | 0,9685 [0,9620; 0,9739] | 0,6984 [0,6527; 0,7511] | 0,6287 | 0,6984 | 0,6554 [0,6197; 0,6910] | 0,9037 [0,8751; 0,9301] | 0,0356 [0,0310; 0,0420] | 0,3196 [0,2468; 0,3938] |
| 2 | B71 original (batch 16) | 0,9644 [0,9569; 0,9707] | 0,6440 [0,5970; 0,6926] | 0,5930 | 0,6440 | 0,6123 [0,5781; 0,6486] | 0,8524 [0,8080; 0,8947] | 0,0112 [0,0087; 0,0167] | 0,2315 [0,1624; 0,3043] |
| 3 | DINOv2 público, protocolo de A71 (batch 16) | 0,9640 [0,9572; 0,9700] | 0,6277 [0,5858; 0,6686] | 0,5846 | 0,6277 | 0,6012 [0,5673; 0,6331] | 0,8326 [0,7903; 0,8724] | 0,0121 [0,0088; 0,0172] | 0,2079 [0,1392; 0,2719] |
| 4 | DINOv2 público, protocolo de B71 (batch 16) | 0,9703 [0,9640; 0,9758] | 0,5786 [0,5454; 0,6136] | 0,5813 | 0,5786 | 0,5799 [0,5458; 0,6126] | 0,8799 [0,8482; 0,9075] | 0,0128 [0,0102; 0,0179] | 0,1599 [0,0919; 0,2252] |

## Contrastes pareados

Bootstrap pareado de 1.000 remuestras, unidad: patient_id. Los baselines (congelados y con fine-tuning) no tienen probabilidades por imagen commiteadas: su comparación con B71-limpio es solo de intervalos (IC 95 % de cada fuente), no un test pareado.

### Canónico (batch 64 + autocast)

| Brazo | Control | Métrica | Δ (brazo − control) [IC 95 %] | El IC excluye 0 | Lectura | Unidad | Fuente |
|---|---|---|---|---|---|---|---|
| A71-limpio | A71 original | F1 macro | -0,0076 [-0,0311; +0,0161] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/A71__clean__vs__A71__original.json` |
| A71-limpio | A71 original | Exactitud balanceada | -0,0011 [-0,0295; +0,0291] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/A71__clean__vs__A71__original.json` |
| A71-limpio | A71 original | Accuracy | -0,0021 [-0,0058; +0,0016] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/A71__clean__vs__A71__original.json` |
| A71-limpio | A71 original | AUC OvR macro | +0,0062 [-0,0072; +0,0219] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/A71__clean__vs__A71__original.json` |
| A71-limpio | A71 original | ECE (menor es mejor) | +0,0008 [-0,0023; +0,0048] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/A71__clean__vs__A71__original.json` |
| A71-limpio | A71 original | MCC | -0,0126 [-0,0581; +0,0337] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/A71__clean__vs__A71__original.json` |
| A71-limpio | DINOv2 público (prot. A71) | F1 macro | +0,0497 [+0,0145; +0,0837] | sí | A71-limpio mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/A71__clean__vs__PUBLIC_A71.json` |
| A71-limpio | DINOv2 público (prot. A71) | Exactitud balanceada | +0,0701 [+0,0225; +0,1208] | sí | A71-limpio mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/A71__clean__vs__PUBLIC_A71.json` |
| A71-limpio | DINOv2 público (prot. A71) | Accuracy | +0,0033 [-0,0016; +0,0077] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/A71__clean__vs__PUBLIC_A71.json` |
| A71-limpio | DINOv2 público (prot. A71) | AUC OvR macro | +0,0762 [+0,0444; +0,1078] | sí | A71-limpio mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/A71__clean__vs__PUBLIC_A71.json` |
| A71-limpio | DINOv2 público (prot. A71) | ECE (menor es mejor) | +0,0220 [+0,0153; +0,0281] | sí | DINOv2 público (prot. A71) mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/A71__clean__vs__PUBLIC_A71.json` |
| A71-limpio | DINOv2 público (prot. A71) | MCC | +0,1042 [+0,0324; +0,1757] | sí | A71-limpio mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/A71__clean__vs__PUBLIC_A71.json` |
| A71 original | DINOv2 público (prot. A71) | F1 macro | +0,0573 [+0,0214; +0,0896] | sí | A71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/A71__original__vs__PUBLIC_A71.json` |
| A71 original | DINOv2 público (prot. A71) | Exactitud balanceada | +0,0711 [+0,0249; +0,1174] | sí | A71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/A71__original__vs__PUBLIC_A71.json` |
| A71 original | DINOv2 público (prot. A71) | Accuracy | +0,0054 [+0,0006; +0,0105] | sí | A71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/A71__original__vs__PUBLIC_A71.json` |
| A71 original | DINOv2 público (prot. A71) | AUC OvR macro | +0,0700 [+0,0371; +0,1054] | sí | A71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/A71__original__vs__PUBLIC_A71.json` |
| A71 original | DINOv2 público (prot. A71) | ECE (menor es mejor) | +0,0212 [+0,0143; +0,0264] | sí | DINOv2 público (prot. A71) mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/A71__original__vs__PUBLIC_A71.json` |
| A71 original | DINOv2 público (prot. A71) | MCC | +0,1168 [+0,0451; +0,1828] | sí | A71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/A71__original__vs__PUBLIC_A71.json` |
| B71-limpio | A71-limpio | F1 macro | -0,0238 [-0,0597; +0,0091] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__A71__clean.json` |
| B71-limpio | A71-limpio | Exactitud balanceada | -0,0440 [-0,0929; -0,0008] | sí | A71-limpio mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__A71__clean.json` |
| B71-limpio | A71-limpio | Accuracy | +0,0005 [-0,0043; +0,0052] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__A71__clean.json` |
| B71-limpio | A71-limpio | AUC OvR macro | -0,0327 [-0,0595; -0,0102] | sí | A71-limpio mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__A71__clean.json` |
| B71-limpio | A71-limpio | ECE (menor es mejor) | -0,0228 [-0,0293; -0,0154] | sí | B71-limpio mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__A71__clean.json` |
| B71-limpio | A71-limpio | MCC | -0,0534 [-0,1272; +0,0139] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__A71__clean.json` |
| B71-limpio | B71 original | F1 macro | +0,0093 [-0,0224; +0,0409] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__B71__original.json` |
| B71-limpio | B71 original | Exactitud balanceada | +0,0052 [-0,0391; +0,0461] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__B71__original.json` |
| B71-limpio | B71 original | Accuracy | +0,0023 [-0,0016; +0,0064] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__B71__original.json` |
| B71-limpio | B71 original | AUC OvR macro | +0,0230 [-0,0079; +0,0574] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__B71__original.json` |
| B71-limpio | B71 original | ECE (menor es mejor) | -0,0011 [-0,0052; +0,0035] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__B71__original.json` |
| B71-limpio | B71 original | MCC | +0,0169 [-0,0494; +0,0805] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__B71__original.json` |
| B71-limpio | DINOv2 público (prot. B71) | F1 macro | +0,0387 [+0,0047; +0,0721] | sí | B71-limpio mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__PUBLIC_B71.json` |
| B71-limpio | DINOv2 público (prot. B71) | Exactitud balanceada | +0,0666 [+0,0302; +0,1033] | sí | B71-limpio mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__PUBLIC_B71.json` |
| B71-limpio | DINOv2 público (prot. B71) | Accuracy | -0,0035 [-0,0084; +0,0010] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__PUBLIC_B71.json` |
| B71-limpio | DINOv2 público (prot. B71) | AUC OvR macro | -0,0032 [-0,0346; +0,0262] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__PUBLIC_B71.json` |
| B71-limpio | DINOv2 público (prot. B71) | ECE (menor es mejor) | -0,0013 [-0,0054; +0,0026] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__PUBLIC_B71.json` |
| B71-limpio | DINOv2 público (prot. B71) | MCC | +0,0822 [+0,0162; +0,1486] | sí | B71-limpio mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__PUBLIC_B71.json` |
| B71 original | A71 original | F1 macro | -0,0408 [-0,0704; -0,0080] | sí | A71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__original__vs__A71__original.json` |
| B71 original | A71 original | Exactitud balanceada | -0,0503 [-0,0919; -0,0054] | sí | A71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__original__vs__A71__original.json` |
| B71 original | A71 original | Accuracy | -0,0039 [-0,0088; +0,0009] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__original__vs__A71__original.json` |
| B71 original | A71 original | AUC OvR macro | -0,0494 [-0,0825; -0,0180] | sí | A71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__original__vs__A71__original.json` |
| B71 original | A71 original | ECE (menor es mejor) | -0,0209 [-0,0269; -0,0136] | sí | B71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__original__vs__A71__original.json` |
| B71 original | A71 original | MCC | -0,0830 [-0,1437; -0,0149] | sí | A71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__original__vs__A71__original.json` |
| B71 original | DINOv2 público (prot. B71) | F1 macro | +0,0294 [-0,0024; +0,0615] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__original__vs__PUBLIC_B71.json` |
| B71 original | DINOv2 público (prot. B71) | Exactitud balanceada | +0,0614 [+0,0243; +0,1026] | sí | B71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__original__vs__PUBLIC_B71.json` |
| B71 original | DINOv2 público (prot. B71) | Accuracy | -0,0057 [-0,0102; -0,0012] | sí | DINOv2 público (prot. B71) mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__original__vs__PUBLIC_B71.json` |
| B71 original | DINOv2 público (prot. B71) | AUC OvR macro | -0,0261 [-0,0636; +0,0065] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__original__vs__PUBLIC_B71.json` |
| B71 original | DINOv2 público (prot. B71) | ECE (menor es mejor) | -0,0003 [-0,0048; +0,0034] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__original__vs__PUBLIC_B71.json` |
| B71 original | DINOv2 público (prot. B71) | MCC | +0,0652 [+0,0015; +0,1315] | sí | B71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__original__vs__PUBLIC_B71.json` |

### Sensibilidad (batch 16, transductivo)

| Brazo | Control | Métrica | Δ (brazo − control) [IC 95 %] | El IC excluye 0 | Lectura | Unidad | Fuente |
|---|---|---|---|---|---|---|---|
| A71 original | DINOv2 público (prot. A71) | F1 macro | +0,0542 [+0,0210; +0,0867] | sí | A71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/contrasts/A71__original__vs__PUBLIC_A71.json` |
| A71 original | DINOv2 público (prot. A71) | Exactitud balanceada | +0,0707 [+0,0252; +0,1189] | sí | A71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/contrasts/A71__original__vs__PUBLIC_A71.json` |
| A71 original | DINOv2 público (prot. A71) | Accuracy | +0,0045 [+0,0000; +0,0092] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/contrasts/A71__original__vs__PUBLIC_A71.json` |
| A71 original | DINOv2 público (prot. A71) | AUC OvR macro | +0,0710 [+0,0385; +0,1063] | sí | A71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/contrasts/A71__original__vs__PUBLIC_A71.json` |
| A71 original | DINOv2 público (prot. A71) | ECE (menor es mejor) | +0,0235 [+0,0167; +0,0294] | sí | DINOv2 público (prot. A71) mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/contrasts/A71__original__vs__PUBLIC_A71.json` |
| A71 original | DINOv2 público (prot. A71) | MCC | +0,1117 [+0,0439; +0,1784] | sí | A71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/contrasts/A71__original__vs__PUBLIC_A71.json` |
| B71 original | A71 original | F1 macro | -0,0431 [-0,0734; -0,0079] | sí | A71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/contrasts/B71__original__vs__A71__original.json` |
| B71 original | A71 original | Exactitud balanceada | -0,0543 [-0,0968; -0,0093] | sí | A71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/contrasts/B71__original__vs__A71__original.json` |
| B71 original | A71 original | Accuracy | -0,0041 [-0,0090; +0,0008] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/contrasts/B71__original__vs__A71__original.json` |
| B71 original | A71 original | AUC OvR macro | -0,0512 [-0,0850; -0,0207] | sí | A71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/contrasts/B71__original__vs__A71__original.json` |
| B71 original | A71 original | ECE (menor es mejor) | -0,0245 [-0,0307; -0,0164] | sí | B71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/contrasts/B71__original__vs__A71__original.json` |
| B71 original | A71 original | MCC | -0,0881 [-0,1500; -0,0169] | sí | A71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/contrasts/B71__original__vs__A71__original.json` |
| B71 original | DINOv2 público (prot. B71) | F1 macro | +0,0324 [-0,0006; +0,0644] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/contrasts/B71__original__vs__PUBLIC_B71.json` |
| B71 original | DINOv2 público (prot. B71) | Exactitud balanceada | +0,0654 [+0,0275; +0,1076] | sí | B71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/contrasts/B71__original__vs__PUBLIC_B71.json` |
| B71 original | DINOv2 público (prot. B71) | Accuracy | -0,0059 [-0,0103; -0,0015] | sí | DINOv2 público (prot. B71) mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/contrasts/B71__original__vs__PUBLIC_B71.json` |
| B71 original | DINOv2 público (prot. B71) | AUC OvR macro | -0,0274 [-0,0664; +0,0054] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/contrasts/B71__original__vs__PUBLIC_B71.json` |
| B71 original | DINOv2 público (prot. B71) | ECE (menor es mejor) | -0,0016 [-0,0058; +0,0028] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/contrasts/B71__original__vs__PUBLIC_B71.json` |
| B71 original | DINOv2 público (prot. B71) | MCC | +0,0716 [+0,0066; +0,1375] | sí | B71 original mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/contrasts/B71__original__vs__PUBLIC_B71.json` |

### B71-limpio contra el fine-tuning (pareado por id, ISIC 2020)

Sección: análisis posterior a la ejecución (post-hoc), no parte del registro original. Probas: probas de val reconstruidas por inferencia desde el checkpoint de la mejor epoca; verificadas contra las metricas registradas. checkpoint_epoch=null; identidad del checkpoint inferida por coincidencia de métricas. El fine-tuning usa la mejor época elegida en val (optimista); B71-limpio no tiene fine-tuning ni selección de época en val. La época se tomó de `mejor_epoca` del JSON del fine-tuning (el .pth no la trae); el sha256 del checkpoint está en el sidecar. Pareo por id de imagen, bootstrap por `patient_id`, 1.000 remuestras, semilla 20260923. p bootstrap aproximado = 2·min(P(Δ≤0), P(Δ≥0)) sobre las 1.000 remuestras pareadas, con piso 1/1000 (no es un p exacto). Holm sobre los 8 modelos de cada métrica; un modelo ausente cuenta como p = 1.

| Control (fine-tuning) | Métrica | Δ (B71-limpio − FT) [IC 95 %] | El IC excluye 0 | Lectura | p bootstrap | p Holm |
|---|---|---|---|---|---|---|
| AlexNet (fine-tuning) | F1 macro | +0,0421 [+0,0071; +0,0750] | sí | B71-limpio (canónico) mejor (el IC excluye 0) | 0,0180 | 0,0360 |
| VGG19 (fine-tuning) | F1 macro | +0,0340 [-0,0016; +0,0671] | no | sin diferencia demostrable (el IC incluye 0) | 0,0660 | 0,0660 |
| ResNet-50 (fine-tuning) | F1 macro | +0,0578 [+0,0256; +0,0912] | sí | B71-limpio (canónico) mejor (el IC excluye 0) | 0,0010 | 0,0080 |
| EfficientNet-B0 (fine-tuning) | F1 macro | +0,0720 [+0,0381; +0,1037] | sí | B71-limpio (canónico) mejor (el IC excluye 0) | 0,0010 | 0,0080 |
| ViT-B/16 (fine-tuning) | F1 macro | +0,0473 [+0,0123; +0,0788] | sí | B71-limpio (canónico) mejor (el IC excluye 0) | 0,0040 | 0,0120 |
| SimSiam ResNet-50 (fine-tuning) | F1 macro | +0,0729 [+0,0390; +0,1068] | sí | B71-limpio (canónico) mejor (el IC excluye 0) | 0,0010 | 0,0080 |
| DINOv2 ViT-S/14 (fine-tuning) | F1 macro | +0,1184 [+0,0823; +0,1545] | sí | B71-limpio (canónico) mejor (el IC excluye 0) | 0,0010 | 0,0080 |
| DINOv2 ViT-B/14 (fine-tuning) | F1 macro | +0,0818 [+0,0407; +0,1209] | sí | B71-limpio (canónico) mejor (el IC excluye 0) | 0,0010 | 0,0080 |
| AlexNet (fine-tuning) | Exactitud balanceada | -0,1571 [-0,2134; -0,1015] | sí | AlexNet (fine-tuning) mejor (el IC excluye 0) | 0,0010 | 0,0080 |
| VGG19 (fine-tuning) | Exactitud balanceada | -0,0922 [-0,1526; -0,0392] | sí | VGG19 (fine-tuning) mejor (el IC excluye 0) | 0,0010 | 0,0080 |
| ResNet-50 (fine-tuning) | Exactitud balanceada | -0,1318 [-0,1795; -0,0852] | sí | ResNet-50 (fine-tuning) mejor (el IC excluye 0) | 0,0010 | 0,0080 |
| EfficientNet-B0 (fine-tuning) | Exactitud balanceada | -0,1628 [-0,2179; -0,1127] | sí | EfficientNet-B0 (fine-tuning) mejor (el IC excluye 0) | 0,0010 | 0,0080 |
| ViT-B/16 (fine-tuning) | Exactitud balanceada | -0,1297 [-0,1835; -0,0801] | sí | ViT-B/16 (fine-tuning) mejor (el IC excluye 0) | 0,0010 | 0,0080 |
| SimSiam ResNet-50 (fine-tuning) | Exactitud balanceada | -0,1552 [-0,2080; -0,1039] | sí | SimSiam ResNet-50 (fine-tuning) mejor (el IC excluye 0) | 0,0010 | 0,0080 |
| DINOv2 ViT-S/14 (fine-tuning) | Exactitud balanceada | -0,0898 [-0,1573; -0,0283] | sí | DINOv2 ViT-S/14 (fine-tuning) mejor (el IC excluye 0) | 0,0100 | 0,0200 |
| DINOv2 ViT-B/14 (fine-tuning) | Exactitud balanceada | +0,0380 [-0,0328; +0,1066] | no | sin diferencia demostrable (el IC incluye 0) | 0,2920 | 0,2920 |
| AlexNet (fine-tuning) | AUC OvR macro | -0,0330 [-0,0630; -0,0044] | sí | AlexNet (fine-tuning) mejor (el IC excluye 0) | 0,0220 | 0,1540 |
| VGG19 (fine-tuning) | AUC OvR macro | -0,0171 [-0,0494; +0,0123] | no | sin diferencia demostrable (el IC incluye 0) | 0,2920 | 0,2960 |
| ResNet-50 (fine-tuning) | AUC OvR macro | -0,0256 [-0,0545; +0,0024] | no | sin diferencia demostrable (el IC incluye 0) | 0,0880 | 0,2960 |
| EfficientNet-B0 (fine-tuning) | AUC OvR macro | -0,0239 [-0,0547; +0,0028] | no | sin diferencia demostrable (el IC incluye 0) | 0,1120 | 0,2960 |
| ViT-B/16 (fine-tuning) | AUC OvR macro | -0,0264 [-0,0583; -0,0003] | sí | ViT-B/16 (fine-tuning) mejor (el IC excluye 0) | 0,0480 | 0,2880 |
| SimSiam ResNet-50 (fine-tuning) | AUC OvR macro | -0,0254 [-0,0588; +0,0018] | no | sin diferencia demostrable (el IC incluye 0) | 0,0740 | 0,2960 |
| DINOv2 ViT-S/14 (fine-tuning) | AUC OvR macro | +0,0457 [+0,0006; +0,0865] | sí | B71-limpio (canónico) mejor (el IC excluye 0) | 0,0480 | 0,2880 |
| DINOv2 ViT-B/14 (fine-tuning) | AUC OvR macro | +0,0939 [+0,0390; +0,1506] | sí | B71-limpio (canónico) mejor (el IC excluye 0) | 0,0010 | 0,0080 |
| AlexNet (fine-tuning) | MCC | -0,0197 [-0,0949; +0,0491] | no | sin diferencia demostrable (el IC incluye 0) | 0,5400 | 1,0000 |
| VGG19 (fine-tuning) | MCC | +0,0042 [-0,0707; +0,0738] | no | sin diferencia demostrable (el IC incluye 0) | 0,9180 | 1,0000 |
| ResNet-50 (fine-tuning) | MCC | +0,0113 [-0,0531; +0,0781] | no | sin diferencia demostrable (el IC incluye 0) | 0,7700 | 1,0000 |
| EfficientNet-B0 (fine-tuning) | MCC | +0,0099 [-0,0617; +0,0749] | no | sin diferencia demostrable (el IC incluye 0) | 0,7940 | 1,0000 |
| ViT-B/16 (fine-tuning) | MCC | +0,0005 [-0,0687; +0,0614] | no | sin diferencia demostrable (el IC incluye 0) | 0,9400 | 1,0000 |
| SimSiam ResNet-50 (fine-tuning) | MCC | +0,0149 [-0,0569; +0,0815] | no | sin diferencia demostrable (el IC incluye 0) | 0,6820 | 1,0000 |
| DINOv2 ViT-S/14 (fine-tuning) | MCC | +0,0884 [+0,0099; +0,1647] | sí | B71-limpio (canónico) mejor (el IC excluye 0) | 0,0280 | 0,1960 |
| DINOv2 ViT-B/14 (fine-tuning) | MCC | +0,1318 [+0,0415; +0,2197] | sí | B71-limpio (canónico) mejor (el IC excluye 0) | 0,0060 | 0,0480 |

### Curva por época del fine-tuning (ISIC 2020)

Receta existente sin cambios; con 1,8 % de positivos la bal_acc en val oscila entre épocas; se reporta la mejor época en val (optimista).

Esta ronda corrió en nuestro pod (GPU RTX PRO 4000), no en el banco de Karen.

Formato: época: bal_acc en val / pérdida de entrenamiento. La curva sale de `curva_epocas` del JSON o, si no existe, del log por época commiteado.

| Modelo | Fuente de la curva | Curva |
|---|---|---|
| AlexNet | json | 1: 0,7757 / 0,5030; 2: 0,7456 / 0,4440; 3: 0,7825 / 0,4146; 4: 0,7801 / 0,3881; 5: 0,8057 / 0,3586; 6: 0,7868 / 0,3216; 7: 0,8065 / 0,3041; 8: 0,7942 / 0,2962 |
| VGG19 | json | 1: 0,5000 / 0,6923; 2: 0,5000 / 0,4523; 3: 0,5115 / 0,4108; 4: 0,5620 / 0,3919; 5: 0,6196 / 0,3661; 6: 0,7334 / 0,3443; 7: 0,7397 / 0,3337; 8: 0,7416 / 0,3233 |
| ResNet-50 | json | 1: 0,7489 / 0,4618; 2: 0,7657 / 0,3851; 3: 0,7811 / 0,3472; 4: 0,7692 / 0,3076; 5: 0,7384 / 0,2539; 6: 0,7599 / 0,2351; 7: 0,7505 / 0,1968; 8: 0,7357 / 0,1841 |
| EfficientNet-B0 | json | 1: 0,7466 / 0,4860; 2: 0,8123 / 0,3568; 3: 0,8009 / 0,3058; 4: 0,7690 / 0,2470; 5: 0,7260 / 0,1914; 6: 0,7161 / 0,1539; 7: 0,7268 / 0,1352; 8: 0,7249 / 0,1229 |
| ViT-B/16 | json | 1: 0,6542 / 0,4486; 2: 0,6495 / 0,4079; 3: 0,7791 / 0,3708; 4: 0,7367 / 0,3265; 5: 0,6904 / 0,3034; 6: 0,7454 / 0,2440; 7: 0,7001 / 0,1841; 8: 0,6822 / 0,1545 |
| SimSiam ResNet-50 | json | 1: 0,6057 / 0,5331; 2: 0,6931 / 0,4968; 3: 0,6784 / 0,4678; 4: 0,7322 / 0,4556; 5: 0,7181 / 0,4332; 6: 0,7879 / 0,4316; 7: 0,7868 / 0,4131; 8: 0,8046 / 0,3994 |
| DINOv2 ViT-S/14 | json | 1: 0,5000 / 0,6907; 2: 0,5037 / 0,5506; 3: 0,5645 / 0,5280; 4: 0,5097 / 0,5123; 5: 0,7392 / 0,4704; 6: 0,5528 / 0,4656; 7: 0,6929 / 0,4615; 8: 0,7232 / 0,4528 |
| DINOv2 ViT-B/14 | log | 1: 0,5000 / 0,5195; 2: 0,5000 / 0,4623; 3: 0,6114 / 0,4442; 4: 0,5431 / 0,4315; 5: 0,5077 / 0,4173; 6: 0,5369 / 0,4035; 7: 0,5431 / 0,3843; 8: 0,5315 / 0,3829 |
