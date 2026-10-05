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
- **Optimista: época elegida en val.** Todas las filas de fine-tuning (HAM10000, ISIC 2019 y la corrida pendiente de ISIC 2020) eligen su mejor época por balanced accuracy sobre val; el plan decía holdout interno y se mantuvo la receta existente por consistencia entre datasets (errata de docs/33). El propuesto no tiene fine-tuning y no se ve afectado.
- **Modelo y protocolo no son lo mismo.** El propuesto usa el protocolo de B71 (features cls4, TTA, probe balanceado); la fila «DINOv2 público, protocolo de B71» es el control con el mismo protocolo y pesos públicos. Los 8 baselines siguen el protocolo de su propio banco (probe logreg).
- Los IC solo existen donde la fuente los publica: los baselines, para exactitud balanceada y F1 macro; B71/A71, para accuracy, exactitud balanceada, F1 macro, AUC y ECE. Precisión macro, recall macro y MCC de los baselines salen exactos de la matriz de confusión y van sin IC; en B71/A71 salen de las probabilidades commiteadas, también sin IC.
- Las filas «pendiente» no tienen número: se listan con su motivo técnico y la ETA declarada por el operador, para que nada parezca olvidado. No se inventa ninguna cifra.

## Congelados (sin fine-tuning)

Orden: F1 macro de mayor a menor; las filas sin número van al final.

### Condición

| # | Modelo | Preentrenamiento | Fine-tuning | Segmentación | Transductivo | Val n | Estado | Nota | Fuente |
|---|---|---|---|---|---|---|---|---|---|
| 1 | B71-limpio (canónico) | auto-supervisado propio (DINOv2 + SimSiam anclado + término de tono, B*) | no | no | no | 6.660 | ok | cómputo canónico (batch 64 + autocast) | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/B71__clean.json` · `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/B71__clean.probas.npz` |
| 2 | DINOv2 público, protocolo de A71 (canónico) | auto-supervisado (DINOv2 público) | no | no | no | 6.660 | ok | cómputo canónico (batch 64 + autocast) | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/PUBLIC_A71.json` · `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/PUBLIC_A71.probas.npz` |
| 3 | DINOv2 público, protocolo de B71 (canónico) | auto-supervisado (DINOv2 público) | no | no | no | 6.660 | ok | cómputo canónico (batch 64 + autocast) | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/PUBLIC_B71.json` · `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/PUBLIC_B71.probas.npz` |
| 4 | ResNet-50 | supervisado (ImageNet) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_resnet50_isic2020.json` |
| 5 | ViT-B/16 | supervisado (ImageNet) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_vit_b16_isic2020.json` |
| 6 | EfficientNet-B0 | supervisado (ImageNet) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_efficientnet_b0_isic2020.json` |
| 7 | DINOv2 ViT-B/14 | auto-supervisado (DINOv2 público) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_dinov2_vitb14_isic2020.json` |
| 8 | DINOv2 ViT-S/14 | auto-supervisado (DINOv2 público) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_dinov2_vits14_isic2020.json` |
| 9 | VGG19 | supervisado (ImageNet) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_vgg19_isic2020.json` |
| 10 | AlexNet | supervisado (ImageNet) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_alexnet_isic2020.json` |
| 11 | SimSiam ResNet-50 | auto-supervisado (SimSiam, ImageNet) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_simsiam_r50_isic2020.json` |
| — | A71-limpio (canónico) | auto-supervisado propio (DINOv2 + SimSiam anclado sin término de tono, A*) | no | no | no | — | pendiente | falta A* s43 limpio (GPU0 en curso); A71-limpio es la sopa de A* s42 + s43 limpios; ETA ~15:00 UTC | `docs/33-banco-completo-multidataset.md` |
| — | B71 original (canónico) | auto-supervisado propio (DINOv2 + SimSiam anclado + término de tono, B*) | no | no | sí | — | pendiente | re-extracción de B71/A71 originales con el cómputo canónico (batch 64 + autocast) en GPU0 después de A* s43; hoy solo existe la corrida de batch 16; ETA ~15:00 UTC | `docs/33-banco-completo-multidataset.md` |
| — | A71 original (canónico) | auto-supervisado propio (DINOv2 + SimSiam anclado sin término de tono, A*) | no | no | sí | — | pendiente | re-extracción de B71/A71 originales con el cómputo canónico (batch 64 + autocast) en GPU0 después de A* s43; hoy solo existe la corrida de batch 16; ETA ~15:00 UTC | `docs/33-banco-completo-multidataset.md` |

### Métricas [IC 95 % donde la fuente lo publica]

| # | Modelo | Accuracy | Exactitud balanceada | Precisión macro | Recall macro | F1 macro | AUC OvR macro | ECE ↓ | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | B71-limpio (canónico) | 0,9671 [0,9600; 0,9731] | 0,6494 [0,6052; 0,6926] | 0,6050 | 0,6494 | 0,6229 [0,5865; 0,6602] | 0,8758 [0,8397; 0,9049] | 0,0111 [0,0087; 0,0164] | 0,2506 [0,1766; 0,3269] |
| 2 | DINOv2 público, protocolo de A71 (canónico) | 0,9634 [0,9563; 0,9694] | 0,6234 [0,5829; 0,6631] | 0,5809 | 0,6234 | 0,5971 [0,5636; 0,6272] | 0,8323 [0,7904; 0,8719] | 0,0119 [0,0086; 0,0170] | 0,1998 [0,1323; 0,2622] |
| 3 | DINOv2 público, protocolo de B71 (canónico) | 0,9706 [0,9643; 0,9759] | 0,5828 [0,5485; 0,6185] | 0,5856 | 0,5828 | 0,5842 [0,5507; 0,6181] | 0,8790 [0,8469; 0,9073] | 0,0125 [0,0100; 0,0177] | 0,1684 [0,1015; 0,2382] |
| 4 | ResNet-50 | 0,9161 | 0,6878 [0,6425; 0,7321] | 0,5448 | 0,6878 | 0,5601 [0,5396; 0,5790] | 0,8297 | 0,0275 | 0,1834 |
| 5 | ViT-B/16 | 0,8883 | 0,7340 [0,6903; 0,7777] | 0,5415 | 0,7340 | 0,5493 [0,5319; 0,5668] | 0,8275 | 0,0598 | 0,1971 |
| 6 | EfficientNet-B0 | 0,9186 | 0,6368 [0,5954; 0,6818] | 0,5352 | 0,6368 | 0,5457 [0,5264; 0,5640] | 0,7607 | 0,0378 | 0,1387 |
| 7 | DINOv2 ViT-B/14 | 0,8994 | 0,6753 [0,6284; 0,7186] | 0,5355 | 0,6753 | 0,5427 [0,5252; 0,5577] | 0,7999 | 0,0559 | 0,1578 |
| 8 | DINOv2 ViT-S/14 | 0,8470 | 0,7572 [0,7132; 0,7981] | 0,5346 | 0,7572 | 0,5266 [0,5119; 0,5393] | 0,8576 | 0,0561 | 0,1886 |
| 9 | VGG19 | 0,8233 | 0,7370 [0,6951; 0,7789] | 0,5286 | 0,7370 | 0,5101 [0,4978; 0,5234] | 0,8128 | 0,0333 | 0,1647 |
| 10 | AlexNet | 0,7940 | 0,7744 [0,7354; 0,8125] | 0,5292 | 0,7744 | 0,5008 [0,4894; 0,5122] | 0,8574 | 0,0442 | 0,1791 |
| 11 | SimSiam ResNet-50 | 0,7518 | 0,7650 [0,7301; 0,8000] | 0,5249 | 0,7650 | 0,4795 [0,4693; 0,4906] | 0,8468 | 0,0851 | 0,1624 |
| — | A71-limpio (canónico) | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |
| — | B71 original (canónico) | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |
| — | A71 original (canónico) | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |

## Con fine-tuning

Orden: F1 macro de mayor a menor; las filas sin número van al final.

### Condición

| # | Modelo | Preentrenamiento | Fine-tuning | Segmentación | Transductivo | Val n | Estado | Nota | Fuente |
|---|---|---|---|---|---|---|---|---|---|
| 1 | DINOv2 ViT-B/14 | auto-supervisado (DINOv2 público) | sí | no | no | 6.660 | ok | época elegida en val (sesgo optimista); mejor época 3 | `resultados/banco_completo/ft_isic2020/json/results_dinov2_vitb14_ft_isic2020.json` |
| — | AlexNet | supervisado (ImageNet) | sí | no | no | — | pendiente | el fine-tuning de los 8 baselines en ISIC 2020 corre en GPU1 después de E2; ETA inicio al terminar E2 (~10:30 UTC); duración no medida; optimista: época elegida en val (receta existente) | `docs/33-banco-completo-multidataset.md` |
| — | VGG19 | supervisado (ImageNet) | sí | no | no | — | pendiente | el fine-tuning de los 8 baselines en ISIC 2020 corre en GPU1 después de E2; ETA inicio al terminar E2 (~10:30 UTC); duración no medida; optimista: época elegida en val (receta existente) | `docs/33-banco-completo-multidataset.md` |
| — | ResNet-50 | supervisado (ImageNet) | sí | no | no | — | pendiente | el fine-tuning de los 8 baselines en ISIC 2020 corre en GPU1 después de E2; ETA inicio al terminar E2 (~10:30 UTC); duración no medida; optimista: época elegida en val (receta existente) | `docs/33-banco-completo-multidataset.md` |
| — | EfficientNet-B0 | supervisado (ImageNet) | sí | no | no | — | pendiente | el fine-tuning de los 8 baselines en ISIC 2020 corre en GPU1 después de E2; ETA inicio al terminar E2 (~10:30 UTC); duración no medida; optimista: época elegida en val (receta existente) | `docs/33-banco-completo-multidataset.md` |
| — | ViT-B/16 | supervisado (ImageNet) | sí | no | no | — | pendiente | el fine-tuning de los 8 baselines en ISIC 2020 corre en GPU1 después de E2; ETA inicio al terminar E2 (~10:30 UTC); duración no medida; optimista: época elegida en val (receta existente) | `docs/33-banco-completo-multidataset.md` |
| — | SimSiam ResNet-50 | auto-supervisado (SimSiam, ImageNet) | sí | no | no | — | pendiente | el fine-tuning de los 8 baselines en ISIC 2020 corre en GPU1 después de E2; ETA inicio al terminar E2 (~10:30 UTC); duración no medida; optimista: época elegida en val (receta existente) | `docs/33-banco-completo-multidataset.md` |
| — | DINOv2 ViT-S/14 | auto-supervisado (DINOv2 público) | sí | no | no | — | pendiente | el fine-tuning de los 8 baselines en ISIC 2020 corre en GPU1 después de E2; ETA inicio al terminar E2 (~10:30 UTC); duración no medida; optimista: época elegida en val (receta existente) | `docs/33-banco-completo-multidataset.md` |

### Métricas [IC 95 % donde la fuente lo publica]

| # | Modelo | Accuracy | Exactitud balanceada | Precisión macro | Recall macro | F1 macro | AUC OvR macro | ECE ↓ | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | DINOv2 ViT-B/14 | 0,9242 | 0,6115 [0,5725; 0,6528] | 0,5316 | 0,6115 | 0,5411 [0,5228; 0,5591] | 0,7820 | 0,2046 | 0,1187 |
| — | AlexNet | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |
| — | VGG19 | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |
| — | ResNet-50 | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |
| — | EfficientNet-B0 | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |
| — | ViT-B/16 | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |
| — | SimSiam ResNet-50 | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |
| — | DINOv2 ViT-S/14 | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |

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
| B71-limpio | DINOv2 público (prot. B71) | F1 macro | +0,0387 [+0,0047; +0,0721] | sí | B71-limpio mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__PUBLIC_B71.json` |
| B71-limpio | DINOv2 público (prot. B71) | Exactitud balanceada | +0,0666 [+0,0302; +0,1033] | sí | B71-limpio mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__PUBLIC_B71.json` |
| B71-limpio | DINOv2 público (prot. B71) | Accuracy | -0,0035 [-0,0084; +0,0010] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__PUBLIC_B71.json` |
| B71-limpio | DINOv2 público (prot. B71) | AUC OvR macro | -0,0032 [-0,0346; +0,0262] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__PUBLIC_B71.json` |
| B71-limpio | DINOv2 público (prot. B71) | ECE (menor es mejor) | -0,0013 [-0,0054; +0,0026] | no | sin diferencia demostrable (el IC incluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__PUBLIC_B71.json` |
| B71-limpio | DINOv2 público (prot. B71) | MCC | +0,0822 [+0,0162; +0,1486] | sí | B71-limpio mejor (el IC excluye 0) | patient_id | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/B71__clean__vs__PUBLIC_B71.json` |

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

### Pendientes

- B71-limpio contra B71 original: pendiente; el contraste B71-limpio contra B71 original exige el original con el mismo cómputo canónico; la corrida de batch 16 no es comparable (cambia el cómputo, no solo los datos); ETA ~15:00 UTC.
- A71-limpio contra A71 original y B71-limpio contra A71-limpio: pendiente; falta A* s43 limpio (GPU0 en curso); A71-limpio es la sopa de A* s42 + s43 limpios; ETA ~15:00 UTC.

### Curva por época del fine-tuning (ISIC 2020)

Receta existente sin cambios; con 1,8 % de positivos la bal_acc en val oscila entre épocas; se reporta la mejor época en val (optimista).

Esta ronda corrió en nuestro pod (GPU RTX PRO 4000), no en el banco de Karen.

Formato: época: bal_acc en val / pérdida de entrenamiento. La curva sale de `curva_epocas` del JSON o, si no existe, del log por época commiteado.

| Modelo | Fuente de la curva | Curva |
|---|---|---|
| AlexNet | — | pendiente: el fine-tuning no ha terminado |
| VGG19 | — | pendiente: el fine-tuning no ha terminado |
| ResNet-50 | — | pendiente: el fine-tuning no ha terminado |
| EfficientNet-B0 | — | pendiente: el fine-tuning no ha terminado |
| ViT-B/16 | — | pendiente: el fine-tuning no ha terminado |
| SimSiam ResNet-50 | — | pendiente: el fine-tuning no ha terminado |
| DINOv2 ViT-S/14 | — | pendiente: el fine-tuning no ha terminado |
| DINOv2 ViT-B/14 | log | 1: 0,5000 / 0,5195; 2: 0,5000 / 0,4623; 3: 0,6114 / 0,4442; 4: 0,5431 / 0,4315; 5: 0,5077 / 0,4173; 6: 0,5369 / 0,4035; 7: 0,5431 / 0,3843; 8: 0,5315 / 0,3829 |
