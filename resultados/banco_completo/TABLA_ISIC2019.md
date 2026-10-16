# Banco completo: ISIC 2019, propuesto y 8 baselines

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
- **Segmentación:** uso de máscaras de lesión como entrada. La columna dice «no» en todas las filas salvo en el bloque exploratorio de HAM10000 (variante Proposed con segmentación (experimental) del bloque E3 de docs/33), que no cambia el propuesto.
- **Transductivo:** el pool de preentrenamiento auto-supervisado contenía las imágenes de val (sin etiquetas). Solo aplica a Proposed sin segmentación / Modelo de control originales en ISIC 2019 e ISIC 2020.
- **Cómputo canónico:** extracción de features con batch 64 y autocast CUDA (el de v7.1). El batch 16 mueve las cifras unas milésimas (orden de reducción en fp16) y se reporta solo como **sensibilidad**.
- **Pesos limpios (sensibilidad):** Proposed sin segmentación / Modelo de control re-preentrenados con un pool SSL sin las imágenes de val de ISIC (docs/33, sección 2). En HAM10000 son un análisis posterior que se lee al lado de la fila registrada de **pesos originales (selección)**; no es selección y no reemplaza al propuesto.
- **Contraste pareado:** diferencia entre dos modelos sobre las mismas imágenes, con IC bootstrap de la diferencia. «Mejor» solo si el IC excluye 0; si lo incluye, «sin diferencia demostrable».
- **Exploratorio:** variantes post-hoc o con selección en otro conjunto; se reportan pero no cambian el propuesto ni se leen como resultado confirmado.

En este val la clase «NV» es el 50,8 % de las imágenes (8 clases): predecir siempre «NV» da una accuracy de 0,5082 y una exactitud balanceada de 0,1250.

## Cómo leer esta tabla

Particiones de val de este banco (no son la misma partición):

- HAM10000, baselines (Kaggle): val n = 2.014; StratifiedGroupKFold por lesion_id (0 lesiones compartidas).
- HAM10000, propuesto (v6/v7): val n = 2.004; partición agrupada por `lesion_id`.
- ISIC 2019, baselines (Kaggle): val n = 5.067; StratifiedKFold estratificado simple SIN grupo de paciente (el release no trae patient_id) - riesgo de fuga entre imagenes del mismo paciente, metricas con sesgo optimista.
- ISIC 2020, baselines (Kaggle): val n = 6.660; StratifiedGroupKFold por patient_id (0 pacientes compartidos).

- **Entre particiones distintas la comparación es solo de intervalos (IC 95 %)**, no un test pareado: HAM10000 usa la partición v6/v7 (n = 2.004) para el propuesto y la de Kaggle (n = 2.014) para los baselines; ISIC 2019 e ISIC 2020 tienen la suya. Los contrastes pareados solo existen donde dos modelos comparten partición e imágenes y hay probabilidades por imagen (docs/33, sección 5); están al final de cada tabla y, todos juntos, en `CONTRASTES_SSL.md` (sin corrección por multiplicidad).
- **Optimista: época elegida en val.** Todas las filas de fine-tuning (HAM10000, ISIC 2019 e ISIC 2020) eligen su mejor época por balanced accuracy sobre val; el plan decía holdout interno y se mantuvo la receta existente por consistencia entre datasets (errata de docs/33). Las de ISIC 2020 corrieron en nuestro pod (GPU RTX PRO 4000), no en el banco de Karen. El propuesto no tiene fine-tuning y no se ve afectado.
- **Modelo y protocolo no son lo mismo.** El propuesto usa el protocolo de Proposed sin segmentación (features cls4, TTA, probe balanceado); la fila «DINOv2 público, protocolo de Proposed sin segmentación» es el control con el mismo protocolo y pesos públicos. Los 8 baselines siguen el protocolo de su propio banco (probe logreg).
- Los IC solo existen donde la fuente los publica: los baselines, para exactitud balanceada y F1 macro; Proposed sin segmentación / Modelo de control, para accuracy, exactitud balanceada, F1 macro, AUC y ECE. Precisión macro, recall macro y MCC de los baselines salen exactos de la matriz de confusión y van sin IC; en Proposed sin segmentación / Modelo de control salen de las probabilidades commiteadas, también sin IC.
- **Pesos limpios en HAM10000: sensibilidad posterior, no es selección.** La fila registrada del propuesto (pesos originales, selección) no se reemplaza. Las filas del bloque «pesos limpios (sensibilidad)» usan la configuración y los umbrales de operación del holdout original, ajustados con los pesos originales y aplicados sin cambios a los limpios (supuesto declarado), con cómputo canónico; sus contrastes pareados (por `lesion_id`, con MCC) están en `CONTRASTES_SSL.md` y `TABLA_ABLACION.md`.
- Las filas «pendiente» no tienen número: se listan con su motivo técnico (sin ETA inventada), para que nada parezca olvidado. No se inventa ninguna cifra.

## Congelados (sin fine-tuning)

Orden: F1 macro de mayor a menor; las filas sin número van al final.

### Condición

| # | Modelo | Preentrenamiento | Fine-tuning | Segmentación | Transductivo | Val n | Estado | Nota | Fuente |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Proposed sin segmentación original (canónico) | auto-supervisado propio (DINOv2 + SimSiam anclado + término de tono, Proposed sin segmentación) | no | no | sí | 5.067 | ok | cómputo canónico (batch 64 + autocast) | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/B71__original.json` · `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/B71__original.probas.npz` |
| 2 | Proposed sin segmentación (pesos limpios) (canónico) | auto-supervisado propio (DINOv2 + SimSiam anclado + término de tono, Proposed sin segmentación) | no | no | no | 5.067 | ok | cómputo canónico (batch 64 + autocast) | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/B71__clean.json` · `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/B71__clean.probas.npz` |
| 3 | DINOv2 público, protocolo de Proposed sin segmentación (canónico) | auto-supervisado (DINOv2 público) | no | no | no | 5.067 | ok | cómputo canónico (batch 64 + autocast) | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/PUBLIC_B71.json` · `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/PUBLIC_B71.probas.npz` |
| 4 | Modelo de control (pesos limpios) (canónico) | auto-supervisado propio (DINOv2 + SimSiam anclado sin término de tono, Modelo de control) | no | no | no | 5.067 | ok | cómputo canónico (batch 64 + autocast) | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/A71__clean.json` · `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/A71__clean.probas.npz` |
| 5 | Modelo de control original (canónico) | auto-supervisado propio (DINOv2 + SimSiam anclado sin término de tono, Modelo de control) | no | no | sí | 5.067 | ok | cómputo canónico (batch 64 + autocast) | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/A71__original.json` · `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/A71__original.probas.npz` |
| 6 | DINOv2 público, protocolo de Modelo de control (canónico) | auto-supervisado (DINOv2 público) | no | no | no | 5.067 | ok | cómputo canónico (batch 64 + autocast) | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/PUBLIC_A71.json` · `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/PUBLIC_A71.probas.npz` |
| 7 | DINOv2 ViT-B/14 | auto-supervisado (DINOv2 público) | no | no | no | 5.067 | ok | — | `results/baselines_isic2019/results_dinov2_vitb14_isic2019.json` |
| 8 | ViT-B/16 | supervisado (ImageNet) | no | no | no | 5.067 | ok | — | `results/baselines_isic2019/results_vit_b16_isic2019.json` |
| 9 | EfficientNet-B0 | supervisado (ImageNet) | no | no | no | 5.067 | ok | — | `results/baselines_isic2019/results_efficientnet_b0_isic2019.json` |
| 10 | DINOv2 ViT-S/14 | auto-supervisado (DINOv2 público) | no | no | no | 5.067 | ok | — | `results/baselines_isic2019/results_dinov2_vits14_isic2019.json` |
| 11 | ResNet-50 | supervisado (ImageNet) | no | no | no | 5.067 | ok | — | `results/baselines_isic2019/results_resnet50_isic2019.json` |
| 12 | AlexNet | supervisado (ImageNet) | no | no | no | 5.067 | ok | — | `results/baselines_isic2019/results_alexnet_isic2019.json` |
| 13 | VGG19 | supervisado (ImageNet) | no | no | no | 5.067 | ok | — | `results/baselines_isic2019/results_vgg19_isic2019.json` |
| 14 | SimSiam ResNet-50 | auto-supervisado (SimSiam, ImageNet) | no | no | no | 5.067 | ok | — | `results/baselines_isic2019/results_simsiam_r50_isic2019.json` |

### Métricas [IC 95 % donde la fuente lo publica]

| # | Modelo | Accuracy | Exactitud balanceada | Precisión macro | Recall macro | F1 macro | AUC OvR macro | ECE ↓ | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Proposed sin segmentación original (canónico) | 0,7681 [0,7569; 0,7799] | 0,6731 [0,6429; 0,7005] | 0,6733 | 0,6731 | 0,6717 [0,6436; 0,6951] | 0,9543 [0,9499; 0,9584] | 0,0480 [0,0394; 0,0593] | 0,6652 [0,6493; 0,6817] |
| 2 | Proposed sin segmentación (pesos limpios) (canónico) | 0,7659 [0,7543; 0,7776] | 0,6761 [0,6493; 0,7055] | 0,6586 | 0,6761 | 0,6662 [0,6401; 0,6911] | 0,9541 [0,9495; 0,9583] | 0,0498 [0,0408; 0,0609] | 0,6628 [0,6465; 0,6791] |
| 3 | DINOv2 público, protocolo de Proposed sin segmentación (canónico) | 0,7580 [0,7464; 0,7703] | 0,6596 [0,6308; 0,6865] | 0,6609 | 0,6596 | 0,6583 [0,6311; 0,6825] | 0,9478 [0,9421; 0,9534] | 0,0706 [0,0610; 0,0818] | 0,6474 [0,6311; 0,6641] |
| 4 | Modelo de control (pesos limpios) (canónico) | 0,7730 [0,7612; 0,7839] | 0,6829 [0,6529; 0,7084] | 0,6322 | 0,6829 | 0,6549 [0,6278; 0,6771] | 0,9548 [0,9506; 0,9586] | 0,0198 [0,0155; 0,0322] | 0,6698 [0,6529; 0,6849] |
| 5 | Modelo de control original (canónico) | 0,7697 [0,7584; 0,7805] | 0,6739 [0,6455; 0,7000] | 0,6270 | 0,6739 | 0,6481 [0,6218; 0,6719] | 0,9531 [0,9491; 0,9569] | 0,0180 [0,0166; 0,0314] | 0,6649 [0,6496; 0,6801] |
| 6 | DINOv2 público, protocolo de Modelo de control (canónico) | 0,7405 [0,7288; 0,7525] | 0,6444 [0,6178; 0,6732] | 0,6145 | 0,6444 | 0,6281 [0,6037; 0,6534] | 0,9373 [0,9307; 0,9433] | 0,0483 [0,0415; 0,0612] | 0,6234 [0,6068; 0,6400] |
| 7 | DINOv2 ViT-B/14 | 0,6811 | 0,6198 [0,5906; 0,6491] | 0,5283 | 0,6198 | 0,5625 [0,5383; 0,5872] | 0,9198 | 0,1056 | 0,5583 |
| 8 | ViT-B/16 | 0,6538 | 0,5794 [0,5512; 0,6062] | 0,4980 | 0,5794 | 0,5277 [0,5024; 0,5500] | 0,9058 | 0,1157 | 0,5230 |
| 9 | EfficientNet-B0 | 0,6609 | 0,5754 [0,5462; 0,6031] | 0,4947 | 0,5754 | 0,5250 [0,5003; 0,5480] | 0,9106 | 0,1062 | 0,5335 |
| 10 | DINOv2 ViT-S/14 | 0,6529 | 0,5972 [0,5678; 0,6262] | 0,4807 | 0,5972 | 0,5178 [0,4938; 0,5401] | 0,9100 | 0,0675 | 0,5236 |
| 11 | ResNet-50 | 0,6377 | 0,5685 [0,5388; 0,5984] | 0,4635 | 0,5685 | 0,4988 [0,4736; 0,5227] | 0,9058 | 0,0956 | 0,5045 |
| 12 | AlexNet | 0,5739 | 0,5330 [0,5034; 0,5622] | 0,3975 | 0,5330 | 0,4208 [0,4012; 0,4403] | 0,8733 | 0,0500 | 0,4357 |
| 13 | VGG19 | 0,5583 | 0,4966 [0,4676; 0,5246] | 0,3733 | 0,4966 | 0,3909 [0,3731; 0,4074] | 0,8599 | 0,0537 | 0,4172 |
| 14 | SimSiam ResNet-50 | 0,4784 | 0,4940 [0,4663; 0,5221] | 0,3359 | 0,4940 | 0,3286 [0,3126; 0,3434] | 0,8391 | 0,2215 | 0,3397 |

## Con fine-tuning

Orden: F1 macro de mayor a menor; las filas sin número van al final.

### Condición

| # | Modelo | Preentrenamiento | Fine-tuning | Segmentación | Transductivo | Val n | Estado | Nota | Fuente |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ViT-B/16 | supervisado (ImageNet) | sí | no | no | 5.067 | ok | época elegida en val (sesgo optimista) | `results/finetune/isic2019/results_vit_b16_ft_isic2019.json` |
| 2 | ResNet-50 | supervisado (ImageNet) | sí | no | no | 5.067 | ok | época elegida en val (sesgo optimista) | `results/finetune/isic2019/results_resnet50_ft_isic2019.json` |
| 3 | EfficientNet-B0 | supervisado (ImageNet) | sí | no | no | 5.067 | ok | época elegida en val (sesgo optimista) | `results/finetune/isic2019/results_efficientnet_b0_ft_isic2019.json` |
| 4 | VGG19 | supervisado (ImageNet) | sí | no | no | 5.067 | ok | época elegida en val (sesgo optimista); mejor época = última (8/8), curva sin converger | `results/finetune/isic2019/results_vgg19_ft_isic2019.json` |
| 5 | AlexNet | supervisado (ImageNet) | sí | no | no | 5.067 | ok | época elegida en val (sesgo optimista) | `results/finetune/isic2019/results_alexnet_ft_isic2019.json` |
| 6 | SimSiam ResNet-50 | auto-supervisado (SimSiam, ImageNet) | sí | no | no | 5.067 | ok | época elegida en val (sesgo optimista); mejor época = última (8/8), curva sin converger | `results/finetune/isic2019/results_simsiam_r50_ft_isic2019.json` |
| 7 | DINOv2 ViT-B/14 | auto-supervisado (DINOv2 público) | sí | no | no | 5.067 | ok | época elegida en val (sesgo optimista); mejor época = última (8/8), curva sin converger | `results/finetune/isic2019/results_dinov2_vitb14_ft_isic2019.json` |
| 8 | DINOv2 ViT-S/14 | auto-supervisado (DINOv2 público) | sí | no | no | 5.067 | ok | época elegida en val (sesgo optimista); mejor época = última (8/8), curva sin converger | `results/finetune/isic2019/results_dinov2_vits14_ft_isic2019.json` |

### Métricas [IC 95 % donde la fuente lo publica]

| # | Modelo | Accuracy | Exactitud balanceada | Precisión macro | Recall macro | F1 macro | AUC OvR macro | ECE ↓ | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ViT-B/16 | 0,8433 | 0,8045 [0,7796; 0,8275] | 0,7896 | 0,8045 | 0,7962 [0,7749; 0,8157] | 0,9768 | 0,0429 | 0,7740 |
| 2 | ResNet-50 | 0,7833 | 0,7578 [0,7330; 0,7823] | 0,6731 | 0,7578 | 0,7080 [0,6836; 0,7287] | 0,9639 | 0,0190 | 0,6899 |
| 3 | EfficientNet-B0 | 0,7527 | 0,7155 [0,6893; 0,7417] | 0,5965 | 0,7155 | 0,6418 [0,6171; 0,6659] | 0,9542 | 0,0180 | 0,6499 |
| 4 | VGG19 | 0,7107 | 0,7012 [0,6751; 0,7246] | 0,5672 | 0,7012 | 0,6126 [0,5907; 0,6343] | 0,9457 | 0,0448 | 0,6002 |
| 5 | AlexNet | 0,6489 | 0,6421 [0,6149; 0,6679] | 0,4884 | 0,6421 | 0,5278 [0,5075; 0,5483] | 0,9233 | 0,0237 | 0,5284 |
| 6 | SimSiam ResNet-50 | 0,6485 | 0,6066 [0,5794; 0,6303] | 0,4674 | 0,6066 | 0,4998 [0,4801; 0,5197] | 0,9161 | 0,0220 | 0,5152 |
| 7 | DINOv2 ViT-B/14 | 0,6004 | 0,5501 [0,5217; 0,5764] | 0,4242 | 0,5501 | 0,4545 [0,4326; 0,4750] | 0,8931 | 0,0197 | 0,4589 |
| 8 | DINOv2 ViT-S/14 | 0,5940 | 0,5332 [0,5067; 0,5598] | 0,4198 | 0,5332 | 0,4462 [0,4245; 0,4662] | 0,8862 | 0,0245 | 0,4520 |

## Sensibilidad (cómputo no canónico: batch 16, transductivo)

Orden: F1 macro de mayor a menor; las filas sin número van al final.

### Condición

| # | Modelo | Preentrenamiento | Fine-tuning | Segmentación | Transductivo | Val n | Estado | Nota | Fuente |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Proposed sin segmentación original (batch 16) | auto-supervisado propio (DINOv2 + SimSiam anclado + término de tono, Proposed sin segmentación) | no | no | sí | 5.067 | ok | sensibilidad: cómputo no canónico (batch 16); transductivo | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/B71__original.json` · `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/B71__original.probas.npz` |
| 2 | DINOv2 público, protocolo de Proposed sin segmentación (batch 16) | auto-supervisado (DINOv2 público) | no | no | no | 5.067 | ok | sensibilidad: cómputo no canónico (batch 16) | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/PUBLIC_B71.json` · `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/PUBLIC_B71.probas.npz` |
| 3 | Modelo de control original (batch 16) | auto-supervisado propio (DINOv2 + SimSiam anclado sin término de tono, Modelo de control) | no | no | sí | 5.067 | ok | sensibilidad: cómputo no canónico (batch 16); transductivo | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/A71__original.json` · `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/A71__original.probas.npz` |
| 4 | DINOv2 público, protocolo de Modelo de control (batch 16) | auto-supervisado (DINOv2 público) | no | no | no | 5.067 | ok | sensibilidad: cómputo no canónico (batch 16) | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/PUBLIC_A71.json` · `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/PUBLIC_A71.probas.npz` |

### Métricas [IC 95 % donde la fuente lo publica]

| # | Modelo | Accuracy | Exactitud balanceada | Precisión macro | Recall macro | F1 macro | AUC OvR macro | ECE ↓ | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Proposed sin segmentación original (batch 16) | 0,7699 [0,7588; 0,7819] | 0,6791 [0,6500; 0,7052] | 0,6752 | 0,6791 | 0,6756 [0,6482; 0,6977] | 0,9545 [0,9501; 0,9585] | 0,0482 [0,0402; 0,0599] | 0,6677 [0,6526; 0,6839] |
| 2 | DINOv2 público, protocolo de Proposed sin segmentación (batch 16) | 0,7576 [0,7456; 0,7697] | 0,6606 [0,6317; 0,6870] | 0,6613 | 0,6606 | 0,6590 [0,6314; 0,6833] | 0,9476 [0,9420; 0,9533] | 0,0711 [0,0610; 0,0816] | 0,6471 [0,6305; 0,6634] |
| 3 | Modelo de control original (batch 16) | 0,7711 [0,7596; 0,7823] | 0,6781 [0,6501; 0,7020] | 0,6310 | 0,6781 | 0,6522 [0,6256; 0,6759] | 0,9531 [0,9492; 0,9569] | 0,0215 [0,0182; 0,0338] | 0,6669 [0,6513; 0,6819] |
| 4 | DINOv2 público, protocolo de Modelo de control (batch 16) | 0,7407 [0,7288; 0,7527] | 0,6428 [0,6151; 0,6715] | 0,6135 | 0,6428 | 0,6269 [0,6013; 0,6512] | 0,9372 [0,9305; 0,9432] | 0,0490 [0,0416; 0,0614] | 0,6238 [0,6068; 0,6405] |

## Contrastes pareados

Bootstrap pareado de 1.000 remuestras, unidad: imagen (sin grupo real). Los baselines (congelados y con fine-tuning) no tienen probabilidades por imagen commiteadas: su comparación con Proposed sin segmentación (pesos limpios) es solo de intervalos (IC 95 % de cada fuente), no un test pareado.

### Canónico (batch 64 + autocast)

| Brazo | Control | Métrica | Δ (brazo − control) [IC 95 %] | El IC excluye 0 | Lectura | Unidad | Fuente |
|---|---|---|---|---|---|---|---|
| Modelo de control (pesos limpios) | Modelo de control original | F1 macro | +0,0069 [-0,0151; +0,0280] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/A71__clean__vs__A71__original.json` |
| Modelo de control (pesos limpios) | Modelo de control original | Exactitud balanceada | +0,0090 [-0,0127; +0,0324] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/A71__clean__vs__A71__original.json` |
| Modelo de control (pesos limpios) | Modelo de control original | Accuracy | +0,0034 [-0,0057; +0,0122] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/A71__clean__vs__A71__original.json` |
| Modelo de control (pesos limpios) | Modelo de control original | AUC OvR macro | +0,0017 [-0,0008; +0,0042] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/A71__clean__vs__A71__original.json` |
| Modelo de control (pesos limpios) | Modelo de control original | ECE (menor es mejor) | +0,0018 [-0,0090; +0,0094] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/A71__clean__vs__A71__original.json` |
| Modelo de control (pesos limpios) | Modelo de control original | MCC | +0,0049 [-0,0079; +0,0176] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/A71__clean__vs__A71__original.json` |
| Modelo de control (pesos limpios) | DINOv2 público (prot. Modelo de control) | F1 macro | +0,0268 [+0,0011; +0,0506] | sí | Modelo de control (pesos limpios) mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/A71__clean__vs__PUBLIC_A71.json` |
| Modelo de control (pesos limpios) | DINOv2 público (prot. Modelo de control) | Exactitud balanceada | +0,0385 [+0,0112; +0,0647] | sí | Modelo de control (pesos limpios) mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/A71__clean__vs__PUBLIC_A71.json` |
| Modelo de control (pesos limpios) | DINOv2 público (prot. Modelo de control) | Accuracy | +0,0326 [+0,0197; +0,0452] | sí | Modelo de control (pesos limpios) mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/A71__clean__vs__PUBLIC_A71.json` |
| Modelo de control (pesos limpios) | DINOv2 público (prot. Modelo de control) | AUC OvR macro | +0,0175 [+0,0120; +0,0235] | sí | Modelo de control (pesos limpios) mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/A71__clean__vs__PUBLIC_A71.json` |
| Modelo de control (pesos limpios) | DINOv2 público (prot. Modelo de control) | ECE (menor es mejor) | -0,0285 [-0,0414; -0,0134] | sí | Modelo de control (pesos limpios) mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/A71__clean__vs__PUBLIC_A71.json` |
| Modelo de control (pesos limpios) | DINOv2 público (prot. Modelo de control) | MCC | +0,0464 [+0,0280; +0,0636] | sí | Modelo de control (pesos limpios) mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/A71__clean__vs__PUBLIC_A71.json` |
| Modelo de control original | DINOv2 público (prot. Modelo de control) | F1 macro | +0,0199 [-0,0051; +0,0438] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/A71__original__vs__PUBLIC_A71.json` |
| Modelo de control original | DINOv2 público (prot. Modelo de control) | Exactitud balanceada | +0,0295 [+0,0005; +0,0551] | sí | Modelo de control original mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/A71__original__vs__PUBLIC_A71.json` |
| Modelo de control original | DINOv2 público (prot. Modelo de control) | Accuracy | +0,0292 [+0,0168; +0,0409] | sí | Modelo de control original mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/A71__original__vs__PUBLIC_A71.json` |
| Modelo de control original | DINOv2 público (prot. Modelo de control) | AUC OvR macro | +0,0158 [+0,0108; +0,0215] | sí | Modelo de control original mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/A71__original__vs__PUBLIC_A71.json` |
| Modelo de control original | DINOv2 público (prot. Modelo de control) | ECE (menor es mejor) | -0,0303 [-0,0406; -0,0135] | sí | Modelo de control original mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/A71__original__vs__PUBLIC_A71.json` |
| Modelo de control original | DINOv2 público (prot. Modelo de control) | MCC | +0,0415 [+0,0238; +0,0571] | sí | Modelo de control original mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/A71__original__vs__PUBLIC_A71.json` |
| Proposed sin segmentación (pesos limpios) | Modelo de control (pesos limpios) | F1 macro | +0,0112 [-0,0131; +0,0369] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__clean__vs__A71__clean.json` |
| Proposed sin segmentación (pesos limpios) | Modelo de control (pesos limpios) | Exactitud balanceada | -0,0068 [-0,0338; +0,0187] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__clean__vs__A71__clean.json` |
| Proposed sin segmentación (pesos limpios) | Modelo de control (pesos limpios) | Accuracy | -0,0071 [-0,0180; +0,0039] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__clean__vs__A71__clean.json` |
| Proposed sin segmentación (pesos limpios) | Modelo de control (pesos limpios) | AUC OvR macro | -0,0007 [-0,0042; +0,0030] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__clean__vs__A71__clean.json` |
| Proposed sin segmentación (pesos limpios) | Modelo de control (pesos limpios) | ECE (menor es mejor) | +0,0301 [+0,0132; +0,0414] | sí | Modelo de control (pesos limpios) mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__clean__vs__A71__clean.json` |
| Proposed sin segmentación (pesos limpios) | Modelo de control (pesos limpios) | MCC | -0,0070 [-0,0219; +0,0084] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__clean__vs__A71__clean.json` |
| Proposed sin segmentación (pesos limpios) | Proposed sin segmentación original | F1 macro | -0,0055 [-0,0232; +0,0155] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__clean__vs__B71__original.json` |
| Proposed sin segmentación (pesos limpios) | Proposed sin segmentación original | Exactitud balanceada | +0,0030 [-0,0161; +0,0245] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__clean__vs__B71__original.json` |
| Proposed sin segmentación (pesos limpios) | Proposed sin segmentación original | Accuracy | -0,0022 [-0,0116; +0,0079] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__clean__vs__B71__original.json` |
| Proposed sin segmentación (pesos limpios) | Proposed sin segmentación original | AUC OvR macro | -0,0002 [-0,0030; +0,0026] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__clean__vs__B71__original.json` |
| Proposed sin segmentación (pesos limpios) | Proposed sin segmentación original | ECE (menor es mejor) | +0,0018 [-0,0092; +0,0113] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__clean__vs__B71__original.json` |
| Proposed sin segmentación (pesos limpios) | Proposed sin segmentación original | MCC | -0,0024 [-0,0153; +0,0121] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__clean__vs__B71__original.json` |
| Proposed sin segmentación (pesos limpios) | DINOv2 público (prot. Proposed sin segmentación) | F1 macro | +0,0079 [-0,0143; +0,0284] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__clean__vs__PUBLIC_B71.json` |
| Proposed sin segmentación (pesos limpios) | DINOv2 público (prot. Proposed sin segmentación) | Exactitud balanceada | +0,0165 [-0,0029; +0,0351] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__clean__vs__PUBLIC_B71.json` |
| Proposed sin segmentación (pesos limpios) | DINOv2 público (prot. Proposed sin segmentación) | Accuracy | +0,0079 [-0,0047; +0,0193] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__clean__vs__PUBLIC_B71.json` |
| Proposed sin segmentación (pesos limpios) | DINOv2 público (prot. Proposed sin segmentación) | AUC OvR macro | +0,0063 [+0,0021; +0,0108] | sí | Proposed sin segmentación (pesos limpios) mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__clean__vs__PUBLIC_B71.json` |
| Proposed sin segmentación (pesos limpios) | DINOv2 público (prot. Proposed sin segmentación) | ECE (menor es mejor) | -0,0208 [-0,0326; -0,0079] | sí | Proposed sin segmentación (pesos limpios) mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__clean__vs__PUBLIC_B71.json` |
| Proposed sin segmentación (pesos limpios) | DINOv2 público (prot. Proposed sin segmentación) | MCC | +0,0154 [-0,0023; +0,0317] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__clean__vs__PUBLIC_B71.json` |
| Proposed sin segmentación original | Modelo de control original | F1 macro | +0,0236 [-0,0002; +0,0477] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__original__vs__A71__original.json` |
| Proposed sin segmentación original | Modelo de control original | Exactitud balanceada | -0,0008 [-0,0263; +0,0265] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__original__vs__A71__original.json` |
| Proposed sin segmentación original | Modelo de control original | Accuracy | -0,0016 [-0,0126; +0,0097] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__original__vs__A71__original.json` |
| Proposed sin segmentación original | Modelo de control original | AUC OvR macro | +0,0012 [-0,0023; +0,0048] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__original__vs__A71__original.json` |
| Proposed sin segmentación original | Modelo de control original | ECE (menor es mejor) | +0,0300 [+0,0127; +0,0391] | sí | Modelo de control original mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__original__vs__A71__original.json` |
| Proposed sin segmentación original | Modelo de control original | MCC | +0,0003 [-0,0154; +0,0167] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__original__vs__A71__original.json` |
| Proposed sin segmentación original | DINOv2 público (prot. Proposed sin segmentación) | F1 macro | +0,0134 [-0,0095; +0,0356] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__original__vs__PUBLIC_B71.json` |
| Proposed sin segmentación original | DINOv2 público (prot. Proposed sin segmentación) | Exactitud balanceada | +0,0135 [-0,0095; +0,0356] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__original__vs__PUBLIC_B71.json` |
| Proposed sin segmentación original | DINOv2 público (prot. Proposed sin segmentación) | Accuracy | +0,0101 [-0,0026; +0,0219] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__original__vs__PUBLIC_B71.json` |
| Proposed sin segmentación original | DINOv2 público (prot. Proposed sin segmentación) | AUC OvR macro | +0,0065 [+0,0024; +0,0111] | sí | Proposed sin segmentación original mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__original__vs__PUBLIC_B71.json` |
| Proposed sin segmentación original | DINOv2 público (prot. Proposed sin segmentación) | ECE (menor es mejor) | -0,0226 [-0,0331; -0,0101] | sí | Proposed sin segmentación original mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__original__vs__PUBLIC_B71.json` |
| Proposed sin segmentación original | DINOv2 público (prot. Proposed sin segmentación) | MCC | +0,0178 [+0,0003; +0,0353] | sí | Proposed sin segmentación original mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/B71__original__vs__PUBLIC_B71.json` |

### Sensibilidad (batch 16, transductivo)

| Brazo | Control | Métrica | Δ (brazo − control) [IC 95 %] | El IC excluye 0 | Lectura | Unidad | Fuente |
|---|---|---|---|---|---|---|---|
| Modelo de control original | DINOv2 público (prot. Modelo de control) | F1 macro | +0,0253 [-0,0002; +0,0501] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/contrasts/A71__original__vs__PUBLIC_A71.json` |
| Modelo de control original | DINOv2 público (prot. Modelo de control) | Exactitud balanceada | +0,0353 [+0,0086; +0,0612] | sí | Modelo de control original mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/contrasts/A71__original__vs__PUBLIC_A71.json` |
| Modelo de control original | DINOv2 público (prot. Modelo de control) | Accuracy | +0,0304 [+0,0182; +0,0420] | sí | Modelo de control original mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/contrasts/A71__original__vs__PUBLIC_A71.json` |
| Modelo de control original | DINOv2 público (prot. Modelo de control) | AUC OvR macro | +0,0159 [+0,0108; +0,0216] | sí | Modelo de control original mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/contrasts/A71__original__vs__PUBLIC_A71.json` |
| Modelo de control original | DINOv2 público (prot. Modelo de control) | ECE (menor es mejor) | -0,0275 [-0,0386; -0,0113] | sí | Modelo de control original mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/contrasts/A71__original__vs__PUBLIC_A71.json` |
| Modelo de control original | DINOv2 público (prot. Modelo de control) | MCC | +0,0431 [+0,0259; +0,0595] | sí | Modelo de control original mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/contrasts/A71__original__vs__PUBLIC_A71.json` |
| Proposed sin segmentación original | Modelo de control original | F1 macro | +0,0234 [-0,0001; +0,0490] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/contrasts/B71__original__vs__A71__original.json` |
| Proposed sin segmentación original | Modelo de control original | Exactitud balanceada | +0,0010 [-0,0236; +0,0274] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/contrasts/B71__original__vs__A71__original.json` |
| Proposed sin segmentación original | Modelo de control original | Accuracy | -0,0012 [-0,0120; +0,0103] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/contrasts/B71__original__vs__A71__original.json` |
| Proposed sin segmentación original | Modelo de control original | AUC OvR macro | +0,0014 [-0,0022; +0,0050] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/contrasts/B71__original__vs__A71__original.json` |
| Proposed sin segmentación original | Modelo de control original | ECE (menor es mejor) | +0,0267 [+0,0104; +0,0376] | sí | Modelo de control original mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/contrasts/B71__original__vs__A71__original.json` |
| Proposed sin segmentación original | Modelo de control original | MCC | +0,0008 [-0,0145; +0,0171] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/contrasts/B71__original__vs__A71__original.json` |
| Proposed sin segmentación original | DINOv2 público (prot. Proposed sin segmentación) | F1 macro | +0,0166 [-0,0073; +0,0386] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/contrasts/B71__original__vs__PUBLIC_B71.json` |
| Proposed sin segmentación original | DINOv2 público (prot. Proposed sin segmentación) | Exactitud balanceada | +0,0186 [-0,0047; +0,0408] | no | sin diferencia demostrable (el IC incluye 0) | image | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/contrasts/B71__original__vs__PUBLIC_B71.json` |
| Proposed sin segmentación original | DINOv2 público (prot. Proposed sin segmentación) | Accuracy | +0,0122 [+0,0002; +0,0243] | sí | Proposed sin segmentación original mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/contrasts/B71__original__vs__PUBLIC_B71.json` |
| Proposed sin segmentación original | DINOv2 público (prot. Proposed sin segmentación) | AUC OvR macro | +0,0068 [+0,0026; +0,0114] | sí | Proposed sin segmentación original mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/contrasts/B71__original__vs__PUBLIC_B71.json` |
| Proposed sin segmentación original | DINOv2 público (prot. Proposed sin segmentación) | ECE (menor es mejor) | -0,0229 [-0,0322; -0,0102] | sí | Proposed sin segmentación original mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/contrasts/B71__original__vs__PUBLIC_B71.json` |
| Proposed sin segmentación original | DINOv2 público (prot. Proposed sin segmentación) | MCC | +0,0206 [+0,0036; +0,0376] | sí | Proposed sin segmentación original mejor (el IC excluye 0) | image | `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/contrasts/B71__original__vs__PUBLIC_B71.json` |
