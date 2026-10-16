# Banco completo: HAM10000, propuesto y 8 baselines

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
- **Transductivo:** el pool de preentrenamiento auto-supervisado contenía las imágenes de val (sin etiquetas). Solo aplica a Proposed sin segmentación / control SSL sin consistencia de tono originales en ISIC 2019 e ISIC 2020.
- **Cómputo canónico:** extracción de features con batch 64 y autocast CUDA (el de v7.1). El batch 16 mueve las cifras unas milésimas (orden de reducción en fp16) y se reporta solo como **sensibilidad**.
- **Pesos limpios (sensibilidad):** Proposed sin segmentación / control SSL sin consistencia de tono re-preentrenados con un pool SSL sin las imágenes de val de ISIC (docs/33, sección 2). En HAM10000 son un análisis posterior que se lee al lado de la fila registrada de **pesos originales (selección)**; no es selección y no reemplaza al propuesto.
- **Contraste pareado:** diferencia entre dos modelos sobre las mismas imágenes, con IC bootstrap de la diferencia. «Mejor» solo si el IC excluye 0; si lo incluye, «sin diferencia demostrable».
- **Exploratorio:** variantes post-hoc o con selección en otro conjunto; se reportan pero no cambian el propuesto ni se leen como resultado confirmado.

En este val la clase «nv» es el 67,7 % de las imágenes (7 clases): predecir siempre «nv» da una accuracy de 0,6768 y una exactitud balanceada de 0,1429.

## Cómo leer esta tabla

Particiones de val de este banco (no son la misma partición):

- HAM10000, baselines (Kaggle): val n = 2.014; StratifiedGroupKFold por lesion_id (0 lesiones compartidas).
- HAM10000, propuesto (v6/v7): val n = 2.004; partición agrupada por `lesion_id`.
- ISIC 2019, baselines (Kaggle): val n = 5.067; StratifiedKFold estratificado simple SIN grupo de paciente (el release no trae patient_id) - riesgo de fuga entre imagenes del mismo paciente, metricas con sesgo optimista.
- ISIC 2020, baselines (Kaggle): val n = 6.660; StratifiedGroupKFold por patient_id (0 pacientes compartidos).

- **Entre particiones distintas la comparación es solo de intervalos (IC 95 %)**, no un test pareado: HAM10000 usa la partición v6/v7 (n = 2.004) para el propuesto y la de Kaggle (n = 2.014) para los baselines; ISIC 2019 e ISIC 2020 tienen la suya. Los contrastes pareados solo existen donde dos modelos comparten partición e imágenes y hay probabilidades por imagen (docs/33, sección 5); están al final de cada tabla y, todos juntos, en `CONTRASTES_SSL.md` (sin corrección por multiplicidad).
- **Optimista: época elegida en val.** Todas las filas de fine-tuning (HAM10000, ISIC 2019 e ISIC 2020) eligen su mejor época por balanced accuracy sobre val; el plan decía holdout interno y se mantuvo la receta existente por consistencia entre datasets (errata de docs/33). Las de ISIC 2020 corrieron en nuestro pod (GPU RTX PRO 4000), no en el banco de Karen. El propuesto no tiene fine-tuning y no se ve afectado.
- **Modelo y protocolo no son lo mismo.** El propuesto usa el protocolo de Proposed sin segmentación (features cls4, TTA, probe balanceado); la fila «DINOv2 público, protocolo de Proposed sin segmentación» es el control con el mismo protocolo y pesos públicos. Los 8 baselines siguen el protocolo de su propio banco (probe logreg).
- Los IC solo existen donde la fuente los publica: los baselines, para exactitud balanceada y F1 macro; Proposed sin segmentación / control SSL sin consistencia de tono, para accuracy, exactitud balanceada, F1 macro, AUC y ECE. Precisión macro, recall macro y MCC de los baselines salen exactos de la matriz de confusión y van sin IC; en Proposed sin segmentación / control SSL sin consistencia de tono salen de las probabilidades commiteadas, también sin IC.
- **Pesos limpios en HAM10000: sensibilidad posterior, no es selección.** La fila registrada del propuesto (pesos originales, selección) no se reemplaza. Las filas del bloque «pesos limpios (sensibilidad)» usan la configuración y los umbrales de operación del holdout original, ajustados con los pesos originales y aplicados sin cambios a los limpios (supuesto declarado), con cómputo canónico; sus contrastes pareados (por `lesion_id`, con MCC) están en `CONTRASTES_SSL.md` y `TABLA_ABLACION.md`.
- Las filas «pendiente» no tienen número: se listan con su motivo técnico (sin ETA inventada), para que nada parezca olvidado. No se inventa ninguna cifra.

## Congelados (sin fine-tuning)

Orden: F1 macro de mayor a menor; las filas sin número van al final.

### Condición

| # | Modelo | Preentrenamiento | Fine-tuning | Segmentación | Transductivo | Val n | Estado | Nota | Fuente |
|---|---|---|---|---|---|---|---|---|---|
| 1 | control SSL sin consistencia de tono (ablación sin término de tono) | auto-supervisado propio (DINOv2 + SimSiam anclado sin término de tono, experimento de control SSL sin consistencia de tono) | no | no | no | 2.004 | ok | pesos originales (selección); probe logreg C1; precisión, recall y MCC desde las probabilidades | `resultados/clasificacion/v71/v71_final_eval.json` · `resultados/clasificacion/v71/v71_final_eval.probas.npz` |
| 2 | Proposed sin segmentación (propuesto) | auto-supervisado propio (DINOv2 + SimSiam anclado + término de tono, experimento SSL con consistencia de tono) | no | no | no | 2.004 | ok | pesos originales (selección); probe logreg C1; precisión, recall y MCC desde las probabilidades; cómputo canónico = S1 (batch 64 + autocast, el de v7.1). F1 macro de Proposed sin segmentación en S0/S1/S2: 0,7106 / 0,7141 / 0,7089 (el cómputo de extracción mueve la cifra) | `resultados/clasificacion/v71/v71_final_eval.json` · `resultados/clasificacion/v71/v71_final_eval.probas.npz` |
| 3 | DINOv2 público, su mejor protocolo v7.1 | auto-supervisado (DINOv2 público) | no | no | no | 2.004 | ok | pesos originales (selección); probe logreg C1; precisión, recall y MCC desde las probabilidades | `resultados/clasificacion/v71/v71_final_eval.json` · `resultados/clasificacion/v71/v71_final_eval.probas.npz` |
| 4 | DINOv2 público, protocolo de Proposed sin segmentación | auto-supervisado (DINOv2 público) | no | no | no | 2.004 | ok | pesos originales (selección); probe logreg C1; precisión, recall y MCC desde las probabilidades | `resultados/clasificacion/v71/v71_final_eval.json` · `resultados/clasificacion/v71/v71_final_eval.probas.npz` |
| 5 | DINOv2 público, protocolo de control SSL sin consistencia de tono | auto-supervisado (DINOv2 público) | no | no | no | 2.004 | ok | pesos originales (selección); probe logreg C1; precisión, recall y MCC desde las probabilidades | `resultados/clasificacion/v71/v71_final_eval.json` · `resultados/clasificacion/v71/v71_final_eval.probas.npz` |
| 6 | DINOv2 ViT-B/14 | auto-supervisado (DINOv2 público) | no | no | no | 2.014 | ok | — | `results/baselines/results_dinov2_vitb14.json` |
| 7 | DINOv2 ViT-S/14 | auto-supervisado (DINOv2 público) | no | no | no | 2.014 | ok | — | `results/baselines/results_dinov2_vits14.json` |
| 8 | EfficientNet-B0 | supervisado (ImageNet) | no | no | no | 2.014 | ok | — | `results/baselines/results_efficientnet_b0.json` |
| 9 | ViT-B/16 | supervisado (ImageNet) | no | no | no | 2.014 | ok | — | `results/baselines/results_vit_b16.json` |
| 10 | ResNet-50 | supervisado (ImageNet) | no | no | no | 2.014 | ok | — | `results/baselines/results_resnet50.json` |
| 11 | AlexNet | supervisado (ImageNet) | no | no | no | 2.014 | ok | — | `results/baselines/results_alexnet.json` |
| 12 | VGG19 | supervisado (ImageNet) | no | no | no | 2.014 | ok | — | `results/baselines/results_vgg19.json` |
| 13 | SimSiam ResNet-50 | auto-supervisado (SimSiam, ImageNet) | no | no | no | 2.014 | ok | — | `results/baselines/results_simsiam_r50.json` |

### Métricas [IC 95 % donde la fuente lo publica]

| # | Modelo | Accuracy | Exactitud balanceada | Precisión macro | Recall macro | F1 macro | AUC OvR macro | ECE ↓ | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | control SSL sin consistencia de tono (ablación sin término de tono) | 0,8398 [0,8229; 0,8553] | 0,7305 [0,6844; 0,7719] | 0,7197 | 0,7305 | 0,7229 [0,6795; 0,7592] | 0,9644 [0,9558; 0,9718] | 0,0154 [0,0135; 0,0339] | 0,6996 |
| 2 | Proposed sin segmentación (propuesto) | 0,8378 [0,8218; 0,8533] | 0,6964 [0,6504; 0,7408] | 0,7381 | 0,6964 | 0,7141 [0,6703; 0,7547] | 0,9602 [0,9516; 0,9680] | 0,0529 [0,0428; 0,0695] | 0,6901 |
| 3 | DINOv2 público, su mejor protocolo v7.1 | 0,8268 [0,8104; 0,8423] | 0,6622 [0,6160; 0,7058] | 0,7226 | 0,6622 | 0,6857 [0,6412; 0,7208] | 0,9598 [0,9516; 0,9665] | 0,0702 [0,0605; 0,0871] | 0,6683 |
| 4 | DINOv2 público, protocolo de Proposed sin segmentación | 0,8278 [0,8119; 0,8438] | 0,6659 [0,6176; 0,7102] | 0,7097 | 0,6659 | 0,6837 [0,6381; 0,7202] | 0,9619 [0,9549; 0,9680] | 0,0688 [0,0573; 0,0846] | 0,6712 |
| 5 | DINOv2 público, protocolo de control SSL sin consistencia de tono | 0,7959 [0,7774; 0,8124] | 0,6174 [0,5714; 0,6632] | 0,6547 | 0,6174 | 0,6320 [0,5875; 0,6710] | 0,9432 [0,9339; 0,9519] | 0,0806 [0,0684; 0,0987] | 0,6110 |
| 6 | DINOv2 ViT-B/14 | 0,7468 | 0,6179 [0,5763; 0,6601] | 0,5984 | 0,6179 | 0,5951 [0,5508; 0,6351] | 0,9070 | 0,1617 | 0,5634 |
| 7 | DINOv2 ViT-S/14 | 0,7110 | 0,5753 [0,5305; 0,6169] | 0,5429 | 0,5753 | 0,5474 [0,5037; 0,5838] | 0,9052 | 0,1489 | 0,5137 |
| 8 | EfficientNet-B0 | 0,7344 | 0,5857 [0,5472; 0,6233] | 0,5192 | 0,5857 | 0,5464 [0,5089; 0,5786] | 0,9170 | 0,1245 | 0,5375 |
| 9 | ViT-B/16 | 0,7229 | 0,5674 [0,5228; 0,6122] | 0,5301 | 0,5674 | 0,5402 [0,4971; 0,5795] | 0,9199 | 0,1685 | 0,5243 |
| 10 | ResNet-50 | 0,7314 | 0,5716 [0,5332; 0,6150] | 0,4980 | 0,5716 | 0,5280 [0,4917; 0,5633] | 0,9084 | 0,0973 | 0,5384 |
| 11 | AlexNet | 0,6663 | 0,5892 [0,5478; 0,6311] | 0,4496 | 0,5892 | 0,4929 [0,4584; 0,5215] | 0,8961 | 0,0793 | 0,4862 |
| 12 | VGG19 | 0,6182 | 0,5147 [0,4724; 0,5546] | 0,3771 | 0,5147 | 0,4151 [0,3858; 0,4404] | 0,8702 | 0,0707 | 0,4140 |
| 13 | SimSiam ResNet-50 | 0,4896 | 0,4942 [0,4531; 0,5394] | 0,3354 | 0,4942 | 0,3169 [0,2939; 0,3405] | 0,8291 | 0,2465 | 0,3186 |

## Con fine-tuning

Orden: F1 macro de mayor a menor; las filas sin número van al final.

### Condición

| # | Modelo | Preentrenamiento | Fine-tuning | Segmentación | Transductivo | Val n | Estado | Nota | Fuente |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ViT-B/16 | supervisado (ImageNet) | sí | no | no | 2.014 | ok | época elegida en val (sesgo optimista) | `results/finetune/ham10000/results_vit_b16_ft.json` |
| 2 | ResNet-50 | supervisado (ImageNet) | sí | no | no | 2.014 | ok | época elegida en val (sesgo optimista) | `results/finetune/ham10000/results_resnet50_ft.json` |
| 3 | EfficientNet-B0 | supervisado (ImageNet) | sí | no | no | 2.014 | ok | época elegida en val (sesgo optimista) | `results/finetune/ham10000/results_efficientnet_b0_ft.json` |
| 4 | VGG19 | supervisado (ImageNet) | sí | no | no | 2.014 | ok | época elegida en val (sesgo optimista) | `results/finetune/ham10000/results_vgg19_ft.json` |
| 5 | AlexNet | supervisado (ImageNet) | sí | no | no | 2.014 | ok | época elegida en val (sesgo optimista) | `results/finetune/ham10000/results_alexnet_ft.json` |
| 6 | SimSiam ResNet-50 | auto-supervisado (SimSiam, ImageNet) | sí | no | no | 2.014 | ok | época elegida en val (sesgo optimista) | `results/finetune/ham10000/results_simsiam_r50_ft.json` |
| 7 | DINOv2 ViT-S/14 | auto-supervisado (DINOv2 público) | sí | no | no | 2.014 | ok | época elegida en val (sesgo optimista) | `results/finetune/ham10000/results_dinov2_vits14_ft.json` |
| 8 | DINOv2 ViT-B/14 | auto-supervisado (DINOv2 público) | sí | no | no | 2.014 | ok | época elegida en val (sesgo optimista) | `results/finetune/ham10000/results_dinov2_vitb14_ft.json` |

### Métricas [IC 95 % donde la fuente lo publica]

| # | Modelo | Accuracy | Exactitud balanceada | Precisión macro | Recall macro | F1 macro | AUC OvR macro | ECE ↓ | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ViT-B/16 | 0,8073 | 0,7523 [0,7122; 0,7912] | 0,6994 | 0,7523 | 0,7058 [0,6593; 0,7431] | 0,9613 | 0,0461 | 0,6721 |
| 2 | ResNet-50 | 0,8004 | 0,7393 [0,7001; 0,7809] | 0,6512 | 0,7393 | 0,6867 [0,6476; 0,7226] | 0,9548 | 0,0522 | 0,6526 |
| 3 | EfficientNet-B0 | 0,7800 | 0,7208 [0,6801; 0,7602] | 0,6066 | 0,7208 | 0,6503 [0,6074; 0,6870] | 0,9508 | 0,0370 | 0,6302 |
| 4 | VGG19 | 0,7438 | 0,7143 [0,6734; 0,7510] | 0,5713 | 0,7143 | 0,6246 [0,5874; 0,6591] | 0,9443 | 0,0568 | 0,5936 |
| 5 | AlexNet | 0,7195 | 0,6702 [0,6325; 0,7093] | 0,5311 | 0,6702 | 0,5794 [0,5470; 0,6103] | 0,9335 | 0,0357 | 0,5488 |
| 6 | SimSiam ResNet-50 | 0,6877 | 0,6782 [0,6396; 0,7174] | 0,4724 | 0,6782 | 0,5323 [0,5007; 0,5634] | 0,9243 | 0,0246 | 0,5198 |
| 7 | DINOv2 ViT-S/14 | 0,6390 | 0,6207 [0,5808; 0,6598] | 0,4471 | 0,6207 | 0,4953 [0,4643; 0,5252] | 0,9147 | 0,0536 | 0,4657 |
| 8 | DINOv2 ViT-B/14 | 0,5929 | 0,5325 [0,4903; 0,5724] | 0,3950 | 0,5325 | 0,4268 [0,3970; 0,4545] | 0,8881 | 0,0601 | 0,4149 |

## Pesos limpios (sensibilidad)

Orden: F1 macro de mayor a menor; las filas sin número van al final.

### Condición

| # | Modelo | Preentrenamiento | Fine-tuning | Segmentación | Transductivo | Val n | Estado | Nota | Fuente |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Proposed sin segmentación, pesos con exclusión de validación (canónico) | auto-supervisado propio (DINOv2 + SimSiam anclado + término de tono, experimento SSL con consistencia de tono) | no | no | no | 2.004 | ok | pesos limpios (pool SSL sin las imágenes de val de ISIC); análisis de sensibilidad posterior, no es selección: configuración y umbrales del holdout original (ajustados con los pesos originales y aplicados sin cambios); cómputo canónico (batch 64 + autocast); se lee al lado de la fila registrada de pesos originales, que no se reemplaza | `resultados/banco_completo/ham_clean/ham_clean_val.json` · `resultados/banco_completo/ham_clean/ham_clean_val.probas.npz` |
| 2 | control SSL sin consistencia de tono, pesos con exclusión de validación (canónico) | auto-supervisado propio (DINOv2 + SimSiam anclado sin término de tono, experimento de control SSL sin consistencia de tono) | no | no | no | 2.004 | ok | pesos limpios (pool SSL sin las imágenes de val de ISIC); análisis de sensibilidad posterior, no es selección: configuración y umbrales del holdout original (ajustados con los pesos originales y aplicados sin cambios); cómputo canónico (batch 64 + autocast); se lee al lado de la fila registrada de pesos originales, que no se reemplaza | `resultados/banco_completo/ham_clean/ham_clean_val.json` · `resultados/banco_completo/ham_clean/ham_clean_val.probas.npz` |

### Métricas [IC 95 % donde la fuente lo publica]

| # | Modelo | Accuracy | Exactitud balanceada | Precisión macro | Recall macro | F1 macro | AUC OvR macro | ECE ↓ | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Proposed sin segmentación, pesos con exclusión de validación (canónico) | 0,8323 [0,8159; 0,8483] | 0,6936 [0,6472; 0,7395] | 0,7339 | 0,6936 | 0,7104 [0,6672; 0,7497] | 0,9612 [0,9545; 0,9673] | 0,0581 [0,0470; 0,0733] | 0,6790 |
| 2 | control SSL sin consistencia de tono, pesos con exclusión de validación (canónico) | 0,8428 [0,8258; 0,8588] | 0,7103 [0,6650; 0,7524] | 0,6964 | 0,7103 | 0,7025 [0,6606; 0,7432] | 0,9630 [0,9548; 0,9705] | 0,0156 [0,0140; 0,0334] | 0,7047 |

## Exploratorio (segmentación, no cambia el propuesto)

Orden: F1 macro de mayor a menor; las filas sin número van al final.

### Condición

| # | Modelo | Preentrenamiento | Fine-tuning | Segmentación | Transductivo | Val n | Estado | Nota | Fuente |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Proposed con segmentación (exploratorio) | auto-supervisado propio (DINOv2 + SimSiam anclado + término de tono, experimento SSL con consistencia de tono) | no | sí (Proposed con segmentación (experimental), exploratorio) | no | 2.004 | ok | exploratorio, no demostrable de forma robusta: Δ F1 macro frente a Proposed sin segmentación +0,0300 [-0,0046; +0,0715] en el cómputo canónico; el IC pareado excluye 0 en 2 de 3 cómputos; no reemplaza al propuesto (TABLA_ABLACION.md) | `resultados/banco_completo/e3b_bs64/e3b_summary.json` · `resultados/banco_completo/e3/robustez/e3_s1_bs64_autocast/e3_final_eval.json` |

### Métricas [IC 95 % donde la fuente lo publica]

| # | Modelo | Accuracy | Exactitud balanceada | Precisión macro | Recall macro | F1 macro | AUC OvR macro | ECE ↓ | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Proposed con segmentación (exploratorio) | 0,8633 [0,8456; 0,8815] | 0,7186 [0,6648; 0,7791] | 0,7847 | 0,7186 | 0,7441 [0,6913; 0,7884] | 0,9659 [0,9577; 0,9745] | 0,0483 [0,0382; 0,0662] | 0,7355 [0,7019; 0,7689] |

## Contrastes pareados (partición v6/v7, n = 2.004)

Contraste solo contra baselines CONGELADOS: el fine-tuning sigue en la partición de Kaggle y no es pareado. Los 8 baselines se re-evaluaron sobre la partición v6/v7 con el mismo probe (E3b, `resultados/banco_completo/e3b_bs64/e3b_summary.json`); sus cifras difieren de las de la tabla de arriba (partición de Kaggle). `*` = el IC pareado excluye 0. Δ = Proposed sin segmentación − baseline (positivo favorece a Proposed sin segmentación).

### Proposed sin segmentación frente a cada baseline congelado (F1 macro)

| Baseline (congelado) | F1 macro (v6/v7) | Δ vs Proposed sin segmentación (1 vista, sin TTA) | Δ vs Proposed sin segmentación (TTA, canónico) | Δ vs Proposed con segmentación (1 vista) | Δ vs Proposed con segmentación (TTA) |
|---|---|---|---|---|---|
| AlexNet | 0,4728 | +0,2067 [+0,1613; +0,2564] * | +0,2413 [+0,1942; +0,2898] * | +0,2059 [+0,1607; +0,2551] * | +0,2504 [+0,2050; +0,3010] * |
| VGG19 | 0,3810 | +0,2985 [+0,2518; +0,3466] * | +0,3331 [+0,2844; +0,3800] * | +0,2213 [+0,1751; +0,2688] * | +0,2658 [+0,2212; +0,3180] * |
| ResNet-50 | 0,4670 | +0,2125 [+0,1587; +0,2653] * | +0,2471 [+0,1969; +0,2943] * | +0,1635 [+0,1155; +0,2157] * | +0,2080 [+0,1569; +0,2640] * |
| EfficientNet-B0 | 0,5041 | +0,1754 [+0,1176; +0,2301] * | +0,2100 [+0,1579; +0,2609] * | +0,1273 [+0,0780; +0,1807] * | +0,1718 [+0,1227; +0,2235] * |
| ViT-B/16 | 0,5633 | +0,1163 [+0,0608; +0,1748] * | +0,1509 [+0,1005; +0,2068] * | +0,1072 [+0,0629; +0,1557] * | +0,1517 [+0,1089; +0,2013] * |
| SimSiam ResNet-50 | 0,5369 | +0,1426 [+0,0934; +0,1938] * | +0,1772 [+0,1278; +0,2299] * | +0,1103 [+0,0609; +0,1645] * | +0,1548 [+0,1119; +0,2065] * |
| DINOv2 ViT-S/14 | 0,5508 | +0,1287 [+0,0801; +0,1800] * | +0,1633 [+0,1198; +0,2066] * | +0,0607 [+0,0100; +0,1120] * | +0,1052 [+0,0594; +0,1562] * |
| DINOv2 ViT-B/14 | 0,6002 | +0,0793 [+0,0349; +0,1251] * | +0,1139 [+0,0728; +0,1585] * | +0,0829 [+0,0379; +0,1291] * | +0,1274 [+0,0819; +0,1749] * |

Proposed sin segmentación por encima del baseline con IC que excluye 0, por variante: Proposed sin segmentación (1 vista, sin TTA): 8/8; Proposed sin segmentación (TTA, canónico): 8/8; Proposed con segmentación (1 vista): 8/8; Proposed con segmentación (TTA): 8/8. Las otras métricas (exactitud balanceada, recall de melanoma) y las 120 comparaciones están en `CONTRASTES_SSL.md`.

### Proposed con segmentación (experimental) (segmentación, exploratorio) sobre cada baseline congelado (F1 macro)

| Brazo | Control | Métrica | Δ (brazo − control) [IC 95 %] | El IC excluye 0 | Lectura | Unidad | Fuente |
|---|---|---|---|---|---|---|---|
| AlexNet + Proposed con segmentación (experimental) | AlexNet | F1 macro | +0,0210 [-0,0121; +0,0536] | no | sin diferencia demostrable (el IC incluye 0) | lesion_id | `resultados/banco_completo/e3b_bs64/e3b_summary.json` |
| VGG19 + Proposed con segmentación (experimental) | VGG19 | F1 macro | +0,0973 [+0,0577; +0,1352] | sí | VGG19 + Proposed con segmentación (experimental) mejor (el IC excluye 0) | lesion_id | `resultados/banco_completo/e3b_bs64/e3b_summary.json` |
| EfficientNet-B0 + Proposed con segmentación (experimental) | EfficientNet-B0 | F1 macro | +0,0683 [+0,0204; +0,1125] | sí | EfficientNet-B0 + Proposed con segmentación (experimental) mejor (el IC excluye 0) | lesion_id | `resultados/banco_completo/e3b_bs64/e3b_summary.json` |
| ResNet-50 + Proposed con segmentación (experimental) | ResNet-50 | F1 macro | +0,0692 [+0,0344; +0,1046] | sí | ResNet-50 + Proposed con segmentación (experimental) mejor (el IC excluye 0) | lesion_id | `resultados/banco_completo/e3b_bs64/e3b_summary.json` |
| ViT-B/16 + Proposed con segmentación (experimental) | ViT-B/16 | F1 macro | +0,0292 [-0,0164; +0,0724] | no | sin diferencia demostrable (el IC incluye 0) | lesion_id | `resultados/banco_completo/e3b_bs64/e3b_summary.json` |
| SimSiam ResNet-50 + Proposed con segmentación (experimental) | SimSiam ResNet-50 | F1 macro | +0,0524 [+0,0132; +0,0929] | sí | SimSiam ResNet-50 + Proposed con segmentación (experimental) mejor (el IC excluye 0) | lesion_id | `resultados/banco_completo/e3b_bs64/e3b_summary.json` |
| DINOv2 ViT-S/14 + Proposed con segmentación (experimental) | DINOv2 ViT-S/14 | F1 macro | +0,0882 [+0,0477; +0,1330] | sí | DINOv2 ViT-S/14 + Proposed con segmentación (experimental) mejor (el IC excluye 0) | lesion_id | `resultados/banco_completo/e3b_bs64/e3b_summary.json` |
| DINOv2 ViT-B/14 + Proposed con segmentación (experimental) | DINOv2 ViT-B/14 | F1 macro | +0,0166 [-0,0193; +0,0508] | no | sin diferencia demostrable (el IC incluye 0) | lesion_id | `resultados/banco_completo/e3b_bs64/e3b_summary.json` |

Proposed con segmentación (experimental) sube el F1 macro de los 8 baselines en 8 de 8 casos; el IC excluye 0 en 5 de 8: VGG19, EfficientNet-B0, ResNet-50, SimSiam ResNet-50, DINOv2 ViT-S/14. Es exploratorio (docs/33, erratas).
