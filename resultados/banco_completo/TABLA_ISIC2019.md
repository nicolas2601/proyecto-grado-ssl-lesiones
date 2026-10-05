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
- **Segmentación:** uso de máscaras de lesión como entrada. No se usa en este banco: la columna dice «no» en todas las filas (las variantes con segmentación son el bloque exploratorio E3 de docs/33).
- **Transductivo:** el pool de preentrenamiento auto-supervisado contenía las imágenes de val (sin etiquetas). Solo aplica a B71/A71 originales en ISIC 2019 e ISIC 2020.

En este val la clase «NV» es el 50,8 % de las imágenes (8 clases): predecir siempre «NV» da una accuracy de 0,5082 y una exactitud balanceada de 0,1250.

## Cómo leer esta tabla

Particiones de val de este banco (no son la misma partición):

- HAM10000, baselines (Kaggle): val n = 2.014; StratifiedGroupKFold por lesion_id (0 lesiones compartidas).
- HAM10000, propuesto (v6/v7): val n = 2.004; partición agrupada por `lesion_id`.
- ISIC 2019, baselines (Kaggle): val n = 5.067; StratifiedKFold estratificado simple SIN grupo de paciente (el release no trae patient_id) - riesgo de fuga entre imagenes del mismo paciente, metricas con sesgo optimista.
- ISIC 2020, baselines (Kaggle): val n = 6.660; StratifiedGroupKFold por patient_id (0 pacientes compartidos).

- **Entre particiones distintas la comparación es solo de intervalos (IC 95 %)**, no un test pareado: HAM10000 usa la partición v6/v7 (n = 2.004) para el propuesto y la de Kaggle (n = 2.014) para los baselines; ISIC 2019 e ISIC 2020 tienen la suya. Los contrastes pareados se harán solo donde dos modelos comparten partición e imágenes (docs/33, sección 5).
- **El fine-tuning elige su mejor época sobre val** en las filas ya commiteadas (HAM10000 e ISIC 2019): su número es optimista. La corrida de ISIC 2020 (pendiente) elige en el holdout interno (20 % del train).
- Los IC solo existen donde la fuente los publica: los baselines, para exactitud balanceada y F1 macro; B71/A71, para accuracy, exactitud balanceada, F1 macro, AUC y ECE. Precisión macro, recall macro y MCC de los baselines salen exactos de la matriz de confusión y van sin IC; en B71/A71 salen de las probabilidades commiteadas, también sin IC.
- Las filas «pendiente» y «no ejecutado todavia» no tienen número: se listan para que nada parezca olvidado. No se inventa ninguna cifra.

## Congelados (sin fine-tuning)

Orden: F1 macro de mayor a menor; las filas sin número van al final.

### Condición

| # | Modelo | Preentrenamiento | Fine-tuning | Segmentación | Transductivo | Val n | Estado | Nota | Fuente |
|---|---|---|---|---|---|---|---|---|---|
| 1 | DINOv2 ViT-B/14 | auto-supervisado (DINOv2 público) | no | no | no | 5.067 | ok | — | `results/baselines_isic2019/results_dinov2_vitb14_isic2019.json` |
| 2 | ViT-B/16 | supervisado (ImageNet) | no | no | no | 5.067 | ok | — | `results/baselines_isic2019/results_vit_b16_isic2019.json` |
| 3 | EfficientNet-B0 | supervisado (ImageNet) | no | no | no | 5.067 | ok | — | `results/baselines_isic2019/results_efficientnet_b0_isic2019.json` |
| 4 | DINOv2 ViT-S/14 | auto-supervisado (DINOv2 público) | no | no | no | 5.067 | ok | — | `results/baselines_isic2019/results_dinov2_vits14_isic2019.json` |
| 5 | ResNet-50 | supervisado (ImageNet) | no | no | no | 5.067 | ok | — | `results/baselines_isic2019/results_resnet50_isic2019.json` |
| 6 | AlexNet | supervisado (ImageNet) | no | no | no | 5.067 | ok | — | `results/baselines_isic2019/results_alexnet_isic2019.json` |
| 7 | VGG19 | supervisado (ImageNet) | no | no | no | 5.067 | ok | — | `results/baselines_isic2019/results_vgg19_isic2019.json` |
| 8 | SimSiam ResNet-50 | auto-supervisado (SimSiam, ImageNet) | no | no | no | 5.067 | ok | — | `results/baselines_isic2019/results_simsiam_r50_isic2019.json` |
| — | B71-limpio (B* s42 limpio, config fija) | auto-supervisado propio (DINOv2 + SimSiam anclado + término de tono, B*) | no | no | no | — | pendiente | E1 de docs/33 | `docs/33-banco-completo-multidataset.md` |
| — | A71-limpio (sopa A* s42+s43 limpios) | auto-supervisado propio (DINOv2 + SimSiam anclado sin término de tono, A*) | no | no | no | — | pendiente | E1 de docs/33 | `docs/33-banco-completo-multidataset.md` |
| — | B71 original | auto-supervisado propio (DINOv2 + SimSiam anclado + término de tono, B*) | no | no | sí | — | pendiente | transductivo: imágenes de val en el pool SSL, sin etiquetas (E1) | `docs/33-banco-completo-multidataset.md` |
| — | A71 original | auto-supervisado propio (DINOv2 + SimSiam anclado sin término de tono, A*) | no | no | sí | — | pendiente | transductivo: imágenes de val en el pool SSL, sin etiquetas (E1) | `docs/33-banco-completo-multidataset.md` |

### Métricas [IC 95 % donde la fuente lo publica]

| # | Modelo | Accuracy | Exactitud balanceada | Precisión macro | Recall macro | F1 macro | AUC OvR macro | ECE ↓ | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | DINOv2 ViT-B/14 | 0,6811 | 0,6198 [0,5906; 0,6491] | 0,5283 | 0,6198 | 0,5625 [0,5383; 0,5872] | 0,9198 | 0,1056 | 0,5583 |
| 2 | ViT-B/16 | 0,6538 | 0,5794 [0,5512; 0,6062] | 0,4980 | 0,5794 | 0,5277 [0,5024; 0,5500] | 0,9058 | 0,1157 | 0,5230 |
| 3 | EfficientNet-B0 | 0,6609 | 0,5754 [0,5462; 0,6031] | 0,4947 | 0,5754 | 0,5250 [0,5003; 0,5480] | 0,9106 | 0,1062 | 0,5335 |
| 4 | DINOv2 ViT-S/14 | 0,6529 | 0,5972 [0,5678; 0,6262] | 0,4807 | 0,5972 | 0,5178 [0,4938; 0,5401] | 0,9100 | 0,0675 | 0,5236 |
| 5 | ResNet-50 | 0,6377 | 0,5685 [0,5388; 0,5984] | 0,4635 | 0,5685 | 0,4988 [0,4736; 0,5227] | 0,9058 | 0,0956 | 0,5045 |
| 6 | AlexNet | 0,5739 | 0,5330 [0,5034; 0,5622] | 0,3975 | 0,5330 | 0,4208 [0,4012; 0,4403] | 0,8733 | 0,0500 | 0,4357 |
| 7 | VGG19 | 0,5583 | 0,4966 [0,4676; 0,5246] | 0,3733 | 0,4966 | 0,3909 [0,3731; 0,4074] | 0,8599 | 0,0537 | 0,4172 |
| 8 | SimSiam ResNet-50 | 0,4784 | 0,4940 [0,4663; 0,5221] | 0,3359 | 0,4940 | 0,3286 [0,3126; 0,3434] | 0,8391 | 0,2215 | 0,3397 |
| — | B71-limpio (B* s42 limpio, config fija) | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |
| — | A71-limpio (sopa A* s42+s43 limpios) | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |
| — | B71 original | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |
| — | A71 original | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |

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
