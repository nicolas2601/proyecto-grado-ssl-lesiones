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
- **Segmentación:** uso de máscaras de lesión como entrada. No se usa en este banco: la columna dice «no» en todas las filas (las variantes con segmentación son el bloque exploratorio E3 de docs/33).
- **Transductivo:** el pool de preentrenamiento auto-supervisado contenía las imágenes de val (sin etiquetas). Solo aplica a B71/A71 originales en ISIC 2019 e ISIC 2020.

En este val la clase «benigno» es el 98,2 % de las imágenes (2 clases): predecir siempre «benigno» da una accuracy de 0,9817 y una exactitud balanceada de 0,5000.

## Cómo leer esta tabla

Particiones de val de este banco (no son la misma partición):

- HAM10000, baselines (Kaggle): val n = 2.014; StratifiedGroupKFold por lesion_id (0 lesiones compartidas).
- HAM10000, propuesto (v6/v7): val n = 2.004; partición agrupada por `lesion_id`.
- ISIC 2019, baselines (Kaggle): val n = 5.067; StratifiedKFold estratificado simple SIN grupo de paciente (el release no trae patient_id) - riesgo de fuga entre imagenes del mismo paciente, metricas con sesgo optimista.
- ISIC 2020, baselines (Kaggle): val n = 6.660; StratifiedGroupKFold por patient_id (0 pacientes compartidos).

- **Entre particiones distintas la comparación es solo de intervalos (IC 95 %)**, no un test pareado: HAM10000 usa la partición v6/v7 (n = 2.004) para el propuesto y la de Kaggle (n = 2.014) para los baselines; ISIC 2019 e ISIC 2020 tienen la suya. Los contrastes pareados se harán solo donde dos modelos comparten partición e imágenes (docs/33, sección 5).
- **Optimista: época elegida en val.** Todas las filas de fine-tuning (HAM10000, ISIC 2019 y la corrida pendiente de ISIC 2020) eligen su mejor época por balanced accuracy sobre val; el plan decía holdout interno y se mantuvo la receta existente por consistencia entre datasets (errata de docs/33). El propuesto no tiene fine-tuning y no se ve afectado.
- Los IC solo existen donde la fuente los publica: los baselines, para exactitud balanceada y F1 macro; B71/A71, para accuracy, exactitud balanceada, F1 macro, AUC y ECE. Precisión macro, recall macro y MCC de los baselines salen exactos de la matriz de confusión y van sin IC; en B71/A71 salen de las probabilidades commiteadas, también sin IC.
- Las filas «pendiente» y «no ejecutado todavia» no tienen número: se listan para que nada parezca olvidado. No se inventa ninguna cifra.

## Congelados (sin fine-tuning)

Orden: F1 macro de mayor a menor; las filas sin número van al final.

### Condición

| # | Modelo | Preentrenamiento | Fine-tuning | Segmentación | Transductivo | Val n | Estado | Nota | Fuente |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ResNet-50 | supervisado (ImageNet) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_resnet50_isic2020.json` |
| 2 | ViT-B/16 | supervisado (ImageNet) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_vit_b16_isic2020.json` |
| 3 | EfficientNet-B0 | supervisado (ImageNet) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_efficientnet_b0_isic2020.json` |
| 4 | DINOv2 ViT-B/14 | auto-supervisado (DINOv2 público) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_dinov2_vitb14_isic2020.json` |
| 5 | DINOv2 ViT-S/14 | auto-supervisado (DINOv2 público) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_dinov2_vits14_isic2020.json` |
| 6 | VGG19 | supervisado (ImageNet) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_vgg19_isic2020.json` |
| 7 | AlexNet | supervisado (ImageNet) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_alexnet_isic2020.json` |
| 8 | SimSiam ResNet-50 | auto-supervisado (SimSiam, ImageNet) | no | no | no | 6.660 | ok | — | `results/baselines_isic2020/results_simsiam_r50_isic2020.json` |
| — | B71-limpio (B* s42 limpio, config fija) | auto-supervisado propio (DINOv2 + SimSiam anclado + término de tono, B*) | no | no | no | — | pendiente | E1 de docs/33 | `docs/33-banco-completo-multidataset.md` |
| — | A71-limpio (sopa A* s42+s43 limpios) | auto-supervisado propio (DINOv2 + SimSiam anclado sin término de tono, A*) | no | no | no | — | pendiente | E1 de docs/33 | `docs/33-banco-completo-multidataset.md` |
| — | B71 original | auto-supervisado propio (DINOv2 + SimSiam anclado + término de tono, B*) | no | no | sí | — | pendiente | transductivo: imágenes de val en el pool SSL, sin etiquetas (E1) | `docs/33-banco-completo-multidataset.md` |
| — | A71 original | auto-supervisado propio (DINOv2 + SimSiam anclado sin término de tono, A*) | no | no | sí | — | pendiente | transductivo: imágenes de val en el pool SSL, sin etiquetas (E1) | `docs/33-banco-completo-multidataset.md` |

### Métricas [IC 95 % donde la fuente lo publica]

| # | Modelo | Accuracy | Exactitud balanceada | Precisión macro | Recall macro | F1 macro | AUC OvR macro | ECE ↓ | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ResNet-50 | 0,9161 | 0,6878 [0,6425; 0,7321] | 0,5448 | 0,6878 | 0,5601 [0,5396; 0,5790] | 0,8297 | 0,0275 | 0,1834 |
| 2 | ViT-B/16 | 0,8883 | 0,7340 [0,6903; 0,7777] | 0,5415 | 0,7340 | 0,5493 [0,5319; 0,5668] | 0,8275 | 0,0598 | 0,1971 |
| 3 | EfficientNet-B0 | 0,9186 | 0,6368 [0,5954; 0,6818] | 0,5352 | 0,6368 | 0,5457 [0,5264; 0,5640] | 0,7607 | 0,0378 | 0,1387 |
| 4 | DINOv2 ViT-B/14 | 0,8994 | 0,6753 [0,6284; 0,7186] | 0,5355 | 0,6753 | 0,5427 [0,5252; 0,5577] | 0,7999 | 0,0559 | 0,1578 |
| 5 | DINOv2 ViT-S/14 | 0,8470 | 0,7572 [0,7132; 0,7981] | 0,5346 | 0,7572 | 0,5266 [0,5119; 0,5393] | 0,8576 | 0,0561 | 0,1886 |
| 6 | VGG19 | 0,8233 | 0,7370 [0,6951; 0,7789] | 0,5286 | 0,7370 | 0,5101 [0,4978; 0,5234] | 0,8128 | 0,0333 | 0,1647 |
| 7 | AlexNet | 0,7940 | 0,7744 [0,7354; 0,8125] | 0,5292 | 0,7744 | 0,5008 [0,4894; 0,5122] | 0,8574 | 0,0442 | 0,1791 |
| 8 | SimSiam ResNet-50 | 0,7518 | 0,7650 [0,7301; 0,8000] | 0,5249 | 0,7650 | 0,4795 [0,4693; 0,4906] | 0,8468 | 0,0851 | 0,1624 |
| — | B71-limpio (B* s42 limpio, config fija) | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |
| — | A71-limpio (sopa A* s42+s43 limpios) | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |
| — | B71 original | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |
| — | A71 original | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |

## Con fine-tuning

Orden: F1 macro de mayor a menor; las filas sin número van al final.

### Condición

| # | Modelo | Preentrenamiento | Fine-tuning | Segmentación | Transductivo | Val n | Estado | Nota | Fuente |
|---|---|---|---|---|---|---|---|---|---|
| — | 8 baselines con fine-tuning | varios | sí | no | no | — | no ejecutado todavia | G1.3 de docs/33; optimista: época elegida en val (receta existente) | `docs/33-banco-completo-multidataset.md` |

### Métricas [IC 95 % donde la fuente lo publica]

| # | Modelo | Accuracy | Exactitud balanceada | Precisión macro | Recall macro | F1 macro | AUC OvR macro | ECE ↓ | MCC |
|---|---|---|---|---|---|---|---|---|---|
| — | 8 baselines con fine-tuning | no ejecutado todavia | no ejecutado todavia | no ejecutado todavia | no ejecutado todavia | no ejecutado todavia | no ejecutado todavia | no ejecutado todavia | no ejecutado todavia |
