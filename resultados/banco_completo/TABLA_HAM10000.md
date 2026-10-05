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
- **Segmentación:** uso de máscaras de lesión como entrada. No se usa en este banco: la columna dice «no» en todas las filas (las variantes con segmentación son el bloque exploratorio E3 de docs/33).
- **Transductivo:** el pool de preentrenamiento auto-supervisado contenía las imágenes de val (sin etiquetas). Solo aplica a B71/A71 originales en ISIC 2019 e ISIC 2020.

En este val la clase «nv» es el 67,7 % de las imágenes (7 clases): predecir siempre «nv» da una accuracy de 0,6768 y una exactitud balanceada de 0,1429.

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
| 1 | A71 (ablación sin término de tono) | auto-supervisado propio (DINOv2 + SimSiam anclado sin término de tono, A*) | no | no | no | 2.004 | ok | probe logreg C1; precisión, recall y MCC desde las probabilidades | `resultados/clasificacion/v71/v71_final_eval.json` · `resultados/clasificacion/v71/v71_final_eval.probas.npz` |
| 2 | B71 (propuesto) | auto-supervisado propio (DINOv2 + SimSiam anclado + término de tono, B*) | no | no | no | 2.004 | ok | probe logreg C1; precisión, recall y MCC desde las probabilidades | `resultados/clasificacion/v71/v71_final_eval.json` · `resultados/clasificacion/v71/v71_final_eval.probas.npz` |
| 3 | DINOv2 público, su mejor protocolo v7.1 | auto-supervisado (DINOv2 público) | no | no | no | 2.004 | ok | probe logreg C1; precisión, recall y MCC desde las probabilidades | `resultados/clasificacion/v71/v71_final_eval.json` · `resultados/clasificacion/v71/v71_final_eval.probas.npz` |
| 4 | DINOv2 público, protocolo de B71 | auto-supervisado (DINOv2 público) | no | no | no | 2.004 | ok | probe logreg C1; precisión, recall y MCC desde las probabilidades | `resultados/clasificacion/v71/v71_final_eval.json` · `resultados/clasificacion/v71/v71_final_eval.probas.npz` |
| 5 | DINOv2 público, protocolo de A71 | auto-supervisado (DINOv2 público) | no | no | no | 2.004 | ok | probe logreg C1; precisión, recall y MCC desde las probabilidades | `resultados/clasificacion/v71/v71_final_eval.json` · `resultados/clasificacion/v71/v71_final_eval.probas.npz` |
| 6 | DINOv2 ViT-B/14 | auto-supervisado (DINOv2 público) | no | no | no | 2.014 | ok | — | `results/baselines/results_dinov2_vitb14.json` |
| 7 | DINOv2 ViT-S/14 | auto-supervisado (DINOv2 público) | no | no | no | 2.014 | ok | — | `results/baselines/results_dinov2_vits14.json` |
| 8 | EfficientNet-B0 | supervisado (ImageNet) | no | no | no | 2.014 | ok | — | `results/baselines/results_efficientnet_b0.json` |
| 9 | ViT-B/16 | supervisado (ImageNet) | no | no | no | 2.014 | ok | — | `results/baselines/results_vit_b16.json` |
| 10 | ResNet-50 | supervisado (ImageNet) | no | no | no | 2.014 | ok | — | `results/baselines/results_resnet50.json` |
| 11 | AlexNet | supervisado (ImageNet) | no | no | no | 2.014 | ok | — | `results/baselines/results_alexnet.json` |
| 12 | VGG19 | supervisado (ImageNet) | no | no | no | 2.014 | ok | — | `results/baselines/results_vgg19.json` |
| 13 | SimSiam ResNet-50 | auto-supervisado (SimSiam, ImageNet) | no | no | no | 2.014 | ok | — | `results/baselines/results_simsiam_r50.json` |
| — | B71-limpio (B* s42 limpio, config fija) | auto-supervisado propio (DINOv2 + SimSiam anclado + término de tono, B*) | no | no | no | — | pendiente | E1 de docs/33 | `docs/33-banco-completo-multidataset.md` |
| — | A71-limpio (sopa A* s42+s43 limpios) | auto-supervisado propio (DINOv2 + SimSiam anclado sin término de tono, A*) | no | no | no | — | pendiente | E1 de docs/33 | `docs/33-banco-completo-multidataset.md` |

### Métricas [IC 95 % donde la fuente lo publica]

| # | Modelo | Accuracy | Exactitud balanceada | Precisión macro | Recall macro | F1 macro | AUC OvR macro | ECE ↓ | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A71 (ablación sin término de tono) | 0,8398 [0,8229; 0,8553] | 0,7305 [0,6844; 0,7719] | 0,7197 | 0,7305 | 0,7229 [0,6795; 0,7592] | 0,9644 [0,9558; 0,9718] | 0,0154 [0,0135; 0,0339] | 0,6996 |
| 2 | B71 (propuesto) | 0,8378 [0,8218; 0,8533] | 0,6964 [0,6504; 0,7408] | 0,7381 | 0,6964 | 0,7141 [0,6703; 0,7547] | 0,9602 [0,9516; 0,9680] | 0,0529 [0,0428; 0,0695] | 0,6901 |
| 3 | DINOv2 público, su mejor protocolo v7.1 | 0,8268 [0,8104; 0,8423] | 0,6622 [0,6160; 0,7058] | 0,7226 | 0,6622 | 0,6857 [0,6412; 0,7208] | 0,9598 [0,9516; 0,9665] | 0,0702 [0,0605; 0,0871] | 0,6683 |
| 4 | DINOv2 público, protocolo de B71 | 0,8278 [0,8119; 0,8438] | 0,6659 [0,6176; 0,7102] | 0,7097 | 0,6659 | 0,6837 [0,6381; 0,7202] | 0,9619 [0,9549; 0,9680] | 0,0688 [0,0573; 0,0846] | 0,6712 |
| 5 | DINOv2 público, protocolo de A71 | 0,7959 [0,7774; 0,8124] | 0,6174 [0,5714; 0,6632] | 0,6547 | 0,6174 | 0,6320 [0,5875; 0,6710] | 0,9432 [0,9339; 0,9519] | 0,0806 [0,0684; 0,0987] | 0,6110 |
| 6 | DINOv2 ViT-B/14 | 0,7468 | 0,6179 [0,5763; 0,6601] | 0,5984 | 0,6179 | 0,5951 [0,5508; 0,6351] | 0,9070 | 0,1617 | 0,5634 |
| 7 | DINOv2 ViT-S/14 | 0,7110 | 0,5753 [0,5305; 0,6169] | 0,5429 | 0,5753 | 0,5474 [0,5037; 0,5838] | 0,9052 | 0,1489 | 0,5137 |
| 8 | EfficientNet-B0 | 0,7344 | 0,5857 [0,5472; 0,6233] | 0,5192 | 0,5857 | 0,5464 [0,5089; 0,5786] | 0,9170 | 0,1245 | 0,5375 |
| 9 | ViT-B/16 | 0,7229 | 0,5674 [0,5228; 0,6122] | 0,5301 | 0,5674 | 0,5402 [0,4971; 0,5795] | 0,9199 | 0,1685 | 0,5243 |
| 10 | ResNet-50 | 0,7314 | 0,5716 [0,5332; 0,6150] | 0,4980 | 0,5716 | 0,5280 [0,4917; 0,5633] | 0,9084 | 0,0973 | 0,5384 |
| 11 | AlexNet | 0,6663 | 0,5892 [0,5478; 0,6311] | 0,4496 | 0,5892 | 0,4929 [0,4584; 0,5215] | 0,8961 | 0,0793 | 0,4862 |
| 12 | VGG19 | 0,6182 | 0,5147 [0,4724; 0,5546] | 0,3771 | 0,5147 | 0,4151 [0,3858; 0,4404] | 0,8702 | 0,0707 | 0,4140 |
| 13 | SimSiam ResNet-50 | 0,4896 | 0,4942 [0,4531; 0,5394] | 0,3354 | 0,4942 | 0,3169 [0,2939; 0,3405] | 0,8291 | 0,2465 | 0,3186 |
| — | B71-limpio (B* s42 limpio, config fija) | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |
| — | A71-limpio (sopa A* s42+s43 limpios) | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |

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
