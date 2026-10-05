# Contrastes pareados del banco completo

Todos los contrastes pareados del banco, con dirección y si el IC excluye 0. «Δ» es brazo − control; el signo ya está resuelto para las métricas donde menor es mejor (ECE) en la columna «Lectura». Los números salen de los JSON; nada se recalcula aquí.

Generado por `scripts/build_bench_contrasts.py` desde los JSON del repositorio. No editar a mano. Plan y reglas: `docs/33-banco-completo-multidataset.md`.

## Advertencia de multiplicidad

Este documento lista 300 contrastes (el IC excluye 0 en 178) **sin corrección por multiplicidad**: cada IC es de 95 %, así que aun sin ningún efecto real se esperaría que cerca del 5 % (≈ 15) excluyera 0. Además comparten imágenes y no son independientes, así que esa cifra es solo orientativa. Se leen por familia y con la pregunta registrada de cada una, nunca escogiendo los que salen bien.

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

### HAM10000: B71 frente a los 8 baselines congelados (4 variantes × 3 métricas)

E3b, cómputo canónico (batch 64 + autocast), HAM10000 val n = 2.004, bootstrap pareado por `lesion_id`. Fuente: `resultados/banco_completo/e3b_bs64/e3b_summary.json`. Solo baselines CONGELADOS re-evaluados sobre la partición v6/v7.

| Brazo | Control | Métrica | Δ (brazo − control) [IC 95 %] | El IC excluye 0 | Lectura |
|---|---|---|---|---|---|
| B71 (1 vista, sin TTA) | AlexNet | F1 macro | +0,2067 [+0,1613; +0,2564] | sí | B71 (1 vista, sin TTA) mejor (el IC excluye 0) |
| B71 (1 vista, sin TTA) | AlexNet | Exactitud balanceada | +0,1164 [+0,0698; +0,1695] | sí | B71 (1 vista, sin TTA) mejor (el IC excluye 0) |
| B71 (1 vista, sin TTA) | AlexNet | Recall de melanoma | +0,0360 [-0,0573; +0,1302] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 + F4 (1 vista) | AlexNet + F4 | F1 macro | +0,2059 [+0,1607; +0,2551] | sí | B71 + F4 (1 vista) mejor (el IC excluye 0) |
| B71 + F4 (1 vista) | AlexNet + F4 | Exactitud balanceada | +0,1600 [+0,1135; +0,2115] | sí | B71 + F4 (1 vista) mejor (el IC excluye 0) |
| B71 + F4 (1 vista) | AlexNet + F4 | Recall de melanoma | +0,0450 [-0,0280; +0,1233] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 (TTA, canónico) | AlexNet | F1 macro | +0,2413 [+0,1942; +0,2898] | sí | B71 (TTA, canónico) mejor (el IC excluye 0) |
| B71 (TTA, canónico) | AlexNet | Exactitud balanceada | +0,1510 [+0,1052; +0,2044] | sí | B71 (TTA, canónico) mejor (el IC excluye 0) |
| B71 (TTA, canónico) | AlexNet | Recall de melanoma | +0,0135 [-0,0737; +0,0988] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 + F4 (TTA) | AlexNet + F4 | F1 macro | +0,2504 [+0,2050; +0,3010] | sí | B71 + F4 (TTA) mejor (el IC excluye 0) |
| B71 + F4 (TTA) | AlexNet + F4 | Exactitud balanceada | +0,1990 [+0,1543; +0,2490] | sí | B71 + F4 (TTA) mejor (el IC excluye 0) |
| B71 + F4 (TTA) | AlexNet + F4 | Recall de melanoma | +0,0495 [-0,0376; +0,1275] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 (1 vista, sin TTA) | VGG19 | F1 macro | +0,2985 [+0,2518; +0,3466] | sí | B71 (1 vista, sin TTA) mejor (el IC excluye 0) |
| B71 (1 vista, sin TTA) | VGG19 | Exactitud balanceada | +0,2412 [+0,1909; +0,2954] | sí | B71 (1 vista, sin TTA) mejor (el IC excluye 0) |
| B71 (1 vista, sin TTA) | VGG19 | Recall de melanoma | +0,1036 [+0,0088; +0,1950] | sí | B71 (1 vista, sin TTA) mejor (el IC excluye 0) |
| B71 + F4 (1 vista) | VGG19 + F4 | F1 macro | +0,2213 [+0,1751; +0,2688] | sí | B71 + F4 (1 vista) mejor (el IC excluye 0) |
| B71 + F4 (1 vista) | VGG19 + F4 | Exactitud balanceada | +0,1921 [+0,1469; +0,2435] | sí | B71 + F4 (1 vista) mejor (el IC excluye 0) |
| B71 + F4 (1 vista) | VGG19 + F4 | Recall de melanoma | +0,0991 [+0,0048; +0,1896] | sí | B71 + F4 (1 vista) mejor (el IC excluye 0) |
| B71 (TTA, canónico) | VGG19 | F1 macro | +0,3331 [+0,2844; +0,3800] | sí | B71 (TTA, canónico) mejor (el IC excluye 0) |
| B71 (TTA, canónico) | VGG19 | Exactitud balanceada | +0,2758 [+0,2200; +0,3309] | sí | B71 (TTA, canónico) mejor (el IC excluye 0) |
| B71 (TTA, canónico) | VGG19 | Recall de melanoma | +0,0811 [-0,0052; +0,1606] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 + F4 (TTA) | VGG19 + F4 | F1 macro | +0,2658 [+0,2212; +0,3180] | sí | B71 + F4 (TTA) mejor (el IC excluye 0) |
| B71 + F4 (TTA) | VGG19 + F4 | Exactitud balanceada | +0,2311 [+0,1860; +0,2880] | sí | B71 + F4 (TTA) mejor (el IC excluye 0) |
| B71 + F4 (TTA) | VGG19 + F4 | Recall de melanoma | +0,1036 [+0,0044; +0,2019] | sí | B71 + F4 (TTA) mejor (el IC excluye 0) |
| B71 (1 vista, sin TTA) | EfficientNet-B0 | F1 macro | +0,1754 [+0,1176; +0,2301] | sí | B71 (1 vista, sin TTA) mejor (el IC excluye 0) |
| B71 (1 vista, sin TTA) | EfficientNet-B0 | Exactitud balanceada | +0,1494 [+0,0914; +0,2078] | sí | B71 (1 vista, sin TTA) mejor (el IC excluye 0) |
| B71 (1 vista, sin TTA) | EfficientNet-B0 | Recall de melanoma | +0,1622 [+0,0805; +0,2398] | sí | B71 (1 vista, sin TTA) mejor (el IC excluye 0) |
| B71 + F4 (1 vista) | EfficientNet-B0 + F4 | F1 macro | +0,1273 [+0,0780; +0,1807] | sí | B71 + F4 (1 vista) mejor (el IC excluye 0) |
| B71 + F4 (1 vista) | EfficientNet-B0 + F4 | Exactitud balanceada | +0,1085 [+0,0605; +0,1617] | sí | B71 + F4 (1 vista) mejor (el IC excluye 0) |
| B71 + F4 (1 vista) | EfficientNet-B0 + F4 | Recall de melanoma | +0,0811 [-0,0000; +0,1594] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 (TTA, canónico) | EfficientNet-B0 | F1 macro | +0,2100 [+0,1579; +0,2609] | sí | B71 (TTA, canónico) mejor (el IC excluye 0) |
| B71 (TTA, canónico) | EfficientNet-B0 | Exactitud balanceada | +0,1840 [+0,1331; +0,2384] | sí | B71 (TTA, canónico) mejor (el IC excluye 0) |
| B71 (TTA, canónico) | EfficientNet-B0 | Recall de melanoma | +0,1396 [+0,0577; +0,2121] | sí | B71 (TTA, canónico) mejor (el IC excluye 0) |
| B71 + F4 (TTA) | EfficientNet-B0 + F4 | F1 macro | +0,1718 [+0,1227; +0,2235] | sí | B71 + F4 (TTA) mejor (el IC excluye 0) |
| B71 + F4 (TTA) | EfficientNet-B0 + F4 | Exactitud balanceada | +0,1475 [+0,0978; +0,1969] | sí | B71 + F4 (TTA) mejor (el IC excluye 0) |
| B71 + F4 (TTA) | EfficientNet-B0 + F4 | Recall de melanoma | +0,0856 [-0,0040; +0,1732] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 (1 vista, sin TTA) | ResNet-50 | F1 macro | +0,2125 [+0,1587; +0,2653] | sí | B71 (1 vista, sin TTA) mejor (el IC excluye 0) |
| B71 (1 vista, sin TTA) | ResNet-50 | Exactitud balanceada | +0,2100 [+0,1582; +0,2689] | sí | B71 (1 vista, sin TTA) mejor (el IC excluye 0) |
| B71 (1 vista, sin TTA) | ResNet-50 | Recall de melanoma | +0,1532 [+0,0687; +0,2398] | sí | B71 (1 vista, sin TTA) mejor (el IC excluye 0) |
| B71 + F4 (1 vista) | ResNet-50 + F4 | F1 macro | +0,1635 [+0,1155; +0,2157] | sí | B71 + F4 (1 vista) mejor (el IC excluye 0) |
| B71 + F4 (1 vista) | ResNet-50 + F4 | Exactitud balanceada | +0,1488 [+0,0991; +0,2033] | sí | B71 + F4 (1 vista) mejor (el IC excluye 0) |
| B71 + F4 (1 vista) | ResNet-50 + F4 | Recall de melanoma | +0,0901 [+0,0222; +0,1627] | sí | B71 + F4 (1 vista) mejor (el IC excluye 0) |
| B71 (TTA, canónico) | ResNet-50 | F1 macro | +0,2471 [+0,1969; +0,2943] | sí | B71 (TTA, canónico) mejor (el IC excluye 0) |
| B71 (TTA, canónico) | ResNet-50 | Exactitud balanceada | +0,2446 [+0,1963; +0,2960] | sí | B71 (TTA, canónico) mejor (el IC excluye 0) |
| B71 (TTA, canónico) | ResNet-50 | Recall de melanoma | +0,1306 [+0,0484; +0,2137] | sí | B71 (TTA, canónico) mejor (el IC excluye 0) |
| B71 + F4 (TTA) | ResNet-50 + F4 | F1 macro | +0,2080 [+0,1569; +0,2640] | sí | B71 + F4 (TTA) mejor (el IC excluye 0) |
| B71 + F4 (TTA) | ResNet-50 + F4 | Exactitud balanceada | +0,1878 [+0,1344; +0,2490] | sí | B71 + F4 (TTA) mejor (el IC excluye 0) |
| B71 + F4 (TTA) | ResNet-50 + F4 | Recall de melanoma | +0,0946 [+0,0148; +0,1744] | sí | B71 + F4 (TTA) mejor (el IC excluye 0) |
| B71 (1 vista, sin TTA) | ViT-B/16 | F1 macro | +0,1163 [+0,0608; +0,1748] | sí | B71 (1 vista, sin TTA) mejor (el IC excluye 0) |
| B71 (1 vista, sin TTA) | ViT-B/16 | Exactitud balanceada | +0,0875 [+0,0316; +0,1475] | sí | B71 (1 vista, sin TTA) mejor (el IC excluye 0) |
| B71 (1 vista, sin TTA) | ViT-B/16 | Recall de melanoma | +0,0045 [-0,0796; +0,0995] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 + F4 (1 vista) | ViT-B/16 + F4 | F1 macro | +0,1072 [+0,0629; +0,1557] | sí | B71 + F4 (1 vista) mejor (el IC excluye 0) |
| B71 + F4 (1 vista) | ViT-B/16 + F4 | Exactitud balanceada | +0,0968 [+0,0533; +0,1422] | sí | B71 + F4 (1 vista) mejor (el IC excluye 0) |
| B71 + F4 (1 vista) | ViT-B/16 + F4 | Recall de melanoma | +0,0405 [-0,0430; +0,1263] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 (TTA, canónico) | ViT-B/16 | F1 macro | +0,1509 [+0,1005; +0,2068] | sí | B71 (TTA, canónico) mejor (el IC excluye 0) |
| B71 (TTA, canónico) | ViT-B/16 | Exactitud balanceada | +0,1221 [+0,0699; +0,1776] | sí | B71 (TTA, canónico) mejor (el IC excluye 0) |
| B71 (TTA, canónico) | ViT-B/16 | Recall de melanoma | -0,0180 [-0,1053; +0,0755] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 + F4 (TTA) | ViT-B/16 + F4 | F1 macro | +0,1517 [+0,1089; +0,2013] | sí | B71 + F4 (TTA) mejor (el IC excluye 0) |
| B71 + F4 (TTA) | ViT-B/16 + F4 | Exactitud balanceada | +0,1358 [+0,0920; +0,1838] | sí | B71 + F4 (TTA) mejor (el IC excluye 0) |
| B71 + F4 (TTA) | ViT-B/16 + F4 | Recall de melanoma | +0,0450 [-0,0407; +0,1300] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 (1 vista, sin TTA) | SimSiam ResNet-50 | F1 macro | +0,1426 [+0,0934; +0,1938] | sí | B71 (1 vista, sin TTA) mejor (el IC excluye 0) |
| B71 (1 vista, sin TTA) | SimSiam ResNet-50 | Exactitud balanceada | +0,1208 [+0,0676; +0,1735] | sí | B71 (1 vista, sin TTA) mejor (el IC excluye 0) |
| B71 (1 vista, sin TTA) | SimSiam ResNet-50 | Recall de melanoma | +0,1171 [+0,0440; +0,1923] | sí | B71 (1 vista, sin TTA) mejor (el IC excluye 0) |
| B71 + F4 (1 vista) | SimSiam ResNet-50 + F4 | F1 macro | +0,1103 [+0,0609; +0,1645] | sí | B71 + F4 (1 vista) mejor (el IC excluye 0) |
| B71 + F4 (1 vista) | SimSiam ResNet-50 + F4 | Exactitud balanceada | +0,0926 [+0,0384; +0,1477] | sí | B71 + F4 (1 vista) mejor (el IC excluye 0) |
| B71 + F4 (1 vista) | SimSiam ResNet-50 + F4 | Recall de melanoma | +0,0856 [-0,0001; +0,1610] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 (TTA, canónico) | SimSiam ResNet-50 | F1 macro | +0,1772 [+0,1278; +0,2299] | sí | B71 (TTA, canónico) mejor (el IC excluye 0) |
| B71 (TTA, canónico) | SimSiam ResNet-50 | Exactitud balanceada | +0,1554 [+0,1041; +0,2078] | sí | B71 (TTA, canónico) mejor (el IC excluye 0) |
| B71 (TTA, canónico) | SimSiam ResNet-50 | Recall de melanoma | +0,0946 [+0,0128; +0,1673] | sí | B71 (TTA, canónico) mejor (el IC excluye 0) |
| B71 + F4 (TTA) | SimSiam ResNet-50 + F4 | F1 macro | +0,1548 [+0,1119; +0,2065] | sí | B71 + F4 (TTA) mejor (el IC excluye 0) |
| B71 + F4 (TTA) | SimSiam ResNet-50 + F4 | Exactitud balanceada | +0,1316 [+0,0818; +0,1848] | sí | B71 + F4 (TTA) mejor (el IC excluye 0) |
| B71 + F4 (TTA) | SimSiam ResNet-50 + F4 | Recall de melanoma | +0,0901 [-0,0001; +0,1741] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 (1 vista, sin TTA) | DINOv2 ViT-S/14 | F1 macro | +0,1287 [+0,0801; +0,1800] | sí | B71 (1 vista, sin TTA) mejor (el IC excluye 0) |
| B71 (1 vista, sin TTA) | DINOv2 ViT-S/14 | Exactitud balanceada | +0,0775 [+0,0303; +0,1289] | sí | B71 (1 vista, sin TTA) mejor (el IC excluye 0) |
| B71 (1 vista, sin TTA) | DINOv2 ViT-S/14 | Recall de melanoma | +0,0270 [-0,0608; +0,1081] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 + F4 (1 vista) | DINOv2 ViT-S/14 + F4 | F1 macro | +0,0607 [+0,0100; +0,1120] | sí | B71 + F4 (1 vista) mejor (el IC excluye 0) |
| B71 + F4 (1 vista) | DINOv2 ViT-S/14 + F4 | Exactitud balanceada | +0,0426 [-0,0129; +0,0974] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 + F4 (1 vista) | DINOv2 ViT-S/14 + F4 | Recall de melanoma | +0,0180 [-0,0637; +0,0921] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 (TTA, canónico) | DINOv2 ViT-S/14 | F1 macro | +0,1633 [+0,1198; +0,2066] | sí | B71 (TTA, canónico) mejor (el IC excluye 0) |
| B71 (TTA, canónico) | DINOv2 ViT-S/14 | Exactitud balanceada | +0,1121 [+0,0703; +0,1574] | sí | B71 (TTA, canónico) mejor (el IC excluye 0) |
| B71 (TTA, canónico) | DINOv2 ViT-S/14 | Recall de melanoma | +0,0045 [-0,0769; +0,0825] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 + F4 (TTA) | DINOv2 ViT-S/14 + F4 | F1 macro | +0,1052 [+0,0594; +0,1562] | sí | B71 + F4 (TTA) mejor (el IC excluye 0) |
| B71 + F4 (TTA) | DINOv2 ViT-S/14 + F4 | Exactitud balanceada | +0,0815 [+0,0312; +0,1350] | sí | B71 + F4 (TTA) mejor (el IC excluye 0) |
| B71 + F4 (TTA) | DINOv2 ViT-S/14 + F4 | Recall de melanoma | +0,0225 [-0,0675; +0,1035] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 (1 vista, sin TTA) | DINOv2 ViT-B/14 | F1 macro | +0,0793 [+0,0349; +0,1251] | sí | B71 (1 vista, sin TTA) mejor (el IC excluye 0) |
| B71 (1 vista, sin TTA) | DINOv2 ViT-B/14 | Exactitud balanceada | +0,0317 [-0,0148; +0,0774] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 (1 vista, sin TTA) | DINOv2 ViT-B/14 | Recall de melanoma | +0,0090 [-0,0676; +0,0834] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 + F4 (1 vista) | DINOv2 ViT-B/14 + F4 | F1 macro | +0,0829 [+0,0379; +0,1291] | sí | B71 + F4 (1 vista) mejor (el IC excluye 0) |
| B71 + F4 (1 vista) | DINOv2 ViT-B/14 + F4 | Exactitud balanceada | +0,0752 [+0,0223; +0,1240] | sí | B71 + F4 (1 vista) mejor (el IC excluye 0) |
| B71 + F4 (1 vista) | DINOv2 ViT-B/14 + F4 | Recall de melanoma | +0,0766 [+0,0044; +0,1496] | sí | B71 + F4 (1 vista) mejor (el IC excluye 0) |
| B71 (TTA, canónico) | DINOv2 ViT-B/14 | F1 macro | +0,1139 [+0,0728; +0,1585] | sí | B71 (TTA, canónico) mejor (el IC excluye 0) |
| B71 (TTA, canónico) | DINOv2 ViT-B/14 | Exactitud balanceada | +0,0663 [+0,0226; +0,1151] | sí | B71 (TTA, canónico) mejor (el IC excluye 0) |
| B71 (TTA, canónico) | DINOv2 ViT-B/14 | Recall de melanoma | -0,0135 [-0,0914; +0,0588] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 + F4 (TTA) | DINOv2 ViT-B/14 + F4 | F1 macro | +0,1274 [+0,0819; +0,1749] | sí | B71 + F4 (TTA) mejor (el IC excluye 0) |
| B71 + F4 (TTA) | DINOv2 ViT-B/14 + F4 | Exactitud balanceada | +0,1141 [+0,0652; +0,1639] | sí | B71 + F4 (TTA) mejor (el IC excluye 0) |
| B71 + F4 (TTA) | DINOv2 ViT-B/14 + F4 | Recall de melanoma | +0,0811 [-0,0001; +0,1641] | no | sin diferencia demostrable (el IC incluye 0) |

### HAM10000: F4 sobre cada baseline congelado (exploratorio)

E3b, cómputo canónico (batch 64 + autocast), HAM10000 val n = 2.004, bootstrap pareado por `lesion_id`. Fuente: `resultados/banco_completo/e3b_bs64/e3b_summary.json`. Solo baselines CONGELADOS re-evaluados sobre la partición v6/v7.

| Brazo | Control | Métrica | Δ (brazo − control) [IC 95 %] | El IC excluye 0 | Lectura |
|---|---|---|---|---|---|
| AlexNet + F4 | AlexNet | F1 macro | +0,0210 [-0,0121; +0,0536] | no | sin diferencia demostrable (el IC incluye 0) |
| AlexNet + F4 | AlexNet | Exactitud balanceada | -0,0258 [-0,0608; +0,0080] | no | sin diferencia demostrable (el IC incluye 0) |
| AlexNet + F4 | AlexNet | Recall de melanoma | -0,0225 [-0,0880; +0,0429] | no | sin diferencia demostrable (el IC incluye 0) |
| VGG19 + F4 | VGG19 | F1 macro | +0,0973 [+0,0577; +0,1352] | sí | VGG19 + F4 mejor (el IC excluye 0) |
| VGG19 + F4 | VGG19 | Exactitud balanceada | +0,0669 [+0,0274; +0,1094] | sí | VGG19 + F4 mejor (el IC excluye 0) |
| VGG19 + F4 | VGG19 | Recall de melanoma | -0,0090 [-0,0866; +0,0692] | no | sin diferencia demostrable (el IC incluye 0) |
| EfficientNet-B0 + F4 | EfficientNet-B0 | F1 macro | +0,0683 [+0,0204; +0,1125] | sí | EfficientNet-B0 + F4 mejor (el IC excluye 0) |
| EfficientNet-B0 + F4 | EfficientNet-B0 | Exactitud balanceada | +0,0587 [+0,0166; +0,1037] | sí | EfficientNet-B0 + F4 mejor (el IC excluye 0) |
| EfficientNet-B0 + F4 | EfficientNet-B0 | Recall de melanoma | +0,0676 [+0,0099; +0,1285] | sí | EfficientNet-B0 + F4 mejor (el IC excluye 0) |
| ResNet-50 + F4 | ResNet-50 | F1 macro | +0,0692 [+0,0344; +0,1046] | sí | ResNet-50 + F4 mejor (el IC excluye 0) |
| ResNet-50 + F4 | ResNet-50 | Exactitud balanceada | +0,0790 [+0,0463; +0,1148] | sí | ResNet-50 + F4 mejor (el IC excluye 0) |
| ResNet-50 + F4 | ResNet-50 | Recall de melanoma | +0,0495 [-0,0233; +0,1194] | no | sin diferencia demostrable (el IC incluye 0) |
| ViT-B/16 + F4 | ViT-B/16 | F1 macro | +0,0292 [-0,0164; +0,0724] | no | sin diferencia demostrable (el IC incluye 0) |
| ViT-B/16 + F4 | ViT-B/16 | Exactitud balanceada | +0,0085 [-0,0362; +0,0549] | no | sin diferencia demostrable (el IC incluye 0) |
| ViT-B/16 + F4 | ViT-B/16 | Recall de melanoma | -0,0495 [-0,1122; +0,0219] | no | sin diferencia demostrable (el IC incluye 0) |
| SimSiam ResNet-50 + F4 | SimSiam ResNet-50 | F1 macro | +0,0524 [+0,0132; +0,0929] | sí | SimSiam ResNet-50 + F4 mejor (el IC excluye 0) |
| SimSiam ResNet-50 + F4 | SimSiam ResNet-50 | Exactitud balanceada | +0,0459 [+0,0072; +0,0858] | sí | SimSiam ResNet-50 + F4 mejor (el IC excluye 0) |
| SimSiam ResNet-50 + F4 | SimSiam ResNet-50 | Recall de melanoma | +0,0180 [-0,0471; +0,0816] | no | sin diferencia demostrable (el IC incluye 0) |
| DINOv2 ViT-S/14 + F4 | DINOv2 ViT-S/14 | F1 macro | +0,0882 [+0,0477; +0,1330] | sí | DINOv2 ViT-S/14 + F4 mejor (el IC excluye 0) |
| DINOv2 ViT-S/14 + F4 | DINOv2 ViT-S/14 | Exactitud balanceada | +0,0528 [+0,0109; +0,0994] | sí | DINOv2 ViT-S/14 + F4 mejor (el IC excluye 0) |
| DINOv2 ViT-S/14 + F4 | DINOv2 ViT-S/14 | Recall de melanoma | -0,0045 [-0,0658; +0,0535] | no | sin diferencia demostrable (el IC incluye 0) |
| DINOv2 ViT-B/14 + F4 | DINOv2 ViT-B/14 | F1 macro | +0,0166 [-0,0193; +0,0508] | no | sin diferencia demostrable (el IC incluye 0) |
| DINOv2 ViT-B/14 + F4 | DINOv2 ViT-B/14 | Exactitud balanceada | -0,0257 [-0,0614; +0,0069] | no | sin diferencia demostrable (el IC incluye 0) |
| DINOv2 ViT-B/14 + F4 | DINOv2 ViT-B/14 | Recall de melanoma | -0,0811 [-0,1449; -0,0189] | sí | DINOv2 ViT-B/14 mejor (el IC excluye 0) |

### E3b: baselines que quedan por debajo de B71 con IC que excluye 0, por versión

Resumen de 8 baselines por celda; el detalle del canónico está arriba.

| Variante de B71 | Métrica | Canónico (batch 64) | Batch 32 (ya commiteado) |
|---|---|---|---|
| B71 (1 vista, sin TTA) | F1 macro | 8/8 | 8/8 |
| B71 (1 vista, sin TTA) | Exactitud balanceada | 7/8 | 7/8 |
| B71 (1 vista, sin TTA) | Recall de melanoma | 4/8 | 4/8 |
| B71 (TTA, canónico) | F1 macro | 8/8 | 8/8 |
| B71 (TTA, canónico) | Exactitud balanceada | 8/8 | 8/8 |
| B71 (TTA, canónico) | Recall de melanoma | 3/8 | 2/8 |
| B71 + F4 (1 vista) | F1 macro | 8/8 | 8/8 |
| B71 + F4 (1 vista) | Exactitud balanceada | 7/8 | 7/8 |
| B71 + F4 (1 vista) | Recall de melanoma | 3/8 | 5/8 |
| B71 + F4 (TTA) | F1 macro | 8/8 | 8/8 |
| B71 + F4 (TTA) | Exactitud balanceada | 8/8 | 8/8 |
| B71 + F4 (TTA) | Recall de melanoma | 2/8 | 4/8 |

### HAM10000: MCC puntual de los baselines sin y con F4 (E3b)

MCC de E3b: solo cifras puntuales de cada fila. Los JSON commiteados de E3 y E3b no traen contrastes pareados de MCC (se commitearon antes del commit 1d7e447, que agrega el MCC a esos contrastes): la última columna es la resta de dos puntos, SIN IC y NO es un contraste. El contraste pareado de MCC exige recomputar E3 y E3b con el código actual; ETA no declarada por el operador.

| Baseline | MCC sin F4 | MCC con F4 | Diferencia puntual (sin IC) |
|---|---|---|---|
| AlexNet | 0,4808 | 0,4870 | +0,0062 |
| VGG19 | 0,3975 | 0,4615 | +0,0641 |
| EfficientNet-B0 | 0,4932 | 0,5592 | +0,0660 |
| ResNet-50 | 0,4716 | 0,5429 | +0,0713 |
| ViT-B/16 | 0,5558 | 0,5843 | +0,0285 |
| SimSiam ResNet-50 | 0,5014 | 0,5553 | +0,0539 |
| DINOv2 ViT-S/14 | 0,5250 | 0,5810 | +0,0559 |
| DINOv2 ViT-B/14 | 0,5649 | 0,5741 | +0,0092 |

### HAM10000: F4 − B71, cómputo S0: batch 16 + autocast

Bootstrap pareado por `lesion_id`, n = 2.004. Fuente: `resultados/banco_completo/e3/e3_final_eval.json`.

| Brazo | Control | Métrica | Δ (brazo − control) [IC 95 %] | El IC excluye 0 | Lectura |
|---|---|---|---|---|---|
| F4 (todas las lesiones) | B71 | F1 macro | +0,0360 [+0,0020; +0,0778] | sí | F4 (todas las lesiones) mejor (el IC excluye 0) |
| F4 (todas las lesiones) | B71 | Exactitud balanceada | +0,0276 [-0,0102; +0,0688] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (todas las lesiones) | B71 | Recall de melanoma | +0,0360 [-0,0154; +0,0876] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (tercil L* más oscuro) | B71 | F1 macro | +0,0730 [+0,0052; +0,1482] | sí | F4 (tercil L* más oscuro) mejor (el IC excluye 0) |
| F4 (tercil L* más oscuro) | B71 | Exactitud balanceada | +0,0805 [+0,0095; +0,1617] | sí | F4 (tercil L* más oscuro) mejor (el IC excluye 0) |
| F4 (tercil L* más oscuro) | B71 | Recall de melanoma | +0,0303 [-0,0617; +0,1081] | no | sin diferencia demostrable (el IC incluye 0) |

### HAM10000: F4 − B71, cómputo S1: batch 64 + autocast (canónico, el de v7.1)

Bootstrap pareado por `lesion_id`, n = 2.004. Fuente: `resultados/banco_completo/e3/robustez/e3_s1_bs64_autocast/e3_final_eval.json`.

| Brazo | Control | Métrica | Δ (brazo − control) [IC 95 %] | El IC excluye 0 | Lectura |
|---|---|---|---|---|---|
| F4 (todas las lesiones) | B71 | F1 macro | +0,0300 [-0,0046; +0,0715] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (todas las lesiones) | B71 | Exactitud balanceada | +0,0222 [-0,0159; +0,0624] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (todas las lesiones) | B71 | Recall de melanoma | +0,0135 [-0,0446; +0,0714] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (tercil L* más oscuro) | B71 | F1 macro | +0,0625 [-0,0070; +0,1382] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (tercil L* más oscuro) | B71 | Exactitud balanceada | +0,0738 [+0,0022; +0,1544] | sí | F4 (tercil L* más oscuro) mejor (el IC excluye 0) |
| F4 (tercil L* más oscuro) | B71 | Recall de melanoma | -0,0101 [-0,1068; +0,0714] | no | sin diferencia demostrable (el IC incluye 0) |

### HAM10000: F4 − B71, cómputo S2: batch 64, fp32

Bootstrap pareado por `lesion_id`, n = 2.004. Fuente: `resultados/banco_completo/e3/robustez/e3_s2_bs64_fp32/e3_final_eval.json`.

| Brazo | Control | Métrica | Δ (brazo − control) [IC 95 %] | El IC excluye 0 | Lectura |
|---|---|---|---|---|---|
| F4 (todas las lesiones) | B71 | F1 macro | +0,0393 [+0,0039; +0,0824] | sí | F4 (todas las lesiones) mejor (el IC excluye 0) |
| F4 (todas las lesiones) | B71 | Exactitud balanceada | +0,0318 [-0,0069; +0,0749] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (todas las lesiones) | B71 | Recall de melanoma | +0,0360 [-0,0154; +0,0876] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (tercil L* más oscuro) | B71 | F1 macro | +0,0705 [-0,0004; +0,1490] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (tercil L* más oscuro) | B71 | Exactitud balanceada | +0,0839 [+0,0009; +0,1713] | sí | F4 (tercil L* más oscuro) mejor (el IC excluye 0) |
| F4 (tercil L* más oscuro) | B71 | Recall de melanoma | +0,0303 [-0,0617; +0,1081] | no | sin diferencia demostrable (el IC incluye 0) |

### HAM10000: F4 − B71, semilla de B* 42 (= cómputo canónico S1)

Bootstrap pareado por `lesion_id`, n = 2.004. Fuente: `resultados/banco_completo/e3/robustez/e3_s1_bs64_autocast/e3_final_eval.json`.

| Brazo | Control | Métrica | Δ (brazo − control) [IC 95 %] | El IC excluye 0 | Lectura |
|---|---|---|---|---|---|
| F4 (todas las lesiones) | B71 | F1 macro | +0,0300 [-0,0046; +0,0715] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (todas las lesiones) | B71 | Exactitud balanceada | +0,0222 [-0,0159; +0,0624] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (todas las lesiones) | B71 | Recall de melanoma | +0,0135 [-0,0446; +0,0714] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (tercil L* más oscuro) | B71 | F1 macro | +0,0625 [-0,0070; +0,1382] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (tercil L* más oscuro) | B71 | Exactitud balanceada | +0,0738 [+0,0022; +0,1544] | sí | F4 (tercil L* más oscuro) mejor (el IC excluye 0) |
| F4 (tercil L* más oscuro) | B71 | Recall de melanoma | -0,0101 [-0,1068; +0,0714] | no | sin diferencia demostrable (el IC incluye 0) |

### HAM10000: F4 − B71, semilla de B* 43

Bootstrap pareado por `lesion_id`, n = 2.004. Fuente: `resultados/banco_completo/e3/robustez/e3_rob_s43/e3_final_eval.json`.

| Brazo | Control | Métrica | Δ (brazo − control) [IC 95 %] | El IC excluye 0 | Lectura |
|---|---|---|---|---|---|
| F4 (todas las lesiones) | B71 | F1 macro | +0,0092 [-0,0339; +0,0499] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (todas las lesiones) | B71 | Exactitud balanceada | -0,0041 [-0,0473; +0,0354] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (todas las lesiones) | B71 | Recall de melanoma | +0,0000 [-0,0467; +0,0455] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (tercil L* más oscuro) | B71 | F1 macro | +0,0005 [-0,0648; +0,0611] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (tercil L* más oscuro) | B71 | Exactitud balanceada | -0,0028 [-0,0750; +0,0506] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (tercil L* más oscuro) | B71 | Recall de melanoma | +0,0404 [-0,0220; +0,1009] | no | sin diferencia demostrable (el IC incluye 0) |

### HAM10000: F4 − B71, semilla de B* 44

Bootstrap pareado por `lesion_id`, n = 2.004. Fuente: `resultados/banco_completo/e3/robustez/e3_rob_s44/e3_final_eval.json`.

| Brazo | Control | Métrica | Δ (brazo − control) [IC 95 %] | El IC excluye 0 | Lectura |
|---|---|---|---|---|---|
| F4 (todas las lesiones) | B71 | F1 macro | +0,0261 [-0,0039; +0,0577] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (todas las lesiones) | B71 | Exactitud balanceada | +0,0128 [-0,0168; +0,0440] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (todas las lesiones) | B71 | Recall de melanoma | +0,0180 [-0,0346; +0,0696] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (tercil L* más oscuro) | B71 | F1 macro | +0,0425 [-0,0065; +0,0924] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (tercil L* más oscuro) | B71 | Exactitud balanceada | +0,0227 [-0,0300; +0,0720] | no | sin diferencia demostrable (el IC incluye 0) |
| F4 (tercil L* más oscuro) | B71 | Recall de melanoma | +0,0707 [-0,0208; +0,1609] | no | sin diferencia demostrable (el IC incluye 0) |

### HAM10000: v7 y v7.1 (48 contrastes)

v7.1 en HAM10000, bootstrap pareado por `lesion_id`, n = 2.004. Fuente: `resultados/clasificacion/v71/v71_final_eval.json`.

| Brazo | Control | Métrica | Δ (brazo − control) [IC 95 %] | El IC excluye 0 | Lectura |
|---|---|---|---|---|---|
| A71 | DINOv2 público (prot. A71) | F1 macro | +0,0909 [+0,0499; +0,1321] | sí | A71 mejor (el IC excluye 0) |
| A71 | DINOv2 público (prot. A71) | Exactitud balanceada | +0,1131 [+0,0728; +0,1553] | sí | A71 mejor (el IC excluye 0) |
| A71 | DINOv2 público (prot. A71) | AUC OvR macro | +0,0212 [+0,0108; +0,0303] | sí | A71 mejor (el IC excluye 0) |
| A71 | DINOv2 público (prot. A71) | ECE (menor es mejor) | -0,0651 [-0,0770; -0,0414] | sí | A71 mejor (el IC excluye 0) |
| A71 | DINOv2 público (prot. A71) | Recall de melanoma | +0,1261 [+0,0524; +0,1949] | sí | A71 mejor (el IC excluye 0) |
| A71 | DINOv2 público (prot. A71) | AUC de melanoma | +0,0709 [+0,0488; +0,0929] | sí | A71 mejor (el IC excluye 0) |
| B71 | DINOv2 público (prot. B71) | F1 macro | +0,0304 [-0,0074; +0,0686] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 | DINOv2 público (prot. B71) | Exactitud balanceada | +0,0305 [-0,0097; +0,0712] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 | DINOv2 público (prot. B71) | AUC OvR macro | -0,0018 [-0,0081; +0,0043] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 | DINOv2 público (prot. B71) | ECE (menor es mejor) | -0,0159 [-0,0296; -0,0008] | sí | B71 mejor (el IC excluye 0) |
| B71 | DINOv2 público (prot. B71) | Recall de melanoma | +0,0180 [-0,0417; +0,0833] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 | DINOv2 público (prot. B71) | AUC de melanoma | +0,0090 [-0,0084; +0,0255] | no | sin diferencia demostrable (el IC incluye 0) |
| A71 | A v7 | F1 macro | +0,0765 [+0,0362; +0,1215] | sí | A71 mejor (el IC excluye 0) |
| A71 | A v7 | Exactitud balanceada | +0,1284 [+0,0864; +0,1737] | sí | A71 mejor (el IC excluye 0) |
| A71 | A v7 | AUC OvR macro | +0,0025 [-0,0045; +0,0084] | no | sin diferencia demostrable (el IC incluye 0) |
| A71 | A v7 | ECE (menor es mejor) | +0,0019 [-0,0147; +0,0138] | no | sin diferencia demostrable (el IC incluye 0) |
| A71 | A v7 | Recall de melanoma | +0,0991 [+0,0424; +0,1568] | sí | A71 mejor (el IC excluye 0) |
| A71 | A v7 | AUC de melanoma | +0,0181 [+0,0053; +0,0305] | sí | A71 mejor (el IC excluye 0) |
| B71 | B v7 | F1 macro | +0,0535 [+0,0105; +0,0913] | sí | B71 mejor (el IC excluye 0) |
| B71 | B v7 | Exactitud balanceada | +0,0637 [+0,0166; +0,1054] | sí | B71 mejor (el IC excluye 0) |
| B71 | B v7 | AUC OvR macro | -0,0028 [-0,0090; +0,0030] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 | B v7 | ECE (menor es mejor) | +0,0204 [+0,0032; +0,0322] | sí | B v7 mejor (el IC excluye 0) |
| B71 | B v7 | Recall de melanoma | +0,0270 [-0,0348; +0,0880] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 | B v7 | AUC de melanoma | -0,0079 [-0,0220; +0,0072] | no | sin diferencia demostrable (el IC incluye 0) |
| A71 | DINOv2 público (v7) | F1 macro | +0,1071 [+0,0675; +0,1475] | sí | A71 mejor (el IC excluye 0) |
| A71 | DINOv2 público (v7) | Exactitud balanceada | +0,1381 [+0,0976; +0,1792] | sí | A71 mejor (el IC excluye 0) |
| A71 | DINOv2 público (v7) | AUC OvR macro | +0,0387 [+0,0279; +0,0493] | sí | A71 mejor (el IC excluye 0) |
| A71 | DINOv2 público (v7) | ECE (menor es mejor) | -0,1153 [-0,1266; -0,0878] | sí | A71 mejor (el IC excluye 0) |
| A71 | DINOv2 público (v7) | Recall de melanoma | +0,1577 [+0,0829; +0,2286] | sí | A71 mejor (el IC excluye 0) |
| A71 | DINOv2 público (v7) | AUC de melanoma | +0,0836 [+0,0571; +0,1086] | sí | A71 mejor (el IC excluye 0) |
| B71 | DINOv2 público (v7) | F1 macro | +0,0983 [+0,0568; +0,1435] | sí | B71 mejor (el IC excluye 0) |
| B71 | DINOv2 público (v7) | Exactitud balanceada | +0,1040 [+0,0597; +0,1503] | sí | B71 mejor (el IC excluye 0) |
| B71 | DINOv2 público (v7) | AUC OvR macro | +0,0344 [+0,0244; +0,0451] | sí | B71 mejor (el IC excluye 0) |
| B71 | DINOv2 público (v7) | ECE (menor es mejor) | -0,0779 [-0,0935; -0,0581] | sí | B71 mejor (el IC excluye 0) |
| B71 | DINOv2 público (v7) | Recall de melanoma | +0,1036 [+0,0201; +0,1765] | sí | B71 mejor (el IC excluye 0) |
| B71 | DINOv2 público (v7) | AUC de melanoma | +0,0689 [+0,0406; +0,0960] | sí | B71 mejor (el IC excluye 0) |
| B71 | A71 | F1 macro | -0,0088 [-0,0457; +0,0270] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 | A71 | Exactitud balanceada | -0,0341 [-0,0752; +0,0064] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 | A71 | AUC OvR macro | -0,0042 [-0,0115; +0,0037] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 | A71 | ECE (menor es mejor) | +0,0374 [+0,0156; +0,0490] | sí | A71 mejor (el IC excluye 0) |
| B71 | A71 | Recall de melanoma | -0,0541 [-0,1149; +0,0093] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 | A71 | AUC de melanoma | -0,0146 [-0,0300; -0,0013] | sí | A71 mejor (el IC excluye 0) |
| DINOv2 público (mejor prot. v7.1) | DINOv2 público (v7) | F1 macro | +0,0699 [+0,0324; +0,1062] | sí | DINOv2 público (mejor prot. v7.1) mejor (el IC excluye 0) |
| DINOv2 público (mejor prot. v7.1) | DINOv2 público (v7) | Exactitud balanceada | +0,0699 [+0,0312; +0,1088] | sí | DINOv2 público (mejor prot. v7.1) mejor (el IC excluye 0) |
| DINOv2 público (mejor prot. v7.1) | DINOv2 público (v7) | AUC OvR macro | +0,0341 [+0,0246; +0,0446] | sí | DINOv2 público (mejor prot. v7.1) mejor (el IC excluye 0) |
| DINOv2 público (mejor prot. v7.1) | DINOv2 público (v7) | ECE (menor es mejor) | -0,0605 [-0,0749; -0,0430] | sí | DINOv2 público (mejor prot. v7.1) mejor (el IC excluye 0) |
| DINOv2 público (mejor prot. v7.1) | DINOv2 público (v7) | Recall de melanoma | +0,0811 [+0,0092; +0,1508] | sí | DINOv2 público (mejor prot. v7.1) mejor (el IC excluye 0) |
| DINOv2 público (mejor prot. v7.1) | DINOv2 público (v7) | AUC de melanoma | +0,0559 [+0,0332; +0,0788] | sí | DINOv2 público (mejor prot. v7.1) mejor (el IC excluye 0) |

### ISIC 2019: canónico (batch 64 + autocast)

Bootstrap pareado por imagen (sin grupo real), 1.000 remuestras. Fuente: `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2019/contrasts/A71__original__vs__PUBLIC_A71.json` y hermanos.

| Brazo | Control | Métrica | Δ (brazo − control) [IC 95 %] | El IC excluye 0 | Lectura |
|---|---|---|---|---|---|
| A71 original | DINOv2 público (prot. A71) | F1 macro | +0,0199 [-0,0051; +0,0438] | no | sin diferencia demostrable (el IC incluye 0) |
| A71 original | DINOv2 público (prot. A71) | Exactitud balanceada | +0,0295 [+0,0005; +0,0551] | sí | A71 original mejor (el IC excluye 0) |
| A71 original | DINOv2 público (prot. A71) | Accuracy | +0,0292 [+0,0168; +0,0409] | sí | A71 original mejor (el IC excluye 0) |
| A71 original | DINOv2 público (prot. A71) | AUC OvR macro | +0,0158 [+0,0108; +0,0215] | sí | A71 original mejor (el IC excluye 0) |
| A71 original | DINOv2 público (prot. A71) | ECE (menor es mejor) | -0,0303 [-0,0406; -0,0135] | sí | A71 original mejor (el IC excluye 0) |
| A71 original | DINOv2 público (prot. A71) | MCC | +0,0415 [+0,0238; +0,0571] | sí | A71 original mejor (el IC excluye 0) |
| B71-limpio | B71 original | F1 macro | -0,0055 [-0,0232; +0,0155] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | B71 original | Exactitud balanceada | +0,0030 [-0,0161; +0,0245] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | B71 original | Accuracy | -0,0022 [-0,0116; +0,0079] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | B71 original | AUC OvR macro | -0,0002 [-0,0030; +0,0026] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | B71 original | ECE (menor es mejor) | +0,0018 [-0,0092; +0,0113] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | B71 original | MCC | -0,0024 [-0,0153; +0,0121] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | DINOv2 público (prot. B71) | F1 macro | +0,0079 [-0,0143; +0,0284] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | DINOv2 público (prot. B71) | Exactitud balanceada | +0,0165 [-0,0029; +0,0351] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | DINOv2 público (prot. B71) | Accuracy | +0,0079 [-0,0047; +0,0193] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | DINOv2 público (prot. B71) | AUC OvR macro | +0,0063 [+0,0021; +0,0108] | sí | B71-limpio mejor (el IC excluye 0) |
| B71-limpio | DINOv2 público (prot. B71) | ECE (menor es mejor) | -0,0208 [-0,0326; -0,0079] | sí | B71-limpio mejor (el IC excluye 0) |
| B71-limpio | DINOv2 público (prot. B71) | MCC | +0,0154 [-0,0023; +0,0317] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | A71 original | F1 macro | +0,0236 [-0,0002; +0,0477] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | A71 original | Exactitud balanceada | -0,0008 [-0,0263; +0,0265] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | A71 original | Accuracy | -0,0016 [-0,0126; +0,0097] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | A71 original | AUC OvR macro | +0,0012 [-0,0023; +0,0048] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | A71 original | ECE (menor es mejor) | +0,0300 [+0,0127; +0,0391] | sí | A71 original mejor (el IC excluye 0) |
| B71 original | A71 original | MCC | +0,0003 [-0,0154; +0,0167] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | DINOv2 público (prot. B71) | F1 macro | +0,0134 [-0,0095; +0,0356] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | DINOv2 público (prot. B71) | Exactitud balanceada | +0,0135 [-0,0095; +0,0356] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | DINOv2 público (prot. B71) | Accuracy | +0,0101 [-0,0026; +0,0219] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | DINOv2 público (prot. B71) | AUC OvR macro | +0,0065 [+0,0024; +0,0111] | sí | B71 original mejor (el IC excluye 0) |
| B71 original | DINOv2 público (prot. B71) | ECE (menor es mejor) | -0,0226 [-0,0331; -0,0101] | sí | B71 original mejor (el IC excluye 0) |
| B71 original | DINOv2 público (prot. B71) | MCC | +0,0178 [+0,0003; +0,0353] | sí | B71 original mejor (el IC excluye 0) |

### ISIC 2019: sensibilidad (batch 16, transductivo)

Bootstrap pareado por imagen (sin grupo real), 1.000 remuestras. Fuente: `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2019/contrasts/A71__original__vs__PUBLIC_A71.json` y hermanos.

| Brazo | Control | Métrica | Δ (brazo − control) [IC 95 %] | El IC excluye 0 | Lectura |
|---|---|---|---|---|---|
| A71 original | DINOv2 público (prot. A71) | F1 macro | +0,0253 [-0,0002; +0,0501] | no | sin diferencia demostrable (el IC incluye 0) |
| A71 original | DINOv2 público (prot. A71) | Exactitud balanceada | +0,0353 [+0,0086; +0,0612] | sí | A71 original mejor (el IC excluye 0) |
| A71 original | DINOv2 público (prot. A71) | Accuracy | +0,0304 [+0,0182; +0,0420] | sí | A71 original mejor (el IC excluye 0) |
| A71 original | DINOv2 público (prot. A71) | AUC OvR macro | +0,0159 [+0,0108; +0,0216] | sí | A71 original mejor (el IC excluye 0) |
| A71 original | DINOv2 público (prot. A71) | ECE (menor es mejor) | -0,0275 [-0,0386; -0,0113] | sí | A71 original mejor (el IC excluye 0) |
| A71 original | DINOv2 público (prot. A71) | MCC | +0,0431 [+0,0259; +0,0595] | sí | A71 original mejor (el IC excluye 0) |
| B71 original | A71 original | F1 macro | +0,0234 [-0,0001; +0,0490] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | A71 original | Exactitud balanceada | +0,0010 [-0,0236; +0,0274] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | A71 original | Accuracy | -0,0012 [-0,0120; +0,0103] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | A71 original | AUC OvR macro | +0,0014 [-0,0022; +0,0050] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | A71 original | ECE (menor es mejor) | +0,0267 [+0,0104; +0,0376] | sí | A71 original mejor (el IC excluye 0) |
| B71 original | A71 original | MCC | +0,0008 [-0,0145; +0,0171] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | DINOv2 público (prot. B71) | F1 macro | +0,0166 [-0,0073; +0,0386] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | DINOv2 público (prot. B71) | Exactitud balanceada | +0,0186 [-0,0047; +0,0408] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | DINOv2 público (prot. B71) | Accuracy | +0,0122 [+0,0002; +0,0243] | sí | B71 original mejor (el IC excluye 0) |
| B71 original | DINOv2 público (prot. B71) | AUC OvR macro | +0,0068 [+0,0026; +0,0114] | sí | B71 original mejor (el IC excluye 0) |
| B71 original | DINOv2 público (prot. B71) | ECE (menor es mejor) | -0,0229 [-0,0322; -0,0102] | sí | B71 original mejor (el IC excluye 0) |
| B71 original | DINOv2 público (prot. B71) | MCC | +0,0206 [+0,0036; +0,0376] | sí | B71 original mejor (el IC excluye 0) |

### ISIC 2020: canónico (batch 64 + autocast)

Bootstrap pareado por `patient_id`, 1.000 remuestras. Fuente: `resultados/banco_completo/isic_eval/limpio_canonico_bs64/isic2020/contrasts/A71__original__vs__PUBLIC_A71.json` y hermanos.

| Brazo | Control | Métrica | Δ (brazo − control) [IC 95 %] | El IC excluye 0 | Lectura |
|---|---|---|---|---|---|
| A71 original | DINOv2 público (prot. A71) | F1 macro | +0,0573 [+0,0214; +0,0896] | sí | A71 original mejor (el IC excluye 0) |
| A71 original | DINOv2 público (prot. A71) | Exactitud balanceada | +0,0711 [+0,0249; +0,1174] | sí | A71 original mejor (el IC excluye 0) |
| A71 original | DINOv2 público (prot. A71) | Accuracy | +0,0054 [+0,0006; +0,0105] | sí | A71 original mejor (el IC excluye 0) |
| A71 original | DINOv2 público (prot. A71) | AUC OvR macro | +0,0700 [+0,0371; +0,1054] | sí | A71 original mejor (el IC excluye 0) |
| A71 original | DINOv2 público (prot. A71) | ECE (menor es mejor) | +0,0212 [+0,0143; +0,0264] | sí | DINOv2 público (prot. A71) mejor (el IC excluye 0) |
| A71 original | DINOv2 público (prot. A71) | MCC | +0,1168 [+0,0451; +0,1828] | sí | A71 original mejor (el IC excluye 0) |
| B71-limpio | B71 original | F1 macro | +0,0093 [-0,0224; +0,0409] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | B71 original | Exactitud balanceada | +0,0052 [-0,0391; +0,0461] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | B71 original | Accuracy | +0,0023 [-0,0016; +0,0064] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | B71 original | AUC OvR macro | +0,0230 [-0,0079; +0,0574] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | B71 original | ECE (menor es mejor) | -0,0011 [-0,0052; +0,0035] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | B71 original | MCC | +0,0169 [-0,0494; +0,0805] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | DINOv2 público (prot. B71) | F1 macro | +0,0387 [+0,0047; +0,0721] | sí | B71-limpio mejor (el IC excluye 0) |
| B71-limpio | DINOv2 público (prot. B71) | Exactitud balanceada | +0,0666 [+0,0302; +0,1033] | sí | B71-limpio mejor (el IC excluye 0) |
| B71-limpio | DINOv2 público (prot. B71) | Accuracy | -0,0035 [-0,0084; +0,0010] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | DINOv2 público (prot. B71) | AUC OvR macro | -0,0032 [-0,0346; +0,0262] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | DINOv2 público (prot. B71) | ECE (menor es mejor) | -0,0013 [-0,0054; +0,0026] | no | sin diferencia demostrable (el IC incluye 0) |
| B71-limpio | DINOv2 público (prot. B71) | MCC | +0,0822 [+0,0162; +0,1486] | sí | B71-limpio mejor (el IC excluye 0) |
| B71 original | A71 original | F1 macro | -0,0408 [-0,0704; -0,0080] | sí | A71 original mejor (el IC excluye 0) |
| B71 original | A71 original | Exactitud balanceada | -0,0503 [-0,0919; -0,0054] | sí | A71 original mejor (el IC excluye 0) |
| B71 original | A71 original | Accuracy | -0,0039 [-0,0088; +0,0009] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | A71 original | AUC OvR macro | -0,0494 [-0,0825; -0,0180] | sí | A71 original mejor (el IC excluye 0) |
| B71 original | A71 original | ECE (menor es mejor) | -0,0209 [-0,0269; -0,0136] | sí | B71 original mejor (el IC excluye 0) |
| B71 original | A71 original | MCC | -0,0830 [-0,1437; -0,0149] | sí | A71 original mejor (el IC excluye 0) |
| B71 original | DINOv2 público (prot. B71) | F1 macro | +0,0294 [-0,0024; +0,0615] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | DINOv2 público (prot. B71) | Exactitud balanceada | +0,0614 [+0,0243; +0,1026] | sí | B71 original mejor (el IC excluye 0) |
| B71 original | DINOv2 público (prot. B71) | Accuracy | -0,0057 [-0,0102; -0,0012] | sí | DINOv2 público (prot. B71) mejor (el IC excluye 0) |
| B71 original | DINOv2 público (prot. B71) | AUC OvR macro | -0,0261 [-0,0636; +0,0065] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | DINOv2 público (prot. B71) | ECE (menor es mejor) | -0,0003 [-0,0048; +0,0034] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | DINOv2 público (prot. B71) | MCC | +0,0652 [+0,0015; +0,1315] | sí | B71 original mejor (el IC excluye 0) |

### ISIC 2020: sensibilidad (batch 16, transductivo)

Bootstrap pareado por `patient_id`, 1.000 remuestras. Fuente: `resultados/banco_completo/isic_eval/originals_bs16_transductivo/isic2020/contrasts/A71__original__vs__PUBLIC_A71.json` y hermanos.

| Brazo | Control | Métrica | Δ (brazo − control) [IC 95 %] | El IC excluye 0 | Lectura |
|---|---|---|---|---|---|
| A71 original | DINOv2 público (prot. A71) | F1 macro | +0,0542 [+0,0210; +0,0867] | sí | A71 original mejor (el IC excluye 0) |
| A71 original | DINOv2 público (prot. A71) | Exactitud balanceada | +0,0707 [+0,0252; +0,1189] | sí | A71 original mejor (el IC excluye 0) |
| A71 original | DINOv2 público (prot. A71) | Accuracy | +0,0045 [+0,0000; +0,0092] | no | sin diferencia demostrable (el IC incluye 0) |
| A71 original | DINOv2 público (prot. A71) | AUC OvR macro | +0,0710 [+0,0385; +0,1063] | sí | A71 original mejor (el IC excluye 0) |
| A71 original | DINOv2 público (prot. A71) | ECE (menor es mejor) | +0,0235 [+0,0167; +0,0294] | sí | DINOv2 público (prot. A71) mejor (el IC excluye 0) |
| A71 original | DINOv2 público (prot. A71) | MCC | +0,1117 [+0,0439; +0,1784] | sí | A71 original mejor (el IC excluye 0) |
| B71 original | A71 original | F1 macro | -0,0431 [-0,0734; -0,0079] | sí | A71 original mejor (el IC excluye 0) |
| B71 original | A71 original | Exactitud balanceada | -0,0543 [-0,0968; -0,0093] | sí | A71 original mejor (el IC excluye 0) |
| B71 original | A71 original | Accuracy | -0,0041 [-0,0090; +0,0008] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | A71 original | AUC OvR macro | -0,0512 [-0,0850; -0,0207] | sí | A71 original mejor (el IC excluye 0) |
| B71 original | A71 original | ECE (menor es mejor) | -0,0245 [-0,0307; -0,0164] | sí | B71 original mejor (el IC excluye 0) |
| B71 original | A71 original | MCC | -0,0881 [-0,1500; -0,0169] | sí | A71 original mejor (el IC excluye 0) |
| B71 original | DINOv2 público (prot. B71) | F1 macro | +0,0324 [-0,0006; +0,0644] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | DINOv2 público (prot. B71) | Exactitud balanceada | +0,0654 [+0,0275; +0,1076] | sí | B71 original mejor (el IC excluye 0) |
| B71 original | DINOv2 público (prot. B71) | Accuracy | -0,0059 [-0,0103; -0,0015] | sí | DINOv2 público (prot. B71) mejor (el IC excluye 0) |
| B71 original | DINOv2 público (prot. B71) | AUC OvR macro | -0,0274 [-0,0664; +0,0054] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | DINOv2 público (prot. B71) | ECE (menor es mejor) | -0,0016 [-0,0058; +0,0028] | no | sin diferencia demostrable (el IC incluye 0) |
| B71 original | DINOv2 público (prot. B71) | MCC | +0,0716 [+0,0066; +0,1375] | sí | B71 original mejor (el IC excluye 0) |

### ISIC 2019: B71-limpio frente a los baselines (solo intervalos)

B71-limpio: F1 macro 0,6662 [0,6401; 0,6911]. Los baselines no tienen probabilidades por imagen commiteadas: solo se comparan intervalos. La comparación mezcla modelo y protocolo (ver «Cómo leer este documento»).

| Baseline congelado | F1 macro [IC 95 %] | Lectura |
|---|---|---|
| DINOv2 ViT-B/14 | 0,5625 [0,5383; 0,5872] | los IC no se solapan (solo de intervalos, no es un test pareado) |
| ViT-B/16 | 0,5277 [0,5024; 0,5500] | los IC no se solapan (solo de intervalos, no es un test pareado) |
| EfficientNet-B0 | 0,5250 [0,5003; 0,5480] | los IC no se solapan (solo de intervalos, no es un test pareado) |
| DINOv2 ViT-S/14 | 0,5178 [0,4938; 0,5401] | los IC no se solapan (solo de intervalos, no es un test pareado) |
| ResNet-50 | 0,4988 [0,4736; 0,5227] | los IC no se solapan (solo de intervalos, no es un test pareado) |
| AlexNet | 0,4208 [0,4012; 0,4403] | los IC no se solapan (solo de intervalos, no es un test pareado) |
| VGG19 | 0,3909 [0,3731; 0,4074] | los IC no se solapan (solo de intervalos, no es un test pareado) |
| SimSiam ResNet-50 | 0,3286 [0,3126; 0,3434] | los IC no se solapan (solo de intervalos, no es un test pareado) |

### ISIC 2020: B71-limpio frente a los baselines (solo intervalos)

B71-limpio: F1 macro 0,6229 [0,5865; 0,6602]. Los baselines no tienen probabilidades por imagen commiteadas: solo se comparan intervalos. La comparación mezcla modelo y protocolo (ver «Cómo leer este documento»).

| Baseline congelado | F1 macro [IC 95 %] | Lectura |
|---|---|---|
| ResNet-50 | 0,5601 [0,5396; 0,5790] | los IC no se solapan (solo de intervalos, no es un test pareado) |
| ViT-B/16 | 0,5493 [0,5319; 0,5668] | los IC no se solapan (solo de intervalos, no es un test pareado) |
| EfficientNet-B0 | 0,5457 [0,5264; 0,5640] | los IC no se solapan (solo de intervalos, no es un test pareado) |
| DINOv2 ViT-B/14 | 0,5427 [0,5252; 0,5577] | los IC no se solapan (solo de intervalos, no es un test pareado) |
| DINOv2 ViT-S/14 | 0,5266 [0,5119; 0,5393] | los IC no se solapan (solo de intervalos, no es un test pareado) |
| VGG19 | 0,5101 [0,4978; 0,5234] | los IC no se solapan (solo de intervalos, no es un test pareado) |
| AlexNet | 0,5008 [0,4894; 0,5122] | los IC no se solapan (solo de intervalos, no es un test pareado) |
| SimSiam ResNet-50 | 0,4795 [0,4693; 0,4906] | los IC no se solapan (solo de intervalos, no es un test pareado) |

### Contrastes pendientes

ETA declarada por el operador el 2026-10-05; no se mide desde el repositorio.

| Contraste | Motivo | ETA |
|---|---|---|
| D3 − D2 en PAD `target_test` (primario) | los pesos de D2 y D3 (GPU) y la única lectura de `target_test` de PAD no han terminado; el split de PAD ya existe (`resultados/banco_completo/pad_da/target_split.json`) | ~10:30 UTC |
| Demás contrastes pareados entre D0, D1, D2 y D3 (descriptivos) | los pesos de D2 y D3 (GPU) y la única lectura de `target_test` de PAD no han terminado; el split de PAD ya existe (`resultados/banco_completo/pad_da/target_split.json`) | ~10:30 UTC |
| A71-limpio contra A71, y B71-limpio contra A71-limpio | falta A* s43 limpio (GPU0 en curso); A71-limpio es la sopa de A* s42 + s43 limpios | ~15:00 UTC |
| B71-limpio contra B71 original (predicción registrada) | el contraste B71-limpio contra B71 original exige el original con el mismo cómputo canónico; la corrida de batch 16 no es comparable (cambia el cómputo, no solo los datos) | ~15:00 UTC |
| Contrastes pareados de MCC en HAM10000 (E3 y E3b) | recomputar E3 y E3b con el código actual (1d7e447) | no declarada por el operador |
