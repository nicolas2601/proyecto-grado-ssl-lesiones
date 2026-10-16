# Tabla completa de resultados de clasificación (HAM10000)

Números copiados de los JSON del repositorio el 2026-10-02 (fuentes al final). Esta tabla NO se genera con script: si un JSON cambia, hay que actualizarla a mano.

## 1. Qué es nuestro modelo (Proposed sin segmentación, el propuesto final)

1. **Punto de partida:** DINOv2 ViT-B/14 público (Meta), una red preentrenada en millones de imágenes generales, sin dermatología.
2. **Adaptación auto-supervisada (sin etiquetas):** con SimSiam, la red aprende a dar la misma representación a dos versiones aumentadas de la misma lesión. Nunca ve el diagnóstico. Se entrenan solo los bloques 8-11 y las LayerNorm, y dos anclas la mantienen cerca del DINOv2 público ("SimSiam anclado").
3. **Término de tono (el aporte propio de la tesis):** la misma imagen con el tono de piel cambiado debe dar la misma representación, para no depender del color de la piel. **Proposed sin segmentación** = con término de tono (propuesto). **Modelo de control** = el mismo modelo sin él (ablación).
4. **Clasificador:** el backbone queda congelado y encima se entrena solo una regresión logística con las etiquetas. Las etiquetas nunca tocan el backbone.
5. **Protocolo v7.1 de Proposed sin segmentación:** pesos de Proposed sin segmentación semilla 42, features = CLS de las 4 últimas capas, pesos de clase balanceados (las clases raras pesan más) y TTA (se promedian 8 rotaciones y volteos). Se eligió en un holdout interno; validación se leyó una sola vez.

## 2. Qué significa cada métrica

Todas van de 0 a 1. Mayor es mejor, salvo ECE (↓), donde menor es mejor.

- **Accuracy (exactitud):** porcentaje total de aciertos. Engaña con este desbalance: ~67 % de las imágenes son nevus, así que decir siempre "nevus" ya da ~0,67.
- **Recall (sensibilidad):** de las lesiones que de verdad son de una clase, qué fracción detecta el modelo. **Recall macro** = el promedio de las 7 clases (= exactitud balanceada, la métrica oficial de ISIC 2018).
- **Precisión:** de las veces que el modelo dice "esta clase", qué fracción acierta. **Precisión macro** = el promedio de las 7 clases.
- **F1-score:** media armónica de precisión y recall; solo es alto si los dos lo son. **F1 macro** = el F1 de las 7 clases promediado sin pesar por tamaño. Es la **métrica principal**: el dermatofibroma (23 imágenes) vale lo mismo que el nevus.
- **AUC:** probabilidad de que el modelo le dé más puntaje de una clase a una imagen de esa clase que a una que no lo es (promedio de las 7). 0,5 = azar, 1 = perfecto. No depende de umbral.
- **ECE (↓):** cuánto se aleja la confianza del modelo de su acierto real. ECE 0,05 ≈ cuando dice "90 % seguro", acierta cerca del 85 %.
- **Recall melanoma:** de los melanomas reales, cuántos detecta. Es la métrica clínica clave: un melanoma no detectado es el error grave.
- **IC 95 %:** intervalo de confianza por bootstrap. Si los IC de dos modelos se solapan, no se puede afirmar que uno gane: es empate.

## 3. Cómo se clasifica cada modelo

- **Preentrenamiento supervisado:** la red se preentrenó con etiquetas (ImageNet): AlexNet, VGG19, ResNet-50, EfficientNet-B0, ViT-B/16.
- **Preentrenamiento auto-supervisado (SSL, sin etiquetas):** DINOv2 ViT-S/14, DINOv2 ViT-B/14, SimSiam ResNet-50 y nuestro modelo.
- **Congelado (sin fine-tuning):** la red no se toca; solo se entrena el clasificador lineal de encima.
- **Fine-tuning supervisado:** se reentrena toda la red con las etiquetas de HAM10000.

## 4. Ranking general (mejor a peor por F1 macro)

| # | Modelo | Preentrenamiento | Fine-tuning con etiquetas | Accuracy | Recall macro | Precisión macro | F1 macro [IC 95 %] | AUC | ECE (↓) | Recall melanoma | Frente a Proposed sin segmentación |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Proposed sin segmentación — nuestro propuesto final** | auto-supervisado + nuestra adaptación SSL | no (congelado) | 0.838 | 0.696 | — | **0.714** [0.670; 0.755] | 0.960 | 0.053 | 0.532 | — |
| 2 | ViT-B/16 | supervisado | sí | 0.807 | 0.752 | 0.699 | 0.706 [0.659; 0.743] | 0.961 | 0.046 | 0.703 | empate |
| 3 | ResNet-50 | supervisado | sí | 0.800 | 0.739 | 0.651 | 0.687 [0.648; 0.723] | 0.955 | 0.052 | 0.689 | empate |
| 4 | EfficientNet-B0 | supervisado | sí | 0.780 | 0.721 | 0.607 | 0.650 [0.607; 0.687] | 0.951 | 0.037 | 0.698 | empate |
| 5 | VGG19 | supervisado | sí | 0.744 | 0.714 | 0.571 | 0.625 [0.587; 0.659] | 0.944 | 0.057 | 0.707 | Proposed sin segmentación gana |
| 6 | DINOv2 ViT-B/14 | auto-supervisado | no (congelado) | 0.747 | 0.618 | 0.598 | 0.595 [0.551; 0.635] | 0.907 | 0.162 | 0.545 | Proposed sin segmentación gana |
| 7 | AlexNet | supervisado | sí | 0.719 | 0.670 | 0.531 | 0.579 [0.547; 0.610] | 0.933 | 0.036 | 0.689 | Proposed sin segmentación gana |
| 8 | DINOv2 ViT-S/14 | auto-supervisado | no (congelado) | 0.711 | 0.575 | 0.543 | 0.547 [0.504; 0.584] | 0.905 | 0.149 | 0.532 | Proposed sin segmentación gana |
| 9 | EfficientNet-B0 | supervisado | no (congelado) | 0.734 | 0.586 | 0.519 | 0.546 [0.509; 0.579] | 0.917 | 0.125 | 0.505 | Proposed sin segmentación gana |
| 10 | ViT-B/16 | supervisado | no (congelado) | 0.723 | 0.567 | 0.530 | 0.540 [0.497; 0.579] | 0.920 | 0.168 | 0.581 | Proposed sin segmentación gana |
| 11 | SimSiam ResNet-50 | auto-supervisado | sí | 0.688 | 0.678 | 0.472 | 0.532 [0.501; 0.563] | 0.924 | 0.025 | 0.590 | Proposed sin segmentación gana |
| 12 | ResNet-50 | supervisado | no (congelado) | 0.731 | 0.572 | 0.498 | 0.528 [0.492; 0.563] | 0.908 | 0.097 | 0.631 | Proposed sin segmentación gana |
| 13 | DINOv2 ViT-S/14 | auto-supervisado | sí | 0.639 | 0.621 | 0.447 | 0.495 [0.464; 0.525] | 0.915 | 0.054 | 0.577 | Proposed sin segmentación gana |
| 14 | AlexNet | supervisado | no (congelado) | 0.666 | 0.589 | 0.450 | 0.493 [0.458; 0.521] | 0.896 | 0.079 | 0.590 | Proposed sin segmentación gana |
| 15 | DINOv2 ViT-B/14 | auto-supervisado | sí | 0.593 | 0.532 | 0.395 | 0.427 [0.397; 0.454] | 0.888 | 0.060 | 0.658 | Proposed sin segmentación gana |
| 16 | VGG19 | supervisado | no (congelado) | 0.618 | 0.515 | 0.377 | 0.415 [0.386; 0.440] | 0.870 | 0.071 | 0.514 | Proposed sin segmentación gana |
| 17 | SimSiam ResNet-50 | auto-supervisado | no (congelado) | 0.490 | 0.494 | 0.335 | 0.317 [0.294; 0.341] | 0.829 | 0.247 | 0.459 | Proposed sin segmentación gana |

"Frente a Proposed sin segmentación": "empate" = los IC de F1 macro se solapan; "Proposed sin segmentación gana" = el IC de Proposed sin segmentación queda entero por encima.

**Fuera del ranking** (re-corrida exploratoria del fine-tuning de DINOv2 con un learning rate 10 veces menor, 5e-6; no es parte del banco registrado):

| Modelo | Preentrenamiento | Fine-tuning | Accuracy | Recall macro | Precisión macro | F1 macro [IC 95 %] | AUC | ECE (↓) | Recall melanoma | Frente a Proposed sin segmentación |
|---|---|---|---|---|---|---|---|---|---|---|
| DINOv2 ViT-S/14, LR 5e-6 | auto-supervisado | sí | 0.805 | 0.731 | 0.687 | 0.690 [0.645; 0.729] | 0.960 | 0.052 | 0.721 | empate |
| DINOv2 ViT-B/14, LR 5e-6 | auto-supervisado | sí | 0.792 | 0.762 | 0.650 | 0.684 [0.647; 0.716] | 0.963 | 0.048 | 0.761 | empate |

### Cómo leerlo

- **Proposed sin segmentación queda 1.º en F1 macro y en accuracy**, pero **empata** con los mejores fine-tuning supervisados (ViT-B/16, ResNet-50, EfficientNet-B0): los IC se solapan, así que no se puede decir que les gane.
- **Proposed sin segmentación les gana con IC separados a 13 de 16 modelos externos**: todos los congelados, y el fine-tuning de VGG19, AlexNet, SimSiam y DINOv2 (con el LR registrado).
- **La debilidad de Proposed sin segmentación es el recall:** recall macro 0.696 y recall de melanoma 0.532, contra 0.70-0.76 de los fine-tuning supervisados. Proposed sin segmentación acierta más en total porque es más preciso con los nevus, pero se le escapan más melanomas.
- **Proposed sin segmentación no usa ninguna etiqueta para ajustar la red**, y los fine-tuning sí. Que empate con ellos sin usar etiquetas en el backbone es el resultado defendible.

## 5. Estudio de ablación del propuesto (¿qué ingrediente sirvió?)

Una ablación le quita al modelo sus ingredientes uno por uno para ver cuál produce la mejora. Proposed sin segmentación tiene tres: **SSL anclado**, **término de tono** y **protocolo v7.1**. Todo se mide en las mismas 2004 imágenes de val, así que estas comparaciones sí son pareadas.

### 5.1 Escalera de ingredientes

| Configuración | SSL anclado | Término de tono | Protocolo v7.1 | Accuracy | Recall macro | F1 macro [IC 95 %] | AUC | ECE (↓) | Recall melanoma |
|---|---|---|---|---|---|---|---|---|---|
| DINOv2 público, probe base | no | no | no | 0.782 | 0.592 | 0.616 [0.569; 0.655] | 0.926 | 0.131 | 0.428 |
| DINOv2 público + protocolo de Proposed sin segmentación | no | no | sí | 0.828 | 0.666 | 0.684 [0.638; 0.720] | 0.962 | 0.069 | 0.514 |
| Modelo de control s42, probe base | sí | no | no | 0.832 | 0.602 | 0.646 [0.598; 0.689] | 0.962 | 0.014 | 0.487 |
| Proposed sin segmentación s42, probe base | sí | sí | no | 0.831 | 0.633 | 0.661 [0.616; 0.704] | 0.963 | 0.033 | 0.505 |
| Modelo de control (Modelo de control sin tono + su mejor protocolo) | sí | no | sí | 0.840 | 0.731 | **0.723** [0.680; 0.759] | 0.964 | 0.015 | 0.586 |
| **Proposed sin segmentación — propuesto final** | sí | sí | sí | 0.838 | 0.696 | 0.714 [0.670; 0.755] | 0.960 | 0.053 | 0.532 |

### 5.2 Contrastes pareados (diferencia de F1 macro [IC 95 %])

| Pregunta | Diferencia de F1 macro | Veredicto |
|---|---|---|
| ¿Sirve el SSL anclado? Modelo de control contra el público con el mismo protocolo | +0.091 [+0.050; +0.132] | **sí, mejora** |
| ¿Sirve el SSL anclado? Proposed sin segmentación contra el público con el mismo protocolo | +0.030 [−0.007; +0.069] | sin diferencia (pero mejora la calibración: ECE −0.016 [−0.030; −0.001]) |
| ¿Sirve el protocolo v7.1? Modelo de control contra Modelo de control con probe base | +0.077 [+0.036; +0.122] | **sí, mejora** |
| ¿Sirve el protocolo v7.1? Proposed sin segmentación contra Proposed sin segmentación con probe base | +0.054 [+0.011; +0.091] | **sí, mejora** |
| ¿Sirve el término de tono? Proposed sin segmentación contra Modelo de control, mismo probe, semilla 42 | +0.014 [−0.019; +0.054] | sin diferencia |
| ¿Sirve el término de tono? Proposed sin segmentación contra Modelo de control, mismo probe, semilla 43 | −0.021 [−0.058; +0.016] | sin diferencia |
| ¿Sirve el término de tono? Proposed sin segmentación contra Modelo de control, mismo probe, semilla 44 | −0.012 [−0.048; +0.027] | sin diferencia |
| Proposed sin segmentación contra Modelo de control (cambian tono y protocolo a la vez; no aísla el tono) | −0.009 [−0.046; +0.027] | sin diferencia (Modelo de control mejor calibrado: ECE +0.037 [+0.016; +0.049] a favor de Modelo de control) |
| Todo junto: Proposed sin segmentación contra el público con probe base | +0.098 [+0.057; +0.144] | **sí, mejora** |

### 5.3 Qué sirvió y qué no

1. **Sirvió el SSL anclado.** Adaptar DINOv2 a las lesiones sin etiquetas mejora el F1 macro en +0.091 con el mismo protocolo (Modelo de control). Es la mejora más grande.
2. **Sirvió el protocolo v7.1** (features, pesos de clase, TTA): +0.05 a +0.08 sin reentrenar nada.
3. **El término de tono NO sirvió para clasificar.** En la única prueba limpia (Proposed sin segmentación contra Modelo de control, todo lo demás igual) da "sin diferencia" en las tres semillas, y el signo cambia de una semilla a otra.
4. **Por eso el mejor de nuestros modelos es Modelo de control** (la ablación, sin tono): 0.723 contra 0.714 de Proposed sin segmentación, aunque la diferencia no es demostrable. Proposed sin segmentación sale 1.º en el ranking por el SSL anclado y el protocolo, no por el término de tono.
5. **Qué se puede afirmar en la tesis:** "la adaptación SSL anclada sobre DINOv2 mejora la clasificación frente al DINOv2 público y empata con el fine-tuning supervisado sin usar etiquetas en el backbone". **Qué no se puede afirmar:** "el término de tono mejora la clasificación".

## 6. Cuidado al comparar

- Hay dos particiones de val, las dos agrupadas por `lesion_id`: la de Kaggle (baselines y fine-tuning, n = 2014) y la de v7 (nuestro modelo, n = 2004). Entre particiones las filas no son pareadas, así que "Frente a Proposed sin segmentación" compara intervalos y no es un test pareado. Los tests pareados válidos son los de la sección 5.
- El fine-tuning supervisado elige su mejor época mirando val, así que su número es optimista.
- La precisión macro de Proposed sin segmentación y Modelo de control aparece como "—" porque no está guardada en los JSON. Se puede calcular a partir de las probabilidades guardadas (`v71_final_eval.probas.npz`).

## Fuentes

- Congelados: `results/baselines/results_*.json` (probe con mejor exactitud balanceada, la logística en todos), resumidos en `resultados/baselines/README.md`.
- Fine-tuning: `results/finetune/ham10000/results_*_ft.json`; LR reducido: `results/finetune_lrprobe/*.json`.
- Nuestro modelo y la ablación: `resultados/clasificacion/v71/v71_final_eval.json` (resumido en `v71/README.md`); término de tono por semilla: `resultados/clasificacion/v7/final_eval.json`, `v7/s43/final_eval.json`, `v7/s44/final_eval.json`.
