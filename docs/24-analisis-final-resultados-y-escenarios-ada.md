# Análisis final de resultados y escenarios para segmentación y ADA

> **Documento histórico.** Describe una etapa anterior del proyecto y fue superado. Los resultados vigentes están en el [README](../README.md) y en los documentos 27 a 33. Se conserva para mostrar cómo se llegó al diseño actual.

| Campo | Valor |
|---|---|
| Fecha de corte | 2026-09-23 |
| Rama consolidada | `proyecto-de-grado` (snapshot de `proposed-cluster`) |
| Commit base de evidencia | `39e0b19` (`proposed-cluster`) |
| Corrida de segmentación | `2026-09-21-final`, preset `cluster-final`, commit de ejecución `d81a492` |
| Banco clasificador | Artefactos en `resultados/clasificacion/`, documentación final hasta `c7dce58` |

## 1. Propósito y reglas de lectura

Este documento consolida los resultados finales disponibles en `proposed-cluster`, explica las limitaciones que cambian su interpretación y propone tres escenarios concretos para decidir cómo continuar la tesis.

La fuente de verdad son los artefactos fechados de las corridas finales de segmentación y clasificación. Los informes anteriores se conservan como historia del proyecto, pero **no reemplazan esos artefactos**. Cada afirmación se clasifica como:

- **Dato observado:** valor presente en un JSON, CSV, log o manifiesto de la corrida.
- **Interpretación:** explicación compatible con los datos, todavía no demostrada como causa.
- **No establecido:** la corrida no permite sostener esa conclusión.

Las métricas redondeadas en las tablas se muestran con la precisión publicada en los artefactos. Los valores con más dígitos están en los JSON y CSV referenciados.

## 2. Resumen ejecutivo

1. La segmentación sin etiquetas produce Dice medio de **0,784658 en HAM10000** y **0,717677 en ISIC 2018**. Frente a SegRNorm, mejora el Dice en HAM10000; en ISIC 2018 empata en Dice medio, pero obtiene mejor HD95. El límite de área aumentado a 0,75 no mejoró el resultado y queda descartado como cambio por defecto.
2. En HAM10000, el Dice es **0,861643 en el tercil claro de L* y 0,750279 en el oscuro**. La diferencia directa calculada desde los promedios sin redondear es 0,1113648455 (0,1114 redondeado). Esto es una señal de disparidad de segmentación dentro de la evaluación HAM; no equivale a fototipo clínico ni, por sí solo, demuestra generalización a una población colombiana o a imágenes no vistas.
3. En ISIC 2018, la diferencia por tono es menor y el análisis por tamaño es mixto. El documento informa que el tono está disponible para 722 de 2.594 imágenes y llama externas a las 1.872 restantes; sin embargo, el código que agrupa los resultados no verifica directamente la pertenencia al pool por ID. La afirmación de independencia externa requiere una atestación ID por ID.
4. El banco clasificador compara DINOv2 público con cuatro brazos SimSiam de dominio, todos inicializados desde los pesos públicos DINOv2. El brazo de pares por tono pierde frente a su control plain en macro-F1 y balanced accuracy en ViT-S y ViT-B. Los intervalos de AUC y ECE entre proposed y plain incluyen cero.
5. No se demuestra que la adaptación por tono reduzca la brecha de clasificación. Los seis intervalos por estrato se solapan; además, el grupo `light` tiene 183 imágenes frente a 820 `dark` y 957 `medium`.
6. Existe una relación real entre segmentación y el entrenamiento del brazo propuesto: las pseudo-máscaras se usan para generar imágenes con desplazamiento de tono. Los falsos negativos del segmentador podrían hacer que píxeles de lesión se modifiquen. En cambio, el banco final evalúa cada backbone sobre la imagen RGB original, sin aplicar una máscara en inferencia. La relación causal entre error de máscara y caída de clasificación es plausible, pero todavía debe probarse.
7. El siguiente paso con mayor valor es auditar qué píxeles se modificaron en los pares por tono según la calidad de máscara. No se recomienda otra corrida grande antes de esa auditoría.

## 3. Fuente de verdad y estado de los informes

### 3.1 Rama y artefactos

**Base verificada:** el análisis toma los artefactos de `proposed-cluster` hasta `c7dce58`; esta rama los conserva y agrega el índice de consolidación. Los artefactos principales son:

| Resultado | Ruta de referencia |
|---|---|
| Segmentación final | `plan-segmentacion/camino-b-cnn/resultados/2026-09-21-final/` |
| Estratos de tono corregidos | `plan-segmentacion/camino-b-cnn/resultados/2026-09-21-final/strata_v3/` y `resultados/tono/strata_v3/` |
| Banco clasificador completo | `resultados/clasificacion/bank.json` |
| Resumen de banco | `resultados/clasificacion/bank.md` |
| Contrastes pareados | `resultados/clasificacion/bootstrap.json` |
| Métricas por estrato | `resultados/clasificacion/strata.json` |
| Manifiestos de entrenamiento | `resultados/clasificacion/runs/*/run.json` |
| Interpretación consolidada de resultados | `resultados/README.md` |

La segmentación final se ejecutó con preset `cluster-final` y commit `d81a492`; el resumen registra su finalización el 2026-09-21. El banco clasificador final aparece en commits del 2026-09-22 y su intervalo bootstrap se documentó en `c7dce58`.

### 3.2 Documentos anteriores que no sustituyen a la corrida final

| Documento | Estado que representa | Uso correcto en este análisis |
|---|---|---|
| `README.md` | Portada mínima: objetivos, arquitectura y equipo | Las tablas largas de baselines viven en [`resultados/baselines/README.md`](../resultados/baselines/README.md), que es donde las compara el test de frescura. Usar `resultados/` y este informe para la lectura final. |
| `docs/11-resultados-finales.md` | Corrida v4 con ResNet-50 y dos semillas | Histórico, con protocolo distinto al banco final. |
| `docs/08-estado-actual-y-next-steps.md` | Estado de agosto y banco de baselines previo | Histórico; no representa las corridas de septiembre. |
| `docs/22-pipeline-propuesto-segmentacion-tono-clasificador.md` | Corrida preliminar de 4 horas-GPU | No es el resultado final. El brazo propuesto tenía un término de pérdida duplicado; está corregido en el commit `08727e8`. |
| `INFORME_ACADEMICO.md` | Informe fechado antes del cierre del banco final | Debe reconciliarse con los artefactos de septiembre; CIRCLe y el banco tone-paired son experimentos diferentes. |
| `resultados/README.md` y `resultados/segmentacion/README.md` | Resúmenes de los artefactos finales | Son los resúmenes de referencia, contrastados aquí con sus JSON, CSV, logs y manifests. |

La corrida preliminar de 4 horas-GPU reportaba un resultado favorable al brazo propuesto, pero no debe presentarse como evidencia final. Con el error corregido y un presupuesto de 12 horas-GPU por brazo, la dirección se invierte.

## 4. Segmentación sin máscaras ni etiquetas

### 4.1 Protocolo final

El preentrenamiento MoCo usa un pool de **58.457 imágenes sin etiquetas**: HAM10000 (10.015), ISIC 2019 sin las imágenes de HAM (15.316) e ISIC 2020 (33.126), deduplicadas por ID. En la última ronda de pseudo-máscaras del modelo de segmentación (`pseudo_r2`), 2.508 quedan excluidas (4,2903 %: 877 sin región y 1.631 por área); el total menos las excluidas es 55.949. Es un conteo distinto del exportador de entradas de tono: `resultados/README.md` y los manifests del banco informan 55.034 imágenes/pares elegibles y 3.423 en cuarentena para la transformación de tono. No se deben combinar esos dos filtros como si fueran la misma etapa. El resumen de buckets `plan-segmentacion/camino-b-cnn/resultados/2026-09-21-final/fst/fst_summary.json` registra 54.978 con estado `ok` y 3.479 en cuarentena. La diferencia de 56 respecto al exportador de tono no se reconcilia en los artefactos comprometidos; se conserva como discrepancia abierta y no se atribuye a una causa no medida.

Las máscaras de referencia de ISIC 2018 y HAM10000 se usan solo al evaluar, no al entrenar el pipeline final. La configuración exacta está en `plan-segmentacion/comun/configs/cluster-final.json`.

| Etapa | Configuración final |
|---|---|
| Entrada y campo | 224 × 224; 2 % reservado como pseudo-validación; umbral mínimo de campo 0,06 y margen 0,04. |
| MoCo v2 | ResNet-50 desde cero, 30 épocas, batch 128 por GPU, AdamW, LR 5e-4, weight decay 1e-4, warm-up 2, cola 16.384, momentum 0,999, temperatura 0,2, split-BN 4. |
| CCAM | Cabeza 3 × 3 sobre layer3 y layer4 congeladas, 5 épocas, batch 64, LR 1e-3, alpha 0,25. |
| Pseudo-máscaras | Fondo < 0,35; lesión > 0,65; ignorar valores intermedios; área entre 0,003 y 0,6 del campo; ignorar tinta y brillos. |
| modelo de segmentación | U-Net, 32 canales base, 7,8 M parámetros; BCE + Dice sobre píxeles confiables; Adam, LR 1e-4; tres rondas de 20 épocas, batch 32 por GPU. |
| Hardware y evaluación | Dos RTX PRO 4000 de 24 GB, DDP, bf16; umbral de máscara 0,5; HD95 y bootstrap de 1.000 remuestreos. |

La corrida final detectó 7.528 campos circulares y 50.929 rectangulares, sin fallback. La exclusión de pseudo-máscaras fue 8,4 % en ronda 0, 4,4 % en ronda 1 y 4,3 % en ronda 2. El proceso completó 16 etapas con código 0 y 9.283 segundos de cómputo en la corrida buena.

### 4.2 Métricas globales

| Corrida final | Dataset | n | Dice medio [IC 95 %] | Mediana | p10 | IoU | Sensibilidad | Especificidad | HD95 medio | HD95 válido |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `2026-09-21-final` | ISIC 2018 | 2.594 | 0,717677 [0,709009; 0,725988] | 0,792476 | 0,401546 | 0,601656 | 0,783126 | 0,972528 | 26,0505 px | 2.589 |
| `2026-09-21-final` | HAM10000 | 10.015 | 0,784658 [0,781202; 0,788259] | 0,851865 | 0,507469 | 0,678390 | 0,776668 | 0,981815 | 22,4397 px | 10.007 |

Los intervalos de estas medias se calcularon con bootstrap percentil de 1.000 remuestreos y semilla fija. Los archivos fuente son `evaluacion_isic2018/summary.json` y `evaluacion_ham10000/summary.json` bajo `2026-09-21-final/`.

**Métricas complementarias de los mismos summaries:**

| Dataset | Accuracy de píxel [IC 95 %] | Dice p5 / p25 / p75 / p90 | Desv. estándar Dice | Cobertura predicha / referencia |
|---|---|---|---:|---|
| ISIC 2018 | 0,910359 [0,905920; 0,914803] | 0,239192 / 0,599069 / 0,893175 / 0,936916 | 0,225154 | 0,169654 / 0,214019 |
| HAM10000 | 0,902874 [0,900663; 0,905178] | 0,376217 / 0,712818 / 0,919059 / 0,948081 | 0,186772 | 0,197699 / 0,266385 |

La accuracy de píxel está dominada por el fondo y no debe sustituir Dice/IoU ni la revisión de bordes.

### 4.3 Corridas y comparación con pisos no supervisados

| Corrida | Commit | Configuración que cambió | Dataset | Dice [IC 95 %] | HD95 medio | Lectura |
|---|---|---|---|---|---:|---|
| Piloto | `f19ab56` | Detector de campo v1; sin supresión de tinta/brillos; área mínima de candidato 0,01. | ISIC 2018 | 0,652 [0,643; 0,661] | 38,2 px | Inferior al final. |
| Piloto | `f19ab56` | Igual. | HAM10000 | 0,759 [0,755; 0,762] | No calculado en esa tabla. | Inferior al final. |
| Final | `d81a492` | Detector de campo v2; área mínima 0,003; supresión de tinta/brillos; MoCo 30 épocas; tres rondas de modelo de segmentación. | ISIC 2018 | 0,718 [0,709; 0,726] | 26,1 px | Configuración principal. |
| Final | `d81a492` | Igual. | HAM10000 | 0,785 [0,781; 0,788] | 22,4 px | Configuración principal. |
| Ablación v2 | `91814a9` | Solo cambia `pseudo.max_area_fraction` de 0,6 a 0,75. | ISIC 2018 | 0,716; Δ pareado −0,0015 [−0,0027; −0,0004] | Δ −0,22 [−0,58; +0,13] | Pierde ligeramente en Dice; empata en HD95. |
| Ablación v2 | `91814a9` | Igual. | HAM10000 | 0,783; Δ pareado −0,0021 [−0,0026; −0,0017] | Δ +0,17 [+0,08; +0,27] | Pierde en Dice y HD95. |

La v2 incluyó 1.087 lesiones grandes adicionales en las pseudo-máscaras de ronda 0, sin desbordar el campo, pero redujo levemente el resultado neto. Se cierra el cambio del tope de área como configuración por defecto.

| Diferencia final − baseline, pareada por imagen | Δ Dice [IC 95 %] | Δ HD95 [IC 95 %] | Conclusión |
|---|---|---|---|
| ISIC 2018 − SegRNorm | +0,002 [−0,007; +0,011] | −7,37 [−8,60; −6,12] px | Empate en solapamiento; mejora del borde. |
| ISIC 2018 − SegXNorm | +0,022 [+0,013; +0,032] | −6,95 [−8,18; −5,73] px | Gana en ambas. |
| HAM10000 − SegRNorm | +0,020 [+0,016; +0,023] | −0,21 [−0,69; +0,28] px | Gana en Dice; empate en HD95. |
| HAM10000 − SegXNorm | +0,052 [+0,049; +0,056] | −0,51 [−0,98; −0,02] px | Gana en ambas, HD95 por margen pequeño. |

La afirmación correcta no es “el modelo de segmentación supera a todos los pisos en todo”: en ISIC 2018 empata con SegRNorm en Dice. El HD95 sí mejora. En HAM10000 el Dice aumenta, pero el percentil 10 de Dice de SegRNorm es 0,530 frente a 0,507 del modelo de segmentación.

### 4.4 Ablaciones adicionales

**Test-time augmentation por cuatro volteos**, sin reentrenar:

| Dataset | Δ Dice [IC 95 %] | Δ HD95 [IC 95 %] | Interpretación |
|---|---|---|---|
| ISIC 2018 | +0,0003 [−0,0002; +0,0007] | −0,32 [−0,61; −0,04] px | Empate en Dice; mejora pequeña del borde. |
| HAM10000 | +0,0002 [+0,0000; +0,0004] | −0,08 [−0,16; −0,01] px | Dice en el límite de redondeo: `resultados/segmentacion/README.md` lo llama empate y el bootstrap de 10.000 remuestreos en `2026-09-21-v2/resultados.md` lo marca como gana. |

No se recomienda TTA como valor por defecto: el cambio de Dice es marginal y cuesta cuatro veces más cómputo de inferencia; la mejora de HD95 sí es consistente en el resumen final. La diferencia de veredicto HAM entre los dos resúmenes debe resolverse usando un único protocolo y precisión no redondeada.

**Dice por tamaño de lesión, corrida final:**

| Cobertura de la máscara de referencia | ISIC 2018 | HAM10000 |
|---|---:|---:|
| < 10 % | 0,646 | 0,750 |
| 10–25 % | 0,793 | 0,850 |
| 25–40 % | 0,782 | 0,794 |
| 40–60 % | 0,733 | 0,724 |
| ≥ 60 % | 0,684 | 0,642 |

El límite de área 0,6 parece excluir algunas lesiones que ocupan gran parte del campo; elevarlo a 0,75 no mejoró el rendimiento agregado. La hipótesis de ajustar el límite no se debe volver a presentar como solución probada.

### 4.5 Terciles de luminancia y estimación ITA en segmentación

> **Nota: estimación ITA no validada (no es FST).** Lo que aquí aparece como «FST», «fototipo» o «bucket Fitzpatrick» por imagen es el ITA con los umbrales fijos de Groh (`src/dermssl/fitzpatrick.py`), y no está validado. Contra FST asignado por expertos acierta 28,5–30,8 % frente a 32,9 % de la baseline trivial en Fitzpatrick17k-C (11.098 imágenes, docs/20-validacion-externa-fst-fitzpatrick17k.md); en PAD-UFES-20, el ITA con umbrales fijos acierta 22,9 % frente a 51,3 % de la baseline de mayoría (Alipour et al., *Skin Research and Technology* 2026, PMC13459198; FST asignado por dermatólogo según ese artículo). Los terciles `tone_tercile` son terciles de luminancia (mediana de L\* de la piel), no tono de piel.

La primera versión de la medición por buckets ITA queda retirada. ITA se vuelve inestable cuando `b*` se acerca a cero y el signo de `b*` separaba grupos con luminosidad similar; el 40,7 % de las imágenes tenía `|b*| < 5`. Además, `b* ≤ 0` tenía 17,10 % de melanoma frente a 7,57 % cuando `b* > 0`, por lo que la primera brecha mezclaba tono con prevalencia diagnóstica.

La medición corregida exige mediana de `b* ≥ 5` para ITA y usa como estrato principal terciles de la mediana `L*` de piel. Los cortes del pool son 67,25 y 78,59. Son **terciles relativos de luminancia del pool**, no fototipos clínicos.

| Dataset | Estrato | n | Dice [IC 95 %] | IoU | HD95 mediana |
|---|---|---:|---|---:|---:|
| HAM10000 | Claro | 864 | 0,8616 [0,8535; 0,8698] | 0,7737 | 9,2 px |
| HAM10000 | Medio | 4.876 | 0,8030 [0,7983; 0,8079] | 0,6988 | 12,0 px |
| HAM10000 | Oscuro | 4.082 | 0,7503 [0,7440; 0,7565] | 0,6358 | 14,3 px |
| HAM10000 | Sin tono asignado | 193 | 0,7028 [0,6527; 0,7509] | 0,6365 | 19,6 px |
| ISIC 2018 | Claro | 260 | 0,8205 [0,8020; 0,8377] | 0,7183 | 11,1 px |
| ISIC 2018 | Medio | 304 | 0,8024 [0,7834; 0,8194] | 0,6954 | 13,1 px |
| ISIC 2018 | Oscuro | 158 | 0,7651 [0,7372; 0,7885] | 0,6461 | 15,5 px |
| ISIC 2018 | Sin tono asignado | 1.872 | 0,6856 [0,6751; 0,6964] | 0,5665 | 13,9 px |

La diferencia claro–oscuro publicada es 0,1114 en HAM10000 y 0,0554 en ISIC 2018. Los intervalos por estrato de HAM10000 no se solapan y dentro de las siete clases el Dice cae de claro a oscuro en seis. El JSON no incluye un intervalo directo para la diferencia claro–oscuro; los intervalos no solapados son evidencia descriptiva, no un contraste pareado del gap. La asociación tampoco demuestra causalidad ni generalización poblacional.

El control por diagnóstico conserva la caída claro→oscuro en las clases mayoritarias: `nv` 0,872/0,840/0,801; `mel` 0,892/0,831/0,741; `bkl` 0,886/0,754/0,637; `bcc` 0,716/0,610/0,517 (claro/medio/oscuro). Las celdas pequeñas se omiten si n<30; no atribuir el mismo grado de evidencia a `akiec`, `df` y `vasc` sin revisar sus conteos.

El control por tamaño en ISIC 2018 es mixto: el grupo oscuro tiene 2,5 veces más lesiones grandes; la diferencia se concentra en cobertura de referencia 0,25–0,60, con brecha de Dice 0,1023 (claro n=113, oscuro n=45), pero no se publica un intervalo directo para esa diferencia. La evidencia no permite declarar ausencia del efecto en ISIC 2018 ni una brecha externa confirmada.

**Advertencia de pertenencia al pool:** los README informan 722 imágenes ISIC 2018 dentro del pool y 1.872 fuera. Sin embargo, `eval_report.py` crea parte de esa separación a partir de disponibilidad de tono, no mediante una comprobación directa del ID de cada imagen frente al manifiesto de entrenamiento. En este informe, 722/1.872 se conserva como partición **reportada por el README**, no como membresía externa ya verificada por ID. Se requiere una atestación para afirmar sin reservas que las 1.872 son completamente no vistas.

El umbral de pseudo-máscara y la generación automática son más frágiles en tinta, pelo, gel, reglas y lesiones pálidas. El sesgo dominante es la subsegmentación en ISIC 2018 (cobertura predicha 0,17 frente a 0,21 de referencia). Una dilatación de 8–12 píxeles a 224 se propone en los resultados para proteger el borde al estimar piel; todavía requiere una prueba controlada.

## 5. Banco clasificador final

### 5.1 Diseño de la comparación

La evaluación usa HAM10000 con siete clases `dx`, división `StratifiedGroupKFold` por `lesion_id`, semilla 42, primer pliegue: 8.011 imágenes para entrenar el probe y 2.004 para validación, sin lesión compartida entre ambos grupos.

| Brazo | Backbone de inicio | Vista de preentrenamiento | Papel | Presupuesto |
|---|---|---|---|---:|
| `dinov2_public_vits14` | DINOv2 ViT-S/14 público | Ninguna adaptación propia | Referencia pequeña | No se entrena en el banco. |
| `dinov2_public_vitb14` | DINOv2 ViT-B/14 público | Ninguna adaptación propia | Referencia grande | No se entrena en el banco. |
| `ssl_plain_vits14` | DINOv2 público ViT-S/14 | SimSiam de dominio, dos vistas | Control pequeño | 12,0 horas-GPU. |
| `ssl_proposed_vits14` | DINOv2 público ViT-S/14 | Original + copia con desplazamiento de tono | Propuesto pequeño | 12,0 horas-GPU. |
| `ssl_plain_vitb14` | DINOv2 público ViT-B/14 | SimSiam de dominio, dos vistas | Control grande | 12,0 horas-GPU. |
| `ssl_proposed_vitb14` | DINOv2 público ViT-B/14 | Original + copia con desplazamiento de tono | Propuesto grande | 12,0 horas-GPU. |

Los cuatro brazos adaptados parten de DINOv2 LVD-142M; **no** son SSL entrenados desde cero. A igualdad de 12 horas-GPU, el ViT-B procesa menos pasos y muestras que el ViT-S. Los brazos pareados leen original y transformada y también procesan aproximadamente 2,8–3,3 % menos muestras que su control plain. Esto es igualdad de presupuesto de cómputo, no de número de muestras.

La validación de clasificación se hace con imágenes RGB originales. Las pseudo-máscaras intervienen antes, al generar las transformaciones de tono del brazo propuesto. Esto da un camino posible para que los errores de segmentación afecten al preentrenamiento propuesto, pero no significa que el clasificador final aplique una máscara al recibir cada imagen.

### 5.2 Rendimiento completo a 100 % de etiquetas

Probe principal: `logreg`, fijado antes de cerrar el banco. Mayor es mejor salvo ECE, donde menor es mejor.

| Backbone | Accuracy | Balanced accuracy | Macro-F1 | Precision macro | Recall macro | AUC | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 público ViT-B/14 | 0,781936 | 0,592388 | 0,615836 | 0,650880 | 0,592388 | 0,925710 | 0,130744 |
| DINOv2 público ViT-S/14 | 0,790918 | 0,560276 | 0,582862 | 0,627227 | 0,560276 | 0,932225 | 0,081120 |
| `ssl_plain_vitb14` | 0,798902 | 0,542891 | 0,585593 | 0,654840 | 0,542891 | 0,944703 | 0,016124 |
| `ssl_plain_vits14` | 0,783433 | 0,476614 | 0,530694 | 0,685625 | 0,476614 | 0,930614 | 0,020707 |
| `ssl_proposed_vitb14` | 0,791417 | 0,477350 | 0,523663 | 0,634440 | 0,477350 | 0,943389 | 0,029754 |
| `ssl_proposed_vits14` | 0,775449 | 0,436166 | 0,483981 | 0,598723 | 0,436166 | 0,921161 | 0,038194 |

Lectura: DINOv2 ViT-B lidera en macro-F1 y balanced accuracy. El plain ViT-B tiene AUC más alto y ECE más bajo, pero menor balanced accuracy. No hay un ganador en todas las métricas. No debe usarse accuracy como resumen único debido al desbalance de clases.

### 5.3 Resultados de probes con 10 % y 100 % de etiquetas

El archivo `resultados/clasificacion/bank.md` también resume tres probes. Se preservan sus valores publicados; no deben confundirse con la comparación principal de macro-F1 con `logreg`.

Los valores al 10 % son medias de tres seeds de selección de etiquetas (0, 1, 2). En las celdas al 100 %, las tres semillas producen exactamente el mismo resultado: el n efectivo de entrenamientos completos sigue siendo 1, no 3.

**KNN — AUC:**

| Backbone | 10 % | 100 % |
|---|---:|---:|
| DINOv2 público ViT-B/14 | 0,8121 | 0,8692 |
| DINOv2 público ViT-S/14 | 0,8094 | 0,8749 |
| `ssl_plain_vitb14` | 0,7198 | 0,7335 |
| `ssl_plain_vits14` | 0,7339 | 0,7579 |
| `ssl_proposed_vitb14` | 0,7117 | 0,7440 |
| `ssl_proposed_vits14` | 0,7084 | 0,7323 |

**Regresión logística (`logreg`) — macro-F1:**

| Backbone | 10 % | 100 % |
|---|---:|---:|
| DINOv2 público ViT-B/14 | 0,4486 | 0,6158 |
| DINOv2 público ViT-S/14 | 0,4699 | 0,5829 |
| `ssl_plain_vitb14` | 0,4460 | 0,5856 |
| `ssl_plain_vits14` | 0,3350 | 0,5307 |
| `ssl_proposed_vitb14` | 0,3633 | 0,5237 |
| `ssl_proposed_vits14` | 0,3186 | 0,4840 |

**Regresión logística balanceada (`logreg_balanced`) — macro-F1:**

| Backbone | 10 % | 100 % |
|---|---:|---:|
| DINOv2 público ViT-B/14 | 0,4694 | 0,6131 |
| DINOv2 público ViT-S/14 | 0,4931 | 0,5578 |
| `ssl_plain_vitb14` | 0,4654 | 0,5544 |
| `ssl_plain_vits14` | 0,4093 | 0,5087 |
| `ssl_proposed_vitb14` | 0,4200 | 0,5368 |
| `ssl_proposed_vits14` | 0,4017 | 0,5064 |

El resultado principal preinscrito es `logreg`; el probe balanceado se usa para describir los estratos por tono. No se debe cambiar el probe después de ver cuál favorece a un brazo.

### 5.4 Contrastes pareados y sus intervalos

Bootstrap pareado sobre los mismos índices del holdout de validación (`n_val=2.004`), estratificado por clase, 1.000 remuestreos, semilla `20260923`. Cada delta es brazo − comparador. Para ECE, una diferencia negativa favorece al brazo de la izquierda.

| Contraste | Métrica | Delta | IC 95 % | Lectura publicada |
|---|---|---:|---|---|
| Proposed ViT-S − Plain ViT-S | Macro-F1 | −0,0467 | [−0,0878; −0,0036] | Gana Plain. |
| Proposed ViT-S − Plain ViT-S | Balanced accuracy | −0,0404 | [−0,0769; −0,0060] | Gana Plain. |
| Proposed ViT-S − Plain ViT-S | AUC | −0,0095 | [−0,0216; +0,0045] | No se distingue del ruido. |
| Proposed ViT-S − Plain ViT-S | ECE | +0,0175 | [−0,0036; +0,0293] | No se distingue del ruido. |
| Proposed ViT-B − Plain ViT-B | Macro-F1 | −0,0619 | [−0,1131; −0,0109] | Gana Plain. |
| Proposed ViT-B − Plain ViT-B | Balanced accuracy | −0,0655 | [−0,1119; −0,0193] | Gana Plain. |
| Proposed ViT-B − Plain ViT-B | AUC | −0,0013 | [−0,0070; +0,0054] | No se distingue del ruido. |
| Proposed ViT-B − Plain ViT-B | ECE | +0,0136 | [−0,0068; +0,0254] | No se distingue del ruido. |
| Proposed ViT-S − DINOv2 público ViT-S | Macro-F1 | −0,0989 | [−0,1499; −0,0495] | Gana DINOv2 público. |
| Proposed ViT-S − DINOv2 público ViT-S | Balanced accuracy | −0,1241 | [−0,1711; −0,0802] | Gana DINOv2 público. |
| Proposed ViT-S − DINOv2 público ViT-S | AUC | −0,0111 | [−0,0289; +0,0050] | No se distingue del ruido. |
| Proposed ViT-S − DINOv2 público ViT-S | ECE | −0,0429 | [−0,0596; −0,0216] | Gana Proposed. |
| Proposed ViT-B − DINOv2 público ViT-B | Macro-F1 | −0,0922 | [−0,1473; −0,0420] | Gana DINOv2 público. |
| Proposed ViT-B − DINOv2 público ViT-B | Balanced accuracy | −0,1150 | [−0,1678; −0,0657] | Gana DINOv2 público. |
| Proposed ViT-B − DINOv2 público ViT-B | AUC | +0,0177 | [+0,0065; +0,0306] | Gana Proposed. |
| Proposed ViT-B − DINOv2 público ViT-B | ECE | −0,1010 | [−0,1177; −0,0758] | Gana Proposed. |
| Plain ViT-S − DINOv2 público ViT-S | Macro-F1 | −0,0522 | [−0,1038; −0,0041] | Gana DINOv2 público. |
| Plain ViT-S − DINOv2 público ViT-S | Balanced accuracy | −0,0837 | [−0,1305; −0,0408] | Gana DINOv2 público. |
| Plain ViT-S − DINOv2 público ViT-S | AUC | −0,0016 | [−0,0181; +0,0147] | No se distingue del ruido. |
| Plain ViT-S − DINOv2 público ViT-S | ECE | −0,0604 | [−0,0711; −0,0343] | Gana Plain. |
| Plain ViT-B − DINOv2 público ViT-B | Macro-F1 | −0,0302 | [−0,0753; +0,0161] | No se distingue del ruido. |
| Plain ViT-B − DINOv2 público ViT-B | Balanced accuracy | −0,0495 | [−0,0938; −0,0034] | Gana DINOv2 público. |
| Plain ViT-B − DINOv2 público ViT-B | AUC | +0,0190 | [+0,0074; +0,0320] | Gana Plain. |
| Plain ViT-B − DINOv2 público ViT-B | ECE | −0,1146 | [−0,1239; −0,0882] | Gana Plain. |

### 5.5 Métricas por tercil de luminancia

El estrato se construye con terciles relativos de luminancia (`tone_tercile`), no con tono de piel. Los conteos son `dark` 820, `light` 183, `medium` 957 y `unavailable` 44. `unavailable` no es un fototipo. Las cifras de cada celda son `macro-F1 / AUC` usando `logreg_balanced` al 100 %.

| Backbone | Dark (n=820) | Light (n=183) | Medium (n=957) | Sin tono (n=44) | Brecha max−min F1 | Intervalos extremos separados |
|---|---|---|---|---|---:|---|
| DINOv2 público ViT-S/14 | 0,5221 / 0,9281 | 0,4833 / 0,9083 | 0,5743 / 0,9138 | 0,4014 / 0,8760 | 0,091 | No |
| DINOv2 público ViT-B/14 | 0,6034 / 0,9169 | 0,6175 / 0,9074 | 0,6036 / 0,9293 | 0,4395 / 0,8974 | 0,014 | No |
| `ssl_plain_vits14` | 0,5249 / 0,9324 | 0,4772 / 0,8940 | 0,4841 / 0,9136 | 0,3903 / 0,8448 | 0,048 | No |
| `ssl_proposed_vits14` | 0,5186 / 0,9058 | 0,4214 / 0,8519 | 0,5017 / 0,9111 | 0,3724 / 0,8258 | 0,097 | No |
| `ssl_plain_vitb14` | 0,5230 / 0,9246 | 0,4635 / 0,8942 | 0,5719 / 0,9347 | 0,3985 / 0,9113 | 0,108 | No |
| `ssl_proposed_vitb14` | 0,5309 / 0,9446 | 0,4969 / 0,8845 | 0,5289 / 0,9288 | 0,3710 / 0,8714 | 0,034 | No |

La brecha max–min se calcula entre los tres terciles asignados; `unavailable` queda fuera. Los intervalos de los estratos extremos se solapan en los seis backbones. En consecuencia, no se demuestra que el clasificador tenga una brecha por tono ni que proposed la reduzca. El grupo `light` es mucho menor; cualquier ausencia de diferencia debe interpretarse con esa limitación. El bootstrap de las comparaciones remuestrea imágenes estratificadas por clase, no lesiones como clusters ni entrenamientos independientes; puede subestimar incertidumbre si varias imágenes pertenecen a una misma lesión. El banco final no incluye una matriz de confusión ni métricas completas por clase; por tanto, no permite identificar con precisión qué clases explican la caída del macro-F1. No se deben trasladar a este banco las métricas por clase de la corrida v4.

### 5.6 Reproducibilidad del banco

- Los cuatro brazos adaptados usan DINOv2 LVD-142M como inicialización; no son modelos SSL entrenados desde cero.
- Los cuatro presupuestos son 12,0 horas-GPU, pero cambian pasos y muestras. Proposed ve aproximadamente 2,8–3,3 % menos muestras que Plain.
- Los tres seeds del probe al 100 % producen valores idénticos: el tamaño efectivo de entrenamientos completos es **n=1**. El bootstrap al 100 % usa `seed_used=0` y no sustituye semillas independientes.
- `bank.json` y `bank.md` sí contienen filas agregadas por backbone/probe/fracción/semilla. `scripts/cluster_frozen_bank.py:415-422` puede escribir esas filas resumidas a `bank_rows.csv`, pero el CSV de esta corrida no está versionado. Esos resúmenes no son las predicciones por imagen que necesita el bootstrap: el cálculo requiere `y_true`, probabilidades, buckets de tono y el orden exacto del holdout de validación. Esas entradas no están comprometidas y no hay un comando documentado que regenere el `bootstrap.json` publicado. Por eso las métricas puntuales se pueden auditar desde el banco, pero sus intervalos pareados no se reconstruyen desde Git solamente.
- Los manifests de algunos brazos tienen revision/hash incompletos; los cuatro manifests deben archivarse junto a cada futura corrida.

## 6. Cómo se relacionan las máscaras con la clasificación

El flujo verificado es:

1. `export_tone_inputs.py` exporta pseudo-máscaras y las asocia a imágenes.
2. `tone_transforms.py` carga esas máscaras y crea la vista desplazada por tono.
3. `circle.shift_skin_tone` conserva los píxeles marcados como lesión y modifica los demás.
4. `torch_data.SimSiamPairDataset` entrena el brazo `proposed` con la vista original y la transformada.
5. `cluster_frozen_bank.py` extrae features para el probe de clasificación desde el RGB original; no aplica la máscara al inferir.

**Interpretación:** los falsos negativos del segmentador pueden hacer que píxeles reales de lesión se modifiquen durante el preentrenamiento proposed. Los falsos positivos pueden dejar zonas de piel o fondo fuera de la transformación. Esto es un mecanismo posible para parte de la caída de proposed frente a Plain, no una causa demostrada ni una explicación completa de la clasificación.

## 7. Evidencia técnica y tests

### 7.1 Corrida de segmentación

`final.log` registra finalización de 16 etapas con código 0, 9.283 segundos de cómputo y sin reinicios de pod durante la corrida buena. La bitácora sí registra corridas anteriores terminadas por SIGTERM, no como resultado final.

### 7.2 Tests de software

| Evidencia | Estado correcto |
|---|---|
| `.github/workflows/ci.yml` | Define Ruff y pytest en Python 3.11/3.12/3.13, con xdist, timeout y comprobaciones de notebooks/tablas. Describe el workflow; no demuestra que el commit final tenga un run remoto verde. |
| Smoke de segmentación en `plan-segmentacion/camino-b-cnn/resultados/2026-09-21-final/BITACORA.md:34-47` | 136 tests estáticos pasaron en el piloto `f19ab56`; no es una corrida de toda la suite en `c7dce58`. |
| `docs/15-auditoria-ssl-puro-y-plan-v5.md:276-286` | v5 informó 101 pass y 1 skip; comprobación histórica sin GPU, no CI del tip final. |
| `docs/18-estructura-visual-y-tests-baseline.md:82-85` | 272 pass con `requirements-dev`; 266 pass y 2 módulos omitidos sin scikit-learn. Histórico del banco de baselines. |
| `docs/08-estado-actual-y-next-steps.md:3-11` | Informa 190 tests para v6 en agosto; estado histórico, no prueba que la rama v6 actual esté libre de bloqueos. |
| Verificación local registrada el 2026-09-23 | El comando de CI completo no pudo arrancar: faltaban pytest-xdist y pytest-timeout. Cuatro archivos de pruebas focalizados dieron 56 pass con UTF-8 explícito; con el codepage Windows predeterminado fallaron cuatro lecturas UTF-8 de README. No equivale a la suite completa. |

**Lectura de CI:** el estado remoto sí está verificado y es `failure`. No se encontró un archivo JUnit/pytest comprometido; el log detallado está en el run de GitHub, accesible solo con permisos al repositorio.

## 8. Documentos anteriores y afirmaciones a corregir

| Documento | Por qué queda histórico o necesita conciliación |
|---|---|
| `README.md` y `docs/11-resultados-finales.md` | Resumen v4 de julio; presentan resultados de ResNet-50/LP-FT con dos semillas, distintos de la corrida final del banco. |
| `docs/08-estado-actual-y-next-steps.md` | Estado de agosto con baselines/v5 previos; no describe íntegramente las corridas finales de septiembre. |
| `docs/22-pipeline-propuesto-segmentacion-tono-clasificador.md` | Presenta la corrida preliminar de 4 horas-GPU; la pérdida duplicada está corregida en `08727e8` y la dirección de los resultados cambia en la corrida final. |
| `INFORME_ACADEMICO.md` | Su estado fue redactado antes del cierre del banco final; distinguir el banco tone-paired de una ejecución completa de CIRCLe. |
| `resultados/segmentacion/README.md` | Es el resumen de resultados finales, pero la etiqueta in/out-of-pool del control por disponibilidad de tono necesita atestación de pertenencia por ID. |

Este informe no modifica estos archivos; los identifica como contexto histórico para evitar que un resultado previo se cite como resultado final.

## 9. Evaluación de la literatura frente al proyecto

| Referencia | Aporte | No permite concluir |
|---|---|---|
| Sharma et al., 2025, “Skin lesion segmentation: A systematic review…”, DOI `10.1016/j.compbiomed.2025.110842` | Revisa 108 estudios; organiza FCN, U-Net, atención, GAN, Transformer y Mamba; muestra el uso extendido de U-Net y de pérdidas BCE+Dice. | No es un benchmark controlado entre arquitecturas. Cambian dataset, split, resolución, supervisión y métricas entre estudios. En su sección de métricas, la fórmula bajo “specificity” es la de precision; validar cifras contra artículos originales. |
| Yu et al., 2023, EIU-Net, DOI `10.1016/j.compbiomed.2023.107081` | Variante supervisada de U-Net con mejoras de extracción/fusión; se reporta Dice 0,902 en ISIC 2018. | No prueba equidad por tono. Sus máscaras supervisadas y protocolo no son comparables directamente con el modelo de segmentación pseudo-supervisado del proyecto. |
| Bencevic et al., 2024, DOI `10.1016/j.cmpb.2024.108044` | Analiza sesgo en segmentación por color; muestra que balancear muestras y usar CIELAB no garantizan eliminarlo. | No convierte los terciles L* de este proyecto en fototipos clínicos ni resuelve nuestro mecanismo de transformación. |
| Ye et al., EMBC 2025, DOI `10.1109/EMBC58623.2025.11253341` | ADA activa con dominios objetivo y selección de etiquetas expertas; usa un presupuesto de 10 etiquetas por dominio objetivo. | Es clasificación binaria de melanoma/nevus por dominios de edad/ubicación; no evalúa este banco HAM de siete clases ni establece equidad por tono. |
| SM3, “Self-supervised multi-modality learning for multi-label skin lesion classification”, PDF local | Explora pares clínico/dermatoscópicos del mismo caso y atributos multimodales de Derm7pt. | Requiere modalidades pareadas y etiquetas compatibles; no se transfiere directamente a un modelo con una sola modalidad. |
| Calderón et al. (2021), BILSK, DOI `10.1016/j.cmpbup.2021.100036` | Trabajo UIS sobre HAM10000 con una red bilineal. | Reporta macro-F1 0,81 y recall de melanoma 0,55; el split se describe como aleatorio 80/5/15 sin documentar agrupación por `lesion_id`. El PDF revisado no enlaza GitHub. |

## 10. Tres escenarios de trabajo

### Escenario A — Corregir la dependencia segmentación → transformación de tono

**Objetivo:** probar si máscaras deficientes explican parte de la pérdida de macro-F1/balanced accuracy del brazo `proposed`.

**Cambios y pruebas:**

1. En un subconjunto de train/dev que tenga máscara de referencia, calcular por imagen el porcentaje de píxeles de lesión real fuera de la pseudo-máscara y el porcentaje de lesión que recibe el desplazamiento de tono.
2. Desglosar esos errores por tercil L*, clase diagnóstica y artefactos usando solo train/dev. No usar el test bloqueado para escoger máscaras, filtros, transformaciones ni hiperparámetros.
3. Comparar tres brazos con los mismos IDs y cómputo: Plain; Proposed con pseudo-máscara actual; Proposed con máscara de piel/lesión corregida o revisada. Una máscara real puede usarse como “oráculo” diagnóstico solo en un análisis separado; no es un resultado desplegable ni debe sustituir la máscara predicha en el brazo final.
4. Si la máscara explica una caída, crear un brazo supervisado/fine-tuned con máscaras expertas de entrenamiento, manteniendo pacientes/lesiones del test fuera de entrenamiento y selección.
5. Mantener U-Net como baseline; después probar **un** cambio, primero una mejora de borde/skip attention. No cambiar simultáneamente resolución, pérdida y arquitectura. Congelar el test y evaluarlo una sola vez después de cerrar decisiones.

**Criterio de éxito:** reducir el porcentaje de lesión accidentalmente transformada y mejorar Dice/HD95 en los grupos con peor segmentación, sin una pérdida inaceptable en el conjunto completo. Acordar el margen antes de correr.

**Costo y riesgo:** requiere revisión de máscaras y cambia el régimen sin etiquetas si se hace fine-tuning con máscaras expertas. Puede explicar Proposed-vs-Plain, pero no necesariamente toda la diferencia frente a DINOv2.

### Escenario B — ADA activa de clasificación con un dominio objetivo compatible

**Objetivo:** implementar ADA como en la literatura: el modelo elige ejemplos del dominio objetivo para recibir etiquetas expertas y luego se adapta.

**Cambios y pruebas:**

1. Fuente: HAM10000/ISIC en dermatoscopía. Objetivo candidato: PAD-UFES-20 para fotos clínicas de smartphone, o un dataset dermatoscópico de otro centro si se prioriza mantener modalidad.
2. Acordar con dermatología un mapa de diagnósticos. No asumir que `AK`, `SCC`, `SEK`, `BKL` y `AKIEC` son equivalentes sin validación clínica.
3. Dividir por `patient_id`/`lesion_id`. Reservar un test objetivo que nunca participe en preentrenamiento, selección activa, fine-tuning ni elección de hiperparámetros.
4. Comparar fuente sola, adaptación sin etiquetas, selección aleatoria de etiquetas objetivo y selección activa con el mismo presupuesto. Como mínimo, reportar curva de presupuesto y no usar diez etiquetas como garantía de suficiencia para siete clases.
5. Medir macro-F1, balanced accuracy, recall por clase y, si se define una tarea binaria, AUPRC. Para equidad, medir por tono solo si los conteos por tono × diagnóstico son suficientes y la etiqueta de tono está validada.

**Criterio de éxito:** ADA activa supera al control de selección aleatoria en el test objetivo bloqueado, conserva rendimiento dentro de un margen preacordado y reduce una brecha por tono con incertidumbre estimada por paciente/lesión.

**Candidatos y límites:** PAD-UFES-20 (Brasil, smartphone clínico, 2.298 imágenes, metadata de Fitzpatrick) es el candidato más cercano a la modalidad móvil, pero no es colombiano y sus clases no coinciden exactamente. Argentina/HIBA ofrece imágenes clínicas y dermatoscópicas, pero su cobertura Fitzpatrick está concentrada en II y no tiene representación suficiente V–VI. BCN20000 ofrece volumen dermatoscópico de otro centro, pero no metadatos de tono. DDI puede ser útil para una auditoría clínica por tono, pero su acceso/uso tiene restricciones y cambia la modalidad.

PAD-UFES-20 tiene seis diagnósticos y 26 atributos, incluidos `patient_id`, `lesion_id` y un campo Fitzpatrick; el artículo informa 52 melanomas y no publica en la ficha resumida el conteo completo por fototipo. Para usarlo como target deben verificarse primero esos conteos y el mapa de clases. El dataset argentino/HIBA informa 566 pacientes con fototipo: I=44, II=451, III=66, IV=5, sin casos V–VI en esa tabla. Por eso sirve para adaptación regional/modalidad, pero no para demostrar mejora en fototipos V–VI.

| Dataset candidato | Volumen y modalidad | Posible uso en ADA | Límite que debe quedar en el informe |
|---|---|---|---|
| PAD-UFES-20, Brasil | 2.298 imágenes clínicas smartphone, 1.373 pacientes, 1.641 lesiones; 6 diagnósticos; BCC 845, SCC 192, AK 730, melanoma 52, nevus 244, seborrheic keratosis 235. BCC/SCC/MEL son biopsy-proven; 58,4 % del total tiene biopsia. Metadata incluye patient/lesion IDs y Fitzpatrick. | Fuente dermatoscópica (HAM/ISIC) → target clínico smartphone; simular presupuesto ADA ocultando labels del pool y consultar actively/random. | Brasil, no Colombia. Solo algunas clases coinciden exactamente con HAM; melanoma n=52 y la distribución FST debe inspeccionarse antes de fairness. Dataset Mendeley CC BY 4.0. |
| Argentina/HIBA | 1.616 imágenes: 1.270 dermatoscópicas y 346 clínicas; 623 pacientes, 1.246 lesiones; metadata de tono. | Target regional; comparar adaptación entre clínica/dermatoscopía si los pares/IDs permiten split por paciente. | FST registrado para 566 pacientes: I=44, II=451, III=66, IV=5; no hay V–VI en la tabla publicada. No sustenta una conclusión sobre tonos V–VI. |
| BCN20000, España | 18.946 imágenes dermatoscópicas, 5.583 lesiones, 8 diagnósticos más casos OOD. | Target de otro centro manteniendo modalidad dermatoscópica; simular ADA en su conjunto etiquetado de entrenamiento. | No es colombiano y no aporta estratos de tono. El test oficial no entrega todas las etiquetas diagnósticas. |
| DDI, EE. UU. | 656 imágenes clínicas, 570 pacientes, diagnósticos confirmados por patología y grupos de tono I–II (208), III–IV (241) y V–VI (207), según [Stanford AIMI](https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images). | Auditoría externa por tono, después de tener un modelo/entrada para imágenes clínicas. | No es dermatoscopía; muestra pequeña; Stanford requiere acuerdo de uso no comercial y restringe derivados. Confirmar términos antes de adaptar pesos o generar imágenes. |
| Derm7pt | 1.011 estudios con pares clínico/dermatoscópicos y checklist de siete puntos. | Reproducir o validar componentes tipo SM3 si se quiere explorar aprendizaje multimodal. | No es colombiano ni dataset de ADA por sí solo; requiere mantener juntos los pares por caso y respetar el acceso a imágenes. |

Los tamaños y metadatos citados provienen de las páginas oficiales/artículos enlazados en Referencias. Antes de entrenar, hay que descargar la versión autorizada, confirmar el esquema de clases y generar un split propio por paciente/lesión.

### Escenario C — Cerrar la tesis con los resultados finales actuales

**Objetivo:** producir una conclusión defendible si no hay tiempo, anotadores o una cohorte objetivo adecuada.

**Cambios y pruebas:**

1. Actualizar los documentos principales con las corridas finales y marcar explícitamente v4 y la prueba de 4 h-GPU como históricas/preliminares.
2. Presentar la conclusión de clasificación como resultado negativo: el emparejamiento por tono pierde macro-F1 y balanced accuracy frente a Plain; no se demuestra cambio de AUC/ECE ni de brecha de clasificación.
3. Presentar la brecha de segmentación HAM10000 como una asociación por terciles relativos L*, con las limitaciones de pool/transducción y de medición de tono descritas en este informe.
4. Guardar para el holdout de validación ordenado los vectores de etiquetas reales, probabilidades por brazo y tercil de luminancia, o un artefacto lossless equivalente con su hash. Mantener `bank_rows.csv` como resumen agregado y agregar un comando documentado que regenere `bootstrap.json` desde las predicciones por imagen. No declarar los intervalos plenamente reproducibles hasta versionar esos insumos y el comando.
5. No afirmar que el modelo se validó clínicamente, que generaliza a Colombia o que corrigió la equidad por tono.

**Costo y riesgo:** el más bajo en GPU y anotación. No promete una mejora positiva; sí permite una tesis honesta sobre segmentación por tono, una intervención que no funcionó y lo que falta para probar ADA.

## 11. Secuencia recomendada y puertas de decisión

1. **Puerta de máscara, sin GPU:** completar el diagnóstico de falsos negativos/falsos positivos y píxeles de lesión alterados por tono. Si la máscara se degrada en el grupo de peor Dice, ejecutar Escenario A.
2. **Puerta de datos objetivo:** confirmar permiso, modalidad, IDs de paciente/lesión, diagnósticos y distribución de tono del dataset candidato. Si no existe un mapa de clases confiable o los grupos son demasiado pequeños, no presentar una brecha ADA como resultado.
3. **Puerta de ADA:** comparar active vs random con igual presupuesto y un test objetivo bloqueado. Si active no supera random, no atribuir la ganancia a selección activa.
4. **Puerta de cierre:** si la anotación o validación externa no es viable, ejecutar Escenario C y presentar los resultados negativos sin maquillarlos.

### Recomendación

La secuencia de menor riesgo es **auditar primero las máscaras usadas en los pares de tono**, porque es barata y puede explicar una parte del resultado `proposed`. Si ese mecanismo se confirma y se consiguen máscaras revisadas, probar fine-tuning del U-Net como un brazo supervisado separado. Solo después invertiría en ADA activa de clasificación con target labels. Si no se logra conseguir un target compatible, cerrar la tesis como evaluación rigurosa de la brecha de segmentación y del resultado negativo de la adaptación por tono es preferible a afirmar una mitigación no demostrada.

## 12. Evidencia, pruebas y reproducibilidad pendiente

### Evidencia de entrenamiento

| Elemento | Evidencia disponible | Límite |
|---|---|---|
| Segmentación final | `2026-09-21-final/`, `final.log`, `summary.json`, CSV por imagen y overlays. | No todos los manifests completos están en Git; algunos artefactos permanecen en el NFS del clúster. |
| Banco clasificador | `bank.json`, `bank.md`, `bootstrap.json`, `strata.json`, cuatro `run.json`. | `bank.json` guarda métricas agregadas; no están comprometidos `bank_rows.csv` ni las probabilidades/etiquetas por imagen requeridas para regenerar el bootstrap. |
| Contrastes | `bootstrap.json`, 1.000 remuestreos por clase, índices pareados. | El intervalo remuestrea el holdout de validación, no semillas de entrenamiento; a 100 % etiquetas el n efectivo es 1. |
| Pertenencia al pool | README documenta pool y partición de ISIC 2018. | Falta una atestación ID-a-ID reproducible; la rutina `in_pool/out_of_pool` usa disponibilidad de tono como criterio indirecto. |
| CI remoto del tip final | [Run 35856736972](https://github.com/nicolas2601/proyecto-grado-ssl-lesiones/actions/runs/35856736972), commit `c7dce58`, 2026-09-23; consultado por GitHub CLI con sesión autenticada. | **Falló**: pytest Python 3.11 y 3.12; Ruff, archivos generados y pytest Python 3.13 pasaron. El permalink requiere acceso al repositorio. |
| Corridas externas de Weights & Biases | Reconciliación posterior al commit realizada el 2026-09-23. | Resultados y artefactos verificados en la sección siguiente; dos corridas históricas no tienen `wandb-summary.json` disponible. |

### Reconciliación posterior con W&B (2026-09-23)

Se enumeraron las 34 corridas de `nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion` y se contrastaron las corridas finales con los artefactos comprometidos. Las métricas finales de segmentación y clasificación coinciden; los resultados de piloto y v2 se mantienen separados.

| Evidencia | Resultado de la verificación |
|---|---|
| Segmentación final, HAM10000 | [Run `eval-ham10000` (`7418b1f9a24e`)](https://wandb.ai/nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion/runs/7418b1f9a24e): Dice medio 0,784658, IC 95 % [0,781202; 0,788259], n=10.015, IoU 0,678390 y HD95 22,4397 px. Coincide con `evaluacion_ham10000/summary.json`. |
| Segmentación final, ISIC 2018 | [Run `eval-isic2018` (`769ac99249ac`)](https://wandb.ai/nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion/runs/769ac99249ac): Dice medio 0,717677, IC 95 % [0,709009; 0,725988], n=2.594, IoU 0,601656 y HD95 26,0505 px. Coincide con `evaluacion_isic2018/summary.json`. |
| Banco clasificador | [Artefacto `proposed-cluster-results:v0`](https://wandb.ai/nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion/artifacts/results/proposed-cluster-results/v0) contiene `bank.json`, `bank.md`, `bank_rows.csv`, `strata.json`, `pool_split_hash.json` y los cuatro manifests. Los JSON coinciden semánticamente con los comprometidos; `bank.md` coincide textualmente. El CSV contiene 108 filas que coinciden con las filas de `bank.json`. El run [`stellar-meadow-31` (`j0gjoivw`)](https://wandb.ai/nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion/runs/j0gjoivw) publica la tabla W&B de 108 filas y los mismos resúmenes de brecha por tono. |
| Bootstrap pareado | [Artefacto `proposed-cluster-bootstrap:v0`](https://wandb.ai/nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion/artifacts/results/proposed-cluster-bootstrap/v0) contiene `bootstrap.json`. Sus metadatos y 24 contrastes coinciden semánticamente con el `bootstrap.json` comprometido. Los tamaños y MD5 difieren por la serialización del JSON; el MD5 del archivo descargado coincide con el digest del manifiesto W&B. |
| Checkpoints adaptados | [Artefacto `proposed-cluster-backbones:v0`](https://wandb.ai/nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion/artifacts/model/proposed-cluster-backbones/v0) lista los cuatro brazos, dos archivos por brazo, y `esperado.sha256` (869,4 MB en total). Se verificó su inventario, no se descargaron los pesos para recalcular sus hashes. |

El proyecto conserva corridas de piloto y v2 que no representan la configuración final. Por ejemplo, los runs de evaluación de v2 reportan Dice 0,782519 en HAM10000 y 0,716147 en ISIC 2018; no deben sustituir los dos runs finales anteriores. Dos corridas históricas (`e738a433cb08` y `yv7u1el2`) devolvieron HTTP 404 para `wandb-summary.json`; no se usan como evidencia final.

Los resúmenes W&B de los cuatro brazos SSL reportan los mismos conteos de imágenes procesadas que sus manifests y aproximadamente 12 horas-GPU por brazo. Sus valores de `total_loss`/`ssl_loss` son observaciones periódicas, no promedios de ejecución: `torch_loop.py` los registra cada 50 pasos y el manifest calcula el promedio de todos los pasos (`src/dermssl/pretrain/torch_loop.py:350-358, 407-431`). Por eso esos valores no son directamente comparables. En los runs del modelo de segmentación, `selected` indica si esa época superó el mejor valor de pseudo-validación; que la última época muestre `false` no significa que se descartó el run. La evaluación final carga `best.pt` (`plan-segmentacion/comun/estudiante/student_engine.py:210-218`, `plan-segmentacion/comun/nucleo/selection.py:23-31`).

### Pruebas locales verificadas el 2026-09-23

- El comando completo de CI no pudo iniciarse porque el entorno no tenía `pytest-xdist` (`-n`, `--dist`, `--max-worker-restart`) y el `pytest.ini` exige `pytest-timeout`.
- Diagnóstico focalizado con el `pytest.ini` del repositorio omitido mediante `-c NUL`; las variables se fijaron en PowerShell como `CI_REQUIRE_FULL_DEPS=1`, `WANDB_MODE=disabled` y `WANDB_SILENT=true`:

```powershell
python -m pytest -c NUL -q -rs tests/test_feature_cache.py tests/test_results_tables.py tests/test_readme_tables_fresh.py tests/test_resultados_readme_fresh.py
```

Resultado observado: **52 passed, 4 failed**. Los cuatro fallos fueron `UnicodeDecodeError` al leer Markdown UTF-8 con el codepage Windows cp1252 (byte `0x8f`, posición 4242).

```powershell
python -X utf8 -m pytest -c NUL -p no:cacheprovider -q -rs tests/test_feature_cache.py tests/test_results_tables.py tests/test_readme_tables_fresh.py tests/test_resultados_readme_fresh.py
```

Resultado observado: **56 passed**. Ambos son registros de esta sesión; la salida completa no está comprometida en el repositorio y el uso de `-c NUL` significa que no equivalen al workflow oficial.
- Los conteos de `docs/08`, `docs/15`, `docs/18` y bitácoras antiguas son evidencia histórica de otros commits, no CI del tip final.

El run remoto citado en la tabla sí ejecutó el workflow completo. La consulta autenticada fue `gh run view 35856736972 --repo nicolas2601/proyecto-grado-ssl-lesiones`; sus logs muestran que cada matriz fallida (Python 3.11.16 y 3.12) informó 10 failed y 1.428 passed. Los 10 fallos derivan de `TypeError: Path.rglob() got an unexpected keyword argument 'recurse_symlinks'` en `src/dermssl/pretrain/config.py:244`; Python 3.13 pasó. Por tanto, **el estado oficial de CI del commit fuente es failure**, no “sin ejecución”. La página del run es privada y solo abre con permisos al repositorio.

## 13. Riesgos técnicos fuera del banco final

La rama `feature/v6-gpu-ssl-pretraining` es divergente y no se debe presentar como parte de la corrida `proposed-cluster`. Su revisión previa encontró, entre otros, un camino DINO que evita el wrapper DDP, estado de resume incompleto y bloqueos de Stage 2 por campos de gate/serialización. Si un escenario vuelve a ejecutar esa rama, debe corregirse y verificarse antes de asignar GPU.

## 14. Referencias

### Resultados internos

- `resultados/README.md` — banco final, presupuesto, contrastes y lectura consolidada.
- `resultados/clasificacion/bank.md` — probes y estratos por tono.
- `resultados/clasificacion/bank.json` — métricas del banco.
- `resultados/clasificacion/bootstrap.json` — intervalos pareados publicados.
- `resultados/clasificacion/strata.json` — métricas e intervalos por estrato.
- `resultados/segmentacion/README.md` — corrida final, ablation v2, baselines, tono y tamaño.
- `plan-segmentacion/camino-b-cnn/resultados/2026-09-21-final/` — logs, summaries, CSV por imagen y overlays.
- `scripts/cluster_frozen_bank.py`, `scripts/run_stage2.py`, `src/dermssl/circle.py`, `src/dermssl/pretrain/tone_transforms.py`, `src/dermssl/pretrain/torch_data.py` — flujo de máscaras y clasificación.

### Literatura revisada

- Sharma, A. L. et al. (2025). “Skin lesion segmentation: A systematic review of computational techniques, tools, and future directions.” *Computers in Biology and Medicine*, 196, 110842. DOI: `10.1016/j.compbiomed.2025.110842`.
- Ye, J. et al. (2025). “Enhancing Skin Lesion Classification Generalization with Active Domain Adaptation.” EMBC 2025. DOI: `10.1109/EMBC58623.2025.11253341`.
- Bencevic et al. (2024). “Understanding skin color bias in deep learning-based skin lesion segmentation.” *Computer Methods and Programs in Biomedicine*, 245, 108044. DOI: `10.1016/j.cmpb.2024.108044`.
- Yu, Z. et al. (2023). “EIU-Net: Enhanced feature extraction and improved skip connections in U-Net for skin lesion segmentation.” *Computers in Biology and Medicine*, 162, 107081. DOI: `10.1016/j.compbiomed.2023.107081`.
- Calderón et al. (2021). BILSK, trabajo UIS sobre HAM10000. DOI: `10.1016/j.cmpbup.2021.100036`. El PDF revisado no enlaza GitHub; el estudio informa split aleatorio 80/5/15 sin documentar agrupación por `lesion_id`, macro-F1 0,81 y recall de melanoma 0,55.
- “Self-supervised multi-modality learning for multi-label skin lesion classification” (PDF local SM3). Requiere pares clínico/dermatoscópicos del mismo caso y etiquetas de atributos; no equivale a la adaptación tone-paired ya probada.
- Pacheco, A. G. C. et al. (2020). “PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones.” *Data in Brief*, 32, 106221. Dataset: `https://data.mendeley.com/datasets/zr7vgbcyr2/1`.
- Dataset de Argentina/HIBA: `https://pmc.ncbi.nlm.nih.gov/articles/PMC10584927/`.
- BCN20000: `https://pmc.ncbi.nlm.nih.gov/articles/PMC11183228/`.
- DDI: `https://ddi-dataset.github.io/`; revisar su acuerdo de uso antes de entrenar o generar derivados.
- Derm7pt (código y enlace al dataset): `https://github.com/jeremykawahara/derm7pt`.
- UIS, “Diseño y desarrollo de un sistema de captura y almacenamiento de imágenes digitales de lesiones cutáneas para dermatoscopios manuales de gama media”: `https://noesis.uis.edu.co/entities/publication/2bf91bf8-aee5-4ce6-98c0-2e3edb0ac00c`.

### PDFs locales revisados

- BILSK/HAM10000: `C:\Users\mpaul\Downloads\1-s2.0-S2666990021000355-main.pdf`.
- SM3 multimodal: `C:\Users\mpaul\Downloads\Self-supervised multi-modality learning for multi-label skin lesion classification - 1-s2.0-S0169260725001464-main.pdf`.
- Sesgo de tono en segmentación: `C:\Users\mpaul\Downloads\1-s2.0-S0169260724000403-main.pdf`.
- EIU-Net: `C:\Users\mpaul\Downloads\1-s2.0-S0010482523005462-main.pdf`.
- Revisión sistemática de segmentación: `C:\Users\mpaul\Downloads\1-s2.0-S001048252501193X-main (2).pdf`.
- Active Domain Adaptation: `C:\Users\mpaul\Downloads\Enhancing_Skin_Lesion_Classification_Generalization_with_Active_Domain_Adaptation.pdf`.



**Base de Concepción:** según la información aportada por el equipo, contiene fotos de lepra tomadas con smartphone y no fotografías dermatoscópicas o clínicas estandarizadas. No es un target etiquetado compatible con las siete clases HAM10000. Solo se consideraría como pool auxiliar sin etiquetas, con permisos confirmados y una ablación explícita de transferencia negativa; no se usaría para entrenar o evaluar el clasificador como si lepra fuera una clase de lesión pigmentada.

**Disponibilidad colombiana:** la búsqueda exploratoria del 2026-09-23 en repositorios institucionales, Zenodo, GitHub y Hugging Face no encontró un dataset colombiano públicamente descargable que combine dermatoscopía, etiquetas compatibles y metadatos de tono. El registro UIS citado arriba describe un sistema de captura; sus documentos están cerrados/por solicitud y no anuncian una descarga de imágenes. Es una posible vía de contacto, no evidencia de una colección abierta. La conclusión es “no encontrado en esta búsqueda”, no “no existe”.

## 15. Cambios necesarios antes de citar este análisis

1. Archivar y versionar las probabilidades y etiquetas por imagen del holdout, el vector de terciles y su orden/hash; añadir un comando verificable que produzca el `bootstrap.json`. Mantener `bank_rows.csv` como resumen agregado y registrar hashes de checkpoint, split y manifiesto transformado por brazo.
2. Crear una atestación de pertenencia al pool por ID antes de presentar una cifra como externa.
3. Mantener los documentos históricos, pero enlazarlos desde un índice actualizado que distinga v4, corrida preliminar, segmentación final y banco clasificador final.
4. Marcar como “no establecido” todo resultado que dependa de intervalos por imagen cuando no haya incertidumbre por entrenamiento o por lesión.
