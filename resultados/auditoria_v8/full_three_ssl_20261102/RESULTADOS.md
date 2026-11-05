# Comparación completa: resultados y recomendaciones

Reporte generado desde `results.verified.json`. Medias descriptivas de tres semillas SSL; cada resultado usa tres particiones CV. PAD y HAM permanecen separados.

| Encoder | Dataset | A0 original | A1 L* −8 | Δ macro-F1 (pp) | Semillas con caída |
|---|---|---:|---:|---:|---:|
| moco | PAD | 0.6438 | 0.6245 | -1.92 | 3/3 |
| moco | HAM | 0.6020 | 0.5687 | -3.33 | 3/3 |
| vicreg | PAD | 0.6381 | 0.6115 | -2.66 | 3/3 |
| vicreg | HAM | 0.5461 | 0.5286 | -1.75 | 2/3 |
| simsiam | PAD | 0.5973 | 0.5366 | -6.07 | 3/3 |
| simsiam | HAM | 0.5179 | 0.4804 | -3.75 | 3/3 |

## Lectura

La perturbación fija de L* sobre piel predicha no muestra una mejora reproducible. El promedio de las tres semillas cae en las seis combinaciones de familia/dataset; hay una única semilla con delta positivo entre los dieciocho contrastes. Los intervalos por semilla están en el JSON: no se ha calculado aquí un intervalo agregado sobre semillas ni se ajustó multiplicidad.

A0 reprodujo exactamente features y probabilidades OOF. La comparación reajustó los lectores de A1 dentro de los folds; no reutilizó probabilidades A0 como resultados transformados. El resultado evalúa esta intervención concreta, no todas las posibles formas de usar segmentación o CIELAB.

## Recomendaciones

1. Conservar MoCo-v3-style con imagen original como candidato principal, VICReg como respaldo y SimSiam como referencia. No sustituir el flujo por la variante L* −8.
2. Revisar casos de desarrollo que pasan de acierto a error, por clase, junto a máscaras, piel predicha y cambio de color. No atribuir todavía la pérdida exclusivamente a segmentación o a color.
3. Antes de otra receta, documentar si la hipótesis es robustez a perturbaciones, augmentación o normalización. Esta prueba fue una perturbación determinista: no estimó un tono objetivo ni demuestra corrección de sesgo.
4. Si se justifica continuar, registrar una ablación pequeña de un solo factor (por ejemplo intensidad menor), preservando A0 y los mismos folds. Evaluar pureza del pool donde existan referencias y usar desarrollo separado para confirmar ajustes; no buscar iterativamente el mejor resultado en los mismos casos como si fuera validación independiente.
5. No repetir entrenamientos de segmentación ni encoders sin evidencia de que sean el origen del fallo. No elegir una semilla favorable aislada.

## Procedencia y límites

La ejecución inicial falló y una reanudación posterior produjo los resultados finales. El runner remoto corrige la comparación de clases como listas y permite reutilizar la caché. El hash de configuración final difiere del lanzamiento inicial; se conserva en el JSON verificado. Se comprobaron los 63 hashes de artefactos finales y los hashes, identidades y máscaras de los 2.662 registros de caché. No se atribuye aquí quién hizo esa reanudación.

Los intervalos son condicionales a predicciones OOF fijas y a los grupos disponibles; la independencia clínica no se acredita de nuevo. Son resultados de desarrollo con selección previa de métodos. Un promedio sobre tres semillas no permite por sí solo inferencia sólida sobre futuros entrenamientos.
