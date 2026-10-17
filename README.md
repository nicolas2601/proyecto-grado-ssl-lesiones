# Proyecto de Grado UNAB — Clasificación de Lesiones Cutáneas con Aprendizaje Autosupervisado (SSL)

## Qué es este proyecto

Un programa de computador que, a partir de una fotografía de una lesión de piel, ayuda a distinguir de qué tipo es (por ejemplo, un lunar benigno o un melanoma). Es una herramienta de investigación para un trabajo de grado; **no es un dispositivo médico y no diagnostica**.

La idea central es no depender de miles de etiquetas hechas por médicos. Primero se deja que el programa "mire" muchas imágenes sin etiquetas para aprender patrones de la piel (aprendizaje autosupervisado, SSL), y después se le enseña a clasificar con pocas etiquetas. El punto de partida es DINOv2, un modelo público que ya sabe ver imágenes en general.

## Cómo leer este repositorio

1. Lee este resumen y el [glosario](docs/10-glosario-eli5.md): explica cada nombre y sigla en lenguaje sencillo.
2. Mira el [resultado principal](#resultado-principal-v7) y la sección [Qué no se puede afirmar](#qué-no-se-puede-afirmar).
3. Si quieres el detalle, sigue el [índice de documentos](docs/README.md) y las [tablas de resultados](resultados/README.md).

### Los nombres que vas a ver

- **Proposed sin segmentación:** la propuesta del proyecto. Mira la fotografía completa y decide la clase.
- **Proposed con segmentación:** la misma propuesta, pero además usa la zona de la lesión. Es **experimental**: no mostró una mejora estable.
- **Modelo de control:** el mismo método pero sin la enseñanza de coherencia frente al tono. Sirve para saber si esa pieza aporta algo; no es una tercera propuesta.
- **Experimento de segmentación 1 a 5:** pruebas de redes que marcan dónde está la lesión sin usar máscaras dibujadas por personas.
- **Prueba de …:** cada pregunta que se evaluó (por ejemplo, "Prueba de pesos limpios").
- **Repetición 42, 43, 44:** el mismo entrenamiento repetido con distinto azar, para saber cuánto cambia el resultado solo por casualidad.

Los códigos antiguos (B71, A71, student, E1, T7, s42…) siguen existiendo únicamente en nombres de archivo y en registros fechados; el glosario tiene la tabla de equivalencias.

## Resultado principal (v7)

La tabla principal es la [combinada de las tres repeticiones](resultados/clasificacion/v7/SEMILLAS.md): cada repetición por separado, la media y el rango (máximo menos mínimo). Detrás están las tablas de la [repetición 42](resultados/clasificacion/v7/README.md), la [43](resultados/clasificacion/v7/s43/README.md) y la [44](resultados/clasificacion/v7/s44/README.md). Todas se generan desde archivos de resultados guardados, y una prueba automática falla si alguien edita un número a mano, así que aquí no se copia ninguna cifra.

Dos tablas generadas más: la del [clasificador](resultados/clasificacion/TABLA_CLASIFICADOR.md) (versiones 6 y 7 lado a lado, con la matriz de confusión, F1, recall, sensibilidad y AUC de melanoma por clase) y la de [segmentación](resultados/segmentacion/TABLA_SEGMENTACION.md) (qué tan bien coinciden las máscaras del modelo con las de referencia).

Se comparan tres modelos sobre las mismas imágenes de validación: DINOv2 público, el modelo de control y Proposed sin segmentación. Se usa un clasificador lineal, una sola evaluación y intervalos de confianza del 95 %. Las predicciones se escribieron antes de correr la evaluación ([docs/28 §5](docs/28-rediseno-brazo-propuesto-v7.md)).

Lectura al 2026-09-29, con las tres repeticiones y sin elegir ninguna (regla en [docs/28 §4.1](docs/28-rediseno-brazo-propuesto-v7.md), extendida a tres repeticiones en [docs/29 §2.5](docs/29-plan-v7.1.md)):

- Proposed sin segmentación le gana al DINOv2 público en F1 macro en las tres repeticiones, con intervalos que excluyen el cero.
- El modelo de control no pierde contra el público en ninguna (predicción de anclaje). La versión estricta de esa predicción se cumple en las repeticiones 43 y 44 pero no en la 42: no es robusta al azar del entrenamiento.
- Proposed sin segmentación y el modelo de control no se distinguen entre sí, y Proposed no calibra mejor que el control.
- La variación entre repeticiones es del orden del efecto medido.

## Qué no se puede afirmar

- **Esto no es un dispositivo médico** y no diagnostica: es un trabajo de investigación.
- **El ajuste de tono no mostró una ventaja demostrable** sobre el modelo de control, y **Proposed con segmentación** es experimental.
- **El tono estimado por color (ITA) no es el fototipo de Fitzpatrick** y las agrupaciones por luminancia no son tono de piel; no se puede concluir que el modelo sea más o menos justo según el tono de piel.
- **Los resultados de un conjunto de datos no se trasladan solos a otro:** en las fotos clínicas de PAD-UFES-20 la ventaja de los modelos adaptados sobre DINOv2 público se invierte.
- **Una cifra aislada no demuestra superioridad:** hay que mirarla con su intervalo de confianza, sus repeticiones y su conjunto de datos. Más detalle en [resultados/README.md, sección 7](resultados/README.md#7-que-no-se-puede-afirmar).

## Objetivos específicos

1. Caracterizar conjuntos de datos de imágenes dermatológicas representativos de lesiones cutáneas relevantes para el contexto clínico del departamento de Santander. Estado: completado. Ver [datasets](docs/02-datasets.md) y [contexto](docs/01-contexto-y-objetivos.md).
2. Diseñar un algoritmo de clasificación de lesiones cutáneas basado en aprendizaje autosupervisado. Estado: completado; diseño vigente en [docs/28](docs/28-rediseno-brazo-propuesto-v7.md).
3. Evaluar el desempeño del algoritmo con métricas de la literatura de análisis de imágenes médicas. Estado: evaluación única v7 hecha y repetida con tres repeticiones (42, 43 y 44) ([plan v7](docs/27-plan-v7-mejoras-ssl.md)).

## Mapa del repositorio

| Carpeta | Qué contiene |
|---|---|
| [`src/dermssl/`](src/dermssl/) | Librería del proyecto: preentrenamiento SSL (`pretrain/`), probes, métricas, bootstrap pareado, evaluación final, procedencia de corridas. |
| [`scripts/`](scripts/) | Puntos de entrada: preentrenamiento, selección de variantes, evaluación final y generadores de tablas (`build_*`). |
| [`resultados/`](resultados/README.md) | Resultados versionados: C1, v7 y bancos congelados. Las tablas se generan desde JSON. |
| [`docs/`](docs/README.md) | Documentación numerada, con índice y estado de cada documento. |
| [`plan-segmentacion/`](plan-segmentacion/README.md) | Segmentación sin etiquetas (CCAM, pseudo-máscaras, remoción de pelo). |
| [`tests/`](tests/) | Suite de pytest; corre sin GPU. |
| [`openspec/`](openspec/) | Especificaciones de cambios (OpenSpec). |
| [`cluster/`](cluster/README.md) | Operación del cluster UNABIA. |
| [`demo/`](demo/README.md) | Demo del modelo. |
| `results/` | Bancos de versiones anteriores (histórico). |

## Equipo

| Nombre | Rol |
|---|---|
| Nicolás Santiago Moreno Monroy | Autor |
| María Paula Saavedra | Autora |
| Andrés Jerez | Director |
| Karen Sánchez — KAUST | Asesora |
| UNAB — Ingeniería de Sistemas | Institución |

## Para quien quiera reproducir la evaluación v7 (requiere conocimientos técnicos)

Requiere HAM10000 en `~/work/datos/ham10000`, los backbones finales Modelo de control y Proposed sin segmentación con su `run.json`, y el holdout y la lista de exclusión de C1. El script se niega a correr sobre validación si el árbol de git está sucio, si los ids de train o de val no coinciden con C1 o si la salida ya existe.

```bash
PYTHONPATH=src python scripts/c1_final_eval.py --split val --single-evaluation \
    --ham-root ~/work/datos/ham10000 --holdout "$S/selection/selection_holdout.json" \
    --plain dinov2_vitb14:"$V/final_A/simsiam_dinov2_backbone_final.pt" \
    --proposed dinov2_vitb14:"$V/final_B/simsiam_dinov2_backbone_final.pt" \
    --exclude-ids-file "$S/selection/pretrain_exclude_ids_finals.txt" \
    --out resultados/clasificacion/v7/final_eval.json
python scripts/build_final_eval_table.py resultados/clasificacion/v7/final_eval.json \
    --out resultados/clasificacion/v7/README.md
```

Tests y lint: `pytest -q` y `ruff check .`
