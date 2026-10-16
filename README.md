# Proyecto de Grado UNAB — Clasificación de Lesiones Cutáneas con Aprendizaje Autosupervisado (SSL)

## Nomenclatura de lectura

**Proposed sin segmentación** clasifica sin usar una máscara de lesión. **Proposed con segmentación** combina imagen completa, recorte y descriptores de lesión; es una variante experimental, sin mejora robusta demostrada. El control SSL sin consistencia de tono no es una tercera propuesta ni la versión segmentada.

Los experimentos de segmentación se identifican por arquitectura, origen de pseudo-máscaras y ronda de aprendizaje. Las rondas no son arquitecturas distintas; los identificadores antiguos se conservan solo donde permiten localizar archivos o reproducir comandos.

Ver [nombres, equivalencias y explicación para lectores técnicos y no técnicos](docs/10-glosario-eli5.md). Los registros históricos conservan sus cifras, predicciones y limitaciones.

Repositorio de trabajo para investigar la clasificación experimental de lesiones cutáneas (HAM10000) con aprendizaje autosupervisado sobre DINOv2. No es un dispositivo de diagnóstico.

## Resultado principal (v7)

La tabla principal es la [combinada de las tres semillas](resultados/clasificacion/v7/SEMILLAS.md): cada semilla por separado, la media y el rango (máx − mín). Detrás están las tablas de la [semilla 42](resultados/clasificacion/v7/README.md) y de las réplicas con [semilla 43](resultados/clasificacion/v7/s43/README.md) y [semilla 44](resultados/clasificacion/v7/s44/README.md). Todas se generan desde los `final_eval.json`, y un test falla si alguien edita un número a mano, así que acá no se copia ninguna cifra.

Dos tablas generadas más, con el mismo test: la del [clasificador](resultados/clasificacion/TABLA_CLASIFICADOR.md) (v6, C1 y v7 lado a lado, y por clase: matriz de confusión, F1, recall y sensibilidad y AUC de melanoma) y la de [segmentación](resultados/segmentacion/TABLA_SEGMENTACION.md) (red de segmentación entrenada con pseudo-máscaras final contra las máscaras de referencia y pseudo-máscaras v7 contra los umbrales registrados en docs/27).

Compara tres brazos sobre la validación del fold 0 de C1: DINOv2 público, SSL anclado sin tono (A\*) y SSL propuesto con tono (B\*). Usa un probe lineal, una sola evaluación e intervalos bootstrap del 95 %. Las predicciones se registraron antes de correrla ([docs/28 §5](docs/28-rediseno-brazo-propuesto-v7.md)).

Lectura al 2026-09-29, con las tres semillas y sin elegir ninguna (regla en [docs/28 §4.1](docs/28-rediseno-brazo-propuesto-v7.md), extendida a tres semillas en [docs/29 §2.5](docs/29-plan-v7.1.md)):
- B\* le gana al DINOv2 público en macro-F1 en las tres semillas, con intervalos que excluyen el cero.
- A\* no pierde contra el público en ninguna (P1). La no inferioridad estricta (P1-NI) se cumple en s43 y s44 pero no en s42: no es robusta a la semilla.
- B\* y A\* no se distinguen entre sí, y B\* no calibra mejor que A\*.
- La variación entre semillas es del orden del efecto medido.

## Objetivos específicos

1. Caracterizar conjuntos de datos de imágenes dermatológicas representativos de lesiones cutáneas relevantes para el contexto clínico del departamento de Santander. Estado: completado. Ver [datasets](docs/02-datasets.md) y [contexto](docs/01-contexto-y-objetivos.md).
2. Diseñar un algoritmo de clasificación de lesiones cutáneas basado en aprendizaje autosupervisado. Estado: completado; diseño vigente en [docs/28](docs/28-rediseno-brazo-propuesto-v7.md).
3. Evaluar el desempeño del algoritmo con métricas de la literatura de análisis de imágenes médicas. Estado: evaluación única v7 hecha; réplica de semilla en curso ([plan v7](docs/27-plan-v7-mejoras-ssl.md)).

## Mapa del repositorio

| Carpeta | Qué contiene |
|---|---|
| [`src/dermssl/`](src/dermssl/) | Librería del proyecto: preentrenamiento SSL (`pretrain/`), probes, métricas, bootstrap pareado, evaluación final, procedencia de corridas. |
| [`scripts/`](scripts/) | Puntos de entrada: preentrenamiento, selección de variantes, evaluación final y generadores de tablas (`build_*`). |
| [`resultados/`](resultados/README.md) | Resultados versionados: C1, v7 y bancos congelados. Las tablas se generan desde JSON. |
| [`docs/`](docs/README.md) | Documentación numerada 00-28, con índice y estado de cada documento. |
| [`plan-segmentacion/`](plan-segmentacion/README.md) | Segmentación sin etiquetas (CCAM, pseudo-máscaras, remoción de pelo). |
| [`tests/`](tests/) | Suite de pytest; corre sin GPU. |
| [`openspec/`](openspec/) | Especificaciones de cambios (OpenSpec). |
| [`cluster/`](cluster/README.md) | Operación del cluster UNABIA. |
| [`demo/`](demo/README.md) | Demo del modelo. |
| `results/` | Bancos de versiones anteriores (histórico). |

## Cómo reproducir la evaluación v7

Requiere HAM10000 en `~/work/datos/ham10000`, los backbones finales A\* y B\* con su `run.json`, y el holdout y la lista de exclusión de C1. El script se niega a correr sobre validación si el árbol de git está sucio, si los ids de train o de val no coinciden con C1 o si la salida ya existe.

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

## Equipo

| Nombre | Rol |
|---|---|
| Nicolás Santiago Moreno Monroy | Autor |
| María Paula Saavedra | Autora |
| Andrés Jerez | Director |
| Karen Sánchez — KAUST | Asesora |
| UNAB — Ingeniería de Sistemas | Institución |
