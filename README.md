# Proyecto de Grado UNAB — Clasificación de Lesiones Cutáneas con Aprendizaje Autosupervisado (SSL)

## Qué es este proyecto

Un programa de computador que, a partir de una fotografía de una lesión de piel, ayuda a distinguir de qué tipo es (por ejemplo, un lunar benigno o un melanoma). Es una herramienta de investigación para un trabajo de grado; **no es un dispositivo médico y no diagnostica**.

La idea central es no depender de miles de etiquetas hechas por médicos. Primero se deja que el programa "mire" muchas imágenes sin etiquetas para aprender patrones de la piel (aprendizaje autosupervisado, SSL), y después se le enseña a clasificar con pocas etiquetas. El punto de partida es DINOv2, un modelo público que ya sabe ver imágenes en general.

## Cómo leer este repositorio

1. Lee este resumen y el [glosario](docs/10-glosario-eli5.md): explica cada nombre y sigla en lenguaje sencillo.
2. Mira la [propuesta actual V10](#método-propuesto-actual-v10) y sus [tablas de desarrollo](resultados/clasificacion/v10/README.md); la [evaluación v7](#resultado-principal-v7) se conserva como registro histórico.
3. Si quieres el detalle, sigue el [índice de documentos](docs/README.md) y las [tablas de resultados](resultados/README.md).

### Los nombres que vas a ver

- **Proposed sin segmentación (v7):** nombre histórico para la variante v7 que clasifica sin usar una máscara de lesión.
- **Proposed con segmentación (v7):** nombre histórico para la variante experimental con máscara, recorte y descriptores; no mostró mejora estable.
- **Propuesta actual V10:** MoCo-v3-style sobre DINOv2 ViT-B/14, con lector CLS4 y salidas PAD6/HAM7. No es SimSiam.
- **SimSiam:** antecedente/comparador histórico; el basename antiguo `simsiam_dinov2_backbone_final.pt` no identifica el método MoCo recuperado.
- **Modelo de control (v7):** comparación histórica de la línea de tono; no es una tercera propuesta ni la variante segmentada.
- **Experimento de segmentación 1 a 5:** pruebas de redes que marcan dónde está la lesión sin usar máscaras dibujadas por personas.
- **Prueba de …:** cada pregunta que se evaluó (por ejemplo, "Prueba de pesos limpios").
- **Repetición 42, 43, 44:** el mismo entrenamiento repetido con distinto azar, para saber cuánto cambia el resultado solo por casualidad.

Los códigos históricos A71/B71 se conservan con su identidad V9 en archivos, registros y selecciones explícitas de la API. El nuevo default es `MOCO_HAM7`; `MOCO_PAD6` selecciona la ontología PAD. No se renombra silenciosamente un modelo antiguo como MoCo.

## Método propuesto actual (V10)

La propuesta V10 es **MoCo-v3-style continued SSL sobre DINOv2 ViT-B/14**: batch de 16 con cuota 12 imágenes de origen + 4 imágenes PAD-UFES-20 sin etiqueta, y lectura CLS4 (tokens CLS normalizados de los bloques 8–11). El lector produce seis salidas PAD (`ACK`, `BCC`, `MEL`, `NEV`, `SCC`, `SEK`) o siete HAM (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`); las etiquetas son salidas, no nodos de entrada. Las ontologías no se fusionan.

| Cohorte de desarrollo | Propuesta macro-F1 | M0 CLS macro-F1 |
|---|---:|---:|
| PAD-UFES-20 | 0.674628 | 0.643762 |
| HAM10000 | 0.620537 | 0.602002 |

Cada cifra propuesta promedia tres encoders entrenados por separado y tres repeticiones de CV agrupada por encoder; **no es un ensemble** ni el resultado de un lector globalmente ajustado para despliegue. Las tablas incluyen P/R/F1 por clase, soporte en imágenes y grupos, accuracy, balanced accuracy, AUC OVR macro y ECE: [resultados PAD6/HAM7](resultados/clasificacion/v10/README.md). Las diferencias son descriptivas; no implican significancia.

**V10: clasificador completo verificado en PyTorch, ONNX y Docker, y API de investigación desplegada.** [Release y archivos](https://github.com/nicolas2601/proyecto-grado-ssl-lesiones/releases/tag/v10.0.0) · [estado API](https://derm.nicolasmoreno.site/api/status) · [uso de los tres formatos](scripts/release_v10/USAGE.md) · [model card](docs/43-moco-model-card-v10.md). PAD es transductivo y HAM corresponde a desarrollo expuesto a selección. El holdout sellado de HAM y `target_test` de PAD no se usaron. No es validación clínica ni un dispositivo médico. Ver [protocolo, procedencia y límites](docs/41-moco-proposed-release-v10.md), [E01 para Pau](docs/42-moco-e01-for-pau.md) y [comparación separada del banco supervisado/arquitecturas](resultados/clasificacion/v10/BANK.md).

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
2. Diseñar un algoritmo de clasificación de lesiones cutáneas basado en aprendizaje autosupervisado. Estado: v7 se conserva como antecedente; propuesta actual V10 en preparación en [docs/41](docs/41-moco-proposed-release-v10.md).
3. Evaluar el desempeño del algoritmo con métricas de la literatura de análisis de imágenes médicas. Estado: evaluación única v7 hecha y repetida con tres repeticiones (42, 43 y 44) ([plan v7](docs/27-plan-v7-mejoras-ssl.md)).

## Mapa del repositorio

| Carpeta | Qué contiene |
|---|---|
| [`src/dermssl/`](src/dermssl/) | Librería del proyecto: preentrenamiento SSL (`pretrain/`), probes, métricas, bootstrap pareado, evaluación final, procedencia de corridas. |
| [`scripts/`](scripts/) | Puntos de entrada: preentrenamiento, selección de variantes, evaluación final y generadores de tablas (`build_*`). |
| [`resultados/`](resultados/README.md) | Resultados versionados: C1, v7 histórico, bancos congelados y tablas OOF de desarrollo V10. |
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
