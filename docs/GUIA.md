# Guía del repositorio

## Navegación de la rama consolidada

Esta rama se llama `proyecto-de-grado` y consolida el snapshot de
`proposed-cluster` con el demo local, los catálogos y la trazabilidad de ramas.

| Necesito... | Ir a... |
|---|---|
| Leer el estado científico final | [`resultados/README.md`](../resultados/README.md) y [`docs/24`](24-analisis-final-resultados-y-escenarios-ada.md) |
| Entender qué rama produjo cada cosa | [`docs/00-fuentes-y-ramas.md`](00-fuentes-y-ramas.md) |
| Encontrar una decisión | [`docs/README.md`](README.md), [`docs/04`](04-decisiones-tecnicas.md) y [`docs/13`](13-por-que-estas-decisiones.md) |
| Ver resultados históricos/baselines | [`results/README.md`](../results/README.md) y las tablas generadas de este README |
| Abrir el demo | [`demo/README.md`](../demo/README.md) |
| Encontrar figuras | [`assets/README.md`](../assets/README.md) |

> Las tablas extensas de este README se conservan porque los tests de frescura
> las comparan contra los JSON generados. Para interpretar el resultado final,
> prevalecen [`resultados/`](../resultados/README.md) y `docs/24`; las cifras de
> corridas anteriores se mantienen como histórico reproducible.

## Resumen ejecutivo

Este repositorio estudia un sistema de apoyo experimental para la clasificación
de lesiones cutáneas mediante aprendizaje autosupervisado, con énfasis en
reproducibilidad, fiabilidad y diferencias de rendimiento por tono de piel. El
proyecto no es un dispositivo de diagnóstico y ninguna cifra sustituye una
validación clínica externa.

| Dato | Estado actual |
|---|---|
| Proyecto | Clasificación de lesiones cutáneas con SSL. |
| Institución | Universidad Autónoma de Bucaramanga, Ingeniería de Sistemas. |
| Autores | Nicolás Santiago Moreno Monroy y María Paula Saavedra. |
| Dirección | Andrés Jerez; asesoría de Karen Sánchez (KAUST). |
| Rama publicada | [`proyecto-de-grado`](https://github.com/nicolas2601/proyecto-grado-ssl-lesiones/tree/proyecto-de-grado). |
| Commit publicado | `bcbe9b4`, consolidación documental y demo local. |

## Resultado vigente

La lectura canónica está en [24 — Análisis final](24-analisis-final-resultados-y-escenarios-ada.md)
y en el [catálogo de resultados](../resultados/README.md). Las cifras de esta
tabla apuntan a artefactos versionados, no a texto transcrito manualmente.

| Línea | Resultado actual | Evidencia directa |
|---|---|---|
| Segmentación final | Dice medio `0,717677` en ISIC 2018 y `0,784658` en HAM10000. | [Resultados de segmentación](../resultados/segmentacion/README.md) · [artefactos finales](../plan-segmentacion/camino-b-cnn/resultados/2026-09-21-final) |
| Generalización | ISIC 2018 es la evaluación externa de segmentación; HAM10000 está dentro del pool sin etiquetas. | [Análisis final §4](24-analisis-final-resultados-y-escenarios-ada.md#4-segmentacion-sin-mascaras-ni-etiquetas) |
| Equidad de segmentación | Brecha claro-oscuro `0,1113648455` en HAM10000. Son terciles relativos de `L*`, no fototipos clínicos. | [Terciles de luminancia y equidad](../resultados/README.md#3-terciles-de-luminancia-y-equidad-de-la-segmentacion) · [`strata_v3`](../resultados/tono/strata_v3) |
| Banco clasificador final | `ssl_proposed_vitb14` macro-F1 `0,523663`; `ssl_plain_vitb14` `0,585593`; DINOv2 público ViT-B/14 `0,615836`. | [Banco sellado](../resultados/clasificacion/bank.md) · [JSON](../resultados/clasificacion/bank.json) |
| Conclusión del brazo propuesto | No gana el banco final en macro-F1 ni balanced accuracy frente a su control. Sus contrastes AUC/ECE no establecen una mejora. | [Bootstrap pareado](../resultados/clasificacion/bootstrap.json) · [Análisis final §5](24-analisis-final-resultados-y-escenarios-ada.md#5-banco-clasificador-final) |
| Preentrenamiento v6 propio | Código y plan preparados; no hay resultado canónico versionado de una corrida final propia. | [Plan v6](16-plan-v6-pretraining-gpu.md) · [Cluster](../cluster/README.md) |

## Cómo leer el proyecto

1. [Estado científico y escenarios](24-analisis-final-resultados-y-escenarios-ada.md).
2. [Resultados finales, JSON y manifiestos](../resultados/README.md).
3. [Pipeline completo](22-pipeline-propuesto-segmentacion-tono-clasificador.md).
4. [Decisiones técnicas](04-decisiones-tecnicas.md) y [rationale](13-por-que-estas-decisiones.md).
5. [Problemas, fiabilidad y sesgo](06-problematicas.md), [fiabilidad](07-fiabilidad-y-sesgo.md).
6. [Bancos históricos](../results/README.md) y [resultados v5](19-resultados-bancos-congelados.md).
7. [Fuentes y ramas](00-fuentes-y-ramas.md).

## Mapa completo de enlaces

### Documentación y decisiones

| Tema | Enlace |
|---|---|
| Índice de todos los documentos | [docs/README.md](README.md) |
| Informe académico integrado | [INFORME_ACADEMICO.md](../INFORME_ACADEMICO.md) |
| Informe histórico v4 | [INFORME.md](../INFORME.md) |
| Contexto y objetivos | [01](01-contexto-y-objetivos.md) |
| Datasets | [02](02-datasets.md) |
| Metodología CRISP-ML(Q) | [03](03-metodologia-crisp-mlq.md) |
| Decisiones técnicas | [04](04-decisiones-tecnicas.md) |
| Pipeline objetivo 2 | [05](05-pipeline-objetivo2.md) |
| Problemáticas abiertas | [06](06-problematicas.md) |
| Fiabilidad y sesgo | [07](07-fiabilidad-y-sesgo.md) |
| Estado y próximos pasos histórico | [08](08-estado-actual-y-next-steps.md) |
| Referencias | [09](09-referencias.md) |
| Glosario | [10](10-glosario-eli5.md) |
| Resultados v4 histórico | [11](11-resultados-finales.md) |
| Model card | [12](12-model-card.md) |
| Por qué estas decisiones | [13](13-por-que-estas-decisiones.md) |
| Mejoras del pipeline | [14](14-mejoras-pipeline.md) |
| Auditoría SSL puro | [15](15-auditoria-ssl-puro-y-plan-v5.md) |
| Plan v6 | [16](16-plan-v6-pretraining-gpu.md) |
| Arquitectura y tests baseline | [18](18-estructura-visual-y-tests-baseline.md) |
| Bancos congelados | [19](19-resultados-bancos-congelados.md) |
| Esquema FST manifest | [20-A](20-esquema-fst-manifest.md) |
| Falsear los guards | [20-B](20-falsear-los-guards.md) |
| Validación externa FST | [20-C](20-validacion-externa-fst-fitzpatrick17k.md) |
| Checkpoint CIRCLe | [21](21-checkpoint-manual-circle-stage2.md) |
| Pipeline propuesto | [22](22-pipeline-propuesto-segmentacion-tono-clasificador.md) |
| Modelo descargable y uso | [23](23-modelo-descargable-y-uso.md) |
| Análisis final | [24](24-analisis-final-resultados-y-escenarios-ada.md) |
| Planes Superpowers | [docs/superpowers/plans/](superpowers/plans) |
| Especificaciones OpenSpec | [openspec/](../openspec) |

### Código, experimentos y operación

| Área | Enlace |
|---|---|
| Código Python | [src/dermssl/](../src/dermssl) |
| Scripts | [scripts/](../scripts) |
| Tests | [tests/](../tests) |
| Notebooks y bancos | [notebooks/README.md](../notebooks/README.md) |
| Baselines de Kaggle | [notebooks/baselines/README.md](../notebooks/baselines/README.md) |
| Bancos y fine-tuning históricos | [results/README.md](../results/README.md) |
| Segmentación Camino B | [plan-segmentacion/README.md](../plan-segmentacion/README.md) |
| Resultados visuales de segmentación | [plan-segmentacion/camino-b-cnn/resultados/README.md](../plan-segmentacion/camino-b-cnn/resultados/README.md) |
| Runbook del cluster | [plan-segmentacion/RUNBOOK_CLUSTER.md](../plan-segmentacion/RUNBOOK_CLUSTER.md) |
| Configuración SLURM y MLflow | [cluster/README.md](../cluster/README.md) |
| Banco de fine-tuning del cluster | [cluster/finetune/README.md](../cluster/finetune/README.md) |
| Worktrees y ramas | [docs/00-fuentes-y-ramas.md](00-fuentes-y-ramas.md) |

### Resultados, imágenes y demo

| Área | Enlace |
|---|---|
| Catálogo de resultados finales | [resultados/README.md](../resultados/README.md) |
| Clasificación final | [resultados/clasificacion/](../resultados/clasificacion) |
| Banco JSON | [bank.json](../resultados/clasificacion/bank.json) |
| Bootstrap pareado | [bootstrap.json](../resultados/clasificacion/bootstrap.json) |
| Datos del sitio/demo | [site_data.json](../resultados/site_data.json) |
| Segmentación final | [resultados/segmentacion/](../resultados/segmentacion) |
| Tono y estratos | [resultados/tono/](../resultados/tono) |
| Resumen de la corrida v4 | [data/RESULTS_SUMMARY.json](../data/RESULTS_SUMMARY.json) |
| Figuras publicables | [assets/README.md](../assets/README.md) |
| Figuras de análisis | [docs/figuras_analisis/](figuras_analisis) |
| Demo local | [demo/README.md](../demo/README.md) |
| Imágenes golden | [demo/models/golden_images/](../demo/models/golden_images) |
| Tests del demo | [test_demo_api.py](../tests/test_demo_api.py), [test_demo_model_service.py](../tests/test_demo_model_service.py), [test_demo_report.py](../tests/test_demo_report.py) |
## Reproducción local

### Dependencias de desarrollo

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### Suite completa

```bash
pytest -q
```

La suite completa verifica contratos, builders, tablas, segmentación, FST,
orquestación, resultados y demo. Los tests que requieren GPU real se marcan y
se ejecutan en el entorno correspondiente del cluster.

### Demo local

```bash
pip install -r demo/requirements-cpu.txt
uvicorn demo.backend.app:app --host 127.0.0.1 --port 8765
```

Abrir <http://127.0.0.1:8765>. El demo requiere el checkpoint externo para
inferencia real, pero puede cargar el reporte y los fixtures sin subir pesos al
repositorio. Consulta [demo/README.md](../demo/README.md).

### Experimentos de cluster

1. Leer [cluster/README.md](../cluster/README.md).
2. Leer [RUNBOOK_CLUSTER.md](../plan-segmentacion/RUNBOOK_CLUSTER.md).
3. Ejecutar primero el smoke test.
4. Registrar commit, preset, hardware, W&B/MLflow y artefactos antes de afirmar
   un resultado.

## Arquitectura del sistema

La figura principal se muestra en [Arquitectura general](../README.md#arquitectura).
Esta sección explica cómo leer cada etapa y enlaza con el código y los
artefactos que la respaldan.

**Cómo leer la figura:**

1. **Datos** — el modelo aprende de ~48-70k imágenes de piel **sin diagnóstico**
   (HAM10000 train + ISIC 2020) tras una etapa de transformación (skin tone Fitzpatrick,
   constancia de color, crops por lesión, augmentation). Las 10 015 imágenes etiquetadas de
   HAM10000 quedan reservadas **únicamente para el examen final**, con split anti-fuga por
   `lesion_id` (cero lesiones compartidas entre train y val).
2. **Pre-entrenamiento SSL (v6)** — dos estrategias compiten con el mismo presupuesto de
   GPU: **DINO continual** (refinar el DINOv2 público con red de segmentación entrenada con pseudo-máscaras/profesor EMA y
   multi-crop) vs. **SimSiam desde cero** (torres siamesas ResNet-50 con stop-gradient y
   detector de colapso). Ninguna ve una sola etiqueta; tracking con MLflow.
3. **Evaluación (protocolo v5)** — el backbone se **congela** (`requires_grad=False` +
   `model.eval()` + `assert trainable == 0`) y se examina con 3 clasificadores ligeros
   (kNN coseno, regresión logística, prototipos) y métricas completas, contra los mismos
   8 baselines con idéntico split, probes y métricas: **la diferencia es atribuible solo
   al backbone**.

*Glosario: SSL = aprendizaje auto-supervisado (sin etiquetas) · EMA = promedio móvil
exponencial · probe = clasificador ligero sobre features congeladas · ECE = error de
calibración esperado.*

> Figuras técnicas de detalle (pipeline del banco de baselines, protocolo de test por
> modelo y arquitecturas v6 por brazo) en
> [18 — Estructura visual y tests](18-estructura-visual-y-tests-baseline.md).
> Lectura de los resultados y la leccion de medicion:
> [19 — Resultados de los bancos congelados](19-resultados-bancos-congelados.md).
> Validacion externa del FST contra Fitzpatrick17k-C y el bug de calibracion del umbral
> ITA: [20 — Validacion externa FST](20-validacion-externa-fst-fitzpatrick17k.md).

> Banco de fine-tuning de los baselines en el cluster (un solo comando para la
> asesora): [cluster/finetune/README.md](../cluster/finetune/README.md).
