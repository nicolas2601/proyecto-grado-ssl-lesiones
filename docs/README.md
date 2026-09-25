# Mapa de documentación

Este índice separa evidencia final, decisiones, operación y documentos
históricos. La entrada principal del repositorio es el

## Ruta recomendada

1. [Fuentes y ramas](00-fuentes-y-ramas.md): por qué existe la rama consolidada
   y de dónde viene cada línea.
2. [Análisis final](24-analisis-final-resultados-y-escenarios-ada.md): lectura
   científica más reciente y límites de interpretación.
3. [Resultados versionados](../resultados/README.md): JSON, bootstrap,
   manifiestos, resúmenes y artefactos visuales.
4. [Pipeline propuesto](22-pipeline-propuesto-segmentacion-tono-clasificador.md):
   descripción de la cadena completa.
5. [Decisiones técnicas](04-decisiones-tecnicas.md) y
   [por qué se eligieron](13-por-que-estas-decisiones.md).
6. [Operación del cluster](../cluster/README.md) y
   [runbook de segmentación](../plan-segmentacion/RUNBOOK_CLUSTER.md).

## Documentos por función

### Contexto y método

| Documento | Uso |
|---|---|
| [01 · Contexto y objetivos](01-contexto-y-objetivos.md) | Problema, objetivos y alcance. |
| [02 · Datasets](02-datasets.md) | Fuentes de datos y restricciones. |
| [03 · CRISP-ML(Q)](03-metodologia-crisp-mlq.md) | Metodología del proyecto. |
| [09 · Referencias](09-referencias.md) | Bibliografía y referencias técnicas. |
| [10 · Glosario](10-glosario-eli5.md) | Términos explicados. |

### Decisiones y arquitectura

| Documento | Estado |
|---|---|
| [04 · Decisiones técnicas](04-decisiones-tecnicas.md) | Registro ADR histórico y vigente por decisión. |
| [13 · Por qué estas decisiones](13-por-que-estas-decisiones.md) | Razonamiento técnico y de recursos. |
| [05 · Pipeline objetivo 2](05-pipeline-objetivo2.md) | Diseño de la primera línea SSL. |
| [16 · Plan v6](16-plan-v6-pretraining-gpu.md) | Diseño de preentrenamiento propio; no implica corrida ejecutada. |
| [18 · Estructura visual y tests](18-estructura-visual-y-tests-baseline.md) | Arquitecturas, protocolos y figuras. |
| [22 · Pipeline propuesto](22-pipeline-propuesto-segmentacion-tono-clasificador.md) | Cadena segmentación → tono → SSL → clasificación. |
| [23 · Modelo descargable](23-modelo-descargable-y-uso.md) | Contrato de uso del artefacto de modelo. |

### Resultados y fiabilidad

| Documento | Lectura correcta |
|---|---|
| [06 · Problemáticas](06-problematicas.md) | Errores, límites y contradicciones detectadas. |
| [07 · Fiabilidad y sesgo](07-fiabilidad-y-sesgo.md) | Riesgos de fiabilidad y fairness. |
| [08 · Estado anterior](08-estado-actual-y-next-steps.md) | Histórico de agosto; no reemplaza el banco de septiembre. |
| [11 · Resultados finales v4](11-resultados-finales.md) | Histórico de la corrida v4. |
| [12 · Model card](12-model-card.md) | Ficha del modelo v4 y condiciones de uso. |
| [14 · Mejoras de pipeline](14-mejoras-pipeline.md) | Mejoras propuestas y su motivación. |
| [15 · Auditoría SSL puro](15-auditoria-ssl-puro-y-plan-v5.md) | Transición v4 → v5. |
| [19 · Bancos congelados](19-resultados-bancos-congelados.md) | Baselines v5 y lecciones de medición. |
| [21 · Checkpoint CIRCLe](21-checkpoint-manual-circle-stage2.md) | Estado manual de la línea CIRCLe. |
| [24 · Análisis final](24-analisis-final-resultados-y-escenarios-ada.md) | Fuente narrativa más reciente. |

### Documentos numerados 20

Los tres archivos con prefijo `20` no son duplicados: se conservan con su
nombre completo para no romper enlaces históricos.

| Archivo | Tema |
|---|---|
| [20 · Esquema FST manifest](20-esquema-fst-manifest.md) | Contrato de datos entre GPU y CPU. |
| [20 · Falsear los guards](20-falsear-los-guards.md) | Lecciones de tests estáticos falsables. |
| [20 · Validación externa FST](20-validacion-externa-fst-fitzpatrick17k.md) | Validación contra Fitzpatrick17k-C y límites de ITA. |

### Planes y especificaciones

| Área | Enlace |
|---|---|
| OpenSpec activo | [`../openspec/`](../openspec/) |
| Planes de ejecución | [`superpowers/plans/`](superpowers/plans/) |
| Plan de segmentación | [`../plan-segmentacion/README.md`](../plan-segmentacion/README.md) |
| Resultados de segmentación | [`../plan-segmentacion/camino-b-cnn/resultados/README.md`](../plan-segmentacion/camino-b-cnn/resultados/README.md) |
| Cluster y SLURM | [`../cluster/README.md`](../cluster/README.md) |
| Figuras de análisis | [`figuras_analisis/`](figuras_analisis/) |

## Regla de actualización

Cuando aparezca una nueva corrida:

1. guardar el artefacto con fecha, commit y preset;
2. actualizar [`resultados/README.md`](../resultados/README.md);
3. actualizar [`docs/24`](24-analisis-final-resultados-y-escenarios-ada.md)
   solo si cambia la interpretación;
4. añadir el enlace aquí y al README raíz;
5. no borrar el documento anterior: marcarlo como histórico.
