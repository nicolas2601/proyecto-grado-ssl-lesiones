# Índice de documentación

## Nomenclatura de lectura

**Proposed sin segmentación** clasifica sin usar una máscara de lesión. **Proposed con segmentación** combina imagen completa, recorte y descriptores de lesión; es una variante experimental, sin mejora robusta demostrada. El Modelo de control no es una tercera propuesta ni la versión segmentada.

Los experimentos de segmentación se identifican por arquitectura, origen de pseudo-máscaras y ronda de aprendizaje. Las rondas no son arquitecturas distintas; los identificadores antiguos se conservan solo donde permiten localizar archivos o reproducir comandos.

Ver [nombres, equivalencias y explicación para lectores técnicos y no técnicos](10-glosario-eli5.md). Los registros históricos conservan sus cifras, predicciones y limitaciones.

Una línea por documento, en orden numérico. La columna Estado indica si el documento es la referencia vigente o si otro lo reemplazó. Los superados se conservan sin editar, porque cuentan cómo se llegó al diseño actual.

Para empezar: [27 · Plan v7](27-plan-v7-mejoras-ssl.md), después [28 · Rediseño v7](28-rediseno-brazo-propuesto-v7.md) y el [resultado v7](../resultados/clasificacion/v7/README.md). Para el cierre de lo que pidieron el profesor y Pau, con veredicto por regla: [tabla de cierre](../resultados/TABLA_CIERRE.md) (generada).

| Doc | Tema | Estado |
|---|---|---|
| [00](00-fuentes-y-ramas.md) | Fuentes y ramas: de dónde viene cada línea de trabajo de la rama consolidada. | Vigente |
| [01](01-contexto-y-objetivos.md) | Contexto, problema, objetivos y alcance. | Vigente |
| [02](02-datasets.md) | Datasets y sus restricciones. | Vigente |
| [03](03-metodologia-crisp-mlq.md) | Metodología CRISP-ML(Q). | Vigente |
| [04](04-decisiones-tecnicas.md) | Registro de decisiones técnicas (ADR). | Vigente como registro |
| [05](05-pipeline-objetivo2.md) | Pipeline del objetivo 2 sobre el notebook v4. | Superado (v4) |
| [06](06-problematicas.md) | Problemáticas abiertas y resueltas. | Vigente como registro |
| [07](07-fiabilidad-y-sesgo.md) | Fiabilidad y sesgo. | Vigente |
| [08](08-estado-actual-y-next-steps.md) | Estado y próximos pasos a agosto de 2026. | Superado (histórico) |
| [09](09-referencias.md) | Referencias bibliográficas. | Vigente |
| [10](10-glosario-eli5.md) | Glosario explicado sencillo. | Vigente |
| [11](11-resultados-finales.md) | Resultados de la corrida definitiva v4. | Superado (v4) |
| [12](12-model-card.md) | Model card del modelo v4. | Superado (v4) |
| [13](13-por-que-estas-decisiones.md) | Por qué se eligieron la arquitectura y el método. | Vigente |
| [14](14-mejoras-pipeline.md) | Mejoras de pipeline propuestas tras v4. | Superado (v4) |
| [15](15-auditoria-ssl-puro-y-plan-v5.md) | Auditoría de SSL puro y plan v5. | Superado (v5) |
| [16](16-plan-v6-pretraining-gpu.md) | Plan v6 de preentrenamiento en GPU con MLflow. | Superado (v6) |
| 17 | Modelos baseline ([docx](17-modelos-baseline.docx)). | Histórico |
| [18](18-estructura-visual-y-tests-baseline.md) | Estructura visual del pipeline y tests de baseline. | Vigente |
| [19](19-resultados-bancos-congelados.md) | Resultados de los bancos congelados v5 y cómo leerlos. | Superado (v5) |
| [20a](20-esquema-fst-manifest.md) | Esquema de `fst_manifest.csv` (costura GPU/CPU). | Vigente |
| [20b](20-falsear-los-guards.md) | Falsear los guards: tests que pasan sin existir. | Vigente |
| [20c](20-validacion-externa-fst-fitzpatrick17k.md) | Validación externa del estimador FST con Fitzpatrick17k. | Vigente |
| [21](21-checkpoint-manual-circle-stage2.md) | Checkpoint manual de la línea CIRCLe (stage 2). | Histórico |
| [22](22-pipeline-propuesto-segmentacion-tono-clasificador.md) | Pipeline propuesto: segmentación, tono y clasificador. | v6; la parte SSL la reemplaza 28 |
| [23](23-modelo-descargable-y-uso.md) | Modelo descargable: qué se publica y cómo se prueba. | Vigente para el artefacto publicado |
| [24](24-analisis-final-resultados-y-escenarios-ada.md) | Análisis de resultados v6 y escenarios ADA. | Superado por el resultado v7 |
| 25 | No existe: el número quedó sin usar. | — |
| [26](26-plan-correccion-segmentacion-y-clasificador.md) | Plan de corrección de segmentación y clasificador. | Superado por 27 |
| [27](27-plan-v7-mejoras-ssl.md) | Plan v7: mejoras de máscaras (unir regiones, quitar pelo), la segmentación como entrada del clasificador y la comparación justa. | Vigente |
| [28](28-rediseno-brazo-propuesto-v7.md) | Rediseño v7 del brazo propuesto, predicciones registradas y réplica de semilla. | Vigente |
| [29](29-plan-v7.1.md) | Plan v7.1: cómo se elige el clasificador final con un conjunto separado y se lee la validación una sola vez. | Vigente |
| [30](30-pad-ufes-zero-shot.md) | PAD-UFES-20 (fotos clínicas): licencia, protocolo, predicciones y resultados de la prueba externa. | Vigente |
| [31](31-t4-shades-of-gray.md) | Prueba de constancia de color (Shades of Gray) antes de analizar la imagen. | Vigente |
| 32 | No existe: el número quedó sin usar. | — |
| [33](33-banco-completo-multidataset.md) | Banco completo: todas las pruebas sobre HAM10000, ISIC 2019, ISIC 2020 y PAD-UFES-20 (pesos limpios, segmentación, matrices de confusión, tono). | Vigente |
| [34](34-entorno-cluster.md) | Entorno del cluster: versiones y comandos para reproducir las corridas. | Vigente |

Otros documentos:

| Documento | Tema |
|---|---|
| [GUIA.md](GUIA.md) | Guía del repositorio y arquitectura del sistema. |
| [INFORME.md](../INFORME.md) | Informe de avance del Objetivo 2 (versión 4, histórico). |
| [INFORME_ACADEMICO.md](../INFORME_ACADEMICO.md) | Informe académico integrado (septiembre de 2026, histórico). |
| [plan-segmentacion-sin-etiquetas.md](plan-segmentacion-sin-etiquetas.md) | Plan ejecutivo de pruebas de segmentación no supervisada por tono. |
| [superpowers/plans/](superpowers/plans/) | Planes de ejecución. |
| [figuras_analisis/](figuras_analisis/) | Figuras de análisis. |

## Regla de actualización

1. Un resultado nuevo se guarda como JSON con su commit y su tabla generada en `resultados/`.
2. Un documento superado no se borra: se marca en esta tabla.
3. Las cifras viven en los JSON y en las tablas generadas; los documentos las enlazan.
