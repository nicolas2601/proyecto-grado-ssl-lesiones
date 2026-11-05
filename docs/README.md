# Índice de documentación

Una línea por documento, en orden numérico. La columna Estado indica si el documento es la referencia vigente o si otro lo reemplazó. Los superados se conservan sin editar, porque cuentan cómo se llegó al diseño actual. Este índice puede quedar detrás de los documentos nuevos: para una versión activa, descubre también los archivos recientes en `docs/` y verifica su fecha y alcance antes de tratarlos como vigentes.

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
| [27](27-plan-v7-mejoras-ssl.md) | Plan v7: segmentación (S1, S2, S2'), C1 y comparativa justa. | Vigente |
| [28](28-rediseno-brazo-propuesto-v7.md) | Rediseño v7 del brazo propuesto, predicciones registradas y réplica de semilla. | Vigente |
| [29](29-plan-v7.1.md) | Plan v7.1: aprovechar más las finales existentes. | Pendiente de clasificación |
| [30](30-pad-ufes-zero-shot.md) | PAD-UFES-20: licencia, protocolo y predicción de la prueba externa sin entrenar. | Registrado, sin correr |
| [31](31-t4-shades-of-gray.md) | T4 exploratorio: constancia de color Shades of Gray antes de DINOv2. | Pendiente de clasificación |
| [33](33-banco-completo-multidataset.md) | Banco completo multidataset: reentrenamiento limpio y pendientes. | Pendiente de clasificación |
| [34](34-entorno-cluster.md) | Entorno del clúster y datos. | Pendiente de clasificación |
| [35](35-auditoria-modular-v8.md) | Auditoría modular v8: vistas reales, piel/ITA y pilotos sin etiquetas. | Registrado; W0 implementado, ejecución real pendiente |
| [36](36-pilotos-v8-desarrollo.md) | Pilotos de desarrollo W0–W2. | Pendiente de clasificación |
| [37](37-s4-inferencia-pareada.md) | S4: inferencia pareada del mismo maestro. | Pendiente de clasificación |
| [38](38-evaluacion-semillas-v8.md) | Evaluación anatómica exploratoria de semillas v8. | Pendiente de clasificación |
| [39](39-resultados-y-puertas-v8.md) | Auditoría modular v8: ejecución y decisiones. | Pendiente de clasificación |
| [40](40-wandb-netrc-configurable-v8.md) | Credenciales W&B desde una ruta netrc configurable. | Pendiente de clasificación |
| [41](41-w3-reproducibilidad-y-protocolo.md) | W3: revisión de reproducibilidad y preparación de comparación. | Pendiente de clasificación |
| [42](42-w3-sled-usl-rpi-texto-completo.md) | W3: revisión de SLED, USL-Net y RPI-Net. | Pendiente de clasificación |
| [43](43-cuatro-metodos-segmentacion-v8.md) | Cuatro adaptaciones de segmentación en v8. | Pendiente de clasificación |
| [44](44-preparacion-cluster-y-cuda-v8.md) | Preparación del clúster y compatibilidad CUDA de RPI. | Pendiente de clasificación |
| [45](45-flujo-preparacion-ejecucion-cuatro-metodos-v8.md) | Flujo de preparación y ejecución de los cuatro métodos v8. | Pendiente de clasificación |
| [46](46-pendientes-por-metodo-v8.md) | Pendientes por método para v8. | Pendiente de clasificación |
| [47](47-criterio-no-supervisado-y-controlador-v8.md) | Criterio no supervisado, pesos públicos y controlador v8. | Pendiente de clasificación |
| [48](48-preparacion-reanudable-pesos-v8.md) | Preparador reanudable de pesos públicos V8, evidencias y bloqueos. | Implementado; despliegue y artefactos pendientes |
| [49](49-revision-fuentes-pesos-reales-v8.md) | Fuentes oficiales fijadas, checksums publicados, licencias y límites semánticos. | Registro histórico; estado actualizado en 51–53 |
| [50](50-intento-transferencia-y-revision-preparacion-v8.md) | Plan académico revisado, snapshot fijado e intento de transferencia real. | Intento histórico; ver actualizaciones en 51–53 |
| [51](51-cierre-preparacion-y-bloqueos-lanzamiento-v8.md) | Cierre de preparación y bloqueos de lanzamiento v8. | Vigente como registro |
| [52](52-simclr-huggingface-validacion-local-v8.md) | SimCLR Hugging Face: validación numérica local de pesos sustitutos. | Vigente |
| [53](53-simclr-despliegue-parche-cluster-v8.md) | Despliegue de parche SimCLR al clúster UNABIA vía terminal. | Vigente |
| [54](54-usl-net-banco-pesos-v8.md) | USL-Net: banco de pesos públicos descargado y verificado en UNABIA. | Vigente |
| [55](55-flujo-v7-para-corridas-v8-en-unabia.md) | Flujo y lecciones v7 aplicadas a corridas v8 en UNABIA. | Vigente |
| [56](56-propuesta-semisupervisada-dermlip-slic.md) | Propuesta semisupervisada: calibración semántica de superpíxeles con DermLIP y destilación docente-estudiante. | Vigente (Propuesta Técnica) |
| [57](57-banco-uslnet-preparacion-v8.md) | Paso 4: banco de pesos USL-Net. | Preparación histórica; verificación posterior en [54](54-usl-net-banco-pesos-v8.md); launch_ready=false |
| [58](58-analisis-casos-fallidos-y-rescate-literatura.md) | Análisis de casos límite y mitigaciones orientadas por literatura. | Trazabilidad científica pendiente en T-002 |
| [59](59-propuesta-arquitectonica-derm-tripleprior.md) | Propuesta Derm-TriplePrior: segmentación semisupervisada con guía multimodal y afinidad morfológica. | Propuesta del 2026-10-24; no implementada |
| [60](60-guia-presentacion-cientifica.md) | Guía de presentación científica. | **ANOTADA, no ejecutar** por decisión del 2026-10-30 |
| [61](61-decision-entregable-algoritmo.md) | Qué se entrega como algoritmo de la tesis. | **Decisión abierta**; no la toma una sesión de trabajo |
| [62](62-validacion-cielab-segmentacion-v8.md) | Validación CIELAB de la segmentación v8 y procedencia de IMA++. | Vigente (2026-10-31) |
| [63](63-instrumento-ita-alcance-y-limites.md) | El instrumento ITA: qué mide, qué no, y la tabla de **cifras retiradas**. | **Vigente y de lectura obligatoria** antes de tocar ITA, FST o la transformación tonal |
| [64](64-aporte-medibilidad-ita-y-desplazamiento-logrado.md) | Aporte propio: la medibilidad del ITA depende del diagnóstico (OR 0,30, p 9e-22, replicado), y el desplazamiento tonal logrado en grados (12,5°). | Vigente (2026-11-05); independiente del resultado del contraste |
| [65](65-resultado-contraste-tono-y-que-prueba.md) | Resultado del contraste pareado de tono: **negativo** (PAD −0,0129 [−0,0249; −0,0015], HAM −0,0112), por qué el experimento es válido, y las tres razones por las que **no prueba la hipótesis de CIRCLe**. | **Vigente (2026-11-05)**; incluye una hipótesis propia refutada y la trampa medida del mapeo DDI→HAM |

**Nota sobre este tramo:** los documentos 59 a 64 estuvieron fuera del índice
hasta el 2026-11-05. Esa ausencia es la causa directa de que se consultara al
clúster por cosas que ya estaban escritas. Si se agrega un documento numerado y
no se agrega su fila acá, el documento no existe en la práctica.

Otros documentos:

| Documento | Tema |
|---|---|
| [GUIA.md](GUIA.md) | Guía del repositorio y arquitectura del sistema. |
| [RUN_CATALOG.md](RUN_CATALOG.md) | Índice de procedimientos de corridas SSL y operación en UNABIA por versión. |
| [plan-segmentacion-sin-etiquetas.md](plan-segmentacion-sin-etiquetas.md) | Plan ejecutivo de pruebas de segmentación no supervisada por tono. |
| [superpowers/plans/](superpowers/plans/) | Planes de ejecución. |
| [figuras_analisis/](figuras_analisis/) | Figuras de análisis. |

## Regla de actualización

1. Un resultado nuevo se guarda como JSON con su commit y su tabla generada en `resultados/`.
2. Un documento superado no se borra: se marca en esta tabla.
3. Las cifras viven en los JSON y en las tablas generadas; los documentos las enlazan.
4. El catálogo de corridas guía hacia el procedimiento de cada versión; verifica siempre el código y el estado vigentes antes de ejecutar.

