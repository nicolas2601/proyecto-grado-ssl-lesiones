# USL-Net: revisión del protocolo de entrenamiento

Fecha: 2026-10-17. Estado: revisión científica y propuesta; no habilita una corrida.

## Qué debía definirse antes del lanzamiento

Las verificaciones de datos, pesos y recursos acreditan condiciones operativas,
pero no justifican el número de épocas ni demuestran convergencia. Haber heredado
cinco épocas de una configuración histórica no bastaba para fijar el presupuesto
científico del CCAM actual. Ese criterio queda corregido mediante esta revisión.

La corrida `usl_ccam-20261017-002` fue entrenamiento real de la cabeza CCAM, no
una prueba planificada ni el entrenamiento completo de USL-Net. Terminó con salida
0 después de cinco épocas. Se conserva como candidato con presupuesto limitado;
no se declara convergente ni se promueve automáticamente para generar pseudomáscaras.
Su registro está en [run_record.md](../../../resultados/preparacion_v8/usl_ccam_launch_20261017/run_record.md).

## Qué hicieron las corridas anteriores

Fuente ejecutada: `plan-segmentacion/camino-b-cnn/resultados/2026-09-21-final/final.log`.
Los presupuestos coinciden con `plan-segmentacion/comun/configs/cluster-final.json`.

| Etapa histórica | Ejecución observada | Interpretación |
|---|---|---|
| MoCo v2 | 30 épocas | Aprendizaje del encoder; no era entrenamiento CCAM. |
| CCAM | 5 épocas; batch global 128; 447 actualizaciones/época | Un encoder MoCo congelado, cabeza 3×3 y Adam. |
| Estudiante | Tres rondas de 20 épocas | Se regeneraron pseudomáscaras y se inicializó un estudiante distinto en cada ronda; no fue una continuación de 60 épocas. |

En las tres rondas del estudiante se seleccionaron checkpoints por pérdida de
validación contra pseudomáscaras, con mejores épocas de índice cero 18, 18 y 15.
Esa selección no acredita calidad anatómica frente a máscaras expertas. El CCAM
histórico disminuyó su pérdida de 1,627846 a 1,013644; la última caída todavía fue
aproximadamente 6,9%. Sus cinco épocas tampoco demostraban una meseta.
En las fuentes ejecutadas revisadas no quedó acreditado el antecedente de pruebas
de 8–10 épocas; esto no demuestra que nunca se hayan realizado.

## Por qué cinco épocas no son comparables entre ambas implementaciones

El CCAM actual aprende una cabeza 1×1 de 9.217 parámetros sobre tres encoders SSL
congelados. Usó AdamW, LR 0,001, decaimiento 0,01, sin scheduler, batch 2 y 34.784
imágenes por época: 17.392 actualizaciones por época y 86.960 en cinco épocas.
El histórico acumuló 2.235: el actual hizo aproximadamente 38,9 veces más
actualizaciones. Esto no implica mayor calidad ni convergencia equivalente,
porque también cambiaron arquitectura, optimizador, fuentes de pesos y batch.

Las pérdidas reales actuales fueron:
`11.906631771180344, 13.675825077666804, 13.71262835921324, 9.441131869576042, 2.424127668683805`.
No muestran una meseta observada. La pérdida total tampoco permite distinguir
por sí sola una mejora útil de activaciones degeneradas. Con batch 2 cada imagen
dispone de un único positivo de otra imagen: el ranking de positivos queda
trivializado. Es una cuestión de diseño que debe resolverse junto al presupuesto,
no una prueba de que el método haya fallado ni una razón suficiente para añadir épocas.

## Qué establece la referencia de USL-Net

La sección 3.5 del [artículo USL-Net](https://arxiv.org/html/2309.13289v3)
establece cinco iteraciones de autoaprendizaje y 300 épocas por iteración del
estudiante. Esas cinco iteraciones no son cinco épocas de CCAM. La referencia
consultada no establece un presupuesto CCAM de cinco épocas ni justifica el
batch o el optimizador usados aquí. Los valores 5×300 son una referencia del
protocolo publicado, no una aprobación automática del presupuesto local.

## Propuesta original y actualización de implementación

Los párrafos siguientes conservan el diagnóstico anterior al checkpoint
reanudable; el estado vigente se precisa al final de esta sección.

La siguiente unidad de trabajo debe resolver conjuntamente política científica,
recuperación y observación dentro de una misma corrida, en lugar de encadenar
pruebas operativas pequeñas. No se modifica ahora ningún config o gate.

1. **Fijar la política antes de entrenar.** Declarar batch efectivo y composición
   contrastiva, LR/optimizador, presupuesto máximo, frecuencia de observación,
   tolerancia de mejora, paciencia y regla de selección del checkpoint. No se
   fijan valores nuevos en esta revisión: faltan una justificación del batch y
   curvas que permitan fundamentar el criterio; copiar 20 o 30 de otra etapa
   volvería a introducir el mismo problema. Si se conserva un presupuesto fijo,
   informar fin de presupuesto y no presentarlo como convergencia.
2. **Hacer recuperable y observable esa ejecución.** Incorporar al CCAM snapshots
   de cabeza, optimizador, RNG, orden/cursor de datos y contadores, junto con
   pérdida total y componentes positivos/negativos, LR y actualizaciones. Observar
   estabilidad de saliencia, proporción de primer plano e incertidumbre en un
   subconjunto fijo del pool de aprendizaje. Ese seguimiento es dentro de muestra;
   no debe llamarse validación independiente ni prueba de calidad anatómica.
3. **Cerrar un único plan serial.** CCAM → pseudomáscaras completas → estudiante
   → predicción del pool de aprendizaje → refinamiento por incertidumbre → siguiente
   ciclo. Declarar si cada estudiante se reinicializa o continúa pesos y por qué;
   no inferir esa política de los ejemplos históricos. El `native_recipe` 5×300
   actual es declarativo: todavía no ejecuta automáticamente los cinco ciclos.

Los puntos anteriores forman una única propuesta de implementación y política,
que debe quedar revisada antes de la próxima ejecución. Los archivos responsables
son `src/dermssl/segmentation_methods/usl_net/seeds.py` (CCAM),
`src/dermssl/segmentation_methods/common/checkpoints.py` (estado recuperable),
`src/dermssl/segmentation_methods/common/training.py` (estudiante), los configs
USL-Net/CCAM y el plan serial del controlador. No se implementan cambios aquí.

El CCAM de la ejecución histórica de cinco épocas guarda únicamente la cabeza final y su linaje: no permite una
reanudación exacta con AdamW/RNG/contadores. Una futura carga de esa cabeza sería
un inicio desde pesos previos de una etapa nueva, no continuación exacta. El
estudiante sí tiene recuperación completa, pero la identidad incluye los hashes
de config, gate y código: cambiar `epochs` invalida la reanudación. Debe fijarse
el presupuesto total antes de dividirlo en sesiones de ejecución recuperables.
Una sesión recuperada y un nuevo ciclo de autoaprendizaje son operaciones distintas.

### Estado vigente después de integrar la configuración (2026-10-17)

El código local ya tiene recuperación CCAM por época con cabeza, AdamW, RNG,
historial y cabeza seleccionada. La identidad vincula hashes de código,
config, manifiesto del banco, checkpoints/sidecars reales de los tres orígenes,
manifiesto de datos, gate y revisión. Los activos del banco se comparan antes y
después de cargarlos. Cambiar esos insumos no es una
recuperación idéntica. Los checkpoints anteriores sin esta identidad no se
aceptan como recuperables por la versión nueva.

El config CCAM deshabilitado fija 30 épocas como techo operativo tomado del
precedente MoCo, explícitamente no medido para CCAM. Mantiene batch 2, LR 0,001,
weight decay 0,01 y límites de 86.400 s/1.024 MiB como planificación. El dato
medido de 1.652,062884596 s para cinco épocas solo permite una extrapolación
lineal de 9.912,377307576 s para 30, no una garantía temporal. La limitación
contrastiva de batch 2 permanece; esta configuración no la resuelve.

Se ejecutaría el presupuesto fijo completo, sin parada temprana. Se conserva
como candidato la primera cabeza con Pearson >= 0,98 entre saliencias de
épocas consecutivas sobre las primeras ocho imágenes de aprendizaje. Se exporta
como `selected.ccam.pt`, separado del `ccam.pt` final de presupuesto. Ambos
conservan época y linaje; `selection.json` declara si hubo candidato o el umbral
nunca se alcanzó. Ese umbral es una heurística de estabilidad dentro de muestra,
no una prescripción del artículo, ni evidencia de convergencia científica o
calidad anatómica. No se usa el holdout experto para elegirlo.

Las métricas locales se reconstruyen desde el historial del checkpoint confirmado
antes de publicar. W&B conserva `epoch` como referencia lógica, sin garantía de
entrega exactamente una vez: una interrupción puede dejar filas externas ausentes.
El siguiente paso es revisar y transportar el ZIP a UNABIA y revisar el gate
contra el nuevo config. El gate antiguo no sirve para este config distinto.
No se habilita, prueba ni ejecuta entrenamiento en esta actualización. El plan
serial del estudiante y la política de continuidad entre ciclos siguen pendientes.

## Separación científica y alcance de esta revisión

Las 2.594 imágenes/máscaras expertas reservadas no se usarán para seleccionar
épocas, umbrales, parada ni checkpoints. Si se requiere validación fuera de muestra
durante desarrollo, debe derivarse del pool de aprendizaje con manifiestos hijos
y procedencia explícita; el manifiesto padre congelado permanece intacto.
La pérdida contra pseudomáscaras y la estabilidad de saliencia son indicadores
del proceso, no sustitutos de una evaluación anatómica reservada.

Se conserva el checkpoint real en UNABIA:
`/home/jovyan/work/proyectonp/runs_v8/usl_ccam-20261017-002/ccam.pt`,
SHA-256 `82f821843a88a1a03165481dc5d6be3d6fcc872d519bb0fd378afd481b44432d`.
No se relanza, modifica ni promueve. La revisión no reabre hashes, manifiestos,
recursos ya revisados, limpieza previa ni búsqueda de metadatos; preserva las
limitaciones clínicas aceptadas. No modifica `readiness.review.json`, gates,
SLED, RPI-Net o TriDerm. No se ejecutaron pruebas ni entrenamiento en este avance.
