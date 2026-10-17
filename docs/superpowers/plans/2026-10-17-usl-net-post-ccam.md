# Plan de corridas USL-Net después de CCAM

Fecha: 2026-10-17. Alcance: preparación; no autoriza entrenamientos adicionales.

**Objetivo:** preparar la generación de pseudomáscaras, el estudiante y sus
refinamientos, y la evaluación final independiente, sin modificar la copia de
código que utiliza CCAM 003.

**Base:** `odd/tasks/v8-cierre-bloqueos-lanzamiento.md`,
`resultados/preparacion_v8/usl_ccam_launch_20261017_003/run_record.md`,
`plan-segmentacion/metodos-v8/launch-preparation/configs/usl_net.draft.json` y el
código de `src/dermssl/segmentation_methods/common/`.

## Qué está listo

- Datos congelados: 34.784 originales de aprendizaje y 2.594 imágenes de
  evaluación, con referencias expertas separadas. No se repiten las auditorías
  de exclusiones, geometría, metadatos ni banco público ya cerradas.
- Banco SimCLR/MoCo v1/MoCo v2 conectado. Se conserva la limitación del espejo
  SimCLR y las limitaciones clínicas aceptadas.
- CCAM 003 enviado con presupuesto fijo de 30 épocas. Su última observación
  documentada acredita proceso y W&B activos, no finalización.
- Las operaciones `generate`, `fit`, `predict` y `refine` existen en
  `scripts/run_segmentation_v8.py`. El estudiante y su recuperación están
  implementados; el refinamiento conserva originales y actualiza pseudomáscaras.
- Existe evaluador separado: `scripts/evaluate_segmentation_v8.py`, con
  predicciones, referencias expertas, protocolo y procedencia ligados por hash.
  Esto acredita implementación disponible, no una evaluación ejecutada.

## Secuencia y dependencias

| Orden | Corrida o decisión | Depende de | Entregable y punto de revisión |
|---|---|---|---|
| 1 | Cerrar CCAM 003 | Salida real del proceso | Historial, `selection.json`, checkpoint final y candidato, sidecars y recursos reales. Revisar antes de promover. |
| 2 | Generar pseudomáscaras iniciales (`generate`) | Cabeza CCAM seleccionada, banco público, config y gate propios | Export de los 34.784 originales, `cases.jsonl`, `summary.json` y `successful_pseudos_manifest.json`. Revisar cobertura e incertidumbre; fijar hash del manifiesto aceptado. |
| 3 | Entrenar estudiante, ciclo 1 (`fit`) | Pseudomáscaras completas aceptadas; presupuesto y gate del estudiante | `final.pt`, procedencia, `resume.pt`, métricas y recursos. Inicialización aleatoria, distinta de la cabeza CCAM. |
| 4 | Predecir sobre aprendizaje (`predict`) y refinar (`refine`) | Estudiante cerrado; manifiesto de pseudomáscaras del ciclo | Probabilidades sobre el mismo aprendizaje y siguiente `manifest.json`, con originales y cohorte preservados. |
| 5 | Repetir ciclos posteriores aprobados | Manifiesto refinado e inicialización explícita por ciclo | Config, gate, checkpoint y registro separados por ciclo; revisión en cada transición. |
| 6 | Predicción final y evaluación reservada | Checkpoint final elegido mediante regla previa | Predicción de las 2.594 imágenes y `evaluation.json`; máscaras expertas únicamente en el evaluador. |

Si CCAM no alcanza Pearson 0,98, `selection.json` informa
`threshold_not_reached`: no se promueve automáticamente el checkpoint de la
época 30. La estabilidad de saliencia tampoco demuestra calidad anatómica.

Las predicciones para refinar corresponden al aprendizaje; las predicciones
finales reservadas son otro export. No se usan los resultados de las 2.594
imágenes para elegir épocas, políticas UM, ciclos ni hiperparámetros.

## Qué preparar ahora

- Especificar nombres y directorios distintos por etapa/ciclo, registros W&B y
  retención de artefactos. Preparar configs deshabilitados fuera del runtime activo.
- Definir presupuesto y regla de selección/parada del estudiante antes de verlo
  contra máscaras expertas. Mantener fijos resolución, originales y comparación.
- Preparar el enlace de checkpoint/provenance/hash/linaje que exigen
  `checkpoint_requirements`: campos con prefijo `ccam_checkpoint` para generar,
  `checkpoint` para predecir e `initial_checkpoint` para continuar entre ciclos.
  Los valores reales se completan solamente cuando existen los artefactos.
- Materializar por tramos el plan serial: el controlador valida entradas reales
  y no fabrica futuros gates ni hashes. `pipeline.draft.json` todavía declara
  `controller_input=false`; no es ejecutable.
- Presupuestar almacenamiento de exports y refinamientos completos, además de
  checkpoints y tracking. El límite de 1.024 MiB de CCAM no se reutiliza: una sola
  matriz float32 de 224×224 por cada original ya ocupa más de 6 GiB, antes de
  mapas adicionales, overlays, válidos y metadatos.

No desplegar cambios sobre los 57 archivos efectivos de la corrida activa.
Toda ejecución posterior permanece en UNABIA; el transporte seguirá por ZIP.

## Decisiones de presupuesto pendientes

El borrador del estudiante contiene batch 2, resolución 224, semilla 42,
AdamW con LR 0,001 y weight decay 0,00001; timeout y salida siguen nulos.
Son valores de borrador, no una configuración aprobada ni recursos medidos.
La receta `native_recipe` de cinco ciclos de 300 épocas es referencia publicada;
no constituye autorización para 1.500 épocas ni una medición de viabilidad.

Definir explícitamente número de ciclos, épocas por ciclo, si se reinicializa o
continúa el estudiante, criterio de selección y límites de tiempo/disco/GPU.
La ejecución no infiere estas decisiones. Una adaptación reducida debe
documentarse como tal; no presentarla como reproducción del presupuesto nativo.

## Brechas reales y cierre

### Preparación materializada (2026-10-17)

Se prepararon cinco configs deshabilitados y la secuencia condicionada en
`plan-segmentacion/metodos-v8/launch-preparation/post-ccam/README.md`.
Incluyen generación, fit ciclo 1, predicción de aprendizaje, refinamiento y
predicción reservada. El evaluador se documenta con su CLI real separado, sin
inventar config ni etapa del controlador. Se propone para revisión tres ciclos
de 20 épocas con reinicialización aleatoria y selección final por presupuesto
fijo; no se considera aprobado ni réplica del presupuesto 5×300 publicado.
Los límites y el cálculo de almacenamiento están documentados como planificación:
64 GiB por export de aprendizaje, 16 GiB por refinado y 320 GiB libres propuestos
para la retención completa. No hay gates creados, hashes futuros fabricados,
despliegue ni pruebas adicionales; CCAM 003 conserva su copia activa.

La brecha inmediata es materializar artefactos y configuración por etapa, no
reescribir los operadores ya implementados. Sigue pendiente una revisión
operativa del estudiante con su presupuesto propio: el consumo de entrenar solo
la cabeza CCAM no mide el entrenamiento de toda la red del estudiante.
También debe quedar resuelta cualquier abstención: el export conserva el
denominador y rechaza promover silenciosamente un pool parcial.

Al cerrar cada etapa se documentan recibo, rutas, hashes, métricas y limitaciones
en el task vigente y GitHub. Solo después se enlaza su salida con la siguiente.
Este plan no habilita gates ni lanza estudiante, refinamientos u otros métodos.
