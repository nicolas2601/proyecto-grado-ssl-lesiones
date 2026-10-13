# 45. Flujo de preparación y ejecución de los cuatro métodos V8

Fecha: 2026-10-13. Rama: `experiment/modular-audit-v8`. Estado: preparación; no habilita corridas.

Este flujo adapta la ejecución por etapas usada en segmentación V7. Mantiene salidas inmutables, pasos ordenados y parada ante error. No reutiliza checkpoints, bancos SSL, pseudo máscaras ni resultados entrenados por el proyecto en V7. Cada método empieza con pesos aleatorios cuando su protocolo no requiere inicialización externa y usa únicamente checkpoints públicos originales cuando su receta sí los requiere.

## Flujo de V7 que sirve de referencia

El driver `plan-segmentacion/camino-b-cnn/entrenamiento/run_pipeline.py` construye y ejecuta etapas en orden; detiene la cadena si una etapa termina con error. El flujo de segmentación documentado es:

`env → cache → contrastive/MoCo → CCAM → pseudo_r0 → student_r1 → pseudo_r1 → student_r2 → pseudo_r2 → student_r3 → FST → evaluación`.

Las máscaras de referencia entran únicamente en evaluación. W&B acompaña las etapas de contraste, CCAM, estudiante y evaluación. La corrida de clasificación de V7 es otra línea y su launcher/checkpoints no se trasladan a V8.

V8 conserva la idea de ejecución secuencial, pero usa su entry point y etapas por método: `scripts/run_segmentation_v8.py`. No se debe invocar el launcher V7 para ejecutar las implementaciones V8.

## Flujo V8 propuesto

```mermaid
flowchart TD
    A["Fijar protocolo, comparación y split"] --> B["Cerrar manifiestos y máscaras de referencia"]
    B --> C["Verificar inicializaciones públicas y procedencia"]
    C --> D{"Puerta común aprobada"}
    D -->|No| X["Detener; resolver pendientes"]
    D -->|Sí| E["Ejecutar un método por vez"]
    E --> F["Generar pseudo máscaras propias del método"]
    F --> G["Revisar hashes, cobertura, abstenciones y recursos"]
    G --> H{"Puerta del método aprobada"}
    H -->|No| X
    H -->|Sí| I["Entrenar etapa dependiente, si aplica"]
    I --> J["Predecir y guardar artefactos en salida nueva"]
    J --> K["Evaluar con referencias reservadas"]
    K --> L{"Informe completo y trazable"}
    L -->|No| X
    L -->|Sí| M["Cerrar método y pasar al siguiente"]
```

### 0. Fijar la comparación antes de ver métricas

Registrar en el protocolo V8:
- qué se compara: pseudo máscaras de los maestros o máscaras finales de los segmentadores entrenados;
- mismo manifiesto elegible, resolución, preprocesamiento, regla de incertidumbre, resolución de evaluación y conjunto de métricas;
- denominador previsto, política para abstenciones y reglas de cierre;
- qué componentes se entrenan desde cero y qué fuentes públicas se requieren por método.

Esta decisión es necesaria porque SLED no tiene etapa de entrenamiento, mientras que USL-Net, RPI-Net y TriDermNet-HAGO sí tienen etapas dependientes de pseudo etiquetas.

### 1. Cerrar datos y evaluación

1. Crear/vincular el manifiesto completo de entrenamiento con IDs únicos, SHA-256 de cada original y exclusiones verificadas.
2. Confirmar la separación por paciente/lesión y que ningún elemento de evaluación aparezca en preentrenamiento, generación de pseudo máscaras o ajuste.
3. Identificar las rutas de las máscaras expertas y reservar una evaluación anatómica independiente. Las máscaras expertas se leen solo en esa etapa.
4. Fijar y verificar presupuesto de almacenamiento. La estimación registrada para V8 es aproximadamente 46,5 GiB para una fuente y 103,8 GiB para cuatro; el límite actual de 4 GiB no alcanza.

La revisión del 13 de octubre verificó integridad y decodificación RGB de 40.896 originales, pero dejó explícitamente pendiente la procedencia de grupos y la reserva anatómica independiente. Esa verificación no sustituye esta puerta.

### 2. Preparar inicializaciones y bancos

Preparar solo las fuentes públicas que cada receta requiera; no usar `salidas/v7`, `salidas/final`, `student_r3`, checkpoints SimSiam adaptados ni pseudo máscaras V7.

| Método | Fuente necesaria según la adaptación V8 |
|---|---|
| SLED | Ningún checkpoint |
| USL-Net | SimCLR, MoCo v1 y MoCo v2 para la fusión/CCAM |
| RPI-Net | MoCo v3, MoCo v2 y DINO original; MoCo v3 aporta dos fuentes (conv1 y CAM) |
| TriDermNet-HAGO | Arquitectura desde cero; no checkpoint público del proyecto |

Para cada checkpoint externo registrar URL/fuente oficial, licencia, arquitectura exacta, preprocesamiento/normalización, SHA-256, conversión reproducible si el formato no coincide, y manifiesto de procedencia. El loader V8 no descarga ni ejecuta código remoto y exige carga estricta; no sustituir DINO original de RPI por DINOv2.

Los bancos pueden prepararse una sola vez y reutilizarse entre métodos únicamente si el hash, la arquitectura y la normalización son exactamente los mismos. Nunca se actualiza el archivo público original durante conversiones.

### 3. Completar la receta y los recursos de cada método

- **SLED:** confirmar presupuesto de tiempo/RAM para las escalas 200–650 sobre el manifiesto completo.
- **USL-Net:** fijar `max_centroid_distance`, ciclos y épocas del CCAM/estudiante; los valores actuales incluyen `null` y límites de ejemplo.
- **RPI-Net:** cerrar número y duración de fases, y validar procedencia de las cuatro pseudo fuentes.
- **TriDermNet-HAGO:** la arquitectura se inicializa desde cero, según la receta original; no se reutilizan checkpoints ni pesos clasificadores de Nico/V7. Declarar de qué maestro V8 aprobado obtiene las pseudo etiquetas, documentar el reemplazo de diagnósticos por cobertura pseudo y fijar el presupuesto de la adaptación para que HAGO supere su calentamiento de 10 épocas. El paper reporta 150 épocas a 256×256 con máscaras expertas y clases diagnósticas; ese presupuesto no se traslada automáticamente a la variante pseudo-supervisada. La configuración de 1 época es solo un marcador de bloqueo.

Registrar por método el dispositivo, timeout, RAM, memoria GPU, espacio de salida y reserva. Los probes sintéticos disponibles miden memoria/tiempo de pasos pequeños; no predicen duración del pool completo.

### 4. Ejecutar por método, con puerta entre etapas

La primera comparación se ejecuta **de forma serial, un método a la vez**. Hay dos RTX disponibles, pero los probes registrados no autorizan concurrencia de corridas completas. Para cada método:

1. Verificar config, hashes, W&B online, manifiesto, gate y carpeta de salida nueva.
2. Ejecutar `check`; confirmar método, etapa, manifiesto y presupuesto.
3. Generar los maestros/pseudo máscaras requeridos:
   - SLED: `generate`;
   - USL-Net: `ccam` y luego `generate`;
   - RPI-Net: `generate` con sus cuatro fuentes;
   - TriDermNet-HAGO: esperar el manifiesto maestro aprobado indicado en su receta; no generar máscaras con una red aleatoria.
4. Inspeccionar manifiesto producido, hashes, cantidad de éxitos/abstenciones, cobertura, salidas y registro W&B. Una abstención no se elimina silenciosamente ni reduce el denominador.
5. Solo si la puerta de esa salida pasa, ejecutar `fit` cuando aplique y luego `predict`.
6. Evaluar con máscaras expertas reservadas, publicar informe y hashes; cerrar el método antes de iniciar el siguiente.

Un error, cambio de manifiesto, fallo de hash, salida parcial o presupuesto excedido detiene la cadena. No se continúa automáticamente con el siguiente método hasta revisar la evidencia del que terminó.

### Recuperación después de una pausa o un crash

El launcher V8 ahora permite reanudar una etapa `fit` de USL-Net, RPI-Net o TriDermNet-HAGO desde el último checkpoint íntegro. No se reanudan automáticamente las etapas `pretrain`, `ccam`, `generate` ni `predict`; una interrupción en ellas conserva la carpeta parcial y exige revisión antes de decidir cómo reiniciar esa etapa en una carpeta nueva. SLED no tiene una etapa `fit`.

En `fit`, el estado atómico `resume.pt` guarda pesos, AdamW, época, cursor y permutación del sampler, métricas parciales, resultados de épocas y RNG de Python, NumPy, CPU y CUDA. También conserva el contador de HAGO. Se escribe al inicio, cada 100 pasos (configurable con `checkpoint_every_steps`), al cierre de época y al recibir SIGTERM/SIGINT en una frontera segura entre lotes. El launcher espera 120 segundos por defecto para que el worker escriba el estado después de una parada ordenada; se puede fijar el margen con `resources.shutdown_grace_seconds`. Un corte abrupto puede repetir como máximo los lotes procesados desde el último guardado; si el proceso cae mientras escribe, el checkpoint anterior permanece íntegro.

Antes de abrir W&B, `--resume` comprueba que el nombre de salida conserva el mismo run ID. El trainer rechaza el estado si cambió la revisión o los hashes de archivos fuente, método/arquitectura, configuración, gate, manifiesto, inicialización o run ID de W&B. No descarga pesos ni consulta V7. La carpeta anterior y sus artefactos se preservan.

Uso, después de inspeccionar logs y resolver la causa de la parada:

```bash
python scripts/run_segmentation_v8.py --method <metodo> --stage fit \
  --config <config.json> --manifest <manifiesto.json> --gate <gate.json> \
  --out <misma-carpeta-de-salida> --wandb-project <proyecto> --resume
```

La opción solo está habilitada para `fit`; no se debe borrar `resume.pt`, renombrar la carpeta, cambiar la receta ni modificar la procedencia para forzar una continuación. Si los hashes no coinciden o falta el estado, se detiene y se abre una nueva ejecución desde la etapa afectada tras revisar la evidencia.

El pipeline de varias etapas aún se ejecuta por comandos secuenciales, uno a la vez; todavía no existe un ledger que salte automáticamente etapas completas ni recupere `pretrain/ccam/generate/predict` por caso. Por eso, la política operativa sigue siendo detener → inspeccionar → corregir → reanudar el `fit` compatible o reiniciar solo la etapa fallida en salida nueva → verificar antes de continuar. El entrenamiento real permanece deshabilitado hasta cerrar las puertas de la sección siguiente.

### Puertas obligatorias para habilitar ejecución real

Por método y por etapa, la evidencia debe coincidir con el SHA del manifiesto e indicar: datos verificados, grupos/exclusiones confirmados, banco y procedencia verificados cuando corresponda, presupuesto medido, smoke aprobado, revisión de código y evaluación anatómica independiente resuelta. Para entrenamiento, todos los campos exigidos por `common/training.py::gate_report` deben ser verdaderos. Después se actualizan configs y gates juntos; no se cambia `execution_enabled` aisladamente.

Los configs publicados aún dicen `execution_enabled=false`, `device=cpu`, `epochs=1`, `batch_size=2`, bancos/checkpoints nulos y `max_output_mb=4096`. Son valores de bloqueo/ejemplo, no una receta de corrida completa. W&B debe quedar online y con el proyecto acordado; la entrega de notificaciones debe confirmarse aparte.

## Evidencia ya disponible y límites

La rama registra las comprobaciones sintéticas y probes acotados, además de la corrección del backward determinista de RPI en CUDA. Estos verifican componentes y recursos pequeños; no demuestran eficacia anatómica ni habilitan el entrenamiento sobre todos los datos. Este documento tampoco inicia pruebas ni corridas.
