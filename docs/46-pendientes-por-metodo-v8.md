# Pendientes por método para V8

Fecha: 2026-10-13. Rama: `experiment/modular-audit-v8`. Estado: preparación. Este documento no habilita ni inicia entrenamiento, auditorías o pruebas.

V7 queda excluida por completo: no se reutilizan sus checkpoints, bancos SSL adaptados, pseudo máscaras ni resultados entrenados. Los modelos aprendibles parten de inicialización aleatoria cuando la receta lo indica; las únicas inicializaciones externas permitidas son pesos públicos originales que la adaptación realmente requiera, con procedencia, licencia, arquitectura, normalización y SHA-256 verificados.

## Estado por método

| Método | Inicialización de V8 | Ya existe | Pendiente antes de una corrida completa |
|---|---|---|---|
| SLED | No necesita pesos ni entrenamiento. | Generador Python, abstenciones y variantes documentadas; etapas `generate/predict`. | Confirmar recursos para el manifiesto completo y congelar el manifiesto/resolución/métricas del protocolo común ya descrito en [45](45-flujo-preparacion-ejecucion-cuatro-metodos-v8.md). La configuración primaria ya queda congelada en `correct_all` + `gaussian_corrected`; `upstream_zero`/`upstream_area` quedan como controles documentados, fuera de la corrida primaria. |
| USL-Net | Encoders públicos SimCLR, MoCo v1 y MoCo v2 en el banco de fuentes; módulos entrenables nuevos desde cero. Nunca usar bancos adaptados de V7. | Generación CCAM y refinamiento para producir pseudo máscaras de otra ronda; el trainer acepta una inicialización iterativa del mismo método con procedencia validada; política UM con área mínima 500 y umbrales .35/.65. | El launcher expone `refine` con gate y comprueba que las predicciones correspondan al manifiesto de entrada. El nuevo controlador serial ejecuta pasos ordenados desde un plan explícito, registra hashes/estado/recursos y recupera `fit` desde `resume.pt`; no crea gates ni infiere el plan científico. Las salidas parciales no `fit` requieren revisión y un directorio nuevo. El corte de distancia al centro queda `null` como decisión V8 previa a resultados; no se descartan muestras por ese criterio. Verificar los tres checkpoints públicos y registrar hashes/normalización. |
| RPI-Net | MoCo v3, MoCo v2 y DINO original públicos; MoCo v3 alimenta dos fuentes. Modelo final nuevo desde cero. No sustituir DINO por DINOv2. | Generación de las cuatro fuentes P1–P4 y entrenamiento del segmentador; se documentaron adaptaciones de polaridad y objetivo Grad-CAM++. | El paper indica 50 épocas iniciales y fases sucesivas de 30, pero no cuántas fases. V8 congela una fase posterior de 30 épocas como adaptación mínima y explícita; no es un número reportado por el artículo. El controlador serial coordina fases y refrescos de P2 si cada paso y gate aparece en el plan; no genera automáticamente el manifiesto ni aprueba gates. Convertir y verificar en el clúster cada archivo oficial exacto (origen, arquitectura, licencia, SHA-256, claves omitidas, normalización y carga estricta). No marcar el banco como listo antes de esa revisión. |
| TriDermNet-HAGO | Arquitectura segmentadora inicializada desde cero, de acuerdo con la receta del paper; no requiere checkpoint público de segmentación ni pesos V7. | Red/objetivos de pseudo-segmentación y HAGO con calentamiento de 10 épocas y métricas de balance registradas por época; entrenamiento acepta un manifiesto con pseudo máscaras, sujeto a fijar y verificar su maestro. | No hay una etapa TriDerm que genere sus propias pseudo máscaras; V8 fija SLED como maestro no supervisado. El trainer bloquea la corrida si falta el SHA-256 exacto del manifiesto y comprueba que cada fila provenga del método fijado. No se usan máscaras expertas ni diagnósticos; el hash se completa después de generar y revisar el manifiesto SLED. Cada época registra fracción de calentamiento, peso HAGO medio, entropía media y fracción de pasos con ruido. La adaptación reemplaza diagnósticos y máscaras expertas por cobertura/área pseudo; se reporta como variante no supervisada, no como reproducción de resultados supervisados. V8 fija 150 épocas a 224px y warmup de 10 como presupuesto de adaptación por pseudo máscaras. |

## Detalle y criterio de cierre

### SLED

SLED produce la máscara directamente, por lo que no debe aparecer una etapa ficticia de entrenamiento o checkpoint. Se conservan sus abstenciones en el denominador: no se eliminan casos difíciles para mejorar métricas. La verificación previa de 40.896 imágenes fue una auditoría ya terminada y no se repite aquí.

**Cierre requerido:** medir en el clúster tiempo, RAM y almacenamiento de `generate` para el manifiesto congelado; verificar espacio libre y salida estimada; fijar los hashes del manifiesto de evaluación y de las salidas. La comparación primaria ya está especificada en [45](45-flujo-preparacion-ejecucion-cuatro-metodos-v8.md): SLED directo frente a predicciones finales, informando cobertura/abstenciones por separado. No ejecutar aún ese cálculo.

### USL-Net

El protocolo del paper usa cinco ciclos de refinamiento y 300 épocas por ciclo. El config conserva esos valores. `scripts/run_segmentation_controller_v8.py` puede recorrer en orden las cinco secuencias si cada paso, manifiesto y gate se declara en el plan; el helper de refinamiento por sí solo no constituye el flujo completo. Cada ciclo necesita identidad propia y guardar como mínimo el manifiesto de entrada, checkpoint, predicciones, pseudo máscaras resultantes, hashes, cobertura y estado. Una interrupción debe poder continuar el ciclo desde un checkpoint íntegro o, si no, reabrir una salida nueva para ese ciclo; nunca mezclar productos parciales.

El paper define un umbral de área de 500 píxeles y los umbrales .35/.65. No fija el límite de distancia centroide que descarta muestras; por tanto no inventarlo a partir de resultados observados. Mantener sin corte implica documentar esa decisión y conservar los casos sin foreground identificable según las reglas existentes.

**Cierre requerido:** preparar/revisar el plan serial de cinco ciclos y gates; concretar procedencia reproducible de los tres encoders públicos; fijar tamaño de entrada y budgets de GPU/disco con lectura actual del clúster. El entrenamiento del segmentador y CCAM parte sin pesos V7.

### RPI-Net

Las fuentes son P1 KMeans de MoCo v3 conv1, P2 Grad-CAM++ de MoCo v3, P3 KMeans de MoCo v2 conv1 y P4 KMeans del DINO original. El objetivo CAM de energía SSL y la orientación por mayoría del borde son adaptaciones, no detalles especificados por el paper. No invertir la polaridad usando etiquetas expertas.

El paper describe 50 épocas iniciales y luego fases de 30 épocas, sin declarar el número de fases. V8 fija una fase posterior (50 + 30 épocas) como adaptación mínima y reproducible. No hay que interpretar el número de fuentes (cuatro) como número de fases. La actualización de P2 debe conservar P1/P3/P4 y registrar hashes por fase. El config fija `phase_count=1` e identifica que es una adaptación V8, no un valor del paper.

**Cierre requerido:** registrar la adaptación 50+30 en protocolo/config y W&B; declarar pasos/gates en el plan y verificar los checkpoints públicos elegidos y su conversión estricta. Mantener P1–P4 separados en manifiestos.

### TriDermNet-HAGO

El paper original usa supervisión experta y clasificación diagnóstica. La variante V8 elimina esas etiquetas y usa una cobertura de lesión pseudo como objetivo auxiliar. No existe una inicialización pública original del segmentador indicada por esa receta; por eso, la red debe empezar aleatoriamente y los campos de checkpoint genérico deben permanecer vacíos. No usar los pesos de clasificación de Nico/V7 como sustituto.

Esta variante sí requiere pseudo máscaras de un maestro. El paper no prescribe uno. V8 elige SLED como maestro sin etiquetas humanas para mantener el flujo totalmente no supervisado. TriDerm usa esas pseudo máscaras como objetivos de entrenamiento: es una adaptación no supervisada basada en pseudo etiquetas, pero queda condicionada por SLED y no debe presentarse como una cuarta salida independiente. Reportar siempre la pareja SLED→TriDerm y no mezclarla con una réplica supervisada del paper. No presentar las cifras supervisadas del paper como resultado esperado de esta variante.

HAGO tiene calentamiento de diez épocas. La configuración V8 ya fija 150 épocas a 224×224 como presupuesto de adaptación no supervisada; no es reproducción del objetivo supervisado original de 150 épocas a 256×256.

**Cierre requerido:** generar pseudo máscaras con SLED, revisar su procedencia/cobertura y congelar el SHA-256 del manifiesto antes de entrenar; verificar presupuesto de GPU/RAM/tiempo y que la receta completa el warmup y registra la evolución de pesos HAGO. TriDerm consume el manifiesto fijo de SLED; no genera etiquetas desde una red aleatoria.

## Compatibilidad de las inicializaciones públicas

La lista de orígenes ya está acotada por método. Se agregó un adaptador estricto para formatos PyTorch de ResNet50 (MoCo v1/v2/v3 y un formato SimCLR ya convertido) y DINO ViT-S/16. No se descargaron checkpoints ni se ejecutó una conversión real: ningún peso se considera verificado hasta probar cada formato contra el archivo oficial exacto y revisar licencia, hash y arquitectura. El código define encoders propios (ResNet50Stages y PatchTransformer) y load_reviewed exige que el campo model del checkpoint coincida estrictamente con esos nombres y formas de parámetros. El script `scripts/prepare_public_ssl_weights_v8.py` convierte solo estructuras reconocidas, falla ante claves tensoriales inesperadas o faltantes, carga con `strict=True`, descarta únicamente cabezas declaradas y registra URL, licencia, hashes, normalización y versión del adaptador. `common/bank.py` exige esa procedencia pública. El formato TensorFlow original de SimCLR tiene una ruta separada (`--format simclr_tensorflow`). Se corrigió el escape de los patrones que reconocen los sufijos numéricos de las capas `conv2d_N` y `batch_normalization_N`; el código oficial usa esos nombres bajo `base_model/`. Aún no se ha ejecutado contra los tensores oficiales, por lo que la conversión sigue sin verificar.

| Origen | Fuente pública candidata | Trabajo técnico pendiente |
|---|---|---|
| SimCLR v1 ResNet50 1x | [Google Research SimCLR](https://github.com/google-research/simclr), archivo oficial [ResNet50_1x.zip](https://storage.cloud.google.com/simclr-gcs/checkpoints/ResNet50_1x.zip) | El README oficial identifica un encoder preentrenado SimCLRv1 con cabeza lineal y sin cabeza de proyección. El adaptador V8 lee el checkpoint TensorFlow, mapea el encoder y excluye la cabeza de clasificación. Falta obtener el archivo en el clúster, revisar los nombres/formas/licencia del artefacto y verificar carga estricta y normalización; el enlace identifica el archivo pero no prueba que la conversión funcione. |
| MoCo v1 ResNet50 | [Checkpoint oficial MoCo v1, 200 épocas](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar) | Candidato compatible con ResNet50. Confirmar que el estado real corresponde al encoder esperado, guardar SHA-256 y licencia del artefacto, normalización y carga estricta. |
| MoCo v2 ResNet50 | [Checkpoint oficial MoCo v2, 800 épocas](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar) | Candidato compatible con ResNet50 y requerido por USL-Net/RPI-Net. Confirmar licencia del artefacto, SHA-256, normalización y carga estricta. |
| MoCo v3 ResNet50 | [Checkpoint oficial ResNet50 de 1000 épocas](https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar), publicado en [CONFIG.md de MoCo v3](https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md) | Candidato para P1/P2 de RPI-Net. Confirmar que coincide con las fuentes del protocolo, licencia del artefacto, SHA-256, normalización y carga estricta. |
| DINO ViT-S/16 | [Backbone oficial `dino_deitsmall16_pretrain.pth`](https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth), publicado en [Facebook Research DINO](https://github.com/facebookresearch/dino) | El README diferencia el backbone preentrenado del checkpoint completo con cabezas teacher/student; se elige el backbone solamente. Falta revisar la estructura real del archivo, licencia del artefacto, SHA-256, normalización y carga estricta. |

Las fuentes anteriores identifican repositorios oficiales, no certifican que el paper de cada método haya usado exactamente esos checkpoints o número de épocas. Se debe contrastar esa selección con los tres papers y guardar URL de archivo, licencia, SHA-256 y normalización en el manifiesto de procedencia. No declarar verificación sin comprobar el archivo exacto. La conversión SimCLR requiere un lector TensorFlow disponible en el entorno del clúster. La alternativa de preentrenar encoders V8 localmente sigue siendo una adaptación distinta y no sustituye en silencio los inicializadores públicos requeridos.

## Bloqueos comunes

1. Congelar manifiesto elegible y partición por paciente/lesión; resolver la ruta y la reserva de máscaras de referencia independientes para evaluación anatómica.
2. Congelar el manifiesto reservado y las reglas del protocolo común descrito en [45](45-flujo-preparacion-ejecucion-cuatro-metodos-v8.md): SLED directo frente a redes finales, TriDerm presentado como SLED→TriDerm, y cobertura/abstenciones separadas de métricas finales.
3. Completar bancos públicos únicamente para USL-Net y RPI-Net. Los candidatos con URL exacta quedan identificados en la tabla; descargar o transferir sus archivos al clúster en una carpeta inmutable cuando se confirme espacio disponible. Calcular SHA-256, registrar licencia/origen y validar cada archivo con carga estricta. SimCLR v1 requiere además el lector TensorFlow; el código no descarga ni ejecuta archivos remotos.
4. Medir recursos actuales del clúster y reemplazar, con evidencia, `device=cpu`, `batch_size=2`, timeout de 180 s y salida máxima de 4096 MiB. Los presupuestos de épocas por método ya están fijados; los límites de recursos siguen siendo provisionales y no describen una corrida completa.
5. Preparar planes seriales de USL/RPI con pasos y gates por ciclo/fase. El controlador registra y recupera `fit` con su checkpoint; las salidas parciales de generación/inferencia requieren revisión manual y una ruta nueva.
6. Mantener `execution_enabled=false` hasta que gates, configuraciones, pesos, datos, evaluación y presupuesto coincidan y estén revisados.

## Orden de trabajo propuesto

1. Congelar comparación, particiones y referencia anatómica.
2. Cerrar procedencia pública de USL/RPI; TriDerm empieza sin checkpoint externo; SLED no usa pesos.
3. Aplicar las decisiones de adaptación V8 ya fijadas: congelar SLED como maestro de TriDerm y registrar explícitamente la fase posterior 50+30 de RPI como adaptación.
4. Preparar y revisar los planes reanudables de USL/RPI, más el manifiesto SLED fijo para TriDerm.
5. Confirmar budgets del clúster y actualizar configs/gates.
6. Cuando todos los cierres estén respaldados, habilitar y ejecutar los métodos serialmente, con revisión entre métodos.

Hasta completar los pasos anteriores, los cuatro métodos siguen en preparación. Este documento no afirma que las corridas estén listas.
\n\nPara el criterio no supervisado, las limitaciones del conversor y el esquema del controlador, ver [47](47-criterio-no-supervisado-y-controlador-v8.md).\n