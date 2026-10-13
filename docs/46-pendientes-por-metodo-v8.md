# Pendientes por método para V8

Fecha: 2026-10-13. Rama: `experiment/modular-audit-v8`. Estado: preparación. Este documento no habilita ni inicia entrenamiento, auditorías o pruebas.

V7 queda excluida por completo: no se reutilizan sus checkpoints, bancos SSL adaptados, pseudo máscaras ni resultados entrenados. Los modelos aprendibles parten de inicialización aleatoria cuando la receta lo indica; las únicas inicializaciones externas permitidas son pesos públicos originales que la adaptación realmente requiera, con procedencia, licencia, arquitectura, normalización y SHA-256 verificados.

## Estado por método

| Método | Inicialización de V8 | Ya existe | Pendiente antes de una corrida completa |
|---|---|---|---|
| SLED | No necesita pesos ni entrenamiento. | Generador Python, abstenciones y variantes documentadas; etapas `generate/predict`. | Confirmar recursos para el manifiesto completo en el clúster; fijar comparación/evaluación común. La configuración primaria ya queda congelada en `correct_all` + `gaussian_corrected`; `upstream_zero`/`upstream_area` quedan como controles documentados, fuera de la corrida primaria. |
| USL-Net | Encoders públicos SimCLR, MoCo v1 y MoCo v2 en el banco de fuentes; módulos entrenables nuevos desde cero. Nunca usar bancos adaptados de V7. | Generación CCAM y refinamiento para producir pseudo máscaras de otra ronda; el trainer acepta una inicialización iterativa del mismo método con procedencia validada; política UM con área mínima 500 y umbrales .35/.65. | El launcher expone `refine` con gate y comprueba que las predicciones correspondan al manifiesto de entrada. El nuevo controlador serial ejecuta pasos ordenados desde un plan explícito, registra hashes/estado/recursos y recupera `fit` desde `resume.pt`; no crea gates ni infiere el plan científico. Las salidas parciales no `fit` requieren revisión y un directorio nuevo. El corte de distancia al centro queda `null` como decisión V8 previa a resultados; no se descartan muestras por ese criterio. Verificar los tres checkpoints públicos y registrar hashes/normalización. |
| RPI-Net | MoCo v3, MoCo v2 y DINO original públicos; MoCo v3 alimenta dos fuentes. Modelo final nuevo desde cero. No sustituir DINO por DINOv2. | Generación de las cuatro fuentes P1–P4 y entrenamiento del segmentador; se documentaron adaptaciones de polaridad y objetivo Grad-CAM++. | El paper indica 50 épocas iniciales y fases sucesivas de 30, pero no cuántas fases. Quedó congelada una fase posterior de 30 épocas como adaptación V8 mínima y explícita; no es un número reportado por el artículo. El controlador serial puede coordinar fases y refrescos de P2 si cada paso y gate aparece en el plan; no genera automáticamente el manifiesto ni aprueba gates. La adaptación V8 queda fijada en una fase posterior de 30 épocas. Convertir y verificar en el clúster cada archivo oficial exacto (origen, arquitectura, licencia, SHA-256, claves omitidas, normalización y carga estricta). No marcar el banco como listo antes de esa revisión. |
| TriDermNet-HAGO | Arquitectura segmentadora inicializada desde cero, de acuerdo con la receta del paper; no requiere checkpoint público de segmentación ni pesos V7. | Red/objetivos de pseudo-segmentación y HAGO con calentamiento de 10 épocas y métricas de balance registradas por época; entrenamiento acepta un manifiesto con pseudo máscaras, sujeto a fijar y verificar su maestro. | No hay una etapa TriDerm que genere sus propias pseudo máscaras; V8 fija SLED como maestro no supervisado. El trainer bloquea la corrida si falta el SHA-256 exacto del manifiesto y comprueba que cada fila provenga del método fijado. No se usan máscaras expertas ni diagnósticos; el hash se completa después de generar y revisar el manifiesto SLED. Cada época ahora registra fracción de calentamiento, peso HAGO medio, entropía media y fracción de pasos con ruido. La adaptación reemplaza diagnósticos y máscaras expertas por cobertura/área pseudo, así que debe reportarse como variante, no como reproducción de los resultados supervisados del paper. V8 fija 150 épocas a 224px como presupuesto de adaptación por pseudo máscaras, con warmup de 10; es una adaptación no supervisada del presupuesto del paper, no una reproducción de su objetivo supervisado. |

## Detalle y criterio de cierre

### SLED

SLED produce la máscara directamente, por lo que no debe aparecer una etapa ficticia de entrenamiento o checkpoint. Se conservan sus abstenciones en el denominador: no se eliminan casos difíciles para mejorar métricas. La verificación previa de 40.896 imágenes fue una auditoría ya terminada y no se repite aquí.

**Cierre requerido:** medir en el clúster tiempo, RAM y almacenamiento de la etapa `generate` para el manifiesto congelado; verificar espacio libre y salida estimada; decidir si la comparación primaria es la máscara SLED frente a máscaras finales de redes, y reportar por separado cobertura/abstención. No lanzar ahora ese cálculo.

### USL-Net

El protocolo del paper usa cinco ciclos de refinamiento y 300 épocas por ciclo. El config conserva esos valores. `scripts/run_segmentation_controller_v8.py` puede recorrer en orden las cinco secuencias si cada paso, manifiesto y gate se declara en el plan; el helper de refinamiento por sí solo no constituye el flujo completo. Cada ciclo necesita identidad propia y guardar como mínimo el manifiesto de entrada, checkpoint, predicciones, pseudo máscaras resultantes, hashes, cobertura y estado. Una interrupción debe poder continuar el ciclo desde un checkpoint íntegro o, si no, reabrir una salida nueva para ese ciclo; nunca mezclar productos parciales.

El paper define un umbral de área de 500 píxeles y los umbrales .35/.65. No fija el límite de distancia centroide que descarta muestras; por tanto no inventarlo a partir de resultados observados. Mantener sin corte implica documentar esa decisión y conservar los casos sin foreground identificable según las reglas existentes.

**Cierre requerido:** preparar/revisar el plan serial de cinco ciclos y gates; concretar procedencia reproducible de los tres encoders públicos; fijar tamaño de entrada y budgets de GPU/disco con lectura actual del clúster. El entrenamiento del segmentador y CCAM parte sin pesos V7.

### RPI-Net

Las fuentes son P1 KMeans de MoCo v3 conv1, P2 Grad-CAM++ de MoCo v3, P3 KMeans de MoCo v2 conv1 y P4 KMeans del DINO original. El objetivo CAM de energía SSL y la orientación por mayoría del borde son adaptaciones, no detalles especificados por el paper. No invertir la polaridad usando etiquetas expertas.

El paper describe 50 épocas iniciales y luego fases de 30 épocas, sin declarar el número de fases. V8 fija una fase posterior (50 + 30 épocas) como adaptación mínima y reproducible. No hay que interpretar el número de fuentes (cuatro) como número de fases. La actualización de P2 debe conservar P1/P3/P4 y registrar hashes por fase. El config fija `phase_count=1` e identifica que es una adaptación V8, no un valor del paper.

**Cierre requerido:** registrar la adaptación de una fase posterior en protocolo/config y W&B; declarar pasos/gates en el plan y verificar los tres modelos/checkpoints públicos y su conversión estricta. Mantener los cuatro orígenes separados en manifiestos.

### TriDermNet-HAGO

El paper original usa supervisión experta y clasificación diagnóstica. La variante V8 elimina esas etiquetas y usa una cobertura de lesión pseudo como objetivo auxiliar. No existe una inicialización pública original del segmentador indicada por esa receta; por eso, la red debe empezar aleatoriamente y los campos de checkpoint genérico deben permanecer vacíos. No usar los pesos de clasificación de Nico/V7 como sustituto.

Esta variante sí requiere pseudo máscaras de un maestro. El paper no prescribe uno. V8 elige SLED como maestro sin etiquetas humanas para mantener el flujo totalmente no supervisado. TriDerm usa esas pseudo máscaras como objetivos de entrenamiento: es una adaptación no supervisada basada en pseudo etiquetas, pero queda condicionada por SLED y no debe presentarse como una cuarta salida independiente. Reportar siempre la pareja SLED→TriDerm y no mezclarla con una réplica supervisada del paper. No presentar las cifras supervisadas del paper como resultado esperado de esta variante.

HAGO tiene calentamiento de diez épocas; un config de una época es un marcador de bloqueo y nunca ejecuta el balance de tareas. La receta original reporta 150 épocas y 256×256 para supervisión experta; esos valores no se trasladan automáticamente a la adaptación pseudo-supervisada de 224×224.

**Cierre requerido:** fijar el maestro/pseudo manifiesto y su procedencia antes de entrenar; declarar si 150 épocas se adopta como presupuesto de adaptación o si se definirá otro criterio de parada; asegurar que la receta supera el warmup y registra evolución de pesos HAGO. Agregar etapa de generación solo si TriDerm va a generar sus propias pseudo máscaras; no generar etiquetas desde una red aleatoria.

## Compatibilidad de las inicializaciones públicas

La lista de orígenes ya está acotada por método. Se agregó un adaptador estricto para formatos PyTorch de ResNet50 (MoCo v1/v2/v3 y un formato SimCLR ya convertido) y DINO ViT-S/16. No se descargaron checkpoints ni se ejecutó una conversión real: ningún peso se considera verificado hasta probar cada formato contra el archivo oficial exacto y revisar licencia, hash y arquitectura. El código define encoders propios (ResNet50Stages y PatchTransformer) y load_reviewed exige que el campo model del checkpoint coincida estrictamente con esos nombres y formas de parámetros. El script `scripts/prepare_public_ssl_weights_v8.py` convierte solo estructuras reconocidas, falla ante claves tensoriales inesperadas o faltantes, carga con `strict=True`, descarta únicamente cabezas declaradas y registra URL, licencia, hashes, normalización y versión del adaptador. `common/bank.py` exige esa procedencia pública. El formato TensorFlow original de SimCLR tiene ahora una ruta separada (`--format simclr_tensorflow`); no se debe confundir con el lector PyTorch ni declararse verificado antes de comprobar un checkpoint oficial.

| Origen | Fuente pública candidata | Trabajo técnico pendiente |
|---|---|---|
| SimCLR v1 ResNet50 1x | [Google Research SimCLR](https://github.com/google-research/simclr), pesos publicados por el proyecto | El checkpoint oficial es TensorFlow. El adaptador V8 ya implementa lectura directa de scopes `base_model/` y mapeo de conv/BN al ResNet local, y excluye la cabeza `head_supervised`; falta ejecutarlo contra el archivo oficial exacto y revisar las formas/normalización. |
| MoCo v1/v2 ResNet50 | [Facebook Research MoCo](https://github.com/facebookresearch/moco); candidato MoCo v2 800 épocas | El adaptador cubre formatos PyTorch reconocidos. Seleccionar el archivo exacto y verificar licencia, hash, normalización y carga estricta. |
| MoCo v3 ResNet50 | [Facebook Research MoCo v3](https://github.com/facebookresearch/moco-v3), con checkpoints oficiales de 100/300/1000 épocas | El adaptador cubre formatos PyTorch reconocidos. Elegir el checkpoint conforme al paper y verificar licencia archivada, hash, normalización y carga estricta. |
| DINO ViT-S/16 | [Facebook Research DINO](https://github.com/facebookresearch/dino), entrada oficial dino_vits16 | Se agregó el mapeo de patch embed, posición, QKV, MLP y normalización. Falta verificarlo contra el archivo oficial exacto, con licencia, hash y carga estricta. |

Las fuentes anteriores identifican repositorios oficiales, no certifican que el paper de cada método haya usado exactamente esos checkpoints o número de épocas. Se debe contrastar esa selección con los tres papers y guardar URL de archivo, licencia, SHA-256 y normalización en el manifiesto de procedencia. No declarar verificación sin comprobar el archivo exacto. La conversión SimCLR requiere un lector TensorFlow disponible en el entorno del clúster. La alternativa de preentrenar encoders V8 localmente sigue siendo una adaptación distinta y no sustituye en silencio los inicializadores públicos requeridos.

## Bloqueos comunes

1. Congelar manifiesto elegible y partición por paciente/lesión; resolver la ruta y la reserva de máscaras de referencia independientes para evaluación anatómica.
2. Acordar el protocolo de comparación: máscaras producidas directamente frente a segmentadores finales; mismas imágenes, resolución, umbrales, métricas, tratamiento de abstenciones y denominador. Distinguir resultados de pseudo máscaras de resultados de segmentación final.
3. Completar bancos públicos únicamente para USL-Net y RPI-Net. Usar los adaptadores estrictos y validar cada archivo exacto antes de aceptar pesos; SimCLR requiere además el lector TensorFlow.
4. Medir recursos del clúster y reemplazar, con evidencia, los marcadores comunes actuales: `device=cpu`, `epochs=1`, `batch_size=2`, timeout 180 s y salida máxima 4096 MiB. Los límites no describen una corrida completa.
5. Preparar planes seriales de USL/RPI con pasos y gates por ciclo/fase. El controlador registra y recupera `fit` con su checkpoint; las salidas parciales de generación/inferencia requieren revisión manual y una ruta nueva.
6. Mantener `execution_enabled=false` hasta que gates, configuraciones, pesos, datos, evaluación y presupuesto coincidan y estén revisados.

## Orden de trabajo propuesto

1. Congelar comparación, particiones y referencia anatómica.
2. Cerrar procedencia pública de USL/RPI; TriDerm empieza sin checkpoint externo; SLED no usa pesos.
3. Resolver las dos decisiones de adaptación no determinadas por los papers: maestro de pseudo máscaras para TriDerm.
4. Preparar y revisar los planes reanudables de USL/RPI, más el manifiesto SLED fijo para TriDerm.
5. Confirmar budgets del clúster y actualizar configs/gates.
6. Cuando todos los cierres estén respaldados, habilitar y ejecutar los métodos serialmente, con revisión entre métodos.

Hasta completar los pasos anteriores, los cuatro métodos siguen en preparación. Este documento no afirma que las corridas estén listas.
\n\nPara el criterio no supervisado, las limitaciones del conversor y el esquema del controlador, ver [47](47-criterio-no-supervisado-y-controlador-v8.md).\n