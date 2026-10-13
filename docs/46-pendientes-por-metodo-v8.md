# Pendientes por método para V8

Fecha: 2026-10-13. Rama: `experiment/modular-audit-v8`. Estado: preparación. Este documento no habilita ni inicia entrenamiento, auditorías o pruebas.

V7 queda excluida por completo: no se reutilizan sus checkpoints, bancos SSL adaptados, pseudo máscaras ni resultados entrenados. Los modelos aprendibles parten de inicialización aleatoria cuando la receta lo indica; las únicas inicializaciones externas permitidas son pesos públicos originales que la adaptación realmente requiera, con procedencia, licencia, arquitectura, normalización y SHA-256 verificados.

## Estado por método

| Método | Inicialización de V8 | Ya existe | Pendiente antes de una corrida completa |
|---|---|---|---|
| SLED | No necesita pesos ni entrenamiento. | Generador Python, abstenciones y variantes documentadas; etapas `generate/predict`. | Confirmar recursos para el manifiesto completo en el clúster; fijar comparación/evaluación común. Aclarar y congelar variantes de centroides/selector en el protocolo. |
| USL-Net | Encoders públicos SimCLR, MoCo v1 y MoCo v2 en el banco de fuentes; módulos entrenables nuevos desde cero. Nunca usar bancos adaptados de V7. | Generación CCAM y refinamiento para producir pseudo máscaras de otra ronda; el trainer acepta una inicialización iterativa del mismo método con procedencia validada; política UM con área mínima 500 y umbrales .35/.65. | El launcher no coordina los cinco ciclos ni enlaza CCAM → máscaras → ajuste del estudiante → siguiente refinamiento. El trainer ya puede iniciar una fase desde un checkpoint anterior del mismo método con provenance validada, pero el controlador serial de ciclos y sus gates por etapa siguen pendientes. Implementar etapas reanudables con manifiesto/hash por ciclo. El paper no da un corte numérico para distancia al centro; mantenerlo sin corte inventado o fijarlo como adaptación antes de mirar resultados. Verificar los tres checkpoints públicos y registrar hashes/normalización. |
| RPI-Net | MoCo v3, MoCo v2 y DINO original públicos; MoCo v3 alimenta dos fuentes. Modelo final nuevo desde cero. No sustituir DINO por DINOv2. | Generación de las cuatro fuentes P1–P4 y entrenamiento del segmentador; se documentaron adaptaciones de polaridad y objetivo Grad-CAM++. | El paper indica 50 épocas iniciales y fases sucesivas de 30, pero no cuántas fases. Quedó congelada una fase posterior de 30 épocas como adaptación V8 mínima y explícita; no es un número reportado por el artículo. El launcher aún no coordina esa fase ni el refresco de P2. El trainer acepta una inicialización iterativa con provenance, pero aún falta el controlador que genere el manifiesto actualizado de cada fase y aplique sus gates. Antes de programar la receta completa hay que fijar y justificar un número de fases como adaptación V8. Verificar origen, arquitectura, hashes y normalización de los pesos públicos. |
| TriDermNet-HAGO | Arquitectura segmentadora inicializada desde cero, de acuerdo con la receta del paper; no requiere checkpoint público de segmentación ni pesos V7. | Red/objetivos de pseudo-segmentación y HAGO con calentamiento de 10 épocas y métricas de balance registradas por época; entrenamiento acepta un manifiesto con pseudo máscaras, sujeto a fijar y verificar su maestro. | No hay etapa que genere esas pseudo máscaras ni se ha declarado un maestro V8. El trainer ahora bloquea la corrida si faltan el método maestro o el SHA-256 exacto del manifiesto; además, comprueba que la procedencia de cada fila coincida con el método fijado. Elegir y congelar ambos antes de entrenar. Cada época ahora registra fracción de calentamiento, peso HAGO medio, entropía media y fracción de pasos con ruido. La adaptación reemplaza diagnósticos y máscaras expertas por cobertura/área pseudo, así que debe reportarse como variante, no como reproducción de los resultados supervisados del paper. El ejemplo actual de 1 época nunca alcanza HAGO. |

## Detalle y criterio de cierre

### SLED

SLED produce la máscara directamente, por lo que no debe aparecer una etapa ficticia de entrenamiento o checkpoint. Se conservan sus abstenciones en el denominador: no se eliminan casos difíciles para mejorar métricas. La verificación previa de 40.896 imágenes fue una auditoría ya terminada y no se repite aquí.

**Cierre requerido:** medir en el clúster tiempo, RAM y almacenamiento de la etapa `generate` para el manifiesto congelado; verificar espacio libre y salida estimada; decidir si la comparación primaria es la máscara SLED frente a máscaras finales de redes, y reportar por separado cobertura/abstención. No lanzar ahora ese cálculo.

### USL-Net

El protocolo del paper usa cinco ciclos de refinamiento y 300 épocas por ciclo. Esos valores aparecen en el config, pero actualmente son descriptivos: `run_segmentation_v8.py` no los orquesta. El helper de refinamiento no equivale al flujo iterativo completo. Cada ciclo necesita identidad propia y guardar como mínimo el manifiesto de entrada, checkpoint, predicciones, pseudo máscaras resultantes, hashes, cobertura y estado. Una interrupción debe poder continuar el ciclo desde un checkpoint íntegro o, si no, reabrir una salida nueva para ese ciclo; nunca mezclar productos parciales.

El paper define un umbral de área de 500 píxeles y los umbrales .35/.65. No fija el límite de distancia centroide que descarta muestras; por tanto no inventarlo a partir de resultados observados. Mantener sin corte implica documentar esa decisión y conservar los casos sin foreground identificable según las reglas existentes.

**Cierre requerido:** implementar el controlador serial de ciclos y su recuperación; concretar procedencia reproducible de los tres encoders públicos; fijar el tamaño de entrada y budgets de GPU/disco con evidencia del clúster. El entrenamiento del segmentador y CCAM parte sin pesos V7.

### RPI-Net

Las fuentes son P1 KMeans de MoCo v3 conv1, P2 Grad-CAM++ de MoCo v3, P3 KMeans de MoCo v2 conv1 y P4 KMeans del DINO original. El objetivo CAM de energía SSL y la orientación por mayoría del borde son adaptaciones, no detalles especificados por el paper. No invertir la polaridad usando etiquetas expertas.

El paper describe 50 épocas iniciales y luego fases de 30 épocas, sin declarar el número de fases. V8 fija una fase posterior (50 + 30 épocas) como adaptación mínima y reproducible. No hay que interpretar el número de fuentes (cuatro) como número de fases. La actualización de P2 debe conservar P1/P3/P4 y registrar hashes por fase. El config fija `phase_count=1` e identifica que es una adaptación V8, no un valor del paper.

**Cierre requerido:** registrar la adaptación de una fase posterior en protocolo/config y W&B; implementar coordinación y recuperación por fase; verificar los tres modelos/checkpoints públicos y la conversión estricta. Mantener los cuatro orígenes separados en manifiestos.

### TriDermNet-HAGO

El paper original usa supervisión experta y clasificación diagnóstica. La variante V8 elimina esas etiquetas y usa una cobertura de lesión pseudo como objetivo auxiliar. No existe una inicialización pública original del segmentador indicada por esa receta; por eso, la red debe empezar aleatoriamente y los campos de checkpoint genérico deben permanecer vacíos. No usar los pesos de clasificación de Nico/V7 como sustituto.

Esta variante sí requiere pseudo máscaras de un maestro, pero el repo todavía no establece cuál. Esa elección afecta dependencia, sesgo y comparabilidad: un maestro único produce una evaluación de TriDerm condicionada por ese maestro; un consenso requiere especificar sus fuentes, regla y posible abstención. No presentar las cifras supervisadas del paper como resultado esperado de esta variante.

HAGO tiene calentamiento de diez épocas; un config de una época es un marcador de bloqueo y nunca ejecuta el balance de tareas. La receta original reporta 150 épocas y 256×256 para supervisión experta; esos valores no se trasladan automáticamente a la adaptación pseudo-supervisada de 224×224.

**Cierre requerido:** fijar el maestro/pseudo manifiesto y su procedencia antes de entrenar; declarar si 150 épocas se adopta como presupuesto de adaptación o si se definirá otro criterio de parada; asegurar que la receta supera el warmup y registra evolución de pesos HAGO. Agregar etapa de generación solo si TriDerm va a generar sus propias pseudo máscaras; no generar etiquetas desde una red aleatoria.

## Bloqueos comunes

1. Congelar manifiesto elegible y partición por paciente/lesión; resolver la ruta y la reserva de máscaras de referencia independientes para evaluación anatómica.
2. Acordar el protocolo de comparación: máscaras producidas directamente frente a segmentadores finales; mismas imágenes, resolución, umbrales, métricas, tratamiento de abstenciones y denominador. Distinguir resultados de pseudo máscaras de resultados de segmentación final.
3. Completar bancos públicos únicamente para USL-Net y RPI-Net. Guardar fuente oficial, licencia, arquitectura, normalización, SHA-256 y cualquier conversión revisada. No descargar ni aceptar pesos hasta verificar estos datos.
4. Medir recursos del clúster y reemplazar, con evidencia, los marcadores comunes actuales: `device=cpu`, `epochs=1`, `batch_size=2`, timeout 180 s y salida máxima 4096 MiB. Los límites no describen una corrida completa.
5. Hacer reanudables las etapas iterativas de USL y RPI y dejar un registro inmutable de cada ciclo/fase. El trainer ya puede cargar una inicialización previa auditada, y `fit` sigue siendo reanudable; todavía no existe el controlador que coordine `ccam`, `generate`, `predict` y refinamiento por ciclo/fase.
6. Mantener `execution_enabled=false` hasta que gates, configuraciones, pesos, datos, evaluación y presupuesto coincidan y estén revisados.

## Orden de trabajo propuesto

1. Congelar comparación, particiones y referencia anatómica.
2. Cerrar procedencia pública de USL/RPI; TriDerm empieza sin checkpoint externo; SLED no usa pesos.
3. Resolver las dos decisiones de adaptación no determinadas por los papers: maestro de pseudo máscaras para TriDerm.
4. Implementar los controladores reanudables de USL/RPI y la etapa de preparación de pseudo etiquetas para TriDerm según la decisión aprobada.
5. Confirmar budgets del clúster y actualizar configs/gates.
6. Cuando todos los cierres estén respaldados, habilitar y ejecutar los métodos serialmente, con revisión entre métodos.

Hasta completar los pasos anteriores, los cuatro métodos siguen en preparación. Este documento no afirma que las corridas estén listas.
