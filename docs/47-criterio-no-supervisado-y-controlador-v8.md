# Criterio no supervisado, pesos públicos y controlador V8

Estado: preparación; no habilita entrenamiento ni descargas. V7 y sus pesos quedan excluidos.

## Criterio metodológico

- En las cuatro adaptaciones, las máscaras expertas y los diagnósticos quedan fuera de todo entrenamiento, selección de checkpoint, ajuste de umbral y decisión de hiperparámetros. Se reservan para evaluación final independiente si el protocolo lo autoriza.
- Una inicialización pública solo se acepta si el checkpoint corresponde a aprendizaje autosupervisado y se usa su encoder. Se omiten cabezas de clasificación, evaluación lineal, proyección y predicción que no formen parte del encoder. La procedencia debe identificar el artefacto exacto, URL oficial, licencia, hash del archivo de origen, arquitectura, normalización y claves excluidas.
- SimCLR, MoCo v1/v2/v3 y DINO son objetivos de representación autosupervisada. Sus repositorios oficiales ([SimCLR](https://github.com/google-research/simclr), [MoCo v3](https://github.com/facebookresearch/moco-v3) y [DINO](https://github.com/facebookresearch/dino)) publican modelos y presupuestos de referencia; esto no implica que un checkpoint concreto ya esté validado ni que todos los formatos sean intercambiables. SimCLR publica los pesos originales en TensorFlow y remite a un conversor PyTorch; V8 implementa una ruta TensorFlow directa para evitar importar ese conversor externo, pendiente aún de validación con el archivo elegido.
- TriDermNet-HAGO del paper usa máscaras expertas y diagnóstico. V8 es una adaptación de pseudo etiquetas: pesos aleatorios al inicio y pseudo máscaras fijadas de SLED como maestro. Se reporta como **SLED→TriDermNet-HAGO**, no como réplica supervisada ni método independiente de SLED. El archivo de configuración debe fijar el hash exacto del manifiesto antes de entrenar. La adaptación debe justificar su presupuesto de épocas; la configuración de una época es solo un bloqueo.
- USL-Net itera cinco ciclos de 300 épocas según la receta registrada. RPI-Net usa 50 épocas iniciales y una fase posterior de 30 como adaptación V8. Mantiene P1/P3/P4 y refresca únicamente P2. Los presupuestos son configuración, no autorización para iniciar.

## Adaptadores de pesos públicos

`scripts/prepare_public_ssl_weights_v8.py` convierte estructuras PyTorch conocidas de ResNet50 (MoCo v1/v2/v3 y checkpoints SimCLR ya convertidos a PyTorch) y DINO ViT-S/16 a los nombres locales de V8. Requiere un checkpoint local, no descarga nada. La conversión falla ante pesos faltantes, formas incompatibles o claves de encoder desconocidas; la carga final es estricta. La procedencia se guarda junto al checkpoint y `common/bank.py` exige los campos de origen y licencia.

Límite actual: la ruta TensorFlow usa `tf.train.load_checkpoint`, ordena y valida los scopes `base_model/`, transforma filtros HWIO→OIHW, inicializa `num_batches_tracked` a cero y registra las cabezas excluidas. Todavía no se ha ejecutado con un archivo oficial; el clúster necesitará disponer de TensorFlow. No se debe marcar SimCLR verificado hasta revisar las formas y cargar el resultado estrictamente. Ningún adaptador se considera verificado hasta ejecutar la conversión sobre cada archivo elegido, cotejar licencia/URL/hash/arquitectura/normalización y revisar que la cabeza excluida no contenga pesos de clasificación que lleguen al modelo.

## Coordinación y recuperación

`scripts/run_segmentation_controller_v8.py` ejecuta en serie una lista ordenada de pasos desde un plan JSON. Cada paso declara método, etapa, configuración, manifiesto congelado, gate revisado, carpeta nueva y, para `refine`, la salida `predict` que le corresponde. El controlador no inventa gates ni elige pseudo etiquetas.

- Sin `--execute`, solo imprime el orden del plan.
- Con `--execute`, registra un ledger atómico, bloquea ejecuciones duplicadas y conserva estado/errores por paso.
- Un paso completado solo se omite al reanudar si el plan, manifiesto, gate, configuración y hashes de salida coinciden.
- Si se interrumpe un `fit`, el mismo plan puede continuar desde `resume.pt`, con identidad cotejada por el trainer. Una salida parcial de `generate`, `predict` o `refine` no se reutiliza: requiere inspección y una ruta de salida nueva.
- El plan enumera explícitamente todos los ciclos/fases. Para USL se encadenan cinco ternas `fit → predict → refine` (más la generación inicial); para RPI se definen la fase inicial y la posterior; TriDerm consume el manifiesto fijo de SLED. Si todavía falta un manifiesto, predicción o gate de un paso futuro, el controlador deja el ledger en `awaiting_review` y se reanuda con el mismo plan (`--resume`) cuando ese archivo esté disponible.
- El controlador registra muestreos de RAM, disco, GPU y RSS del proceso y aplica los límites de tiempo/RSS definidos en cada configuración. Estos valores describen la sesión visible del usuario; no son una reserva de recursos frente a otros jobs del clúster.

El controlador es un motor serial recuperable, no genera automáticamente un plan científico ni aprueba las fases posteriores. Antes de la corrida se necesita un plan concreto con los paths del pod, configuraciones por fase, manifiestos y gates validados.

## Recursos del clúster

El último inventario guardado en `resultados/entorno/hardware.json` se midió el **2026-10-05**: 2 GPU NVIDIA RTX PRO 4000 Blackwell con 24.467 MiB cada una, CPU de 48 hilos lógicos, 123 GiB de RAM, volumen compartido de 40 T y `/dev/shm` de 64 MiB. Es una referencia histórica, no una lectura de disponibilidad actual.

La observación actual de Jupyter no quedó disponible: el navegador rechazó leer el estado retenido tras una entrega de credenciales. Por tanto, el estado actual de procesos, VRAM libre, RAM/disco disponibles y logs del pod queda **sin confirmar**; no se interpreta como clúster vacío. La próxima medición debe ser de solo lectura en el pod y guardar fecha UTC, límites cgroup, procesos existentes, GPU y espacio libre antes de crear el plan de ejecución.

## Lista de cierre previa al entrenamiento

1. Congelar el manifiesto de entrenamiento sin máscaras expertas ni diagnósticos, y registrar su hash.
2. Seleccionar cada checkpoint público con referencia exacta al paper y registrar licencia/origen/normalización/hash.
3. Convertir y verificar los encoders en el pod; resolver el conversor TensorFlow oficial de SimCLR.
4. Terminar las configuraciones por método/fase y quitar los marcadores de una época solo después de fijar presupuestos.
5. Tomar lectura actual de recursos del clúster y revisar salidas estimadas.
6. Crear un plan serial con gates por etapa y revisar su orden sin usar `--execute`.
7. Habilitar ejecución solo tras revisar el plan y los registros previos; iniciar los métodos uno por uno.
