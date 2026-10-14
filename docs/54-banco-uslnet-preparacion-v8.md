# Paso 4: banco de pesos USL-Net

Estado: preparación del candidato completada localmente; carga conjunta en el clúster pendiente. Este trabajo no depende de que termine ISIC2018 y no modifica ni detiene esa descarga.

## Contrato del banco

USL-Net requiere tres encoders congelados y de orígenes distintos: SimCLR, MoCo v1 y MoCo v2. El cargador [`source_bank`](../src/dermssl/segmentation_methods/common/bank.py) rechaza un conjunto incompleto o con orígenes adicionales; carga cada `encoder.pt` con `strict=True`, verifica el SHA-256 del encoder y su procedencia, y aplica la normalización declarada.

La fusión toma las etapas 3 y 4 de cada ResNet50; a 224×224 produce 9.216 canales y una cuadrícula 14×14. SimCLR conserva `unit_rgb` y el maxpool TensorFlow SAME; MoCo v1/v2 usan normalización ImageNet y maxpool simétrico. Se mantiene la adaptación de stride final a 1 para la salida densa.

## Fuentes y rutas candidatas

El manifiesto [`source-bank.usl-net.candidate.json`](../plan-segmentacion/metodos-v8/source-bank.usl-net.candidate.json) contiene exactamente los tres orígenes y las rutas previstas bajo `/home/jovyan/work/public_ssl_v8_preparation/`.

- **SimCLR:** archivo Hugging Face fijado por revisión, 102.542.786 bytes; SHA-256 publicado y calculado coinciden. Conversión local estricta y comparación por etapa pasaron. Su conversión de terceros no acredita igualdad binaria ni paridad numérica con el checkpoint TensorFlow original de Google. La transferencia al clúster no está confirmada.
- **MoCo v1:** descarga, prefijo MD5 publicado y conversión estricta constan en el recibo de preparación del clúster.
- **MoCo v2:** descarga, longitud exacta, prefijo MD5 publicado y conversión estricta constan en el recibo del clúster. El mismo encoder se comparte con RPI-Net; no se reconvierte ni duplica.

Los pesos MoCo se conservan para uso académico no comercial conforme al alcance registrado; los archivos y avisos de atribución deben mantenerse junto con su procedencia.

## Estado y verificación que falta

La especificación del banco está lista, pero `candidate_execution_ready=false`. El runtime `runtime_simclr_same_eed86c8` figura desplegado y verificado en el recibo de la rama. No pude leer el estado vivo de Jupyter en esta sesión: la conexión devolvió `502 Bad Gateway` / `Connection refused`.

Cuando Jupyter responda, el gate del paso 4 es comprobar hashes/rutas de los tres pares encoder-procedencia, cargar `source_bank` con exactamente esos orígenes y hacer un único forward sintético de 224×224 para confirmar `[1, 9216, 14, 14]`. Esta comprobación no usa imágenes reales ni inicia entrenamiento. No repetir conversiones ya registradas.

La máquina de estado está en [`source-bank.usl-net.review.json`](../plan-segmentacion/metodos-v8/source-bank.usl-net.review.json). El resto de bloqueos del proyecto continúa: manifiesto elegible y agrupación anatómica, pseudo-máscaras SLED, límites finales por etapa y plan serial del controlador. `launch_ready=false` y las configuraciones siguen deshabilitadas.
