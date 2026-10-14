> Estado más reciente: [descarga Hugging Face y validación local](52-simclr-huggingface-validacion-local-v8.md). Las afirmaciones siguientes sobre SimCLR pendiente describen el estado previo a esa validación.

# Revisión para preparación e intento de transferencia al clúster

> **Estado actualizado, 2026-10-14 UTC:** el ZIP original de runtime sí quedó subido y extraído en una etapa posterior, con SHA-256 `83dfe51de9b128b7f13780bd21bde08548aa02d76cf038603ebd99891269109d`; sus recibos siguen inmutables. Hay cuatro checkpoints convertidos. La copia `STATUS.md` recibió la corrección exacta de tamaños. La carga de un ZIP nuevo con el parche SimCLR fue rechazada por revisión automática y no se completó; después el navegador bloqueó la descarga del informe y no pudo reanudarse. Ver [51](51-cierre-preparacion-y-bloqueos-lanzamiento-v8.md). El intento fallido descrito a continuación es histórico.

2026-10-13, America/Bogota. Continúa `eb5ca591`.
Estado físico de pesos: no descargados. No se habilitan entrenamiento ni gates.

## Revisión documental completada para preparación académica

Se conserva el inventario de candidatos del documento 49 como referencia. El
nuevo [plan revisado](../plan-segmentacion/metodos-v8/public-weights.reviewed-research.plan.json)
registra quién/cuándo revisó, fuente oficial exacta, selección V8 explícita,
normalización y criterio de licencia para cada archivo. Su aprobación se limita
a descarga y preparación académica no comercial; no es una validación de bytes,
una autorización de redistribución ni una nueva licencia concedida por el proyecto.

Para MoCo v1/v2 se retienen las condiciones CC BY-NC 4.0 declaradas por el README
que publica los checkpoints. No se extiende el relicenciamiento MIT del código a
los pesos. Para MoCo v3 se registra la declaración CC BY-NC 4.0 consistente del
proyecto. Para DINO/SimCLR se usa la declaración Apache 2.0 que acompaña la
publicación oficial, dejando explícita la inferencia de alcance del proyecto.
Estas son decisiones de preparación documentadas; no se afirma haber recibido
una aclaración adicional de los autores ni comprobado un aviso dentro de los
archivos todavía ausentes.

Siguen pendientes la validación física de cada checkpoint, el alcance que se
requiera para cualquier publicación/redistribución posterior, y la semántica de
SimCLR señalada en el documento 49. La aprobación no convierte un prefijo MD5
en un digest completo ni cierra una conversión no ejecutada.

## Paquete de código exacto para el pod

Se preparó un paquete con el adaptador original y los 209 archivos de `src/`
de `8a2f42fe60aa5e29c0dd346cc2e645278897db4d`. Los 210 blobs se contrastaron con
GitHub y sus bytes coinciden. El [manifiesto de fuente](../adapter-source-manifest.json)
contiene sus SHA-256. El preparador ahora admite ese snapshot exportado sin `.git`
solo si el plan fija el SHA-256 exacto del manifiesto, la revisión coincide y
todos los archivos de fuente están presentes, sin archivos añadidos ni cambios.
No se fabrica un repositorio Git ni se modifica el checkout histórico.

Este camino permite llevar el código por la carga de archivos de Jupyter sin
entregar credenciales de GitHub al pod. El adaptador sigue teniendo SHA-256
`7bfa34bb153e22413409fa77c8e245a401688042af0ffa56d095714d41fcf77d`.
La identidad verificada del snapshot se registra en el recibo de conversión.

Los datos del paquete preparado y el alcance exacto de la verificación están en
[cluster-transfer-attempt.json](../resultados/preparacion_v8/staging_20261013/cluster-transfer-attempt.json).
El ZIP es un producto temporal de transferencia, no un checkpoint ni evidencia
de despliegue. Puede reconstruirse desde los archivos fijados y el plan revisado.

## Qué ocurrió en la sesión real de Jupyter

1. La nueva pestaña mostró el formulario de UNABIA. Se usó `browserAuth` con
   usuario/contraseña; no se recibieron valores de credenciales en el modelo.
2. El flujo devolvió `submitted` y después se observó JupyterLab autenticado.
3. La consulta acotada de la carpeta de preparación mostró el conversor externo
   SimCLR existente y ningún checkpoint en el alcance consultado. La carpeta
   conservaba `checkpoints/`, `source/` y `STATUS.md`.
4. Se intentó cargar el paquete nuevo en `public_ssl_v8_preparation` usando el
   selector de archivos. El navegador rechazó la operación por protección de
   datos retenidos tras entrega de credenciales y reinició su kernel de control.
5. La recuperación del navegador devolvió `native credential state cannot be
   safely resumed. Start a new browser runtime to continue.` No se confirmó la
   transferencia ni se ingresaron más credenciales después de ese bloqueo.

Por tanto, no se afirma que el adaptador V8 esté desplegado, no se lanzó ninguna
descarga, no se calcularon hashes de pesos reales y no hubo conversiones de modelos.
El bloqueo actual es el navegador de control después de autenticar, no una prueba
de falta de recursos o de caída del clúster. El 403 de SimCLR no se reintentó.

## Comprobación nueva y siguiente acción concreta

Se comprobaron solo cuatro casos nuevos del verificador de snapshot con archivos
temporales: identidad válida, archivo modificado, archivo añadido y manifiesto
sustituido. Los cuatro pasaron; las suites anteriores no se repitieron. No se
importaron encoders ni se ejecutaron entrenamientos en esas comprobaciones.

Cuando un runtime de navegador permita continuar, la siguiente acción es subir
y contrastar el paquete, extraerlo en una carpeta nueva y ejecutar el preparador
con el plan revisado sobre los cuatro candidatos MoCo/DINO. SimCLR conserva su
bloqueo de acceso y su revisión semántica; no se sustituye por otra familia. Las
conversiones se ejecutan únicamente sobre archivos íntegros y quedan separadas
de cualquier habilitación científica.

## Actualización posterior: transferencia y prueba sintética en UNABIA (2026-10-14)

Lo anterior conserva el relato de la sesión del 2026-10-13 y no describe el estado
actual. En una sesión posterior se transfirió el checkpoint SimCLR PyTorch fijado
de Hugging Face a
`/home/jovyan/work/public_ssl_v8_preparation/staging-03/simclr_r50_1x/resnet50-1x.pth`;
los 102.542.786 bytes y SHA-256
`fdd96bbcc92008232f31e0f7de46d30810e7115a05005d7cff926e78e6fa37ea` coinciden con
el objeto publicado. Se instalaron cinco módulos de código en
`runtime_simclr_same_eed86c8/src/dermssl/` y sus hashes coinciden con la revisión
`eed86c892ef1c0516e21a024a66f0b8788aedd76`.

Después, en el clúster, el banco USL-Net revisado pasó un forward sintético con
`torch.randn` float32 de forma `[1,3,224,224]`, semilla 42, CPU, modo `eval` y
`inference_mode`. La salida fue finita `[1,9216,14,14]`; se leyeron cero imágenes
reales y no comenzó entrenamiento. El recibo se conserva en el clúster, no como
archivo crudo versionado en Git. El mismo recibo está enlazado como artefacto
`usl-net-synthetic-forward-20261014` (tipo `validation-receipt`) en el run nuevo de W&B
[`g8us8pz9`](https://wandb.ai/nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion/runs/g8us8pz9),
con SHA-256 `b392e7d32eba94fe663777a4af754f4f554df50c715baaa0807ae535b7f6a83a`.

Este avance no prueba equivalencia numérica con el checkpoint TensorFlow original
de Google ni habilita entrenamiento. Los gates de datos/evaluación, cuotas y
presupuestos completos, planes seriales, revisión de SLED y configuración
CUDA/launcher permanecen pendientes. Ver el estado consolidado en [54](54-usl-net-banco-pesos-v8.md).

