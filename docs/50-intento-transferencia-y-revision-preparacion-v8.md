# Revisión para preparación e intento de transferencia al clúster

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
