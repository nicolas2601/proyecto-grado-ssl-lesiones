# Preparación reanudable de pesos públicos V8

2026-10-13, America/Bogota. Base: `8a2f42fe60aa5e29c0dd346cc2e645278897db4d`.
Implementación para preparación; no habilita entrenamiento ni declara pesos verificados.

## Qué se implementó

- `scripts/stage_public_ssl_weights_v8.py`: preparación con biblioteca estándar,
  sin importar PyTorch/TensorFlow. Sin opciones de ejecución imprime el inventario,
  sin conexiones de red ni escrituras. `--download` y `--convert` son acciones separadas.
- `plan-segmentacion/metodos-v8/public-weights.plan.json`: cinco candidatos exactos
  ya identificados en el documento 46. MoCo v2 se comparte entre USL y RPI; no se
  duplica la descarga. Los tamaños proceden de las consultas HEAD anteriores,
  no de nuevas consultas. Todas las revisiones siguen `approved=false`.
- Pruebas nuevas de E/S en `tests/test_public_ssl_staging_v8.py`, aisladas de las
  pruebas anteriores de segmentación y sin modelos ni checkpoints reales.

El preparador no sustituye `prepare_public_ssl_weights_v8.py`: lo invoca como
subproceso CPU solo con `--convert`. Se conserva el adaptador original y su SHA-256
`7bfa34bb153e22413409fa77c8e245a401688042af0ffa56d095714d41fcf77d`.
La conversión exige un checkout Git independiente cuyo adaptador y `src/` coincidan
con la base fijada, sin archivos de código locales sin seguimiento. Registra la
revisión real del checkout; un commit posterior que solo agregue este preparador o
documentación es compatible. No se importa ni ejecuta el conversor externo de SimCLR.

### Snapshot exportado sin Git

El preparador también acepta un snapshot si `adapter_source_manifest_sha256` fija
el hash de `adapter-source-manifest.json` en su raíz. Comprueba revisión y todos
los archivos importables, rechazando cambios, ausencias y añadidos. No fabrica
historia Git ni omite el control de identidad. Ver [50](50-intento-transferencia-y-revision-preparacion-v8.md)
para el plan académico revisado y el intento de transferencia real.

## Revisión por artefacto

Copiar el plan una sola vez a un archivo de revisión fuera del checkout. Completar
por candidato los campos `review`: quién/cuándo revisó, identificador y URL de
licencia, explicación de su alcance sobre el archivo de pesos exacto, evidencia
de selección del inicializador, normalización de entrada y su referencia.
Solo después se establece `approved=true`. La herramienta comprueba la presencia
de la revisión; no interpreta jurídicamente una licencia ni certifica su alcance.
No basta con copiar la licencia del código del repositorio a la de sus pesos.

`expected_sha256` es opcional y permanece `null` en el inventario. Si existe un
digest independiente, registrarlo antes de descargar. Sin él, y si tampoco hay prefijo MD5 publicado, la herramienta
registra `integrity=local_sha256_only`: identifica los bytes recibidos y permite
detectar cambios locales posteriores, pero no demuestra autenticidad por una
comparación independiente. Con él se exige coincidencia antes de promover el archivo. Los candidatos MoCo
v1/v2 ahora incluyen `published_md5_prefix`: se compara únicamente ese prefijo
publicado y se registra `matches_published_md5_prefix`, además del SHA-256 local.
No equivale a un MD5 completo independiente ni a autenticidad criptográfica. Ver
[49](49-revision-fuentes-pesos-reales-v8.md) y el inventario de evidencia fijada.

Las revisiones pueden completarse una por una. Cambiar un artefacto que ya tiene
archivos exige otra carpeta de preparación; no se reaprovechan silenciosamente
parciales obtenidos bajo otra especificación. Un bloqueo por revisión pendiente
que no creó archivos sí permite completar la revisión en el mismo workspace.

## Uso desde un checkout de preparación nuevo

El plan revisado y la carpeta `staging-01` se ubican bajo la zona de preparación
existente. Las rutas siguientes presuponen que el archivo revisado ya se creó;
no son instrucciones para escribir sobre el checkout `d55c5cd`.

Solo mostrar los candidatos, sin descargar:

```bash
python scripts/stage_public_ssl_weights_v8.py \
  --plan ~/work/public_ssl_v8_preparation/reviewed-weights.plan.json \
  --workspace ~/work/public_ssl_v8_preparation/staging-01
```

Descargar únicamente el candidato revisado elegido (ejemplo MoCo v1):

```bash
python scripts/stage_public_ssl_weights_v8.py \
  --plan ~/work/public_ssl_v8_preparation/reviewed-weights.plan.json \
  --workspace ~/work/public_ssl_v8_preparation/staging-01 \
  --artifact mocov1_r50_200ep --download
```

Repetir exactamente esa orden reanuda el parcial o reconoce el archivo completo.
`--artifact` puede repetirse para seleccionar varios candidatos; omitirlo selecciona
todos. Un bloqueo individual no impide procesar los demás; la salida global es 2
si algún archivo quedó bloqueado. No se reintenta automáticamente un HTTP 403.

Convertir únicamente un archivo ya descargado, sin conexiones de red:

```bash
python scripts/stage_public_ssl_weights_v8.py \
  --plan ~/work/public_ssl_v8_preparation/reviewed-weights.plan.json \
  --workspace ~/work/public_ssl_v8_preparation/staging-01 \
  --artifact mocov1_r50_200ep --convert
```

Se pueden combinar las dos opciones explícitas. La conversión necesita las
dependencias del adaptador en ese entorno independiente. No instala paquetes.
SimCLR TensorFlow necesita su lector TensorFlow; no es necesario para descargar
o convertir los otros formatos. El ZIP de SimCLR conserva el original y extrae
solo el índice y todos los shards del prefijo declarado, con límite de tamaño;
no ejecuta ni extrae scripts del archivo.

## Reanudación, presupuesto y evidencias

Cada candidato tiene `transfer.json`, `source.*.part` durante transferencia y
`source.*` tras completarla. `weights-ledger.json` resume el estado por candidato.
Un lock `flock` impide dos preparadores sobre la misma carpeta y se libera al
terminar el proceso, incluso si se interrumpe. No se borran locks de otros procesos.

La reanudación exige el SHA-256 del parcial y tamaño guardados, un ETag fuerte,
`If-Range`, respuesta 206 y `Content-Range` exacto. Un servidor que ignora el rango,
cambia el objeto o no permite esa identidad deja el parcial conservado y bloqueado.
Cada bloque se sincroniza antes de registrar progreso. Si una terminación abrupta
ocurre entre la escritura y el registro y ambos difieren, se conserva para revisión;
no se trunca ni se mezcla con otra respuesta. Si el archivo terminó y solo faltó
renombrarlo, se recupera sin descargarlo de nuevo.

Se exige Content-Length y presupuesto por archivo. La reserva libre por defecto
es 1 GiB; es un mínimo de preparación, no una cuota ni reserva efectiva del clúster.
La conversión comprueba espacio para extracción y 256 MiB de encoder; tiene un
límite de 600 s, se ejecuta en CPU y conserva fallos/logs en carpetas de intento.
El timeout de socket y el tiempo máximo por transferencia se pueden fijar con
`--socket-timeout` y `--max-transfer-seconds`. La descarga no tiene bucles de reintento.

Solo se aceptan URLs HTTPS públicas sin credenciales de los hosts previstos.
Se rechazan redirecciones a login, documentos HTML/JSON/XML, URLs firmadas y
mirrors no revisados; no se usan cookies ni se ofrece introducir credenciales.

El checkpoint convertido y su procedencia se publican juntos mediante renombrado
de directorio, después de una salida correcta del adaptador y comprobación de los
hashes. `receipt.json` enlaza fuente, código, encoder, procedencia y shards de TF.
Una conversión completa se reutiliza únicamente con identidad y hashes iguales;
los intentos fallidos quedan preservados. Los estados son `downloaded`,
`converted_strict` o `blocked`; una carga estricta comprueba estructura, no
equivalencia numérica con el encoder original ni calidad anatómica.

No se generan bancos `source_bank` ni gates científicos automáticamente. Después
de revisar las conversiones se enlazan sus pares `encoder.pt`/`encoder.pt.json`
en los bancos con exactamente los orígenes prescritos. Las configuraciones
mantienen `execution_enabled=false`.

## Verificación de esta implementación y bloqueos restantes

El informe de las comprobaciones nuevas está en
[`verification.json`](../resultados/preparacion_v8/staging_20261013/verification.json).
Usan respuestas HTTP simuladas, bytes temporales y un adaptador simulado; no
validan una descarga real ni el mapeo de tensores oficiales. No se repitieron
auditorías, pruebas de los cuatro métodos ni entrenamientos anteriores.

Sigue pendiente recuperar acceso observable a Jupyter, desplegar este preparador
y el adaptador V8 con su código en una carpeta independiente, cerrar cada revisión
de licencia/normalización/selección y obtener los pesos. El 403 de SimCLR no se
ha resuelto con este cambio. La presencia del conversor externo en la carpeta
`source/` no acredita la del adaptador V8. No se accedió al clúster durante esta
implementación y no se afirma un estado nuevo de sus archivos.

También siguen pendientes los manifiestos/particiones y evaluación independiente,
los planes seriales concretos de USL/RPI, el manifiesto SLED para TriDerm y los
presupuestos efectivos. Este preparador no cierra esos requisitos del documento 46.
