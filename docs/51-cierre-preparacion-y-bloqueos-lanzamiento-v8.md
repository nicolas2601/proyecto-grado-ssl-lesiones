> Estado más reciente: [descarga Hugging Face y validación local](52-simclr-huggingface-validacion-local-v8.md). Las afirmaciones siguientes sobre SimCLR pendiente describen el estado previo a esa validación.

# 51. Revisión de preparación y requisitos de lanzamiento V8

2026-10-14 UTC / 2026-10-13 America/Bogota. La revisión reutiliza las auditorías,
pruebas y conversiones guardadas. No inicia entrenamiento ni crea una corrida W&B.
Este documento actualiza el estado histórico de los documentos 46, 49 y 50.

## Resultado

La comparación completa de los cuatro métodos todavía no está lista para
lanzamiento. Hay cuatro checkpoints públicos descargados y convertidos
estrictamente, y un banco candidato RPI comprobado con el cargador existente.
SimCLR, el cierre de datos/evaluación y el plan de recursos siguen pendientes.
Los configs conservan `execution_enabled=false`; no se fabricaron gates.

## Estado físico confirmado

- JupyterLab mostró una sesión autenticada después del formulario seguro.
- Se corrigió la frase de tamaños en `~/work/public_ssl_v8_preparation/STATUS.md`;
  se conserva `STATUS.before-size-note-20261014.md` como copia anterior.
- Los recibos de `staging-01` y `staging-02` se leyeron sin volver a descargar
  archivos, convertir pesos ni repetir hashes de los archivos fuente completos.
- `repo_four_methods_v8_d55c5cd_20261013` continúa en `d55c5cd` y limpio.
  `repo_four_methods_v8_cuda_fix_20261013` también conserva ese HEAD y sus dos
  cambios locales (`rpi_net/model.py` y `test_rpi_net.py`); se registraron sus
  hashes sin editar esos archivos.
- La consulta de lectura W&B confirmó `paulasaavedra1101`, acceso al proyecto
  compartido `dermssl-segmentacion` y cero corridas `running`. El archivo privado
  de credenciales permaneció igual. No se creó una corrida ni una alerta.

La evidencia agregada está en `cluster-readiness.observations.json` dentro de la
carpeta de preparación. No contiene claves ni registros por imagen. Su descarga
al entorno local quedó bloqueada por protección de credenciales; no se declara
que esa transferencia se completó. La recuperación posterior devolvió
`native credential state cannot be safely resumed` y se detuvo.

## Pesos y banco RPI

| Artefacto | Bytes fuente | Estado actual |
|---|---:|---|
| MoCo v1 R50, 200 épocas | 95.347.404 | Descarga y conversión estricta en `staging-01`; MD5 coincide con el prefijo publicado. |
| MoCo v2 R50, 800 épocas | 112.133.139 | Descarga y conversión estricta en `staging-02`; MD5 coincide con el prefijo y el ETag recibido. |
| MoCo v3 R50, 1000 épocas | 272.786.643 | Descarga y conversión estricta en `staging-02`; SHA-256 local registrado; ETag multipart no tratado como digest. |
| DINO original ViT-S/16 | 86.710.517 | Descarga y conversión estricta en `staging-01`; SHA-256 local registrado. |
| SimCLRv1 R50 1x | Desconocidos | Sin checkpoint descargado ni conversión real. |

Los valores anteriores de MoCo v2/v3 eran `1121331339` y `2727866643`.
Cada uno contenía un dígito repetido adicional. Eran aproximadamente 10×, pero
no exactamente: difieren de 10× por −51 y +213 bytes. El error estaba en los
metadatos del plan. Los tamaños corregidos coinciden con los recibos de descarga.

Se creó `source-bank.rpi-net.reviewed-candidate.json` en el clúster:
P1/P2 → MoCo v3, P3 → MoCo v2, P4 → DINO original. El cargador del runtime
inmutable existente comprobó las tres procedencias, hashes de encoders, cargas
estrictas y normalización ImageNet. Fue una comprobación nueva de integración:
cero imágenes leídas, cero forward de encoders, cero entrenamientos y ninguna
reconversión. El banco queda como candidato documentado V8; no demuestra que
los papers usaran esos archivos/épocas exactos ni certifica equivalencia numérica.
No se conectó el banco a un config habilitado.

Los recibos históricos corresponden al adaptador 2 y al runtime anterior.
Los cambios del adaptador 3 tienen su propia revisión y manifiesto; no se
reescriben los recibos ni se atribuyen conversiones antiguas al código nuevo.

## SimCLR: acceso y semántica son dos pendientes

### Acceso al objeto publicado

Un GET acotado, distinto de las consultas HEAD anteriores, devolvió HTTP 403
con XML `Code=AccessDenied`, `Message=Access denied.` desde Google Storage.
Se leyeron hasta 2048 bytes del error; no se guardó un checkpoint. Se registró
`simclr-access-diagnostic-20261014.json`.

El README oficial fijado también publica `gs://simclr-checkpoints/simclrv1`.
Una única consulta de sus metadatos devolvió HTTP 401; se registró en
`simclr-additional-source-metadata-20261014.json`. No se seleccionó otro objeto
ni se intentó eludir autenticación. Estas respuestas no permiten determinar
si el archivo original fue retirado, cambió de permisos o es accesible desde
otro cliente autorizado. No hay evidencia de un fallo de cálculo del clúster.

Para cerrar este punto hace falta acceso legítimo al mismo artefacto, una fuente
oficial restaurada, o un espejo con identidad y licencia revisadas. No sustituir
por SimCLRv2, DINOv2, pesos de V7 ni un preentrenamiento local presentado como
el checkpoint original.

### Perfil espacial preparado

El `resnet.py` oficial usa kernel 3/stride 2/SAME en el maxpool. En una entrada
de stem 112×112, el padding es 0 antes y 1 después; padding simétrico 1 cambia
las posiciones aunque conserve la forma 56×56. Las convoluciones usan padding
fijo explícito, el stride está en la convolución 3×3 del bottleneck y la epsilon
de BatchNorm es 1e−5; esas tres decisiones coinciden estáticamente con el código
local. `last_stride=1` sigue siendo una adaptación densa V8 frente al stride 2
original y no se presenta como reproducción literal.

El adaptador 3 prepara un perfil `tensorflow_same` exclusivo para SimCLR.
MoCo y los segmentadores mantienen el perfil simétrico por defecto. El perfil
se registra en la procedencia y el banco rechaza un SimCLR legacy sin ese dato;
el conversor TensorFlow rechaza el modelo con perfil incorrecto antes de importar
TensorFlow. Se conservan nombres y formas de parámetros.

Hay cinco pruebas nuevas para ventanas pares, tamaños impares/rectangulares,
bordes negativos, esquema de parámetros/perfiles y rechazos antes de leer pesos.
Solo se comprobó sintaxis local. PyTorch no está instalado en este entorno local.
La revisión automática rechazó subir el ZIP de código al clúster por requerir
autorización explícita. No se desplegó el parche ni se ejecutaron esas pruebas.
La equivalencia con TensorFlow y los tensores reales sigue sin verificar.
El lector TensorFlow no estuvo disponible en el inventario anterior; no se
instaló ni confirmó en esta revisión. Se requiere un lector en un entorno
aislado y versionado antes de leer los shards. Las cinco pruebas nuevas no
requieren TensorFlow.

## Puertas antes de una corrida

| Requisito | Evidencia disponible | Falta cerrar |
|---|---|---|
| Integridad de originales | Informe previo: 40.896 verificados, cero fallos. Se reutilizó su `summary.json`. | No repetir esta auditoría. |
| Elegibilidad y separación | Exclusiones históricas preservadas. El informe guardado marca `original_grouping_review=pending`. | Procedencia de grupos paciente/lesión, separación de evaluación y manifiesto elegible completo con prueba válida. |
| Evaluación anatómica | Sigue declarada pendiente en el protocolo. | Reserva independiente y rutas/manifiesto de máscaras expertas exclusivamente para evaluación. |
| Fuentes | Cuatro archivos y banco candidato RPI cargados. | SimCLR real y validación del perfil nuevo; completar el banco USL y la revisión de sus procedencias. |
| Recursos | 48 CPU; `cpu.max=max 100000`, `memory.max=max`; dos RTX PRO 4000 de 24.467 MiB, con 2 MiB usados y 0% utilización en la observación. | Cuota individual de disco, límites operativos y timeout/RSS/VRAM por etapa. Una lectura del sistema compartido no acredita cuota. |
| Salidas | Estimación ya documentada a 224: 46,49 GiB por fuente; 103,82 GiB para generación RPI de cuatro fuentes. | El límite base de 4 GiB es insuficiente. Registrar presupuesto por etapa y acumulado de todos los ciclos conservados. |
| Receta serial | USL 5×300 épocas; RPI 50+30; TriDerm 150 y warmup 10; SLED maestro de TriDerm. | Planes concretos del controlador con configs, manifiestos, gates y salidas nuevas por paso. No inventar hashes ni true en gates. |
| TriDerm | Arquitectura desde cero, maestro SLED fijado. | Pseudo máscaras SLED revisadas y SHA-256 de su manifiesto; no entrenar contra un maestro sin productos verificados. |
| Seguimiento | Cuenta de Paula autenticada y proyecto accesible, cero runs creados. | Comprobar seguimiento de la etapa autorizada cuando se lance; entrega de notificaciones aún no confirmada. |

El loader exige elegibilidad verificada, exclusión de `target_test`, grupos
paciente/lesión y deduplicación preservados. Para `fit`, el gate exige
`data_verified`, `independent_anatomical_evaluation_resolved`,
`resource_budget_measured`, `weights_provenance_verified`, `smoke_passed`,
SHA del manifiesto y revisión. El informe de integridad anterior no completa
automáticamente esos campos. No se crearon archivos gate con aprobaciones falsas.

## Siguiente acción concreta

Autorizar la carga del ZIP del perfil SimCLR en una copia nueva del runtime dentro
de la zona de preparación y ejecutar únicamente las cinco pruebas nuevas. La copia
inmutable y los checkouts `d55c5cd` quedan fuera de ese despliegue. También hace
falta recuperar acceso funcional a Jupyter. Esta autorización no habilita
entrenamiento: después siguen las puertas de acceso a pesos, datos/evaluación,
recursos, manifiestos y planes seriales.

Evidencia pública agregada: [readiness.review.json](../resultados/preparacion_v8/readiness_20261014/readiness.review.json).
Fuente SimCLR fijada: [resnet.py](https://github.com/google-research/simclr/blob/383d4143fd8cf7879ae10f1046a9baeb753ff438/resnet.py)
y [README](https://github.com/google-research/simclr/blob/383d4143fd8cf7879ae10f1046a9baeb753ff438/README.md).
