# Registro de lanzamiento inicial CCAM de USL-Net V8

Estado observado a `2026-10-17T15:44:57.251626Z` (10:44:57, America/Bogota):
CCAM `002` en curso, con dos de cinco épocas completadas según el JSONL real,
34.784 muestras en cada una. No existe checkpoint ni recibo de salida final en
esa observación. W&B estaba `running` a `15:43:51.526199Z`. El intento `001`
falló antes de ejecutar el método. La [copia local de progreso](epoch-progress.snapshot.json)
registra las épocas y sus fuentes; no se interpreta la loss como eficacia de
segmentación ni se afirma finalización. La fuente de la observación anterior es
[state.snapshot.json](state.snapshot.json), copia parseada de evidencia original
del clúster; sus bytes no sustituyen los hashes originales declarados.

## Identidad y alcance

- Familia: segmentación V8 de cuatro métodos; método USL-Net, etapa inicial CCAM.
- Rama: `experiment/modular-audit-v8`. Checkout de ejecución:
  `/home/jovyan/work/proyectonp/repo_v8_canonical`.
- HEAD del checkout: `95a8f602b2fb49304dbbb9a78a18edc43cc3ebba`, con deltas
  transferidos por ZIP. La identidad efectiva del código está fijada por
  `source_hashes` y `launcher_sha256` del recibo. Estado completo del árbol:
  no registrado; no se afirma checkout limpio ni pertenencia de esos deltas al HEAD.
- Autorización: el usuario autorizó avanzar al lanzamiento inicial CCAM y preparar
  los demás métodos mientras se ejecuta. Esa autorización no demuestra ejecución.
- Semilla: 42. Resolución: 224. Batch: 2. Presupuesto: cinco épocas, 86.400 s,
  65.536 MiB RSS, 18.432 MiB GPU más 4.096 MiB de reserva y 1.024 MiB de salida.
  Son límites operativos, no mediciones de la corrida completa.

## Configuración, datos y pesos

Config de ejecución:
`/home/jovyan/work/proyectonp/repo_v8_canonical/plan-segmentacion/metodos-v8/launch-preparation/configs/usl_ccam.run-20261017.json`.
SHA-256 del config en el recibo y en la confirmación de ejecución:
`0547355805167a54d918d06fb490599bd9b5a5d24a41876e612c5e616681def6`.
Gate:
`/home/jovyan/work/validation_v8/usl-ccam-launch-20261017-002/ccam.gate.json`,
SHA-256 `03a86e16611f0d38ded16538784527ac9207b85994cdd68c2f01edd21ccde7df`.
Lanzador SHA-256:
`22b7cc957ad9bb18d38d34edb41acfd944d610603d848cad881f0fc30c611927`.
El comando exacto sin secretos consta en `launch_preparation_record.command` de
la copia local: Python del venv, `scripts/run_segmentation_v8.py --method usl_net
--stage ccam`, con el config, manifiesto, gate y salida absoluta registrados aquí,
y `--wandb-project dermssl-segmentacion`. El recibo fija además los hashes de los
doce módulos ejecutados; no se los sustituye por el commit histórico del checkout.

Se utiliza exclusivamente el manifiesto de aprendizaje congelado
`/home/jovyan/work/validation_v8/accepted-manifests-20261016-002/learning.manifest.json`,
34.784 registros, SHA-256
`7749471410aa98a76dd978cca2ef58f9642d14404530c98e534484ba40a7a6ff`.
ISIC2018 Task1 conserva 2.594 imágenes y máscaras para evaluación; las máscaras
expertas no son entradas de aprendizaje. No existe evaluación nueva en esta etapa.

Banco de pesos:
`/home/jovyan/work/public_ssl_v8_preparation/source-bank.usl-net.reviewed-candidate.json`,
SHA-256 `306136b3abd29ae32ac8be2a1456cf411c3f1422f93b39f05d46636141cce6f6`.
Reúne SimCLR de espejo Hugging Face, MoCo v1 y MoCo v2. No se acredita equivalencia
numérica del espejo con el TensorFlow oficial de SimCLR. Las limitaciones aceptadas
de independencia paciente/lesión e identidad de duplicados permanecen declaradas.
`readiness.review.json` y el recibo histórico de congelación permanecen intactos.

## Limpieza previa y excepción residual

La limpieza operativa, revisada en la copia local del recibo, quedó en
`/home/jovyan/work/validation_v8/pre-real-run-cleanup-20261017-002/cleanup.receipt.json`,
SHA-256 `c1542226a2e724e0e0deb0ecc42e66300f93374d54c71727578e2b4cadc14290`.
Se preservó el archivo de procedencia del intento `001`, SHA-256
`24b7ffd2feabea2e6d04efcf3102defd461c4aa266ae7a63a1f73f5f79628306`.
Se retiraron el recibo sintético local, el dry-run del controlador, la corrida W&B
sintética y su artefacto personalizado. El recibo `001` parcial no se reescribió.

No se afirma limpieza remota absoluta: `all_remote_objects_removed=false`.
W&B rechazó la eliminación directa de un objeto de historial de servicio
`wandb-history`, de 1.054 bytes, ID `QXJ0aWZhY3Q6NDAxNTQ2MzQwMg==`.
Su digest observado cambió de `d1ca41f4e102b11d2f77a1d1c695bc54`
a `9cff0b03bdb1db0b19bc6d0961d3fbad`; su equivalencia de contenido no está
verificada. Ese residuo no es una entrada real de entrenamiento y se conserva
como excepción explícita del servicio, sin nuevas búsquedas de causa.

## Intento `001`: fallo previo al método

Fuente revisada en la copia local del recibo:
`/home/jovyan/work/validation_v8/usl-ccam-launch-20261017-001/`.
Estado final: fallido, salida 1, a `2026-10-17T15:27:18.589037Z`, duración
25,0153 s. PID supervisor: 251854; PID runner: 251855. Cero épocas registradas,
directorio de resultados ausente y GPU observada sin ejecución del método.

El runner emitió `RuntimeError: Tracking unavailable; no method execution started`.
El ID W&B previsto era `3c8f822096eb`; no está confirmada la creación de esa corrida.
No se interpreta este fallo como un resultado, un checkpoint o un entrenamiento parcial.

La causa operativa identificada fue la herencia de `WANDB_SERVICE`: el preflight
W&B 0.30 del proceso padre dejaba una referencia IPC cuyo servicio ya no existía
cuando arrancaba el runner separado. La autenticación online sí funcionó.
La corrección operativa elimina exclusivamente `WANDB_SERVICE` al iniciar supervisor
y runner, conservando autenticación y proyecto; no altera el método ni los datos.
Se prepararon rutas nuevas `002`, sin reutilizar las del intento fallido.

Antes de `001`, una preparación falló por un import ausente antes de crear su
directorio de recibos. Se restauró `wandb_logging.py` sin cambios mediante ZIP,
SHA-256 `63110620b0d78b2b33e65846c8cb8b839ef85dd036875d9399ffa0bff6627c73`.
No se repitió el smoke CUDA por estos errores operativos.

## Entorno y observación histórica anterior al cierre de épocas

La evidencia previa local de preparación registra Python 3.13.15, PyTorch
`2.11.0+cu128`, CUDA 12.8 y RTX PRO 4000 Blackwell. No se presentan como una
medición nueva del intento `002`. La cuota individual de disco/CPU sigue sin
verificar; la capacidad agregada es una referencia de planificación.

El recibo inmutable de preparación `002` se creó a
`2026-10-17T15:30:28.874295Z`; su `training_started=false` describe ese instante
previo al lanzamiento. No se reescribe para representar una observación posterior.
El estado del supervisor registra arranque del runner a
`2026-10-17T15:30:28.968518Z` y `training_started=null`; tampoco es un contador
de progreso del entrenamiento. La confirmación posterior aporta evidencia distinta:

| Observación `2026-10-17T15:33:04.679888Z` | Resultado |
|---|---|
| Supervisor / runner / worker | PID 252264 / 252265 / 252317 |
| Worker | `State: R (running)`, descendiente del runner, RSS 1.888.920 kB |
| GPU 0 | 802 MiB totales observados, utilización 39%; worker 252317, 792 MiB |
| W&B | `db7da6c1e784`, estado `running`, también observado como `Running` en navegador |
| Salida | Existe; no existe `exit.json` |
| Épocas / checkpoint / evaluación | Ninguno acreditado en esta observación |

RSS y memoria GPU son observaciones puntuales, no picos de toda la corrida.
La hora exacta del primer paso del optimizador no está registrada en esta evidencia.
No se publican métricas científicas ni se interpreta la actividad GPU como una
época completada. La URL de tracking es
[USL CCAM 002](https://wandb.ai/nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion/runs/db7da6c1e784).

Artefactos originales de `002`, bajo
`/home/jovyan/work/validation_v8/usl-ccam-launch-20261017-002/`:

- `launch.json`, SHA-256
  `70c8e16eac23e2d1607d71bf75f68b8eb50ef2f4995c66b0ebca3d4091193179`.
- `runtime-confirmation.json`, fuente de la observación viva anterior; su hash
  no está registrado en la copia local, por lo que no se inventa.
- `runner.log` y `supervisor.log`: rutas fijadas por el lanzador; sus contenidos
  completos no fueron revisados por este documentador.

Resultados:
`/home/jovyan/work/proyectonp/runs_v8/usl_ccam-20261017-002`.
Tracking local:
`/home/jovyan/work/proyectonp/runs_v8/usl_ccam-20261017-002-tracking`.
Los bytes originales siguen en UNABIA. Este registro revisa únicamente su copia
JSON local parseada; no opera el clúster ni acredita finalización.

## Progreso efectivo posterior (2026-10-17)

Se revisó [epoch-progress.snapshot.json](epoch-progress.snapshot.json), copia
parseada del recibo
`/home/jovyan/work/validation_v8/usl-ccam-launch-20261017-002/epoch-progress.json`.
A `2026-10-17T15:44:57.251626Z`, el archivo real
`/home/jovyan/work/proyectonp/runs_v8/usl_ccam-20261017-002/metrics.jsonl`
contiene dos épocas completas, cada una con `sample_count=34784`, de cinco
configuradas. El SHA-256 de los bytes JSONL capturados es
`7fb45206fe204e9b77e3ce1731c8cd812e4a1e5351e4598f2c2b43d78c35e67a`;
no es el hash de la serialización local ni fija las escrituras posteriores del log.
Checkpoint y `exit.json` siguen ausentes. La corrida permanece en curso; las
losses se conservan en el JSON fuente sin interpretarlas como calidad anatómica,
evaluación independiente o éxito final.

## Pendientes después del inicio

Al finalizar CCAM, conservar código de salida,
checkpoint, sidecar, hashes y linaje. Después corresponde generar pseudomáscaras
y su manifiesto completo y preparar el plan serial y los gates del estudiante.
No se afirma que USL-Net completo haya comenzado ni terminado.

La preparación paralela de los demás métodos tiene únicamente alcance de
metadata: recibo copiado localmente desde
`/home/jovyan/work/validation_v8/other-methods-preparation-20261017-001/metadata-verification.json`,
SHA-256 `e2261a526ee956bbb8069fd0549bc8f53dbda33dae0f772a33326324d2129819`.
Los cuatro configs permanecen deshabilitados. SLED requiere perfil CPU, gate y
cobertura/procedencia de pseudomáscaras; RPI requiere perfil GPU y gates por fase;
TriDerm depende de la salida SLED. No se ejecutaron nuevos modelos ni pruebas GPU
de esos métodos en esa preparación, y no hay lanzamiento automático de otros métodos.
