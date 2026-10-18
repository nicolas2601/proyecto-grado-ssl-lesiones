# Corrida CCAM V8 `usl_ccam-20261017-003`

## Identidad y estado observado

Familia: segmentación V8; método USL-Net; etapa CCAM. Autorización directa:
«si lanza la corrida», para la propuesta de 30 épocas y comienzo nuevo con semilla
42. No autoriza ejecutar automáticamente pseudomáscaras, estudiante ni otros métodos.

Fuente local: [runtime.initial.snapshot.json](runtime.initial.snapshot.json),
resumen observado por el operador desde UNABIA a las
`2026-10-17T23:17:49.345015+00:00`. El supervisor arrancó a las
`2026-10-17T23:16:02.889029+00:00` (18:16:02, UTC−05:00).
Estado inicial registrado: `runner_process_started_training_unconfirmed`,
`training_started=null`, sin `exit.json` y con cero épocas completas.
La presencia del proceso y uso de GPU no acredita aún actualizaciones del
optimizador ni una corrida completada.

Observación posterior del operador a las `2026-10-17T23:20:53.276878+00:00`,
conservada en [runtime.confirmed.snapshot.json](runtime.confirmed.snapshot.json):
W&B mostraba `Running`, los tres procesos seguían presentes, `metrics` existía,
pero seguían cero épocas completas, `resume_checkpoint_exists=false` y
`exit=null`. El estado continuaba `runner_process_started_training_unconfirmed`,
con `training_started=null`. «confirmed» en el nombre del snapshot identifica
esta observación posterior, no una confirmación de época ni finalización.

Rama de trabajo: `experiment/modular-audit-v8`; HEAD local al preparar el
lanzamiento: `f3d3e94`, que contiene la corrección RNG de `4a6832c`.
El checkout UNABIA conserva HEAD Git `95a8f60` más despliegues ZIP; ese HEAD
no identifica por sí solo los archivos ejecutados. El inventario de lanzamiento
vincula 57 archivos efectivos. No se debe editar esa copia durante la corrida.

## Configuración y entradas

Raíz de lanzamiento: `/home/jovyan/work/validation_v8/usl-ccam-launch-20261017-003/`.

| Archivo dentro de esa raíz | SHA-256 |
|---|---|
| `ccam.run.approved.json` | `30fd4ad9e870f86009e1b591fab05cb84226f66621297912d8349c43bed7756c` |
| `ccam.gate.json` | `d98c73c022d7809ae0b3d4fe94b16aea0ef3f92d6bdf1ef9b96b7ac953189bd0` |
| `launch.json` | `d5f8f9718fdb8e62d191a2cf3662f56444e73f0f9e81e21c04e04f832aeeebb4` |

Configuración activa separada del borrador: `execution_enabled=true`, 30 épocas
fijas, batch 2, resolución 224, semilla 42, comienzo nuevo (`resume_requested=false`),
primera cabeza con Pearson >= 0,98 como candidato, sin parada temprana.
Ese umbral y el presupuesto no acreditan convergencia ni calidad anatómica.
Límites planificados: 24 h, RSS 65.536 MiB, salida/tracking/logs 1.024 MiB,
GPU 18.432 MiB con reserva 4.096 MiB; no son consumos medidos de esta corrida.

Manifiesto de aprendizaje:
`/home/jovyan/work/validation_v8/accepted-manifests-20261016-002/learning.manifest.json`,
SHA-256 `7749471410aa98a76dd978cca2ef58f9642d14404530c98e534484ba40a7a6ff`,
34.784 registros. Se mantienen las exclusiones y limitaciones clínicas aceptadas;
las máscaras expertas reservadas no participan en el ajuste de esta etapa.
Procedencia de pesos: banco de tres encoders revisado, vinculado por el inventario
del lanzamiento; los hashes individuales no están transcritos en el snapshot inicial.

El comando exacto sin secretos está en `launch_summary.command` del snapshot:
`scripts/run_segmentation_v8.py --method usl_net --stage ccam`, con config,
manifiesto, gate y salida anteriores; no lleva `--resume`.

## Limpieza previa, proceso y tracking

Antes de enviar una única ejecución se archivaron y verificaron 27 archivos
diagnósticos (23 archivos de cuatro carpetas de casos y cuatro fixtures).
Recibo: `/home/jovyan/work/validation_v8/usl-ccam-launch-preparation-4a6832c-20261017-001/review/approved-submission/cleanup.receipt.json`,
SHA-256 `0ca336b90aecf3fd999c9b03f7a83b5cc95d961c55650a78cdac091a470922ab`.
Archivo preservado: `/home/jovyan/work/validation_v8/usl-ccam-recovery-verification-6c5627d-20261017-001/archived-before-run003/reviewed-diagnostics.tar.gz`.
SHA-256 del archivo: `edefc6d2b67361a2adedf8113cc02457eed42cc3a3d6e61f69a73e2306e99772`.
Se conservaron ambos recibos diagnósticos y el checkpoint de la corrida real 002.
`cleanup_complete`, `evidence_preserved` y `wandb_cleanup_complete` son `true`
únicamente dentro del alcance registrado de esos cuatro directorios y cuatro fixtures.
La limpieza W&B de este alcance comprende cero runs diagnósticos nuevos; no se
afirma limpieza global de la cuenta ni se reabre la limpieza histórica ya cerrada.

Salida: `/home/jovyan/work/proyectonp/runs_v8/usl_ccam-20261017-003`.
Tracking local: la misma ruta con sufijo `-tracking`.
Supervisor PID 264263; runner PID 264264; hijo de cómputo CUDA observado PID
264315. El snapshot inicial muestra 792 MiB del proceso GPU y utilización GPU
39% inicialmente y 34% en la observación posterior (802/24.467 MiB del dispositivo);
son observaciones puntuales, no picos ni cuota individual.

W&B: [`2761bbcd828a`](https://wandb.ai/nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion/runs/2761bbcd828a),
nombre `usl_ccam-20261017-003`. El ID consta en el log real del runner y el
operador verificó la página autenticada. GPU: NVIDIA RTX PRO 4000 Blackwell;
Python/PyTorch del proceso real: no registrados en este snapshot inicial.

## Pendientes

- Incorporar evidencia posterior de la primera época o un error real, conservando
  la distinción entre envío, proceso y entrenamiento efectivo.
- Al terminar, registrar salida del supervisor, historial, checkpoints finales/
  seleccionados y consumos reales. No hay métricas científicas finales disponibles.
- Revisar curvas y candidato antes de promover pseudomáscaras. Preparar los otros
  métodos en una copia separada, sin alterar la fuente activa; el plan serial del
  estudiante y refinamientos sigue pendiente y no arranca automáticamente.

## Cierre posterior observado

El estado inicial anterior se conserva como observación histórica. La revisión
posterior registra 30 épocas completas y código de salida 0, con candidato de la
época 6 según la regla predeclarada. El historial, límites y fuentes del cierre
están en el [registro de finalización](../usl_ccam_completion_20261017_003/run_record.md).
Completar CCAM no acredita calidad anatómica ni habilita automáticamente las
siguientes etapas. La inspección posterior de ocho originales de aprendizaje
registró abstención en 8/8 tanto para el candidato como para el final; el cierre
documenta el bloqueo de promoción y conserva su alcance acotado.
