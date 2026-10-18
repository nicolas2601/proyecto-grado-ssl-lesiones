# Registro de corrida CCAM estabilizado 005

## Alcance

Se lanzó únicamente la etapa CCAM de USL-Net, variante opt-in `ccam_stabilized_v1`, en UNABIA. No se iniciaron generación de pseudo-máscaras, entrenamiento del estudiante, SLED, RPI-Net ni TriDerm. El config versionado sigue siendo un borrador deshabilitado; el lanzador creó un config aprobado separado y ligado por hash al gate y al manifiesto.

## Corrección y despliegue

El primer intento 004 se detuvo antes de crear el supervisor porque faltaba `protocol_id` en `evaluation_protocol`. Se conservó su recibo en `/home/jovyan/work/validation_v8/usl-ccam-stabilized-launch-20261018-004/` y el log de error en `/home/jovyan/work/validation_v8/usl-ccam-launch-20261017-003/stabilized-deployment-execution-004.log`.

El commit correctivo es `1e66bfd93588cb8b7bb866b25dc327216d19db9e`, publicado en `experiment/modular-audit-v8`. Añade `protocol_id=v8_conservative_retrospective_20261016` al config y asigna recibos/salida nuevos al intento 005. ZIP incremental: `/home/jovyan/work/validation_v8/v8-ccam-retry-1e66bfd.zip`, 5.205 bytes, SHA-256 `3745575f9894cef5f9c136020204bb3d2e1616dac9cd977b7ce237f7bb4f689f`. El bundle incluido tiene SHA-256 `24e3a859a99a977d97f1a8d43f62b88ad15db1ca9cbd8831b88705c938f34bc6` y requiere como base el commit ya desplegado `adf12df244be0251a5c3804d1fd04e756bb75b9a`.

Checkout aislado: `/home/jovyan/work/proyectonp/repo_v8_ccam_stabilized_004`, HEAD `1e66bfd93588cb8b7bb866b25dc327216d19db9e`.

## Identidades y hashes

- Recibos de ejecución: `/home/jovyan/work/validation_v8/usl-ccam-stabilized-launch-20261018-005/`.
- Salida: `/home/jovyan/work/proyectonp/runs_v8/usl_ccam-20261018-005/`.
- Config de ejecución: SHA-256 `4bb0f027e6d12780a1ee762660a1b738c3a9787a8ad456e174ff205b641cafa8`.
- Gate de ejecución: SHA-256 `3754cd94df289365f843a89b7768d9e3dd66c4dde90c3bf38e51aa28a518f923`.
- `launch.json`: SHA-256 `1aea223c9d353ec4b208579d9a19f230d051aace86551cd48a714d2aa1d017ad`.
- Manifiesto de aprendizaje: SHA-256 `7749471410aa98a76dd978cca2ef58f9642d14404530c98e534484ba40a7a6ff`.
- Protocolo retrospectivo: SHA-256 `f4836405a7cbb8309f1757a3488e34ab4fa4f4f02f785b4f72bae79b9d349b03`.
- Banco de pesos: SHA-256 `306136b3abd29ae32ac8be2a1456cf411c3f1422f93b39f05d46636141cce6f6`.
- Recibo de limpieza sintética previo: SHA-256 `0ca336b90aecf3fd999c9b03f7a83b5cc95d961c55650a78cdac091a470922ab`.

## Inicio observado

El 2026-10-18, UNABIA registró supervisor PID `277107` y runner PID `277108`; una captura de estado encontró el runner vivo. El archivo `first-update.json`, emitido después de `optimizer.step`, contiene:

- `status`: `first_optimizer_update_completed`.
- `epoch`: 1.
- `optimizer_update`: 1.
- `sample_count`: 64.
- `effective_batch_size`: 64.
- `adaptation_profile`: `ccam_stabilized_v1`.
- SHA-256 del marcador: `e47fe2c48316130bafe2878e738746bd90a5247ce9f882c665d148240b64a35`.

W&B confirma el run `09a097ab5da3`, nombre `usl_ccam-20261018-005`, estado `running`, creado el `2026-10-18T14:17:45Z`: [abrir corrida](https://wandb.ai/nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion/runs/09a097ab5da3).

El recibo consolidado remoto es `/home/jovyan/work/validation_v8/usl-ccam-stabilized-launch-20261018-005/launch-evidence-005.json`; la consulta de W&B quedó registrada en `wandb-observation-005.log`. `metrics.jsonl` tenía cero bytes al capturar el primer update, por lo que no se reportan aún métricas de época ni medidas completas de RSS/VRAM/tiempo. El status JSON del supervisor conservaba `runner_process_started_training_unconfirmed`; la confirmación de inicio real proviene del marcador posterior al paso del optimizador, el proceso vivo y el estado `running` de W&B.

## Presupuesto, gate y límites

La corrida tiene semilla 42, techo de 30 épocas, batch efectivo 64 y extracción congelada de encoders en microbloques de 2. El techo no significa que se vayan a completar 30 épocas: los criterios de estabilidad seleccionan o detienen la corrida según la evidencia observada.

El gate de esta corrida mantiene `smoke_passed=false` y `resource_budget_measured=false` bajo el waiver explícito autorizado para esta variante. Esos checks no se presentan como aprobados. Las mediciones de tiempo, RSS y VRAM solo se informarán con los recibos producidos durante/tras el entrenamiento.

No se usaron diagnósticos ni máscaras expertas para aprender. La independencia paciente/lesión continúa sin acreditarse para el subconjunto sin metadatos; esta corrida conserva el protocolo retrospectivo y sus limitaciones ya aceptadas. No se alteraron `readiness.review.json` ni gates históricos.
