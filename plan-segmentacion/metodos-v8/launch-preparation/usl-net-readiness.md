# Preparación de USL-Net con datos congelados

Estado vigente (2026-10-17): una pasada CCAM real en GPU completada; gate exclusivo de preparación, ejecución deshabilitada.
Preparación exclusiva de USL-Net; SLED, RPI-Net y TriDermNet-HAGO quedan fuera del alcance de este artefacto.
No hay autorización de entrenamiento. El presupuesto CCAM tiene valores concretos; el recibo GPU y el gate de preparación se documentan al final. El presupuesto del estudiante, la limpieza sintética y la revisión del usuario siguen pendientes. No comenzó entrenamiento.

## Referencias fijadas

Los dos configs conservan el banco ya conectado en
`/home/jovyan/work/public_ssl_v8_preparation/source-bank.usl-net.reviewed-candidate.json`.
Su SHA-256 informativo es `306136b3abd29ae32ac8be2a1456cf411c3f1422f93b39f05d46636141cce6f6`.
Se reutiliza la revisión existente de carga estricta y forward sintético CPU del banco; el preflight inicial no aportó nuevas pruebas de carga, smoke ni forward. La pasada CCAM posterior queda registrada al final.
El sustituto SimCLR de Hugging Face conserva su limitación de equivalencia numérica no verificada contra el TensorFlow original.

Directorio de datos: `/home/jovyan/work/validation_v8/accepted-manifests-20261016-002/`.

| Artefacto | SHA-256 |
|---|---|
| `protocol.json` | `f4836405a7cbb8309f1757a3488e34ab4fa4f4f02f785b4f72bae79b9d349b03` |
| `learning.manifest.json` (34.784 registros) | `7749471410aa98a76dd978cca2ef58f9642d14404530c98e534484ba40a7a6ff` |
| `freeze.receipt.json` | `70adc1177b70da13b0a0a98bfdb1ab8d3456b0dc5d39056d0fc53a6ce1bab548` |

`evaluation_protocol` es el contrato superior que consume el runtime, con identificador
`v8_conservative_retrospective_20261016`. Los pins en `preparation_review` son informativos:
el manifiesto de aprendizaje debe pasarse mediante el argumento CLI correspondiente en una futura ejecución autorizada;
una clave de preparación no lo transmite automáticamente al runner.
El recibo de congelación mantiene `training_consumable=false`: congelar datos y autorizar ejecución son pasos separados.
No se acredita independencia clínica, identidad paciente/lesión completa ni ausencia global de exposición.

## Preflight histórico de candidatos verificado en UNABIA

Estos comandos comprueban identidad CLI/config sin manifiesto y sin lectura de imágenes:

```bash
/home/jovyan/work/proyectonp/.venv/bin/python scripts/run_segmentation_v8.py --method usl_net --stage check --config plan-segmentacion/metodos-v8/launch-preparation/configs/usl_ccam.draft.json
/home/jovyan/work/proyectonp/.venv/bin/python scripts/run_segmentation_v8.py --method usl_net --stage check --config plan-segmentacion/metodos-v8/launch-preparation/configs/usl_net.draft.json
```

El coordinador ejecutó ambos comandos sin manifiesto: salida 0. Comprueban identidad CLI/config y no prueban lectura de imágenes, tensores ni uso del banco en el runtime.
El recibo `/home/jovyan/work/validation_v8/usl-config-binding-20261016-001/metadata-preflight.json`
registra `validate_binding(config, learning_document)` aprobado para los dos candidatos,
con hashes, cohorte y guard de fuentes; también se verificó el hash del JSON del banco.
Por esa evidencia `manifest_bound=true`, con alcance de enlace de metadatos, cohorte,
guard de fuentes y protocolo. Se conservan `execution_enabled=false`, `gate_bound=false`
y `resource_caps_measured=false`. El pin del manifiesto sigue requiriendo el argumento CLI `--manifest`.

La observación de GPU detectó dos RTX PRO 4000 Blackwell, cada una con 24.467 MiB,
con 23.723 y 23.985 MiB libres. Los metadatos reales de PyTorch fueron `2.11.0+cu128`,
CUDA 12.8, disponibilidad verdadera, arquitectura compilada `sm_120` y capacidad 12.0.
Esto no prueba ejecución de kernels, smoke CUDA del método ni memoria pico.
No se revalidaron bytes de imágenes, cargaron tensores ni ejecutaron optimizador, forward o tracking.

La comprobación canónica posterior quedó registrada en
`/home/jovyan/work/validation_v8/usl-config-binding-20261016-001/final-preflight.json`,
SHA-256 `54eee39336b5743092db10d200395b6c00397dd8c77831397855b690662b76a5`.
Su estado es `usl_configs_bound_runtime_blocked`, con
`metadata_and_identity_checks_passed=true` e `initial_ccam_can_start=false`.
Se instalaron los dos configs bajo
`/home/jovyan/work/proyectonp/repo_v8_canonical/plan-segmentacion/metodos-v8/launch-preparation/configs/`:

| Config canónico | SHA-256 |
|---|---|
| `usl_net.draft.json` | `80cd9554a92013c483ee2abee21114a005f47ef84f1316306139e9635c490474` |
| `usl_ccam.draft.json` | `a43879e56767ec00bc73c69df8b3d0273c5d9d66d748aac9c6acc95014c299f7` |

Ambos aprobaron el enlace al documento congelado de 34.784 registros y el pin del JSON
del banco. Las rutas de sus tres checkpoints y tres archivos de procedencia estaban presentes
(6/6); no se volvieron a cargar los pesos. Los comandos CLI mostrados arriba, aplicados
a las rutas canónicas absolutas, terminaron con salida 0, sin `--manifest`.
Se reutilizaron los recibos históricos de carga estricta CPU y forward sintético,
sin prueba nueva del modelo. `readiness.review.json` permaneció sin cambios,
SHA-256 `295ebe945807f6672cb11f18b4e75d2127bba7265bb73ad556d6e7a87f3ae014`;
el recibo de congelación también permaneció sin cambios.

## Pendientes históricos del preflight inicial

- Fijar y revisar las épocas CCAM: `usl_ccam.draft.json` conserva `epochs=null`.
- Medir recursos y fijar timeout y almacenamiento por etapa. Ambos configs conservan
  `timeout_seconds=null` y `max_output_mb=null`; RSS 65.536 MiB, VRAM 20.480 MiB
  y reserva 4.096 MiB son borradores sin medición.
- Verificar el smoke CUDA del método y obtener el gate real con método `usl_ccam`, vinculado a hashes del config y manifiesto,
  datos, recursos, pesos, smoke y limitación retrospectiva con independencia paciente no acreditada.
- Cerrar la limpieza sintética previa a la primera corrida real, preservando sus recibos de procedencia.
- Completar la revisión obligatoria del usuario antes de habilitar ejecución o lanzar.

## Dependencias posteriores del estudiante

El entrenamiento CCAM inicial producirá el checkpoint CCAM, su sidecar, sus hashes y su linaje;
no son requisitos previos para entrenar ese mismo CCAM. Después se necesita el manifiesto
completo de pseudomáscaras y su procedencia antes de entrenar el estudiante.
`native_recipe` registra cinco ciclos de 300 épocas, pero no implementa su orquestación automática:
el plan debe materializar sus etapas, refrescos, gates y salidas antes de ejecutar los ciclos.

`max_centroid_distance=null` es válido y significa ausencia de corte por distancia; no es un bloqueo de preparación.

## Preparación CCAM con diagnóstico GPU acotado (2026-10-16)

Esta sección actualiza el estado anterior sin reemplazar sus recibos históricos.
Solo se modifica `usl_ccam.draft.json`; el config del estudiante permanece intacto.

| Parámetro CCAM | Valor fijado | Naturaleza |
|---|---|---|
| `epochs` | 5 | Referencia histórica `cluster-final.json` y doc 24, con head y batch distintos; no acredita equivalencia con el paper ni suficiencia de convergencia. |
| `resources.timeout_seconds` | 86.400 s | Techo de planificación de 24 horas; no duración medida ni promesa de terminar cinco épocas. |
| `resources.max_output_mb` | 1.024 MiB | Techo operativo basado en la estimación del precheck del runner; no almacenamiento final medido. |
| `resources.max_rss_mb` | 65.536 MiB | Límite conservado; no medición. |
| `resources.max_gpu_memory_mb` | 18.432 MiB | Tope del asignador CUDA de 18 GiB; junto con la reserva suma 22.528 MiB y deja 1.195 MiB respecto a los 23.723 MiB libres observados en GPU 0. No se confunde capacidad total con memoria disponible. |
| `resources.gpu_reserve_mb` | 4.096 MiB | Reserva de planificación; el diagnóstico comprueba disponibilidad antes de cargar el modelo. |

Los cinco ciclos de 300 épocas de `native_recipe` corresponden al estudiante y no son el presupuesto CCAM.
`resource_caps_measured=false` conserva la distinción entre límites de configuración y mediciones del recibo.
`execution_enabled=false` permanece en ambos configs.

El diagnóstico `scripts/probe_v8_usl_ccam_cuda.py` usa dos originales reales del manifiesto
de aprendizaje, resolución 224 y batch 2, el banco de tres encoders y la misma secuencia
de fusión, activación, loss y backward de `fit_ccam`. No crea optimizador, no actualiza
parámetros, no ejecuta épocas, no guarda pesos aprendidos ni inicia W&B. Compara el
SHA-256 de parámetros y buffers antes y después. Su timeout diagnóstico es 600 s,
separado del techo de planificación de 86.400 s.

El intento `/home/jovyan/work/validation_v8/usl-ccam-preparation-20261016-001/receipt.json`
abortó antes de CUDA por ausencia en el checkout parcial de la dependencia histórica
`plan-segmentacion/camino-b-cnn/pseudo-mascaras/ccam_engine.py`.
SHA-256 del recibo fallido: `7a71d27a36993c6aa4057cf9e17c0759a0066df00c99e44f9c65a6d2c1aa474a`.
No produjo pasada de modelo ni medición GPU. Se conservó el intento y se restauró
por ZIP el archivo histórico sin modificaciones, SHA-256
`3250cf52d60e7c3ebaab437c764148c8a88843f497f2df624e9430b4792c06a7`.
No se alteró la implementación del objetivo para eludir la dependencia.

El intento `/home/jovyan/work/validation_v8/usl-ccam-preparation-20261016-002/receipt.json`
rechazó el presupuesto antes de cargar el modelo: 20.000 + 4.096 = 24.096 MiB excedían
los 23.723 MiB libres reales en GPU 0, aunque la capacidad total era 24.467 MiB.
Su pico asignado CUDA fue cero; no fue una prueba del modelo ni un OOM de CCAM.
SHA-256 del recibo: `466fbe655dccad8bb19b9ae5e540e738ce08b7c0cfe4c0bfefc90d936b0dbbb6`.
Se ajustó únicamente el límite de configuración a 18.432 MiB, preservando la reserva.
## Resultado real CCAM de preparación (2026-10-17)

Se completó una única pasada real GPU con el presupuesto revisado en
`/home/jovyan/work/validation_v8/usl-ccam-preparation-20261017-003/`.
Los intentos históricos `001` y `002` se preservan como fallos previos al modelo.

| Artefacto final | SHA-256 |
|---|---|
| `receipt.json` | `ed7275b299a2a8cf237c31d25375148de60b41ac5b9f4e7c42c6726e0cf30d84` |
| `ccam.gate.json` | `bcb9f329dbc0ff9acde28fe9e06f1853fc6a2f67022bd3d1fd74ce411cd16264` |
| Config CCAM, bytes | `37fae7fdcdf52c91d1f2207922620f6146aa82e0030e8f665747fa4a70ac4159` |
| Config CCAM, JSON canónico | `fa6ad6a4c201998509fa00013dd3da9dd834a640f1f5df509a6db93ec05a2202` |
| `scripts/probe_v8_usl_ccam_cuda.py` | `ce3e9ac7a3d41e4fda59c6f9ce2fe62379b8070df62ae815a3eae0fce491b16e` |

Entorno observado: Python 3.13.15, PyTorch `2.11.0+cu128`, CUDA 12.8,
RTX PRO 4000 Blackwell, capacidad 12.0. El asignador CUDA reportó
23.985,875 MiB totales y 23.485,25 MiB libres. Estas cifras tienen un alcance distinto
de la observación anterior de `nvidia-smi` (24.467 MiB totales y 23.723 MiB libres);
se conservan ambas observaciones sin intercambiarlas.

El batch de dos originales de aprendizaje fue `ISIC_0000010` y
`ISIC_0000118_downsampled`, resolución 224, con features `[2, 9216, 14, 14]`.
Carga del modelo: 0,962499836 s; batch: 0,427408971 s; total: 5,205805866 s.
RSS pico: 1.721,578125 MiB; pico CUDA asignado: 396,95263671875 MiB;
pico reservado: 452 MiB. Loss: 4,4416823387146, con dos tensores de gradiente
finitos, sin errores ni OOM. El SHA-256 del estado antes y después fue idéntico:
`4f10c6fc00fadf37157fb09fee2720858161f2a5c63eb294066e5bc4ed5a2050`.

La pasada comprendió forward, loss y backward; no creó optimizador, ejecutó pasos,
guardó checkpoints aprendidos, inició tracking ni entrenamiento. No midió estados AdamW,
duración del conjunto completo ni volumen final de salida. Los cinco epochs y el techo
operativo de 24 horas siguen siendo planificación, no duración medida de una corrida completa.

El gate real queda ligado por hash al config y al manifiesto congelado de 34.784 registros
(`7749471410aa98a76dd978cca2ef58f9642d14404530c98e534484ba40a7a6ff`).
Registra `smoke_passed=true`, `resource_budget_measured=false`, `preparation_only=true`,
`training_authorized=false` y ejecución deshabilitada. Debe rechazar entrenamiento mediante
`gate_report`. El recibo de datos mantiene `training_consumable=false`, sin cambios;
la independencia paciente/lesión no se acredita. La limpieza sintética y la revisión del usuario
siguen pendientes antes de habilitar una corrida real. Los otros métodos no se modifican.

La [copia local de la evidencia parseada](../../../resultados/preparacion_v8/usl_ccam_cuda_20261017/diagnostic.snapshot.json) permite revisar recibo y gate
con sus hashes de origen. Su serialización local difiere de los bytes originales del clúster;
no sustituye esos hashes de origen.
