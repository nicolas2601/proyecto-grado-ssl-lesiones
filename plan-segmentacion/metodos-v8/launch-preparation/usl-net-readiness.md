# Preparación de USL-Net con datos congelados

Estado: `bound_pending_runtime_gates` (2026-10-16).
Preparación exclusiva de USL-Net; SLED, RPI-Net y TriDermNet-HAGO quedan fuera del alcance de este artefacto.
No se puede confirmar inicio de entrenamiento esta noche: siguen pendientes épocas CCAM, presupuestos y medición de recursos, smoke CUDA del método, gates, limpieza sintética y revisión del usuario. No comenzó entrenamiento.

## Referencias fijadas

Los dos configs conservan el banco ya conectado en
`/home/jovyan/work/public_ssl_v8_preparation/source-bank.usl-net.reviewed-candidate.json`.
Su SHA-256 informativo es `306136b3abd29ae32ac8be2a1456cf411c3f1422f93b39f05d46636141cce6f6`.
Se reutiliza la revisión existente de carga estricta y forward sintético CPU del banco; esta preparación no aporta nuevas pruebas de carga, smoke ni forward.
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

## Preflight de candidatos verificado en UNABIA

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

## Pendientes para entrenar CCAM inicialmente

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
