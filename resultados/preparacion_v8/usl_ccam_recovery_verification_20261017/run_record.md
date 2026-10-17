# Verificación GPU acotada de recuperación CCAM V8

## Identidad y fuentes

- Fecha: 2026-10-17; horas de inicio/fin con zona: no registradas en el resumen local.
- Familia: segmentación V8, USL-Net, etapa diagnóstica CCAM; no corrida científica de 30 épocas.
- Rama de trabajo: `experiment/modular-audit-v8`. Base desplegada: `6c5627d` más corrección RNG por ZIP; el HEAD Git remoto sigue en `95a8f602b2fb49304dbbb9a78a18edc43cc3ebba` y no representa todos los archivos desplegados.
- Fuente local revisada: [verification.snapshot.json](verification.snapshot.json), resumen observado por el operador desde UNABIA; no es una copia íntegra de los recibos.
- Raíz remota: `/home/jovyan/work/validation_v8/usl-ccam-recovery-verification-6c5627d-20261017-001/`.
- Recibo inicial: `run/receipt.json`, SHA-256 `d1b980de794f92d92dbb08e537b90740effbe5cdaec88ad65f849bd3571c7c55`.
- Recibo posterior: `run-fixed/receipt.json`, SHA-256 `bd79fd6893b323836e0307e28e71ca06e2a239c2e1b547b62a9508b7a8fca669`.
- Log posterior: `run-fixed.process.log`, dentro de esa raíz.
- Runner `verify.py`: SHA-256 `a7de8ebb3c308bd22251bad9a13396d7da54ce7ef06b5d6ed64b4e4ab57a9bb4`; semilla 42 heredada del config de preparación.

Comando de la segunda pasada, ejecutado en UNABIA con `PYTHONPATH` al `src` canónico y `CUBLAS_WORKSPACE_CONFIG=:4096:8`:

```bash
/home/jovyan/work/proyectonp/.venv/bin/python /home/jovyan/work/validation_v8/usl-ccam-recovery-verification-6c5627d-20261017-001/supervise.py --repo /home/jovyan/work/proyectonp/repo_v8_canonical --output-dir /home/jovyan/work/validation_v8/usl-ccam-recovery-verification-6c5627d-20261017-001/run-fixed
```

## Datos y alcance

Se utilizaron ocho originales reales elegibles de aprendizaje, sin imágenes sintéticas nuevas ni máscaras expertas. Manifiesto padre:
`/home/jovyan/work/validation_v8/accepted-manifests-20261016-002/learning.manifest.json`, SHA-256 `7749471410aa98a76dd978cca2ef58f9642d14404530c98e534484ba40a7a6ff`.
Se ejercitaron los tres encoders congelados, fusión, pérdida, optimizador, RNG y checkpoints reales en GPU. El harness sustituyó únicamente `gate_report`, `read_manifest` y `training_lineage` por controles diagnósticos: esta pasada no verifica el lanzador ni el gate de producción ni la cohorte completa.

El diagnóstico usó dos épocas y umbral de selección `-1` para alcanzar la rama de exportación. El config de producción sigue deshabilitado, con 30 épocas y umbral `0.98`, sin cambios. El gate candidato anterior conservó su SHA-256 `a1fa4b2d4772a71b0f2d3080f2fa0212e85b2fa4709d5467c7f0b620d01ef2cf`; ese valor no es el hash del config. Los hashes completos de pesos/config del diagnóstico no están transcritos en el resumen local; sus fuentes autoritativas son el runner y los recibos remotos. No hubo tracking W&B nuevo.

## Fallo inicial y corrección delimitada

La primera pasada falló al recuperar con `TypeError: RNG state must be a torch.ByteTensor`: `torch.load` con destino CUDA movía a GPU los tensores ByteTensor del RNG, incompatibles con la restauración. Duración del worker: 9.63970499811694 s; supervisor: 10.746103493031114 s. El recibo fallido se conserva.

La corrección carga el payload en CPU, manteniendo `weights_only=True`, y mueve explícitamente `previous_saliency` al dispositivo de cómputo. Solo se cambió `src/dermssl/segmentation_methods/usl_net/seeds.py`, con respaldo del archivo anterior:

| Identidad | SHA-256 |
|---|---|
| Archivo anterior | `b83586ff26d5185d6bef193d81e032988c6442917afaffd57c795ca817df3a75` |
| Archivo corregido desplegado | `46c4cb2049c789c2a38a21261b514e3d071b6239d44c14242c06f1707bbac463` |
| ZIP de corrección | `88caf6bbd0253ce51774ff19c1f4e09b18b04755c2f97154d9484802eb803cf8` |

Árbol de fuente preparado: `356f1d37948682e039beee7f450cfbdd76525744`. No se atribuye este cambio al commit base `6c5627d`.

## Resultado posterior y recursos observados

Estado fuente: `passed_diagnostic_only`; supervisor completado con código 0. Hubo 16 actualizaciones: ocho de referencia continua, cuatro antes de interrumpir y cuatro al recuperar. La interrupción ocurrió después de confirmar la época 1 y antes de publicar métricas (cero filas); la recuperación reconstruyó las épocas `[1, 2]` sin duplicados.

`continuous_vs_recovered_exact` dio `true` en sus ocho claves: cabeza, optimizador, RNG, registros por época, historial de pérdida, estabilidad, saliencia anterior y candidato seleccionado. Las exportaciones cargaron con `weights_only=True`, ligadas al estado guardado; no quedaron temporales de checkpoint.

| Medición | Valor observado | Alcance |
|---|---:|---|
| Worker | 9.222343945875764 s | Diagnóstico posterior completo |
| Supervisor | 10.194587241858244 s | Proceso supervisado posterior |
| RSS pico | 2154.9140625 MiB | Proceso worker |
| CUDA pico asignado | 548.45947265625 MiB | Allocator PyTorch |
| CUDA pico reservado | 676 MiB | Allocator PyTorch; no total de dispositivo |

Entorno observado: Python 3.13.15, PyTorch `2.11.0+cu128`, CUDA 12.8, `cuda:0`, NVIDIA RTX PRO 4000 Blackwell. No se observó OOM. No son mediciones de la corrida completa de 30 épocas ni cuota individual del clúster.

## Selección y límites de interpretación

`selection.selected_epoch=2`, `selection.status=stability_candidate` y `selection.stopping_reason=fixed_budget_exhausted` corresponden al umbral diagnóstico `-1`. `epoch_metrics[1].saliency_stability=0.9772918224334717`, inferior al umbral de producción `0.98`: no se verificó su cruce. La selección coincide con la última época del diagnóstico; no demuestra retención de un candidato frente a épocas posteriores de entrenamiento. Tampoco acredita convergencia, validez científica del umbral ni calidad anatómica.

Los artefactos protegidos conservaron sus hashes antes/después de ambas pasadas: corrida histórica de cinco épocas, configuración/gate candidato, manifiestos congelados y readiness histórico. Se comprobaron inicialmente 55 archivos contra el despliegue; el runner identifica 11 fuentes relevantes. No se habilitó ni inició producción.

## Pendientes y orden siguiente

1. Preparar el vínculo de inventario/código/gate posterior a la corrección RNG, usando este recibo y los pines ya cerrados. El gate de `6c5627d` es anterior a la corrección y no se da por aprobado.
2. Revisar configuración de 30 épocas, rutas de salida/tracking nuevas e inventario exacto de artefactos diagnósticos; preservar primero recibos/hashes y retirar o archivar lo registrado justo antes de la corrida real. Los directorios de checkpoints diagnósticos son `run/continuous/`, `run/interrupted-resumed/`, `run-fixed/continuous/` y `run-fixed/interrupted-resumed/`, relativos a la raíz remota indicada. Los archivos `diagnostic-subset.json` y `diagnostic-not-a-production-gate.json` de ambas pasadas también son fixtures, no manifiestos oficiales. No se elimina nada en este avance.
3. Presentar la habilitación y el lanzamiento real a revisión del usuario; este diagnóstico no los autoriza.
4. Tras CCAM, analizar curvas y candidato antes de promover pseudomáscaras. El plan serial del estudiante/refinamientos permanece pendiente; la referencia de cinco ciclos de 300 épocas no inicia automáticamente ninguna etapa. Los otros métodos no se modifican.
