# V8 — CCAM: política de convergencia y checkpoint reanudable

## Objetivo
Cerrar el hallazgo de Codex del 2026-10-17: CCAM corrió 5 épocas fijas sin un
criterio de convergencia acreditado. La ejecución histórica solo guardó la
cabeza final (no reanudable). Hay que dejar CCAM con: (1) un
checkpoint reanudable completo (modelo + optimizador + scheduler + estado RNG,
igual de robusto que `fit()` en `common/training.py`), y (2) señales de
convergencia reales trackeadas por época (curva de pérdida completa +
estabilidad del mapa de saliencia/activación), para poder justificar con
evidencia en la tesis en qué época conviene detenerse.

## Por qué / contexto
- Quedan ~1 mes hasta la defensa (dato del usuario, 2026-10-17). Prioridad
  explícita: refinar a fondo la línea de segmentación (la dueña de esta tarea
  es la usuaria; Nico refina clasificación por separado sin insumo de
  segmentación por ahora; al final se combina en un "proposed con
  segmentación").
- CCAM `002` ya completó 5 épocas reales (no sintéticas), checkpoint
  `candidato`, pérdida 9,44→2,42 (solo primer/último valor conocido, falta la
  curva completa). Ver `odd/tasks/v8-cierre-bloqueos-lanzamiento.md`, commit
  `012110f`. Explícitamente: "no repetir épocas ni entrenamiento automático"
  hasta resolver esta política.
- `common/training.py::fit()` y `common/checkpoints.py` YA tienen un patrón de
  checkpoint reanudable completo y probado (modelo+optimizador+sampler+RNG,
  guardado atómico vía tmp+os.replace, identidad verificada antes de
  reanudar) usado por usl_net (estudiante), rpi_net y tridermnet_hago. CCAM
  **no** pasa por ese `fit()` genérico — tiene su propio loop en
  `plan-segmentacion/camino-b-cnn/pseudo-mascaras/train_ccam.py` (+
  `ccam_engine.py`, `saliency_prep.py`, `uncertainty_module.py`,
  `make_pseudo_masks.py`, `artifacts.py`), lanzado vía
  `scripts/launch_v8_usl_ccam.py`. El objetivo es llevar CCAM al mismo
  estándar de robustez, reusando el patrón existente en vez de reinventarlo,
  no necesariamente forzarlo a pasar literalmente por `fit()` si su forma de
  entrenamiento (CAM, no segmentación con pseudo-labels) no encaja ahí.
- Modo TDD: **Strict TDD habilitado** (CLAUDE.md). Hay 365 archivos de test en
  `tests/`, incluido `tests/segmentation_v8/`. Seguir RED→GREEN→REFACTOR real,
  sin inventar evidencia de tests pasando.
- Nunca ajustar con las máscaras expertas reservadas (ISIC2018 holdout) en
  ningún punto de este trabajo — es una regla dura del proyecto.

## Alcance de esta tarea
1. Leer y entender el loop real de CCAM (`train_ccam.py`, `ccam_engine.py`,
   `saliency_prep.py`, `uncertainty_module.py`, `launch_v8_usl_ccam.py`) antes
   de tocar nada.
2. Checkpoint reanudable para CCAM: modelo + optimizador + scheduler + estado
   RNG + cursor de época/paso, guardado atómico, con la misma disciplina de
   identidad/lineage que ya usa `common/checkpoints.py` (no reinventar el
   formato si se puede reusar).
3. Tracking por época: curva de pérdida completa (ya existe parcialmente,
   verificar) + una métrica de estabilidad del mapa de saliencia/activación
   CCAM entre épocas consecutivas (IoU o correlación sobre un subconjunto
   fijo) — esto no existía en ningún lado antes de este cierre.
4. Documentar (no necesariamente implementar todavía) la política de
   parada: techo razonable de épocas (anclado al precedente de 30 épocas del
   preentrenamiento MoCo ya usado en este mismo proyecto, no inventar otro
   número), criterio de selección del checkpoint candidato (estabilización de
   saliencia, no último epoch por default).
5. Tests nuevos en `tests/segmentation_v8/` cubriendo lo anterior, siguiendo
   TDD estricto.

## Fuera de alcance (explícitamente, por ahora)
- No relanzar CCAM en UNABIA todavía — esto es solo el código/tests locales.
- No tocar SLED, RPI-Net ni TriDermNet-HAGO en esta tarea (aparte, pendiente
  del mapeo completo en curso).
- No tocar el Experimento 5 (DINOv2+DPT) ni la línea supervisada de respaldo
  todavía.

## Estado
- [x] Explorar el loop real de CCAM. Hallazgo: `train_ccam.py` (camino-b, pre-v8)
      NO es el código que corre en UNABIA — el real es
      `fit_ccam` en `src/dermssl/segmentation_methods/usl_net/seeds.py`,
      invocado por `scripts/run_segmentation_v8.py --stage ccam`. Confirmaba
      exactamente el hallazgo de Codex: un solo `save_checkpoint` al final,
      sin estado de optimizador ni reanudación.
- [x] Checkpoint reanudable para CCAM. `fit_ccam` ahora acepta `resume=True`,
      guarda `resume.pt` atómico después de cada época (cabeza CCAM +
      optimizador + época + historial de pérdida/estabilidad + RNG del
      DataLoader), con identidad verificada (método/revisión/config/manifest/
      gate/tracking_run_id) antes de reanudar. Reusa `atomic_checkpoint`,
      `rng_state`, `restore_rng` de `common/training.py` (antes privados,
      promovidos a públicos — únicos 3 renombres, sin cambiar su
      comportamiento; sus 3 call-sites internos en `fit()` actualizados).
      `scripts/run_segmentation_v8.py` ahora acepta `--resume` también para
      `--stage ccam` (antes solo `fit`).
- [x] Tracking de estabilidad de saliencia por época. Nueva función pura
      `saliency_stability(previous,current)` (correlación de Pearson sobre un
      probe fijo de hasta 8 imágenes, semilla 42) en `usl_net/seeds.py`;
      queda en `metrics.jsonl` como `saliency_stability` por época (`null` en
      la primera, por no tener época previa con la que comparar).
- [x] Política de parada documentada (sin relanzar nada). `RECOMMENDED_EPOCH_CEILING=30`
      en `seeds.py`, anclado al precedente de 30 épocas del preentrenamiento
      MoCo ya usado en este proyecto (docs/55) — no es un número inventado ni
      medido para CCAM específicamente, está declarado como tal. Función pura
      `select_candidate_epoch(stability_history,threshold=.98)` devuelve la
      primera época que cruza el umbral de estabilidad, no la última por
      default. No se invocó contra ninguna corrida real.
- [x] Tests nuevos, TDD estricto, evidencia real.
      `tests/segmentation_v8/test_usl_ccam_fit.py`: 12 tests (unit para
      `saliency_stability`/`select_candidate_epoch` + integración para
      `fit_ccam` con manifiesto/gate sintéticos y encoders de juguete
      parcheados vía `unittest.mock.patch`, sin backbone real).
      RED confirmado: `git stash` de los 3 archivos de implementación →
      `ImportError: cannot import name 'saliency_stability'` (colección
      falla, no solo assert). GREEN confirmado tras restaurar: **12 passed**
      en 3.98s. Suite completa `tests/segmentation_v8/`: **106 passed, 1
      skipped, 12 failed** — los 12 fallos son pre-existentes en
      `test_v8_exclusion_application.py` (error de Windows al duplicar
      handles de subprocess, `OSError: [WinError 50]`), confirmado idéntico
      (mismo conteo, mismas pruebas) con y sin estos cambios vía stash — no
      relacionado con este trabajo. torch no estaba instalado localmente;
      se instaló `torch` (CPU, wheel cp314 real) y `pillow` para poder correr
      la suite — es una dependencia de desarrollo reversible, no toca UNABIA
      ni ningún entorno de cluster.

## Configuración y recuperación integradas (2026-10-17, posterior a la sincronización)

- Implementación local preparada, sin pruebas ni entrenamiento en este avance:
  el CLI pasa los hashes reales del código a la identidad de recuperación;
  su mapa recorre `segmentation_methods/`, incluido `usl_net/seeds.py`.
  También se vinculan el hash del manifiesto del banco y las rutas/hashes de
  los tres checkpoints y sus sidecars reales; se comprueban antes y después
  de cargarlos. Una recuperación exige
  identidad idéntica: no admite los checkpoints antiguos sin este vínculo.
- `resume.pt` se confirma atómicamente antes de publicar métricas. Conserva
  los registros completos por época y la cabeza seleccionada; `metrics.jsonl`
  se reconstruye atómicamente desde esos registros al recuperar. Así el archivo
  local tiene una fila por época confirmada, aunque una interrupción haya
  dejado métricas atrasadas. W&B recibe el número lógico `epoch`, pero no hay
  transacción distribuida: una interrupción antes de enviar puede dejar una
  fila ausente allí; no se promete entrega externa exactamente una vez.
- El borrador sigue con `execution_enabled=false`: 30 épocas fijas, batch 2,
  AdamW LR 0,001 y weight decay explícito 0,01. Las 30 épocas se anclan al
  precedente MoCo y no a una convergencia medida de CCAM. No hay parada temprana.
- `ccam_selection` conecta la primera época con Pearson >= 0,98. La cabeza
  de esa época se conserva dentro de `resume.pt` y al completar se exporta
  a `selected.ccam.pt` con sidecar y época real. `ccam.pt` conserva la cabeza
  final de presupuesto, explícitamente no seleccionada. `selection.json`
  registra ambos hashes; si no se cruza el umbral, declara
  `threshold_not_reached`, sin candidato seleccionado ni promoción automática.
  Las exportaciones son recuperables desde el estado confirmado y rechazan
  sobrescribir archivos pertenecientes a otra identidad.
- El probe son las primeras ocho imágenes del pool de aprendizaje, dentro de
  muestra. El umbral es una heurística técnica, no una regla del artículo ni
  evidencia de calidad anatómica. Batch 2 conserva un único positivo de otra
  imagen; esa limitación científica sigue declarada.
- Se conservan los límites de planificación de 86.400 s y 1.024 MiB. Las cinco
  épocas previas duraron 1.652,062884596 s; la extrapolación lineal a 30 da
  9.912,377307576 s, sin acreditar una medición de 30 épocas. Los archivos de
  cabeza/optimizador y métricas previstos caben holgadamente en el límite
  planificado; no se presenta ese límite como consumo medido.

## Pendiente inmediato

Revisar el cambio completo, preparar ZIP y desplegar en UNABIA. El gate anterior
está vinculado al config anterior: `gate_bound=false` y presupuesto no congelado
hasta revisar el nuevo vínculo. No se alteraron gates ni `readiness.review.json`.
La corrida real de cinco épocas permanece intacta; no se recupera exactamente
desde su cabeza sin optimizador/RNG. No se relanza ni se habilita nada con esta
preparación. El plan serial del estudiante y sus refinamientos sigue pendiente.
La evidencia de pruebas de la sección anterior corresponde al commit previo;
no valida estos cambios nuevos.

## Actualización: diagnóstico real GPU de recuperación (2026-10-17)

La comprobación acotada autorizada detectó un fallo RNG de CUDA en la base
`6c5627d`; se corrigió únicamente `seeds.py`, cargando el payload en CPU y
trasladando la saliencia al dispositivo. Se preservó el recibo fallido. Con
ese overlay, la repetición en UNABIA terminó con código 0 y estado
`passed_diagnostic_only`: cabeza, optimizador, RNG, registros, pérdidas,
estabilidad, saliencia anterior y candidato coincidieron exactamente entre
referencia continua y recuperación; las métricas reconstruidas no se duplicaron.
Se midieron 9,222 s de worker, RSS 2154,914 MiB y CUDA asignado/reservado
548,459/676 MiB. Son dos épocas sobre ocho originales, no la corrida completa.
La exportación se ejercitó con umbral diagnóstico `-1`; el `0.98` de producción
no se cambió ni se cruzó, y no se demuestra retención frente a épocas posteriores.
Pendiente: vincular gate/inventario al overlay, revisar preparación final y
limpieza tardía antes de someter habilitación/lanzamiento a revisión del usuario.
No se modificaron gates, readiness, manifiestos ni la corrida histórica.
[Registro, hashes y limitaciones](../../resultados/preparacion_v8/usl_ccam_recovery_verification_20261017/run_record.md).
