# V8 — CCAM: política de convergencia y checkpoint reanudable

## Objetivo
Cerrar el hallazgo de Codex del 2026-10-17: CCAM corrió 5 épocas fijas derivadas
del presupuesto de tiempo del smoke test, no de un criterio de convergencia.
Solo guarda la cabeza final (no reanudable). Hay que dejar CCAM con: (1) un
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

## Pendiente, fuera de este cierre
- No se relanzó CCAM en UNABIA (fuera de alcance, como se pidió).
- El `probe_images` de estabilidad usa las primeras N imágenes de
  entrenamiento (no un split reservado) — es una señal de monitoreo, no de
  validación; declarado así a propósito.
- `select_candidate_epoch`/`RECOMMENDED_EPOCH_CEILING` quedan como utilidades
  documentadas; nadie los invocó todavía contra una corrida con datos reales,
  porque esta tarea no relanzó entrenamiento.
