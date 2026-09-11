# Checkpoint manual: Stage 2 CIRCLe (baseline vs. regularizacion vs. blend sintetico)

**Estado**: NO ejecutado. Este documento describe el paso manual que un humano con
sesion GPU debe correr; no hay resultados reales todavia. Numbering: usa `21-`
para no chocar con `docs/20-esquema-fst-manifest.md` de PR3 (rama hermana, aun no
fusionada).

## Por que es manual, no CI

Igual que el paso de Kaggle de PR3 (`docs/20`), este entorno de apply no tiene GPU
ni sesion de Kaggle. `src/dermssl/circle.py` (numpy, PR4a) y
`src/dermssl/finetune/circle_torch.py` (torch, PR4b) estan implementados y
probados (TDD real para `circle.py`, `py_compile`/ast estatico para
`circle_torch.py`), pero **nadie corrio todavia** un entrenamiento real de Stage
2. Por proposal.md: "Stage 2 is gated on Stage 1 showing measurable bias" - y
Stage 1 mismo (el `fst_manifest.csv` real de PR3) tampoco corrio todavia. Este
checkpoint queda documentado para cuando ambos pasos se ejecuten.

## Precondicion

Antes de este checkpoint debe existir un `fst_manifest.csv` real (PR3's manual
TODO) y el gate de fairness de Stage 1 (`gates.fairness_gate`) debe reportar
`FAIL` - o, en terminos de codigo, `dermssl.circle.should_run_stage2(gate_status)`
debe devolver `True`. Si Stage 1 reporta `PASS` o `SKIP`, este checkpoint se omite
por diseno (spec.md, circle-adaptation, "Stage-Gate Dependency on Stage 1") - no
es un fallo, es el comportamiento esperado cuando no hay brecha medible que
mitigar.

## Los tres runs que produce este checkpoint

1. **Baseline**: el probe entrenado normalmente (sin CIRCLe), evaluado con
   `metrics.stratified_summary()` sobre el manifest FST real. Es exactamente el
   Stage 1 ya wireado en `scripts/notebook_code_v5.py` (PR2).
2. **Regularization-only**: el mismo probe, entrenado con
   `finetune.circle_torch.stage2_train_step` (termino MSE conjunto entre la
   imagen original y su version desplazada en FST via
   `dermssl.circle.shift_skin_tone`), sin mezclar datos sinteticos en el batch
   de entrenamiento en si - solo el termino de regularizacion.
3. **Synthetic-blend**: igual que (2), pero ademas el batch de entrenamiento
   mezcla imagenes reales y sinteticas (desplazadas en FST) segun
   `dermssl.circle.synthetic_blend_mask` (ratio configurable, default ~50/50,
   shift maximo 2 niveles FST via `dermssl.circle.clamp_fst_shift`).

Cada uno de los tres runs produce sus propias predicciones de validacion, que se
pasan a `metrics.stratified_summary()` por separado - los tres resultados son el
insumo de `metrics.comparison_report()` (Fase 6, PR4c), que calcula la brecha por
variante y la mejora relativa de (2) y (3) contra (1).

## Pasos para un humano con sesion GPU real

1. Confirmar que `fst_manifest.csv` (PR3) existe y que
   `gates.fairness_gate(stratified_summary(...))` en el run de Stage 1 real
   reporta `FAIL` (o una brecha por encima del umbral que se considere "vale la
   pena mitigar").
2. Entrenar el probe baseline (ya existe el camino v5/finetune actual) y guardar
   sus predicciones de validacion + bucket FST por imagen.
3. Entrenar la variante "regularization-only" usando
   `finetune.circle_torch.freeze_backbone`, `build_probe_optimizer` y
   `stage2_train_step` en un loop de entrenamiento real (adaptar
   `finetune/torch_run.py`'s estructura de datos, pero manteniendo el backbone
   congelado - a diferencia de `torch_run.py`, que si entrena el backbone
   completo para el finetune bank).
4. Entrenar la variante "synthetic-blend" igual que (3), agregando el paso de
   mezcla real/sintetica antes de construir cada batch
   (`dermssl.circle.synthetic_blend_mask` + `dermssl.circle.shift_skin_tone`
   aplicado a las imagenes seleccionadas como sinteticas).
5. Correr `metrics.stratified_summary()` sobre las predicciones de validacion de
   cada una de las tres variantes.
6. Pasar los tres resultados a `metrics.comparison_report()` para obtener la
   brecha por variante y la mejora relativa - el criterio de aceptacion primario
   de la tesis (proposal.md, "Success Criteria").
7. Empezar con `--smoke` (pocas epocas, subconjunto chico) para validar el
   wiring antes de un run real, igual que el precedente de
   `scripts/kaggle_fst_manifest.py` y `scripts/pretrain.py`.

## Que NO esta hecho todavia

- No existe un script `scripts/kaggle_circle_stage2.py` dedicado (design.md's
  File Changes table para esta fase solo lista `finetune/circle_torch.py`); un
  humano debe ensamblar el loop de entrenamiento real a partir de las
  primitivas de este modulo cuando llegue el momento, siguiendo el patron de
  `scripts/kaggle_fst_manifest.py` / `finetune/torch_run.py`.
- No hay datos reales, no hay numeros reales, no hay mejora de fairness medida
  todavia en ningun lado de este repositorio para Stage 2.
