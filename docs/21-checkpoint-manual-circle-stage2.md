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
`circle_torch.py`). Tambien existe `scripts/kaggle_circle_stage2.py`, pero
**nadie corrio todavia** un entrenamiento real de Stage 2. Por proposal.md:
"Stage 2 is gated on Stage 1 showing measurable bias" - y
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

## Bundle de entrada

El runner recibe un `.npz` con features del mismo backbone congelado. Debe
contener `x_train_original`, `x_train_shifted`, `y_train`, `x_val`, `y_val`,
`fst_buckets_val` y el escalar `backbone_frozen=True`. Las features desplazadas
deben provenir de imagenes transformadas con `circle.shift_skin_tone` y las
mascaras usadas en Stage 1. El runner rechaza dimensiones desalineadas y bundles
que no certifican el backbone congelado.

## Ejecucion en GPU

1. Confirmar que `fst_manifest.csv` (PR3) existe y que
   `gates.fairness_gate(stratified_summary(...))` en el run de Stage 1 real
   reporta `FAIL` (o una brecha por encima del umbral que se considere "vale la
   pena mitigar").
2. Ejecutar primero smoke sobre datos reales. El resultado queda marcado
   `valid_for_thesis=false`:

   ```bash
   python scripts/kaggle_circle_stage2.py \
       --feature-bundle circle_features.npz \
       --stage1-gate RESULTS_SUMMARY_v5.json \
       --output-json circle_stage2_smoke.json --smoke
   ```

3. Si smoke termina bien, ejecutar la corrida completa quitando `--smoke` y
   fijando epocas, semilla y numero de remuestreos bootstrap. El runner entrena
   baseline, regularizacion y blend sintetico con el mismo probe lineal,
   calcula `stratified_summary()` y produce brechas, mejoras relativas e IC95
   bootstrap pareados mediante `comparison_report()`.

## Que NO esta hecho todavia

- No hay datos reales, no hay numeros reales, no hay mejora de fairness medida
  todavia en ningun lado de este repositorio para Stage 2.
