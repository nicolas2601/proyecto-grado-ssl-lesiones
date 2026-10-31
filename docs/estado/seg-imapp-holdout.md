# Sesion: seg-imapp-holdout  |  2026-10-30 (corregido 2026-10-31)

> **Aviso.** La version anterior de este contrato describia Run 008 como el ultimo trabajo de
> segmentacion. **Era incorrecto.** El estado real son los runs 019 a 023. Lo de abajo reemplaza
> por completo esa version.

## Tarea

Bajar de UNABIA los resultados de los runs 019, 020, 021 y 022, armar la tabla pareada por
semilla con las metricas de seguridad CIELAB, y verificar el estado del run 023.

Acceso de **solo lectura** a UNABIA autorizado para esta tarea. No entrenas, no lanzas, no
modificas nada remoto.

## Estado real, verificado el 2026-10-31

Diseno pareado, un solo factor por semilla:

| Semilla | Referencia | Ablacion |
|---|---|---|
| 42 | **019** | **020** (copy-paste) |
| 43 | **021** | **022** (copy-paste) |

**023** es la ablacion siguiente: ResNet34 + PH2 copy-paste + **IMA++ N=100 imagenes**.

> **Correccion del 2026-10-31.** La version anterior de este contrato decia "IMA++ al 100 %".
> **Era un error de dos ordenes de magnitud.** El sufijo `imapp100` del lanzador significa
> **N = 100 imagenes**, no el 100 % del dataset. Evidencia: `runs/dermlip_abl023_console.log`
> linea 2, *"Loaded Labeled datasets: PH2 (156 train, 39 val), IMA++ audited cohort
> (100 images)"*. Son 100 de 14.967, el **0,67 %**. El `pipeline.log` lo nombra bien:
> "IMA++ N=100 real (s42)". No propagar la cifra vieja a ningun documento.

Lanzador local
`launch_train_023_abl_resnet34_ph2cp_imapp100.sh`, config
`dermlip.selection.runtime.20261028.023.json` (vive en UNABIA, no en el repo). Parametros:
`--backbone resnet34 --flat-lr --superpixel-affinity-weight 0.1
--pseudo-label-threshold-high 0.65 --pseudo-label-threshold-low 0.35 --copy-paste-prob 0.5
--device cuda:1 --execute`.

La usuaria cree que 023 ya corrio en UNABIA. **Verificarlo es parte de esta tarea.** No asumirlo.

El pipeline de evaluacion es `scripts/auto_eval_seed43_pipeline.py` y hace cinco pasos: espera
que terminen los entrenamientos (300 epocas), corre el benchmark PH2 val de 3 grupos
(Natural 39, Sintetico 75, Fondos reparados 25), corre inferencia cruda sobre ISIC 2018 (2.594)
mas evaluacion morfologica pareada, genera metricas estratificadas por tamano y de seguridad
CIELAB, y contrasta 021 vs 022 contra 019 vs 020.

Las metricas CIELAB que ya calcula `scripts/compare_seed42_vs_seed43.py` miden si la mascara
sirve para alimentar CIRCLe: `cielab_median_avail_skin_frac`, `cielab_p05_avail_skin_frac`,
`cielab_mean_unprotected_lesion_frac_of_gt`, `cielab_p95_skin_pool_lesion_contamination` y
`cielab_circle_degenerate_mask_count` (mascara degenerada = area predicha < 0,005 o > 0,95).

## Entra

Local:
- `scripts/compare_seed42_vs_seed43.py` y `scripts/auto_eval_seed43_pipeline.py`
- `launch_train_023_abl_resnet34_ph2cp_imapp100.sh`
- `docs/ESTADO.md`, seccion de segmentacion
- `docs/RUN_CATALOG.md` — procedimiento vigente por familia de corrida
- Archivos `scratch_*` de la raiz

Remoto en UNABIA, **solo lectura**:
- `/home/jovyan/work/dermlip_slic_v8/evaluations/paired-raw-morph-20261027/{021,022}/*/cruda/cases.jsonl`
- `/home/jovyan/work/dermlip_slic_v8/evaluations/ph2_val_3groups_benchmark_20261027/eval_0{19,20,21,22}_*_stratified.json`
- `/home/jovyan/work/dermlip_slic_v8/runs/` — estado de 023
- `/home/jovyan/work/dermlip_slic_v8/runs/dermlip_abl023_console.log`
- El config `dermlip.selection.runtime.20261028.023.json`

Usa el agente `operador-unabia`, que conoce el procedimiento de acceso.

## Sale

Un unico archivo: `docs/estado/seg-imapp-holdout.reporte.md`, con:

1. **Tabla pareada por semilla**, estratificada por tamano de lesion (<1 %, 1-2 %, 2-5 %,
   5-10 %, >=10 %): Dice, precision, razon de area mediana y conteo de ceros, para 019, 020,
   021 y 022. Mas la fila global sobre los 2.594.
2. **Delta del copy-paste y su reproducibilidad**: Delta(020-019) contra Delta(021-022... ojo,
   el orden correcto es 022-021). Declarar si el efecto se reproduce entre semillas o no, y si
   el signo es consistente por estrato.
3. **Tabla de seguridad CIELAB** para los cuatro runs, con las cinco metricas. Veredicto
   explicito: cual de los cuatro produce mascaras aptas para alimentar CIRCLe y por que.
4. **Estado del run 023**: corrio o no, con la evidencia exacta (log, archivos de salida,
   proceso vivo). Si corrio, sus metricas. Si no, que falta para lanzarlo.
5. **Benchmark PH2 de 3 grupos**: resultados de Natural 39, Sintetico 75 y Fondos reparados 25.
6. **Que inventario de artefactos conviene bajar al repo** y con que nombre definitivo, para que
   estos resultados dejen de vivir solo en el cluster.

## Criterios (binarios)

1. Toda cifra sale de un archivo real leido en UNABIA, citado con su ruta completa.
2. El estado de 023 se afirma con evidencia, nunca por suposicion. Si no se puede determinar, se
   declara NO VERIFICADO con el comando que lo comprobaria.
3. La tabla pareada respeta el orden referencia -> ablacion y declara el signo del delta.
4. El veredicto CIELAB dice explicitamente si la mascara sirve para CIRCLe, con el umbral que lo
   justifica.
5. Se distingue lo verificado de lo no verificado en cada punto.
6. No se marca ningun gate como listo.

## Fuera de alcance

- **No entrenar, no evaluar, no lanzar ninguna corrida.** Solo lectura.
- No modificar nada en UNABIA: ni configs, ni gates, ni archivos de resultados.
- No cambiar `execution_enabled`, `launch_ready` ni `training_started`.
- No renombrar, mover ni borrar los archivos `scratch_*`: solo proponer nombres.
- No commitear nada.
- No tocar la linea de clasificacion: es territorio de Nicolas.
- No tocar la documentacion del paper ni los diagramas: estan congelados.
- Si el cluster esta ocupado con una corrida de Nicolas, **no interferir**: reportarlo y parar.

## Densidad: caveman alto

Escribis denso. Sin relleno, sin cortesias, sin preambulos, sin resumir lo que acabas de decir.

- Fuera: "basicamente", "en realidad", "cabe destacar", "es importante notar", "vale la pena".
- Fuera: "claro", "por supuesto", "con gusto", "excelente pregunta".
- Fuera el hedging: o lo afirmas con evidencia, o lo marcas NO VERIFICADO. No hay punto medio.
- No anuncies lo que vas a hacer. Hacelo.
- No cierres con un resumen de lo que ya esta escrito arriba.

**No se recorta**: el porque tecnico, las cifras, los nombres de archivo, los comandos, los
mensajes de error literales, y la distincion verificado / no verificado.

Tablas y listas antes que parrafos cuando el dato es estructurado. Espanol neutral, sin emojis.

---
## Reporte  (lo completa la sesion al terminar)

Entregado: `docs/estado/seg-imapp-holdout.reporte.md` (2026-10-31). Detalle completo de
verificado / no verificado / bloqueos en sus secciones 0 y «Reporte». Resumen:

**Verificado.** Acceso por SSH (alias `unabia`, host `jupyter-nmoreno534`). Las seis entregas
del contrato estan cubiertas salvo el benchmark de 3 grupos de 023, que no existe. Todas las
cifras se recalcularon en local desde los `cases.jsonl` crudos y coinciden con los JSON
estratificados del cluster con **delta maximo 0,000e+00** (13 metricas x 6 ambitos x 5 runs).
El invariante `dice == 2*TP/(gt+pred)` cierra con error maximo 4,8e-12 en 12.970 casos.
**023 corrio y completo**: `summary.json` `status: completed`, `Epoch 300/300`,
`Training completed in 11085.0 seconds`. GPUs libres al observar; no se interfirio con la
linea de Nicolas.

**Hallazgos que corrigen este contrato.**
1. La version previa de este contrato decia «IMA++ al 100 %». Era **incorrecto**:
   `dermlip_abl023_console.log` linea 2 registra `IMA++ audited cohort (100 images)` — 100
   imagenes, el 0,67 % de las 14.967. El sufijo `imapp100` significa `N=100`, no «100 %».
   Ya corregido en la linea 24 y en el aviso de las lineas 26-31.
2. El efecto del copy-paste **no se reproduce** entre semillas en `5-10%` ni en `>=10%` (ahi
   esta el 59,3 % de los casos). El +0,24 / +0,46 global es cancelacion, no ausencia de efecto.
3. El grupo 2 sintetico es **circular** (benchmark hecho con copy-paste para medir copy-paste:
   +34,7 puntos contra +0,24 reales) y su n efectivo es **25**, no 75, por declaracion del
   propio archivo.
4. `scripts/compare_seed42_vs_seed43.py` tiene dos defectos: `np.median` sobre `area_ratio`
   puede dar `NaN` (latente, no activo) y no valida el invariante de Dice.

**No verificado.** Que 100 imagenes componen el cohorte IMA++ de 023 y si solapan con el split
de evaluacion: **ningun recibo de procedencia lista un artefacto de IMA++**. Tampoco los
`morfologica/cases.jsonl`, ni el run 018, ni las cifras de 12.235 / 11.807 candidatos.

**Bloqueos.** 023 no es auditable como ablacion de IMA++ sin congelar ese cohorte con hash y
recibo. Ningun run entrega un pool de piel apto para CIRCLe sin filtro por caso (contaminacion
p95 entre 0,352 y 0,396). `patient_independence_accredited: false` en los cinco runs.

**Alcance respetado.** No se entreno, no se lanzo nada, no se modifico ningun archivo remoto,
no se renombro ni movio ningun `scratch_*`, **ningun gate fue modificado**. No se commiteo:
queda un conflicto abierto entre la linea 97 de este contrato y la instruccion directa de la
usuaria de commitear al terminar.
