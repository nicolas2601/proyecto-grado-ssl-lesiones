# 62. Validación CIELAB de la línea de segmentación v8 y cierre de la procedencia de IMA++

Fecha: 2026-10-31. Rama `experiment/modular-audit-v8`.
Alcance: cerrar los dos bloqueos que impedían declarar validada la línea de segmentación
v8 (runs 019-023). No se entrenó, no se lanzó ninguna corrida, no se cambió ningún gate.
Acceso a UNABIA por SSH de solo lectura (`unabia`, host `jupyter-nmoreno534`).

Antecedente: `docs/estado/seg-imapp-holdout.reporte.md` dejó dos bloqueos abiertos.
Este documento los resuelve y declara con precisión qué queda fuera.

---

## 1. Bloqueo CIELAB: ningún run era apto sin filtro por caso

El reporte previo midió contaminación p95 del pool de piel entre 0,3520 y 0,3955 en los
cinco runs, y concluyó que **ninguno es apto para alimentar CIRCLe sin un filtro previo
por caso**. Esa conclusión era correcta y sigue en pie. Lo que faltaba era medir el filtro:
cuánto cuesta aplicarlo y si el costo está repartido o concentrado.

El filtro se audita con `scripts/audit_cielab_case_filter.py`, que reusa las definiciones
textuales de `scripts/compare_seed42_vs_seed43.py`. Un caso es admisible cuando **no es
degenerado** (área predicha fuera de `[0,005; 0,95]`) **y** su contaminación del pool de
piel no supera `tau`.

`tau = 0,20` está **registrado antes del dato** en `docs/estado/config-modelos-investigacion.reporte.md`
(propuesta P4). No se ajustó contra estos resultados.

### Resultado con `tau = 0,20`

| Run | N | Retenidos | Retención | Descarte degenerado | Descarte contaminación | p95 antes | p95 después | máx. después |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 019 | 2.594 | 2.325 | 89,6 % | 1 | 268 | 0,3538 | 0,1349 | 0,1998 |
| 020 | 2.594 | 2.297 | 88,6 % | 7 | 290 | 0,3955 | 0,1408 | 0,1993 |
| 021 | 2.594 | 2.318 | 89,4 % | 1 | 275 | 0,3693 | 0,1361 | 0,1995 |
| 022 | 2.594 | 2.321 | 89,5 % | 3 | 270 | 0,3557 | 0,1400 | 0,1994 |
| 023 | 2.594 | 2.339 | 90,2 % | 5 | 250 | 0,3520 | 0,1298 | 0,1993 |

**El filtro funciona y es barato en volumen:** retiene entre 88,6 % y 90,2 % de los casos
y baja la contaminación p95 de 0,35-0,40 a 0,13-0,14. El máximo después del filtro es
≤ 0,20 por construcción, así que no queda cola.

### El costo NO está repartido: se concentra en las lesiones grandes

| Run | `<1%` | `1-2%` | `2-5%` | `5-10%` | `>=10%` |
|---|---|---|---|---|---|
| 019 | 48/48 (100 %) | 149/149 (100 %) | 427/427 (100 %) | 432/432 (100 %) | **1.269/1.538 (83 %)** |
| 020 | 47/48 (98 %) | 149/149 (100 %) | 427/427 (100 %) | 430/432 (100 %) | **1.244/1.538 (81 %)** |
| 021 | 48/48 (100 %) | 149/149 (100 %) | 427/427 (100 %) | 431/432 (100 %) | **1.263/1.538 (82 %)** |
| 022 | 48/48 (100 %) | 149/149 (100 %) | 427/427 (100 %) | 431/432 (100 %) | **1.266/1.538 (82 %)** |
| 023 | 48/48 (100 %) | 149/149 (100 %) | 427/427 (100 %) | 430/432 (100 %) | **1.285/1.538 (84 %)** |

Prácticamente **todo el descarte cae en el estrato `>=10%`**, que concentra 1.538 de 2.594
casos (59,3 % del conjunto). Los otros cuatro estratos retienen 98-100 %.

El mecanismo es aritmético, no un defecto del modelo: a mayor área de lesión, el pool de
piel (`1 - área predicha`) es más chico y la masa de falsos negativos pesa más sobre él,
así que la contaminación relativa sube. No es evitable subiendo el Dice.

**Consecuencia que hay que declarar, no esconder:** el pool de tono resultante
**subrepresenta las lesiones grandes en unos 17 puntos**. Si CIRCLe estima estadísticos de
tono sobre ese pool, los estima sobre una cohorte sesgada hacia lesiones chicas. Para
estimar tono de piel eso es defendible —se busca piel limpia—, pero es un sesgo medido y
tiene que ir declarado en la tabla, no en una nota al pie.

### Sensibilidad a `tau` (análisis de sensibilidad, no selección)

Barrido sobre 021. `tau` **no** se re-eligió con esta tabla; se reporta para mostrar que la
elección no cae en un borde.

| `tau` | Retención | p95 después | máx. después |
|---:|---:|---:|---:|
| 0,15 | 86,1 % | 0,1085 | 0,1492 |
| **0,20** | **89,4 %** | **0,1361** | **0,1995** |
| 0,25 | 91,8 % | 0,1630 | 0,2483 |
| 0,30 | 93,4 % | 0,1894 | 0,2998 |
| 0,40 | 95,3 % | 0,2211 | 0,3932 |

El intercambio es suave y **no hay codo**: entre 0,15 y 0,40 la retención sube de forma
monótona sin salto. No existe un `tau` que los datos señalen como óptimo, así que
elegirlo por los datos sería arbitrario disfrazado de empírico. Se mantiene el valor
pre-registrado.

### Verificación de que la auditoría reproduce la tabla publicada

La columna «p95 antes» y los conteos de máscaras degeneradas (1, 7, 1, 3, 5) coinciden
**exactamente** con la tabla de seguridad CIELAB de `seg-imapp-holdout.reporte.md`
sección 3, calculada por otro script y en otro momento. Además el invariante
`dice == 2*TP/(área_gt + área_pred)` se asertó caso por caso y **pasó en los 12.970 casos**
(5 runs x 2.594) con tolerancia 1e-4.

---

## 2. Bloqueo de procedencia: el cohorte IMA++ del run 023

El reporte previo concluyó que «ningún recibo acredita IMA++ en el run 023» y que por eso
023 no era auditable como ablación. **Esa conclusión era parcialmente incorrecta y queda
corregida:** el manifiesto existe y está hasheado. Lo que falta es un eslabón de registro,
no el manifiesto.

### Lo que sí existe

| Hecho | Evidencia |
|---|---|
| Manifiesto congelado del cohorte, 100 entradas | `ima_plusplus/pilot_100_cohort_frozen_manifest.json` en UNABIA |
| `sha256` del cohorte | `a1248973921a8c39294440ec980a94f528cd958af553201ce67aeb8b54c443e1` |
| Copia en el paquete de preparación, **byte-idéntica** | `experiments/023-ph2-ima100-prepared-20261028/inputs/ima100.frozen.json`, mismo sha256, mismos 75.989 bytes |
| `sha256` por imagen y por máscara, anotador, herramienta y nivel de destreza | campos `img_sha256`, `mask_sha256`, `annotator`, `tool`, `skill_level` de cada entrada |
| Recibo de preparación con hashes de config | `experiments/023-ph2-ima100-prepared-20261028/verification.json` |

### El gate decisivo: solapamiento con el split de evaluación

**Solapamiento = 0 casos.** Verificado contra los 2.594 IDs de evaluación de 023, por dos
vías: ID completo y sufijo numérico normalizado. La segunda vía es obligatoria porque una
comparación por sufijo de 7 dígitos ya produjo 1.334 falsos positivos en este proyecto
(ver `docs/46`); acá da cero por ambas.

Se verificó sobre el manifiesto **realmente cargado por el entrenamiento**, no sobre la
copia del paquete. Importa porque son dos rutas distintas y el lanzador no pasa
`--ima-manifest`: `scripts/train_dermlip_semi_023.py:1516` tiene como valor por omisión
`ima_plusplus/pilot_100_cohort_frozen_manifest.json`. Los dos archivos resultaron
byte-idénticos, así que la distinción no cambia el resultado, pero la verificación se hizo
sobre el que corrió.

El script además **no puede sustituir el cohorte en silencio**: exige exactamente 100
casos (`:1134`) y levanta `FileNotFoundError` por cada imagen o máscara ausente
(`:1141`, `:1143`).

**Por lo tanto el Dice medio de 84,16 % de 023 sobrevive como cifra de evaluación**, sujeto
al caveat de independencia por paciente de la sección 3, que es anterior e independiente.

### El eslabón que sigue abierto

Los recibos propios del run **no registran el cohorte**. Verificado con búsqueda exacta: la
única aparición de `imapp` en `runs/dermlip-semi-supervised-v8-...-abl023-.../selection/*.json`
es el nombre del directorio. `input_validation.json` declara `ph2_manifest_sha256`,
`ph2_source_artifact_sha256`, `unlabeled_pool`, `dermlip_model`,
`dermlip_model_manifest_sha256` y `patient_independence_accredited`, y **ningún campo de
IMA++**. El `sha256` del cohorte no aparece en ningún archivo fuera del paquete de
preparación, y el config que usó el run (`e86d2232…`) tampoco lo referencia.

La ligazón entre la corrida y el cohorte se sostiene hoy en el valor por omisión del
script, no en un recibo. **Es un defecto de registro, no un defecto científico**, y se
cierra agregando un campo `ima_cohort_sha256` a `input_validation.json` en la próxima
corrida. Mientras no exista ese campo, la trazabilidad de 023 se acredita citando este
documento.

### Un hallazgo que refuta una premisa previa

`patient_id` y `lesion_id` están **nulos en 99 de las 100 entradas**. La justificación
registrada para traer IMA++ era precisamente que aportaba `patient_id`/`lesion_id` para
acreditar la independencia por paciente que sigue en `false`. **Para este cohorte de 100
imágenes, IMA++ no aporta eso.** Si se quiere usar IMA++ para cerrar independencia por
paciente, hay que verificar antes que los metadatos de paciente estén poblados en el
dataset completo; no alcanza con que el dataset los declare en su publicación.

---

## 3. Lo que esta validación NO cierra

1. **Independencia por paciente.** Sigue en `patient_independence_accredited: false`, con
   `run_type: dermlip_guided_semi_supervised_case_disjoint_patient_unverified` y el
   manifiesto de los 2.594 como `post_run_evaluation_only_candidate`. Ninguna cifra de los
   2.594 puede presentarse como evaluación independiente acreditada. El filtro CIELAB no
   toca este bloqueo.
2. **Independencia visual del pool completo.** `verification.json` lo declara explícito:
   `visual_screening_scope` cubre pHash contra el benchmark, no el pool completo.
   `known_groups.metadata_complete` es `False`.
3. **El copy-paste sigue sin acreditar.** Su efecto se invierte entre semillas en los dos
   estratos mayores. El filtro CIELAB no lo rescata: 020 sigue siendo el peor de los cinco
   también después de filtrar (menor retención, 81 %, y 7 degeneradas).
4. **El benchmark PH2 de 3 grupos de 023 no existe.** Requiere una corrida de evaluación
   nueva, y eso exige autorización explícita por `docs/RUN_CATALOG.md` paso 6.
5. **Dos semillas no alcanzan** para afirmar nada sobre el estrato `>=10%`, que es donde
   está el 59,3 % de los casos y donde el filtro descarta el 17 %.

---

## 4. Veredicto

La línea de segmentación v8 **queda validada para alimentar CIRCLe bajo filtro por caso
con `tau = 0,20`**, con estas tres condiciones declaradas:

1. Se excluyen las máscaras degeneradas y los casos con contaminación > 0,20. Entre 9,8 %
   y 11,4 % del conjunto según el run.
2. Se declara el sesgo de composición: el pool retenido subrepresenta el estrato `>=10%`
   en unos 17 puntos.
3. El resultado se reporta como equidad **entre terciles de luminancia**, nunca como
   mejora en piel oscura (límite de alcance ya fijado en P4).

**Run recomendado: 021.** Mejor balance antes del filtro (1 degenerada, lesión no
protegida 0,1499) y retención de 89,4 % después. 023 retiene algo más (90,2 %) y tiene la
mejor contaminación, pero trae 5 degeneradas contra 1, y su benchmark PH2 de 3 grupos no
existe. **020 queda descartado** en los dos regímenes.

Lo que falta para «lista» en sentido pleno sigue siendo la independencia por paciente,
que no se cierra con corridas: necesita metadatos de paciente poblados y el cruce contra
el inventario histórico.

---

## Artefactos bajados al repo en este trabajo

Con los nombres definitivos del inventario de `seg-imapp-holdout.reporte.md` sección 6.

| Destino | Contenido |
|---|---|
| `resultados/preparacion_v8/dermlip_isic2018_eval_2594_019_resnet34_flatlr_s42_20261027_008/` | `cases.jsonl` (2.594) + `evaluation_report.json` |
| `resultados/preparacion_v8/dermlip_isic2018_eval_2594_020_resnet34_ph2copypaste_s42_20261027_009/` | ídem |
| `resultados/preparacion_v8/dermlip_isic2018_eval_2594_021_resnet34_flatlr_s43_20261028_010/` | ídem |
| `resultados/preparacion_v8/dermlip_isic2018_eval_2594_022_resnet34_ph2copypaste_s43_20261028_011/` | ídem |
| `resultados/preparacion_v8/dermlip_isic2018_eval_2594_023_resnet34_ph2cp_imapp100img_s42_20261031_012/` | ídem |
| `resultados/segmentacion/v8/ph2_val_3groups_benchmark_20261027/` | 4 `eval_*_3groups.json` + 2 `comparacion_*` |
| `resultados/segmentacion/v8/cielab_case_filter_tau020.json` | salida de la auditoría de este documento |

El sufijo `imapp100` se renombró a `imapp100img` a propósito: `imapp100` se lee como
«100 %» y ya produjo exactamente ese error. Son 100 imágenes de 14.967, el 0,67 %.
