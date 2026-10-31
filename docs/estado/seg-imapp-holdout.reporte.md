# Reporte: runs pareados 019-023 y estado de IMA++ en segmentacion

Sesion `impl`, contrato `docs/estado/seg-imapp-holdout.md` (version corregida 2026-10-31).
Fecha de observacion: **2026-10-31**, 15:38-16:20 UTC. Acceso a UNABIA por SSH, **solo lectura**.
No se entreno, no se lanzo nada, no se modifico ningun archivo remoto, no se cambio ningun gate.

Copia local de trabajo de los artefactos bajados:
`C:\Users\mpaul\AppData\Local\Temp\unabia-seg-20261031\` (6,4 MB, 16 archivos).
Script de recalculo: `recompute.py` en esa misma carpeta. Salida: `recomputed.json`.

---

## 0. Integridad y reproducibilidad de las cifras

**VERIFICADO.** Todas las cifras de este reporte se recalcularon en local desde los
`cases.jsonl` crudos y se contrastaron contra los JSON estratificados que ya existian en
UNABIA. Coinciden exactamente.

| Run | N | IDs unicos | GT vacio | `area_ratio` NaN | err. max del invariante de Dice | Violaciones > 1e-4 | max abs delta vs JSON del cluster |
|---|---:|---:|---:|---:|---:|---:|---:|
| 019 | 2.594 | 2.594 | 0 | 0 | 4,736e-12 | 0 | **0,000e+00** |
| 020 | 2.594 | 2.594 | 0 | 0 | 4,845e-12 | 0 | **0,000e+00** |
| 021 | 2.594 | 2.594 | 0 | 0 | 4,113e-12 | 0 | **0,000e+00** |
| 022 | 2.594 | 2.594 | 0 | 0 | 4,727e-12 | 0 | **0,000e+00** |
| 023 | 2.594 | 2.594 | 0 | 0 | 4,205e-12 | 0 | **0,000e+00** |

El invariante es `dice == 2*TP/(area_gt + area_pred)`. Importa porque
`scripts/compare_seed42_vs_seed43.py:24` **no lee** el area predicha del archivo: la
reconstruye como `sens*gt + (1-spec)*(1-gt)`. De esa reconstruccion cuelgan precision, razon de
area y las cinco metricas CIELAB. El invariante cierra con error maximo 4,8e-12, asi que la
reconstruccion es correcta y el redondeo del `jsonl` no la degrada. El script original no chequea
esto; se agrego en el recalculo local.

### Dos defectos del script de referencia

1. **`np.median` sobre `area_ratio` puede devolver `NaN`** — `compare_seed42_vs_seed43.py:29` y
   `:67`. `area_ratio` es `NaN` cuando `gt_area == 0`, y el estrato `<1%` esta definido como
   `0.0 <= gt < 0.01`, asi que admite mascaras GT vacias. Una sola basta para que la celda
   "razon de area mediana" del estrato salga `NaN` sin aviso.
   **Latente, no activo:** los cinco runs tienen 0 casos de GT vacio sobre 2.594, columna
   "GT vacio" de la tabla de arriba. La correccion es `np.nanmedian`.
2. **No valida el invariante de Dice.** Un campo corrupto o un redondeo agresivo en
   `sensitivity`/`specificity` pasaria sin deteccion y contaminaria toda la tabla publicada.

Ninguno de los dos invalida las cifras de hoy. Los dos deben corregirse antes de que el script
genere una tabla de tesis.

### Caveat que afecta a todo el reporte

`patient_independence_accredited: false` en el config de 023 y en `summary.json`. El
`run_type` registrado es literalmente
`dermlip_guided_semi_supervised_case_disjoint_patient_unverified`: disjunto por caso,
**no verificado por paciente**. Y el manifiesto de evaluacion de los 2.594 declara
`"role": "post_run_evaluation_only_candidate"` — es un candidato, no un holdout acreditado.
Deduplicar por caso no equivale a independencia por paciente. Ninguna cifra de los 2.594 de
este reporte puede presentarse como evaluacion independiente acreditada.

---

## 1. Tabla pareada por semilla, estratificada por tamano de lesion

Diseno: un solo factor (copy-paste PH2), dos semillas. Orden **referencia -> ablacion**.

| Semilla | Referencia | Ablacion |
|---|---|---|
| 42 | 019 `abl019-resnet34-flatlr` | 020 `abl020-resnet34-ph2copypaste` |
| 43 | 021 `abl019-resnet34-flatlr-seed43` | 022 `abl020-resnet34-ph2copypaste-seed43` |

### Dice medio

| Estrato | N | Dice 019 | Dice 020 | Delta s42 | Dice 021 | Dice 022 | Delta s43 | Signo |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| <1 % | 48 | 45,86 % | 56,75 % | **+10,89** | 39,03 % | 44,94 % | **+5,91** | consistente |
| 1-2 % | 149 | 72,64 % | 78,57 % | **+5,92** | 65,57 % | 71,05 % | **+5,48** | consistente |
| 2-5 % | 427 | 79,49 % | 82,56 % | **+3,07** | 79,06 % | 79,21 % | +0,15 | consistente |
| 5-10 % | 432 | 83,73 % | 84,22 % | +0,49 | 85,06 % | 84,92 % | −0,14 | **SE INVIERTE** |
| >= 10 % | 1.538 | 85,24 % | 83,73 % | **−1,50** | 85,20 % | 85,26 % | +0,06 | **SE INVIERTE** |
| **Global** | **2.594** | **82,59 %** | **82,82 %** | **+0,24** | **82,18 %** | **82,64 %** | **+0,46** | consistente |

### Precision media, razon de area mediana y conteo de ceros

| Estrato | Prec 019 | Prec 020 | Prec 021 | Prec 022 | AR 019 | AR 020 | AR 021 | AR 022 | Ceros 019/020/021/022 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| <1 % | 34,85 % | 49,71 % | 26,81 % | 34,59 % | 4,54x | 1,50x | 3,64x | 4,31x | 2 / 4 / 1 / 3 |
| 1-2 % | 61,87 % | 73,69 % | 51,71 % | 61,21 % | 1,43x | 1,18x | 1,80x | 1,43x | 0 / 0 / 0 / 0 |
| 2-5 % | 74,19 % | 80,07 % | 71,40 % | 72,82 % | 1,13x | 1,08x | 1,21x | 1,17x | 2 / 4 / 3 / 3 |
| 5-10 % | 85,02 % | 87,09 % | 85,73 % | 85,23 % | 1,01x | 0,98x | 1,03x | 1,04x | 1 / 4 / 1 / 2 |
| >= 10 % | 95,87 % | 96,60 % | 96,09 % | 95,63 % | 0,89x | 0,86x | 0,89x | 0,89x | 0 / 3 / 1 / 1 |
| **Global** | **87,41 %** | **90,11 %** | **86,47 %** | **87,04 %** | **0,96x** | **0,93x** | **0,98x** | **0,97x** | **5 / 15 / 6 / 9** |

Fuentes, rutas completas en UNABIA:

- `/home/jovyan/work/dermlip_slic_v8/evaluations/paired-raw-morph-20261027/019/0ae35137b699d60f/cruda/cases.jsonl`
- `/home/jovyan/work/dermlip_slic_v8/evaluations/paired-raw-morph-20261027/020/55b3c0149906db8d/cruda/cases.jsonl`
- `/home/jovyan/work/dermlip_slic_v8/evaluations/paired-raw-morph-20261027/021/c0167b4735bb35a1/cruda/cases.jsonl`
- `/home/jovyan/work/dermlip_slic_v8/evaluations/paired-raw-morph-20261027/022/02859d9e0f46fab5/cruda/cases.jsonl`

El hash del directorio de evaluacion es el prefijo de 16 hex del `checkpoint_sha256` del
`best_teacher_model.pt` correspondiente. Verificado contra `checkpoint_sha256` en los
`eval_0XX_*_3groups.json`: `0ae35137b699d60f...` para 019, `55b3c0149906db8d...` para 020,
`c0167b4735bb35a1...` para 021, `02859d9e0f46fab5...` para 022. Esto ata cada `cases.jsonl` a un
checkpoint concreto.

---

## 2. Delta del copy-paste y su reproducibilidad

**El efecto NO se reproduce entre semillas en los dos estratos que concentran la masa de casos.**

| Resultado | Evidencia |
|---|---|
| Se reproduce en lesiones chicas | `<1%`, `1-2%` y `2-5%`: delta positivo en ambas semillas, y de magnitud grande en los dos primeros (+10,89/+5,91 y +5,92/+5,48) |
| **Se invierte en lesiones grandes** | `5-10%`: +0,49 (s42) contra −0,14 (s43). `>=10%`: **−1,50 (s42) contra +0,06 (s43)** |
| El global es un promedio enganoso | +0,24 % y +0,46 % globales son la cancelacion de una ganancia grande en lesiones chicas contra una perdida en lesiones grandes. El estrato `>=10%` tiene 1.538 de 2.594 casos, el 59,3 % |
| **El copy-paste empeora los fallos catastroficos en ambas semillas** | Ceros globales: 019 -> 020 pasa de **5 a 15** (x3). 021 -> 022 pasa de **6 a 9**. Direccion consistente en las dos semillas, y el Dice medio lo oculta |
| La correccion de sobre-segmentacion tampoco reproduce | Estrato `<1%`, razon de area: s42 mejora de 4,54x a **1,50x**; s43 **empeora** de 3,64x a 4,31x |

Lectura: con dos semillas, el unico efecto del copy-paste que aguanta es **mejora en lesiones
pequenas (<5 % del area) a cambio de mas fallos de solapamiento cero**. El signo en lesiones
grandes depende de la semilla, asi que no se puede afirmar direccion alli. Dos semillas no
alcanzan para una afirmacion de reproducibilidad; alcanzan para mostrar que el efecto global
cercano a cero no es informativo.

---

## 3. Tabla de seguridad CIELAB y veredicto para CIRCLe

Global sobre los 2.594. Las cinco metricas del contrato, mas la mediana de lesion no protegida.
`mascara degenerada = area predicha < 0,005 o > 0,95`.

| Run | Rol | med. piel disponible | p05 piel disponible | media lesion no protegida | p95 contaminacion del pool de piel | **degeneradas** |
|---|---|---:|---:|---:|---:|---:|
| 019 | ref. s42 | 0,8675 | 0,4830 | 0,1568 | 0,3538 | **1** |
| 020 | copy-paste s42 | 0,8790 | 0,5038 | **0,1811** | **0,3955** | **7** |
| 021 | ref. s43 | 0,8710 | 0,4894 | 0,1499 | 0,3693 | **1** |
| 022 | copy-paste s43 | 0,8689 | 0,4907 | 0,1525 | 0,3557 | **3** |
| 023 | ResNet34 + CP + IMA++ 100 img | 0,8655 | 0,4823 | **0,1181** | **0,3520** | 5 |

**Veredicto: 021 es la mas apta para alimentar CIRCLe; 020 es la menos apta y queda descartada.**

Justificacion por umbral:

- **Mascara degenerada es un fallo duro, no una degradacion.** Con area predicha < 0,005 no hay
  lesion de la cual separar el pool de piel; con > 0,95 no queda piel que muestrear. Esos casos
  no se le pueden pasar a CIRCLe: hay que excluirlos o CIRCLe muestrea ruido. Conteos: 019 y 021
  con 1 de 2.594 (0,04 %), 022 con 3, 023 con 5, **020 con 7 (0,27 %, x7 contra su referencia)**.
- **020 es peor que su propia referencia en las tres metricas de fuga**: lesion no protegida
  0,1811 contra 0,1568, contaminacion p95 0,3955 contra 0,3538, degeneradas 7 contra 1. El
  copy-paste de la semilla 42 degrada la mascara para CIRCLe en todos los ejes relevantes.
- **021 tiene el mejor balance**: 1 degenerada, lesion no protegida 0,1499 (la mejor entre los
  cuatro runs pareados) y contaminacion p95 0,3693.
- **023 gana en fuga pero pierde en degeneradas**: mejor lesion no protegida de los cinco
  (0,1181) y mejor contaminacion p95 (0,3520), pero 5 degeneradas contra 1 de 019.

**Caveat que aplica a los cinco runs, y es el mas importante de esta seccion.** La contaminacion
p95 esta entre 0,3520 y 0,3955 en todos. Significa que en el 5 % peor de los casos, cerca de un
**35 % de la masa del "pool de piel" es en realidad lesion no detectada**. Ningun run entrega un
pool de piel limpio en la cola. Si CIRCLe estima tono desde ese pool sin filtrar, en esa cola
estima tono sobre lesion. **Ninguno de los cinco es apto sin un filtro previo por caso**; el
ranking de arriba ordena cual exige el filtro mas chico, no cual puede usarse crudo.

---

## 4. Estado del run 023

**VERIFICADO: 023 corrio y completo.** La creencia de la usuaria era correcta. Evidencia
concreta, en UNABIA:

| Evidencia | Valor |
|---|---|
| `runs/dermlip-semi-supervised-v8-isic2018-2594-abl023-resnet34-ph2cp-imapp100/selection/summary.json` | `"status": "completed"` |
| `runs/dermlip_abl023_console.log`, ultima linea | `Training completed in 11085.0 seconds. Best PH2 validation Dice: student 0.9387, teacher 0.9374` |
| Misma, penultima | `Epoch 300/300 — sup_loss: 0.1325, ... student_dice: 0.9331, teacher_dice: 0.9362` |
| `runs/dermlip_abl023_pipeline.log` | `Training PID 1653208 has finished at Sat Oct 31 10:24:12 UTC 2026`, luego evaluacion pareada y estratificacion |
| `selection/training_started.json` | recibo con `run_id` `70254fae-2956-4a49-9c8a-d7cf5a5a2f45`, `timestamp_utc` `2026-10-31T07:19:18.851407Z` |
| Artefactos de salida | `best_student_model.pt`, `best_teacher_model.pt`, `last_checkpoint.pt`, `last_checkpoint.receipt.json`, `metrics.jsonl`, `run_identity.json`, `summary.json`, `input_validation.json`, `claims/` |
| Evaluacion | `evaluations/paired-raw-morph-20261027/023/4b0db180963c62eb/cruda/cases.jsonl` y `evaluations/ph2_val_3groups_benchmark_20261027/eval_023_abl_isic2018_2594_stratified.json` (2026-10-31 12:13) |
| Proceso vivo al momento de observar | **ninguno.** `nvidia-smi` a las 15:38 UTC: ambas GPU al 0 %, 2 MiB de 24.467 MiB, sin procesos. No hay `train_dermlip_semi_023.py` en `ps` |

Duracion 11.085 s (3 h 5 min), 300 epocas, semilla 42, `device cuda:1`.

### Metricas de 023 sobre los 2.594

| Estrato | N | Dice 023 | vs 019 | vs 020 | Precision | Razon de area | Ceros |
|---|---:|---:|---:|---:|---:|---:|---:|
| <1 % | 48 | 56,24 % | +10,37 | −0,51 | 44,06 % | 2,05x | 2 |
| 1-2 % | 149 | 72,20 % | −0,44 | **−6,36** | 59,12 % | 1,62x | 0 |
| 2-5 % | 427 | 80,08 % | +0,59 | −2,48 | 70,50 % | 1,36x | 3 |
| 5-10 % | 432 | 85,77 % | +2,04 | +1,55 | 81,34 % | 1,18x | 3 |
| >= 10 % | 1.538 | 86,87 % | +1,63 | **+3,14** | 95,00 % | 0,92x | 2 |
| **Global** | **2.594** | **84,16 %** | **+1,57** | **+1,34** | **85,69 %** | **1,03x** | **10** |

023 es el mejor Dice global de los cinco (84,16 %). Como 023 = 020 + IMA++ con la misma semilla
42, el contraste **023 − 020 aisla el aporte de IMA++**: +1,34 puntos globales, concentrados en
lesiones grandes (+3,14 en `>=10%`) y negativos en lesiones chicas (−6,36 en `1-2%`). Es
decir, IMA++ empuja en direccion opuesta al copy-paste por estrato.

### Correccion al contrato: no es "IMA++ al 100 %"

El contrato (`seg-imapp-holdout.md:24`) describe 023 como "ResNet34 + PH2 copy-paste +
**IMA++ al 100 %**". Es incorrecto. `runs/dermlip_abl023_console.log`, linea 2:

```
2026-10-31 07:19:08,572 [INFO] Loaded Labeled datasets: PH2 (156 train, 39 val), IMA++ audited cohort (100 images)
```

Son **100 imagenes**, no el 100 % del dataset. Sobre 14.967 imagenes de IMA++ eso es el
**0,67 %**. El `pipeline.log` lo describe coherentemente como
`"Piloto desacoplado PH2 (CP p=0.5) + IMA++ N=100 real (s42)"`. El sufijo `imapp100` del nombre
del lanzador y del directorio es la fuente de la ambiguedad: significa `N=100`, no `100 %`.
**023 es un piloto de 100 imagenes, no una corrida con IMA++ completo.** Presentarlo como lo
segundo seria un error de magnitud de dos ordenes.

### Hueco de procedencia de IMA++ en 023

**NO VERIFICADO, y los recibos positivamente no lo acreditan.** Ningun artefacto de IMA++ tiene
hash en la cadena de procedencia de 023. `selection/run_identity.json`, `summary.json` y
`training_started.json` listan en `source_artifact_sha256` unicamente:
`dermlip_model_manifest`, `evaluation_manifest` (+ recibo), `ph2_exclusions`, `ph2_manifest`,
`refit_source_manifest`, `selection_source_manifest`, `terms_acceptance`. El
`pool_provenance` declara 34.330 imagenes no etiquetadas, desglosadas como ISIC2019 14.411 +
ISIC2020 19.919 — **IMA++ no esta ahi**. El config
`dermlip.selection.runtime.20261028.023.json` (sha256
`e86d223244189080b4dad37756f52f148eff85a905efe6528196daafc4ab8be0`) declara `ph2_manifest`
(195 registros), `pool_manifest` (34.330) y `evaluation_manifest` (2.594), y **no tiene ninguna
entrada de IMA++**: ni ruta, ni sha256, ni `record_count`.

El entrenamiento cargo "IMA++ audited cohort (100 images)" segun su propio log, pero **no hay
manifiesto con hash que diga que 100 imagenes fueron, de donde salieron, ni si solapan con el
split de evaluacion**. Se comprueba identificando en `scripts/train_dermlip_semi_023.py` de donde
lee ese cohorte y congelando ese listado con hash y recibo. Hasta entonces 023 no es auditable
como ablacion de IMA++.

### Sobre los gates de 023

El config declara `execution_enabled: true`, `launch_ready: true`, `execution_ready: true`,
`training_started: false`, `verified_for_use: false`,
`patient_independence_accredited: false`. El `training_started: false` del config **no es una
contradiccion**: el config es el artefacto de entrada y no se reescribe despues de la corrida; la
evidencia de ejecucion es el recibo `selection/training_started.json` mas `summary.json`
(`status: completed`). Queda registrado para que nadie lea el config suelto y concluya que 023 no
corrio. **Esta sesion no cambio ningun gate.**

---

## 5. Benchmark PH2 de 3 grupos

Solo existe para 019, 020, 021 y 022. **Para 023 no existe:** `ls` de
`evaluations/ph2_val_3groups_benchmark_20261027/` solo devuelve
`eval_023_abl_isic2018_2594_stratified.json`. 023 no tiene benchmark de 3 grupos.

### Grupo 1 — PH2 val original, n = 39 (fotos reales)

| Metrica | 019 | 020 | 021 | 022 |
|---|---:|---:|---:|---:|
| Dice medio | 0,9359 | 0,9367 | 0,9362 | 0,9363 |
| Dice mediano | 0,9544 | 0,9537 | 0,9559 | 0,9599 |
| Precision media | 0,9340 | 0,9484 | 0,9394 | 0,9397 |
| Sensibilidad media | 0,9502 | 0,9386 | 0,9453 | 0,9456 |
| Razon de area mediana | 1,0099 | 0,9775 | 0,9916 | 0,9966 |
| Solapamiento cero | 0 | 0 | 0 | 0 |
| CIELAB degeneradas | 0 | 0 | 0 | 0 |

**En fotos reales el copy-paste no hace nada.** `delta_dice` registrado en los propios archivos
de comparacion del cluster: **+0,00082** (019 -> 020) y **+0,000077** (021 -> 022). Cuarto y
quinto decimal.

### Grupo 2 — trios sinteticos, 75 imagenes

| Metrica | 019 | 020 | 021 | 022 |
|---|---:|---:|---:|---:|
| Dice medio | 0,5281 | **0,8751** | 0,5854 | **0,8516** |
| Dice mediano | 0,5236 | 0,9594 | 0,6764 | 0,9501 |
| Precision media | 0,4303 | 0,8626 | 0,5005 | 0,8157 |
| Sensibilidad media | 0,8635 | 0,9404 | 0,8484 | 0,9539 |
| Razon de area mediana | 2,5111 | 1,0185 | 1,9030 | 1,0497 |
| Solapamiento cero | 3 | 1 | 7 | 0 |
| Casos con derrame en agujero reparado | 75 | 44 | 63 | 47 |
| Dice, bin 0,5-1,0 | 0,3464 | 0,8408 | 0,4362 | 0,8026 |
| Dice, bin 1,0-2,0 | 0,5233 | 0,8748 | 0,5653 | 0,8596 |
| Dice, bin 2,0-5,0 | 0,7146 | 0,9096 | 0,7547 | 0,8926 |

**Este grupo esta circularmente confundido y no puede sustentar la ablacion.** Dos razones, las
dos verificadas en el propio archivo:

1. **El benchmark esta construido con copy-paste y mide una ablacion de copy-paste.** Las
   ganancias son de **+34,7 puntos** (019 -> 020) y **+26,6 puntos** (021 -> 022). Contra
   **+0,24 y +0,46 puntos** en fotos reales de ISIC 2018 y **+0,0008 y +0,00008** en PH2 val
   real. Una discrepancia de dos ordenes de magnitud. Lo que mide el grupo 2 es cuanto aprendio
   el modelo la distribucion del generador sintetico, no generalizacion.
2. **n = 75 no son 75 muestras independientes.** El archivo lo dice literalmente en su campo
   `Note`: `"25 non-independent host trios (75 images total); fixed host, donor, and position
   within each trio."` El n efectivo es **25**, no 75. Cualquier intervalo de confianza o prueba
   que trate los 75 como independientes esta mal especificado.

**No reportar el delta del grupo 2 como evidencia del beneficio del copy-paste.** Si se reporta,
tiene que ir con las dos advertencias y junto a los numeros reales.

### Grupo 3 — fondos reparados, n = 25

Mide falsos positivos sobre regiones *inpainted* sin lesion.

| Metrica | 019 | 020 | 021 | 022 |
|---|---:|---:|---:|---:|
| Casos con algun pixel FP | **25** | 5 | 23 | 6 |
| Casos con FP dentro de la zona reparada | **25** | 5 | 23 | 5 |
| Tasa media de FP en zona reparada | **0,2413** | 0,0388 | 0,2176 | 0,0392 |
| Tasa maxima de FP en zona reparada | 0,4562 | 0,2896 | **0,7628** | 0,2872 |
| Tasa media de FP fuera de la zona | 0,0180 | 0,0079 | 0,0132 | 0,0088 |
| Probabilidad media dentro de la reparacion | 0,2725 | 0,0413 | 0,2425 | 0,0625 |
| Probabilidad media fuera | 0,0388 | 0,0100 | 0,0297 | 0,0323 |
| **Probabilidad maxima dentro de la reparacion** | 0,9982 | **0,9995** | **1,0000** | **0,9999** |

**Aca si hay una mejora real y reproducible, y es el unico beneficio solido del copy-paste.** Los
modelos de referencia disparan sobre el artefacto de *inpainting* en practicamente todos los
casos (25/25 y 23/25), con probabilidad media 7x mayor dentro de la reparacion que fuera
(0,2725 contra 0,0388 en 019). Entrenar con copy-paste lo corrige: 5/25 y 6/25, y la tasa media
de FP en la zona cae de 0,24 a 0,039 y de 0,22 a 0,039. Direccion y magnitud consistentes en las
dos semillas.

**Pero el peor caso no mejora.** `max_prob_inside_repair_across_25` se queda en ~1,0 en los
cuatro runs, incluidos 020 y 022 (0,9995 y 0,9999). Mejora la media, no la cola: sigue habiendo
al menos un caso donde el modelo asigna confianza casi saturada a una region sin lesion.

Este grupo tambien comparte el generador con el entrenamiento del brazo de copy-paste, asi que
parte de la ganancia es familiaridad con el artefacto. Sigue siendo la medicion mas defendible de
los dos grupos sinteticos, porque mide **falsos positivos sobre region sin lesion**, que es un
fallo real de robustez y no una metrica de solapamiento inflada por el generador.

---

## 6. Inventario de artefactos a bajar al repo, con nombre definitivo

Convencion vigente, leida de `resultados/preparacion_v8/`: directorio
`dermlip_isic2018_eval_2594_<run>_<variante>_<YYYYMMDD>_<NNN>/` con `cases.jsonl` +
`evaluation_report.json`, y `run_record.md` cuando hay narrativa (patron de
`usl_ccam_stabilized_20261018_005/`). El ultimo `NNN` usado es `007`, asi que la numeracion
sigue en `008`.

| Origen en UNABIA | Destino propuesto en el repo |
|---|---|
| `evaluations/paired-raw-morph-20261027/019/0ae35137b699d60f/cruda/cases.jsonl` | `resultados/preparacion_v8/dermlip_isic2018_eval_2594_019_resnet34_flatlr_s42_20261027_008/cases.jsonl` |
| `evaluations/ph2_val_3groups_benchmark_20261027/eval_019_reference_isic2018_2594_stratified.json` | mismo directorio, `evaluation_report.json` |
| `.../020/55b3c0149906db8d/cruda/cases.jsonl` | `resultados/preparacion_v8/dermlip_isic2018_eval_2594_020_resnet34_ph2copypaste_s42_20261027_009/cases.jsonl` |
| `eval_020_abl_isic2018_2594_stratified.json` | mismo directorio, `evaluation_report.json` |
| `.../021/c0167b4735bb35a1/cruda/cases.jsonl` | `resultados/preparacion_v8/dermlip_isic2018_eval_2594_021_resnet34_flatlr_s43_20261028_010/cases.jsonl` |
| `eval_021_isic2018_2594_stratified.json` | mismo directorio, `evaluation_report.json` |
| `.../022/02859d9e0f46fab5/cruda/cases.jsonl` | `resultados/preparacion_v8/dermlip_isic2018_eval_2594_022_resnet34_ph2copypaste_s43_20261028_011/cases.jsonl` |
| `eval_022_isic2018_2594_stratified.json` | mismo directorio, `evaluation_report.json` |
| `.../023/4b0db180963c62eb/cruda/cases.jsonl` | `resultados/preparacion_v8/dermlip_isic2018_eval_2594_023_resnet34_ph2cp_imapp100img_s42_20261031_012/cases.jsonl` |
| `eval_023_abl_isic2018_2594_stratified.json` | mismo directorio, `evaluation_report.json` |
| `eval_019_reference_3groups.json`, `eval_020_candidate_3groups.json`, `eval_021_reference_3groups.json`, `eval_022_candidate_3groups.json`, `comparacion_019_vs_020_ph2_val_3groups.json`, `comparacion_021_vs_022_ph2_val_3groups.json` | `resultados/segmentacion/v8/ph2_val_3groups_benchmark_20261027/` con los mismos nombres |
| `runtime_poolv2_20261022_001/.../dermlip.selection.runtime.20261028.023.json` | `plan-segmentacion/metodos-v8/launch-preparation/configs/dermlip.selection.runtime.20261028.023.json` |
| `runs/dermlip-semi-supervised-v8-...-abl023-.../selection/{run_identity,summary,training_started}.json` | directorio de 023, subcarpeta `procedencia/` |

**El sufijo `imapp100` se renombra a `imapp100img` a proposito.** `imapp100` se lee como "100 %" y
ya genero exactamente ese error en el contrato de esta sesion (seccion 4). El nombre definitivo
tiene que decir que son 100 imagenes.

Dos cosas que **no** conviene bajar al repo como estan:

- Los `morfologica/cases.jsonl` (existen para los cinco runs, mismo tamano que los `cruda/`). El
  contrato no los pedia y no se analizaron. Bajarlos sin analizar agrega 6,5 MB sin lectura.
- `scratch_segs.zip` (154,8 MB). Es el ZIP de mascaras de IMA++. No va al repo; va a un almacen
  de datos con manifiesto y hash, igual que el resto de los datasets, que por convencion no viven
  en `data/`.

### Nota sobre los `scratch_*` de la raiz

El contrato reescrito ya no pide el inventario completo, pero `Entra` todavia los lista. Dato que
importa para el triaje pendiente con Nicolas: **los tres CSV de IMA++ no son archivos sin
seguimiento, son ignorados.** `git check-ignore -v` devuelve `.gitignore:17: *.csv` para
`scratch_IMAplusplus_img_metadata.csv`, `scratch_seg_metadata.csv` e
`scratch_iaa_metrics_image.csv`. No aparecen como `??` en `git status`, asi que una auditoria
basada en `git status` no los ve nunca. Hay tambien un directorio `scratch/` con 11 scripts del
2026-10-28 (`measure_imapp_masks.py`, `verify_imapp_user_claims.py`, `replace_leaked_sample.py`,
`check_visual_duplicates_100.py`, entre otros) que el contrato no menciona. **No se renombro, no
se movio ni se borro nada.**

---

## 7. Que falta, separado por quien depende

### Depende de nosotros

1. **Congelar el cohorte IMA++ de 100 imagenes con manifiesto, hash y recibo.** Sin eso 023 no es
   auditable como ablacion de IMA++ (seccion 4). Leer de donde lo toma
   `scripts/train_dermlip_semi_023.py` y verificar solapamiento contra el split de evaluacion.
2. **Corregir los dos defectos de `scripts/compare_seed42_vs_seed43.py`**: `np.nanmedian` y
   assert del invariante de Dice (seccion 0).
3. **Decidir si el grupo 2 sintetico se reporta y como.** Tal como esta, su delta de +34,7 puntos
   contradice los +0,24 reales y es circular (seccion 5).
4. **Correr el benchmark de 3 grupos para 023**, que no existe. Requiere una corrida de
   evaluacion: **fuera del alcance de esta sesion**.
5. **Ampliar a mas semillas si se quiere afirmar algo sobre lesiones grandes.** Con dos semillas
   el signo en `>=10%` se invierte y ahi esta el 59,3 % de los casos.

### Depende de recursos o autorizacion

6. **Independencia por paciente.** `patient_independence_accredited: false` y
   `run_type: ..._patient_unverified` en los cinco runs. El manifiesto de los 2.594 es
   `post_run_evaluation_only_candidate`. Esto es el bloqueo de fondo y no se cierra con las
   corridas: necesita los `patient_id`/`lesion_id` que IMA++ aporta, mas el cruce contra el
   inventario historico.
7. **Cualquier corrida nueva** (023 con 3 grupos, semillas adicionales, IMA++ completo) requiere
   autorizacion explicita por accion segun `docs/RUN_CATALOG.md` paso 6.

### Estado del clúster al momento de observar

GPU 0 y GPU 1 libres (0 % de uso, 2 MiB de 24.467 MiB, sin procesos). Vivos pero al 0 % de CPU y
sin GPU: `scripts/ssl_runner_v2.py` (3 d 20 h) y `scripts/ssl_runner_gpu1.py` (3 d 2 h), de la
linea de Nicolas, mas `dermssl.hub_keepalive`. **No se interfirio con nada.**

---

## Reporte

**Verificado:**

- Conectividad y lectura en UNABIA: `ssh unabia` (alias Windows), `hostname` =
  `jupyter-nmoreno534`, 2026-10-31 15:38 UTC.
- Estado del clúster: `nvidia-smi` y `ps -eo pid,etime,pcpu,rss,args`. Ambas GPU al 0 %, sin
  procesos de entrenamiento.
- Existencia e inventario de artefactos: `find`/`ls -la` sobre
  `evaluations/paired-raw-morph-20261027/`, `evaluations/ph2_val_3groups_benchmark_20261027/` y
  `runs/`.
- **Todas las cifras de las secciones 1, 3, 4 y 5**: recalculadas en local desde los
  `cases.jsonl` crudos con `recompute.py`, y contrastadas contra los JSON estratificados del
  clúster. **Delta maximo 0,000e+00** en las 13 metricas x 6 ambitos x 5 runs.
- Invariante `dice == 2*TP/(gt+pred)`: error maximo 4,8e-12, cero violaciones sobre 1e-4 en los
  12.970 casos. Valida la reconstruccion del area predicha.
- Las cifras del grupo 1, 2 y 3 de PH2: leidas directo de `eval_0XX_*_3groups.json` y
  `comparacion_*.json`.
- 023 completo: `summary.json` `"status": "completed"`, `Epoch 300/300`,
  `Training completed in 11085.0 seconds`, recibo `training_started.json`.
- IMA++ = 100 imagenes, no 100 %: `runs/dermlip_abl023_console.log` linea 2.
- Ausencia de procedencia de IMA++: `run_identity.json`, `summary.json`,
  `training_started.json` y el config, los cuatro leidos completos. Ninguno lo lista.
- sha256 del config de 023: `sha256sum` =
  `e86d223244189080b4dad37756f52f148eff85a905efe6528196daafc4ab8be0`.
- Los CSV de IMA++ estan ignorados, no sin seguimiento: `git check-ignore -v`.
- Convencion de nombres de `resultados/`: `ls` de `resultados/preparacion_v8/` y de dos de sus
  subdirectorios.

**No verificado:**

- **Que 100 imagenes componen el cohorte IMA++ de 023, y si solapan con el split de evaluacion.**
  Ningun recibo las lista. Se comprueba leyendo de donde las toma
  `scripts/train_dermlip_semi_023.py` (en UNABIA, no en el repo) y cruzando ese listado contra
  `preflight/isic2018-task1-eval-2594-v2/manifest.json`.
- **Benchmark PH2 de 3 grupos para 023.** No existe. Exige una corrida de evaluacion, fuera de
  alcance.
- **Las cifras de 12.235 y 11.807 candidatos de IMA++** que mencionan los comentarios de
  `scripts/check_imapp_cross_unabia.py`. El contrato reescrito ya no las pedia y no se tocaron.
  Siguen NO VERIFICADAS.
- **Los `morfologica/cases.jsonl`** de los cinco runs: existen, no se analizaron. El contrato
  pedia la evaluacion `cruda`.
- **Independencia por paciente de los 2.594.** Declarada `false` en los propios artefactos.
- **Run 018**, que existe en `paired-raw-morph-20261027/018/2b5fb99984e01823/`. Fuera del
  contrato, no se leyo.

**Bloqueos:**

1. **023 no es auditable como ablacion de IMA++** hasta que se congele el cohorte de 100
   imagenes con hash y recibo. Es el bloqueo principal y es nuestro.
2. **El contrato de esta sesion contiene un error de magnitud**: "IMA++ al 100 %" contra las 100
   imagenes reales (0,67 % del dataset). Corregido en la seccion 4. Hay que arreglar
   `seg-imapp-holdout.md:24` y cualquier documento que haya copiado esa descripcion.
3. **El grupo 2 sintetico no puede sustentar la ablacion de copy-paste**: es circular y su n
   efectivo es 25, no 75. Decision pendiente sobre si se reporta.
4. **El efecto del copy-paste no se reproduce entre semillas en los dos estratos mas grandes.**
   No se puede afirmar direccion en `>=10%`, que es el 59,3 % de los casos.
5. **Ningun run entrega un pool de piel apto para CIRCLe sin filtro por caso** (contaminacion p95
   entre 0,352 y 0,396). Afecta al diseno del pipeline segmentacion -> clasificacion, que es
   materia del segundo contrato.
6. **No se commiteo nada**, por `seg-imapp-holdout.md:97`. Conflicto pendiente con la
   instruccion directa de la usuaria de commitear al terminar; sin resolver a proposito.
7. **Ningun gate fue modificado.** `execution_enabled`, `launch_ready`, `training_started`,
   `execution_ready`, `verified_for_use` y `patient_independence_accredited` quedan como estaban.
