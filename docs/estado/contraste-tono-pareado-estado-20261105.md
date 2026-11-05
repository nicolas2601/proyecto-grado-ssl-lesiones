# Contraste pareado de tono: estado real al 2026-11-05

Estado: vigente. Este documento existe para que nadie vuelva a medir en la
direccion equivocada por no saber que ya esta hecho y que no.

Leer antes: `docs/63-instrumento-ita-alcance-y-limites.md`,
`docs/estado/tono-cableado-real-vs-instrumento-v2.md`, `CLAUDE.md` seccion 11.

---

## 1. Nomenclatura, porque "A", "B" y "C" significan tres cosas distintas

Esta confusion costo horas el 2026-11-04. Fijada de una vez:

| donde aparece | que significa ahi |
|---|---|
| `resultados/auditoria_v8/full_three_ssl_20261102/` | **A0** = imagen original, **A1** = perturbacion fija L\* −8 |
| `clf1_clinical_reference_20261024/config/ssl-overlay-v1/` | **A/B/C** = variantes de `lr_multiplier` y `clinical_batch_quota`. Nada que ver con tono |
| las tres variantes de la tesis | **V1** tono sin mascara, **V2** tono + mascara semisupervisada, **V3** tono + mascara no supervisada |

**Nombres a usar de aqui en adelante, que dicen que varia:**

- `ctrl-tono-0`: `tone_consistency_weight = 0.0`, `adaptation_mode = two_view_ssl`
- `tono-026`: peso > 0, `adaptation_mode = original_transformed_consistency`,
  vistas transformadas del espejo del teacher 026

## 2. Que esta verificado y listo

| pieza | estado | evidencia |
|---|---|---|
| pool limpio | verificado por hash | `mixed-fit.jsonl`, sha256 `afcfab26...1eabb`, 34.754 filas = 33.620 `source_fit` + 1.134 `pad_adapt` |
| agrupacion de duplicados | verificada | 33.620 imagenes en **14.004 componentes**; 2.624 componentes con mas de una imagen cubren 22.240 (66 %). Los splits DEBEN agrupar por `identity_component_id`, no por `image_id` |
| espejo PNG base | materializado | `~/work/tone-mirror-20261103/full/mirror`, 33.620 PNG, 2,4 G |
| espejos transformados | materializados | **`tone_s42` a `tone_s47`: seis**, 33.620 PNG cada uno y manifiesto completo. Solo s42/s43/s44 tienen `.log`. Corregido el 2026-11-04; antes esta fila decia tres (ver seccion 3.4) |
| transformada | verificada | `lab_l_b_feathered_v1`, **dos canales L\* y b\***, `b_per_ita_degree 0.5`, `l_per_ita_degree 0.12`, `feather_sigma 2.0`, `max_abs_ita_delta 25.0` |
| mecanismo en el entrenador | cableado | `--tone-consistency-weight`, default 0.0 |
| `paired_factor_gate` | **pasa** tras `2b538df` | ver seccion 4 |

## 3. Los tres bloqueos abiertos

### 3.1 Los dos brazos entrenarian sobre datasets distintos

`ctrl-tono-0` (`ssl_overlay_A_s42_6000_v1`) corrio con **`image_count 34754`**:
los dos roles. El espejo se construyo solo desde `masks.source_fit.jsonl`, asi
que tiene **33.620**. Son 1.134 imagenes de diferencia.

**`paired_factor_gate` NO puede ver esto**: compara diccionarios de
configuracion, nunca cuenta archivos en disco. El contraste habria pasado el
gate y medido dos cosas a la vez.

**Y la salida facil no lo es.** Las 1.134 de `pad_adapt` son PAD-UFES-20,
**fotos clinicas de celular** (`/home/jovyan/work/pad_ufes_20/images/...`,
`PAT_1006_53_385.png`), no dermatoscopia. **Cero tienen mascara del teacher
026** (comprobado por interseccion de ids). Incluirlas exige correr un
segmentador entrenado en dermatoscopia sobre imagenes clinicas, fuera de
dominio, con calidad de mascara desconocida: un confound nuevo dentro del
brazo tratado.

**Recomendacion: re-correr el control sobre `source_fit` solo** (6.000 pasos,
una GPU). Descarta un control ya hecho; a cambio los dos brazos quedan sobre
exactamente las mismas 33.620 imagenes y todo en dominio.

### 3.2 El peso calibrado no corresponde a la configuracion real

`tone_consistency_weight = 0.194` se calibro
(`resultados/preparacion_v8/tone_consistency_calibration_20261105/`) con las
palancas por defecto. El control real usa otras:

| palanca | control real | sondeo de calibracion |
|---|---:|---:|
| `freeze_blocks` | **8** | 0 |
| `layer_decay` | **0.75** | 1.0 |
| `anchor_distill` | **1.0** | 0.0 |
| `anchor_l2sp` | **0.0001** | 0.0 |
| `head_warmup_steps` | **32** | 0 |

Con `anchor_distill = 1.0` hay **otro termino sumado a la misma perdida**, y
con 8 bloques congelados la escala de gradientes cambia. Ademas el control
termina en `ssl_loss −0.766` contra los `−0.022` del sondeo de 50 pasos: la
razon medida a paso cero no describe el regimen de 6.000 pasos.

**Hay que recalibrar con las palancas reales y a mitad de corrida.**

### 3.3 `data_revision` incompatible: causa raiz resuelta el 2026-11-04

El control declara `afcfab26...`. El manifiesto del espejo declara
`source_revision f4fcee05...` y `config.validate_transformed_manifest` aborta
si no coinciden (`config.py:472`, compara contra `cfg.data_revision`).

**No son dos hashes del mismo pool. Uno es un dataset y el otro es un modelo.**

| hash | que es | evidencia |
|---|---|---|
| `afcfab26...` | sha256 de `mixed-fit.jsonl`, el manifiesto de seleccion del pool | `tone_mirror_pool_20261103/receipt.json:20` |
| `f4fcee05...` | sha256 del **checkpoint del teacher 026** (`best_teacher_model.pt`) | `run_record.md:46` lo etiqueta literal; `ph2_benchmark.json:3` como `checkpoint_sha256`; `measure_ph2_hair_removal_shaped_dice.py:27` como `CHECKPOINT_SHA256` |

El campo `source_revision` esta definido de forma coherente en todo el codigo
como **la revision de los datos fuente**: `pool_manifest.build_pool_manifest`
lo recibe como `Mapping[dataset -> revision]`,
`tone_transforms._load_records` filtra por el, y `config.py:472` lo compara
contra `data_revision`. No hay ambiguedad en el codigo.

**El error fue del lado del operador al materializar el espejo:** se paso el
sha del checkpoint del teacher a un campo cuyo contrato es la revision de
datos. Verificado el 2026-11-04 sobre
`~/work/tone-mirror-20261103/full/masks.mirror.jsonl`: las 33.620 filas traen
`source_revision = f4fcee05...`, **y ademas** ya traen la identidad del modelo
en su lugar correcto:

- `producer`: `dermlip_slic_v8:abl026-resnet34-ph2cp-ima0-s42:teacher`
- `version`: `cruda+hair_removal+resize224_bilinear+thr0.5`

Es decir, **el sha del checkpoint en `source_revision` es redundante** y esta
ocupando el campo que deberia llevar la revision de datos. Los borradores de
protocolo ya lo tenian separado bien (`tone-lab-pairs-v1.draft.json` lleva
`pool_manifest_sha256` y `mask_teacher_sha256` como campos distintos); el
codigo nunca los confundio, la materializacion si.

**Por lo tanto el arreglo es de valor, no de esquema.** No hace falta rediseniar
`config.py` ni el manifiesto: hace falta que `source_revision` lleve la
revision del conjunto de datos realmente usado.

**Y ahi 3.3 se une con 3.1.** La revision correcta no es `afcfab26...`, que es
el sha de `mixed-fit.jsonl` con sus 34.754 filas (los dos roles). El espejo se
construyo sobre 33.620. El valor correcto es el hash del subconjunto
`source_fit` efectivamente usado, declarado por **los dos brazos**. Un solo
arreglo cierra 3.1 y 3.3; la causa raiz comun es que el espejo y el control no
se construyeron desde el mismo conjunto declarado.

**Pendiente de decision:** reescribir `source_revision` en los artefactos ya
materializados es una reescritura de registros de procedencia. No se toco nada
sin autorizacion.

### 3.4 Hallazgos colaterales de la verificacion del 2026-11-04

1. **Hay seis espejos transformados, no tres.** `tone_s42` a `tone_s47`, los
   seis con 33.620 PNG y su `tone_transform_manifest.json` completo. Solo
   s42/s43/s44 tienen `.log`. La seccion 2 de este documento decia tres.
2. **421 imagenes del brazo tratado son copias identidad**
   (`fallbacks_by_reason: {ineligible_mask: 421}`, `transformed_fraction
   0.98748`). Es deliberado y el manifiesto lo explica: dejarlas caer haria que
   el brazo tratado entrenara sobre un conjunto distinto al de su control. Pero
   significa que el 1,25 % del brazo tratado no recibe tratamiento, y eso
   diluye el efecto medido. Hay que reportarlo con el delta, no esconderlo.
3. **`ita_delta_applied_abs_mean: 0.0201` estaba mal nombrado, y la cifra real
   es 620 veces mayor.** Corregido el 2026-11-04 (`src/dermssl/tone_augment_lab.py`):
   el campo calculaba una media **con signo** bajo un nombre que prometia
   valores absolutos. Medido sobre los manifiestos ya materializados, sin
   re-materializar nada:

   | semilla | publicado como `abs_mean` | media con signo real | media **absoluta** real | max abs |
   |---|---:|---:|---:|---:|
   | s42 | 0.0201 | 0.0201 | **12.4757** | 24.9999 |
   | s43 | −0.1601 | −0.1601 | **12.4775** | 24.9972 |
   | s44 | −0.0751 | −0.0751 | **12.5248** | 24.9998 |

   **La dosis aplicada es de 12,48 grados de ITA**, no 0,02. Y las tres semillas
   son coherentes entre si: mediana absoluta ~= media absoluta ~= 12,5, que es
   exactamente la mitad del tope `max_abs_ita_delta 25.0`, o sea una
   distribucion uniforme sobre +-25 tal como el diseno pedia. La media con signo
   cerca de cero **prueba que el muestreador no esta sesgado**; no es una dosis
   debil.

   Leer el campo viejo literal habria llevado a afirmar que la transformacion es
   inerte, sobre una corrida que desplaza 12,5 grados de media. **La cifra
   `0.0201` queda retirada**; usar `ita_delta_applied_abs_mean` solo en
   manifiestos posteriores al 2026-11-04, que ahora reportan las tres cifras con
   nombres honestos (`_mean` con signo, `_abs_mean`, `_abs_max`).

6. **El motivo de abstencion se aplanaba bajo una sola etiqueta.** Corregido en
   `tone_augment_pairs.py`: `fallback_reason` estaba fijo en `ineligible_mask`
   sin importar la causa. Las 421 del pool son `degenerate_area`, fallos del
   teacher 026 **propio** (381 con mascara bajo el 1 % del area, algunas vacias;
   40 por encima del 90 %; **cero** por baja confianza, con confianza mediana
   0.8963). Una imagen sin mascara seria `missing_mask`. Son hallazgos distintos
   —uno es un defecto medible de nuestro segmentador, el otro una imagen que
   nunca paso por el— y aplanarlos dejaba la dilucion sin atribuir.
4. **72 % de las imagenes tienen inversion de signo en b\***
   (`images_with_b_sign_flip: 24276` de 33.620). Consistente con la asimetria ya
   trazada a los signos acoplados de Corbin y su umbral analitico
   `b*_inicial < 5 + 0.50*|t|`.
5. **El manifiesto del espejo es de esquema v2**, con `parameters`, `seed` y
   `materialization_step`, que `tone_transforms.generate_tone_transformations`
   no escribe. Lo produjo otra ruta de codigo
   (`tone_mirror_pool_v1.py` en `~/work/tone-mirror-20261103/scripts/`).

## 3.5 El envoltorio del cluster, y por que el control NO se recorta

Verificado el 2026-11-04. Esto refuta la recomendacion de 3.1.

El control no se lanzo con `scripts/pretrain.py` del repo. Lo lanzo
`~/work/clf1_clinical_reference_20261024/scripts/overlay_ssl_runtime.py`
(198 lineas, sha `6aa8787e...`, identico en `scripts/` y en
`scripts_snapshot_20261027_pre_v2/`), un envoltorio **no versionado**.

Sus parametros reales traen dos campos que el repo no tiene:

```json
"sampling_plan": ".../config/ssl-overlay-v1/mixed-uniform.npz",
"sampling_plan_sha256": "d0491ef4...",
"clinical_batch_quota": 4,
"batch_size_per_gpu": 16
```

**`clinical_batch_quota 4` sobre `batch_size_per_gpu 16` reserva 4 de cada 16
posiciones de cada lote al rol clinico.** Las 1.134 de `pad_adapt` no son un 3 %
del pool: son el **25 % de cada paso**, por sobremuestreo. Y la linea 101 dice
para que:

```python
domains = [int(row["role"] == "pad_adapt") for row in rows]
coverage = CoverageState(ids, domains)
```

**El reparto de roles ES el adaptador de dominio**, el que V1 y V2 declaran.

Y la linea 93 lo cierra por codigo:

```python
or cfg.clinical_batch_quota not in (1, 4) or cfg.max_steps not in (40, 6000)
or cfg.seed not in (42, 43, 44) ...
    raise ValueError("Configuration is outside the authorized isolated experiment")
```

**`clinical_batch_quota = 0` esta rechazado.** Recortar el control a `source_fit`
no es inconveniente: el arnes no lo permite, y si lo permitiera borraria el
adaptador de dominio. **Recomendacion corregida: conservar el control tal como
esta** (34.754 imagenes, cuota 4, 6.000 pasos ya pagados) y adaptar el brazo
tratado, no el control.

El envoltorio **no sabe nada de tono**: cero coincidencias de `adaptation_mode`,
`transformed` y `tone_consistency`. Pero el codigo congelado que maneja
(`frozen_experiment_root = /home/jovyan/work/clf1_ssl_phone_20261024`) **si lo
soporta entero**: `config.py`, `levers.py`, `torch_adaptation.py`,
`torch_loop.py` y `torch_simsiam.py` llevan los flags. Falta que el envoltorio
los pase y que su lista blanca admita el brazo tratado. Cambio chico.

**Plan corregido:**

1. Materializar las 1.134 de `pad_adapt` en el espejo por el camino de copia
   identidad con registro de mascara inelegible, el mismo que ya atiende a 421
   imagenes. Sin segmentador fuera de dominio, sin confound nuevo.
2. Calcular la revision de datos sobre **34.754**, no sobre 33.620.
3. Pasar los flags de tono en el envoltorio y extender su lista blanca.

**Nota sobre `wrapper_sha256`:** el sello del control dice `8b34a29c...` y el
archivo hashea `6aa8787e...`. **No es una procedencia rota.** La linea 99 hace
`seal = config_seal(config)`: el campo guarda un sello de la **configuracion**,
no el hash del envoltorio. El nombre invita al error; se estuvo a punto de
reportar una falsa alarma.

## 3.6 La dosis puede estar sub-dimensionada, y es candidata a explicar los nulos

Abierto el 2026-11-04 al cruzar la medicion de la dosis real con la observacion
**#2081** sobre el origen del tope. Es una hipotesis, no un resultado.

**Lo medido** (33.199 imagenes por semilla, seis semillas, parametros identicos
en las seis: `max_abs_ita_delta 25.0`, `b_per_ita_degree 0.5`,
`l_per_ita_degree 0.12`, `feather_sigma 2.0`):

| cantidad | valor |
|---|---:|
| desplazamiento tipico aplicado (media absoluta) | **12,48 grados** |
| tope declarado `MAX_ABS_ITA_DELTA` | 25,0 |
| recorrido fisicamente alcanzable por imagen | **~70 grados** reales (75,4 en S-SYNTH) |

**El desplazamiento tipico es ~18 % de lo que las imagenes soportan fisicamente,
y el tope es ~36 %.**

**De donde sale el 25, segun #2081:**

1. Se derivo de la tabla de **Corbin** (dos categorias FST ~= 18-27 grados,
   mediana 22). Pero el estimador del repo clasifica con la de **Groh**
   (`FST_THRESHOLDS = (40, 23, 12, 0, -25)`), donde dos categorias abarcan 23-37
   grados, mediana **28**. El tope es coherente con una tabla que no usamos.
2. **Y la incoherencia es doble, con la segunda capa mas grave:** el tope se
   justifica como "dos categorias FST", y **#2047** ya establecio que la
   clasificacion ITA->FST no funciona en estos datos (recalibrar converge al
   predictor constante, 53,3 % contra base trivial 55,8 %). El tope esta anclado
   a una categorizacion que el proyecto **ya retiro**. Cambiar 25 por 28
   arreglaria la tabla sin arreglar el fundamento.
3. **Lo que Corbin acota de verdad no son categorias**, es plausibilidad visual:
   *"results beyond this maximum value did not yield satisfactory visual
   outcomes"*. Las categorias eran su forma de acotarlo, no la razon.

**Y `capped: 0` no dice lo que parece.** Aparece en las seis semillas con maximo
absoluto 24,9999. No significa "se probo mas y no hizo falta": el muestreador
sortea **uniformemente dentro** de +-25 en vez de sortear mas amplio y recortar,
asi que el tope nunca recorta porque nunca se le pide mas. **El tope no limita,
define el rango**, y es vinculante por construccion. Es otro campo que invita a
la lectura inversa.

**Por que esto es candidato a explicar los nulos previos.** El brazo B dio
negativo 18/18 y A71 dio −0,1002 en PAD. Esas corridas se descartaron por el
pool viejo y la segmentacion anterior, pero **ninguna de las dos explicaciones
toca la magnitud del tratamiento**. Si la dosis tipica es un sexto del recorrido
disponible, un efecto real podria quedar por debajo del ruido sin que nada en el
pipeline este roto.

**Recomendacion: medir respuesta a la dosis antes de concluir cualquier cosa del
contraste.** Es barato del lado de la materializacion: no requiere segmentador
nuevo, ni mascaras nuevas, ni resolucion nueva — solo volver a correr la
transformacion con un tope mayor sobre el espejo que ya existe. Las seis
semillas actuales no sirven para esto porque son replicas con parametros
identicos; hacen falta espejos a topes distintos.

**Lo que no hay que hacer:** subir el tope a 28 "porque Groh". Eso repite el
error de anclar la dosis a una categorizacion retirada. La cota principiada es
el recorrido alcanzable medido por caso (#2080), y el tope deberia declararse
como **politica conservadora explicita** derivada de plausibilidad visual, no
como traduccion de una tabla.

## 4. El gate, resuelto el 2026-11-05 (`2b538df`)

Habia una contradiccion entre dos reglas del propio repo:

- `levers.Levers.validate` prohibe `tone_consistency_weight > 0` fuera del modo
  pareado, asi que el peso **no puede** ser igual en los dos brazos;
- `TREATMENT_FIELDS` no lo incluia, asi que cualquier diferencia se rechazaba
  como segundo factor.

Ninguna configuracion podia pasar las dos. Resuelto admitiendo el peso en
`TREATMENT_FIELDS`: no es un parametro de optimizacion independiente, porque
**no puede variar sin `adaptation_mode`**, que ya estaba en el conjunto. Es la
dosis del tratamiento.

Admitirlo abria un agujero: con peso cero en los dos brazos, el tratado
cargaria las vistas transformadas y dejaria el termino inerte, pasando todos
los chequeos sin medir nada. Cerrado con comprobaciones explicitas (control
en cero, tratado por encima, campo ausente tratado como cero). Tres tests
nuevos, 17 en verde.

## 5. Lo que no se debe repetir

El 2026-11-04 se gastaron horas en mediciones que no hacian falta. Las causas,
para que queden como precedente y no como anecdota:

1. **Se midio contra v7 sin preguntar cual era la corrida vigente.** v9 no es
   una corrida: es el **release** del clasificador (tag `v9.0.0`, .pt + ONNX +
   Docker, VPS #1953) y el servicio de inferencia de la rama `phone-web`.
2. **Se pulio el solver por imagen de `tone_augment_lab_v2`** —que estima ITA
   para apuntar a un objetivo— cuando el dia anterior, en VPS #2305 y decidido
   con Maria Paula, ese uso del ITA ya estaba **DESCARTADO** (es 22 %
   iluminacion). Solo sobrevive el ITA como **unidad de parametrizacion**: la
   transformacion aplica `delta_b` y `delta_l` en Lab y **nunca estima ITA**.
3. **Se presentaron resultados de octubre como estado actual.** El brazo B
   negativo 18/18 y A71 −0.1002 en PAD son **anteriores** a la segmentacion
   semisupervisada y al pool nuevo. No describen donde estamos.
4. **Se propuso bajar el brazo tratado a L\*-only** sin releer que ya estaba
   medido que L\* solo es peor: Steiger `z = +5.327`, y `eta2(luz)` 0.272
   contra 0.177, es decir 35 % mas contaminado por iluminacion.
5. **Se rehizo el inventario de cableado que ya tenia asignado otra sesion**,
   por leer `started 4d ago` como inactividad cuando es la hora de arranque.
6. **Se lanzo una corrida de 1.500 imagenes con el campo equivocado** como
   hash por imagen (`source_revision`, constante del dataset) y hubo que
   repetirla.

**La regla unica detras de las seis:** antes de medir, leer; antes de afirmar,
verificar de cuando es el dato. Escrita en `CLAUDE.md` seccion 11 y en
`memory/feedback_leer-antes-de-medir.md`.

## 6. Deudas de documentacion

1. `docs/63` seccion 7 tiene una **cita entrecomillada sin fuente**, buscada en
   los tres PDF y no encontrada. Probablemente redactada por una sesion previa
   y entrecomillada despues. **No citarla**; el aporte no la necesita.
2. VPS **#2305** afirma "la luz mueve el instrumento mas que la biologia
   entera", cifra **retirada** en `docs/63` (comparaba rango de medias contra
   max−min crudo; medias contra medias da melanina 34,9 contra luz 19,3).
3. VPS **#2268** da el trio `0.597/0.638/0.610` mezclando datasets y brazos sin
   decirlo. El 0.610 es **PAD de DINO+iBOT**, no HAM de MoCo. Ya hizo perder
   tiempo una vez.
4. El readout de la **cola del 29 de octubre** (siete brazos SSL, pedido por
   Karen) sigue sin registrarse. #2272 lo pidio el 30 de octubre.
