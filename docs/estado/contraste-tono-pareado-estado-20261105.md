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
| espejos transformados | materializados | `tone_s42/s43/s44`, 33.620 PNG cada uno, 2,5 G |
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

### 3.3 `data_revision` incompatible

El control declara `afcfab26...` (sha del manifiesto del pool). El manifiesto
del espejo declara `source_revision f4fcee05...` y
`config.validate_transformed_manifest` aborta si no coinciden.

Si el tratado corre con `f4fcee05...`, difiere del control en `data_revision`,
que **no** esta en `TREATMENT_FIELDS`, y el gate lo rechaza con razon.

Falta entender por que hay dos hashes distintos para el mismo pool antes de
tocar nada.

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
