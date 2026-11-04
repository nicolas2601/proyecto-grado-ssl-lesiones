# Progreso: escalera ITA — cuánto cuesta el error de máscara en grados

Pregunta que responde esta línea: el brazo de segmentación v8, ¿sirve de verdad
como insumo del estimador de tono, o es un adorno? Se contesta midiendo el error
de ITA de cada opción de enmascarado contra un techo de máscara de verdad.

Protocolo de referencia: acta 2026-09-18 (`plan-segmentacion/README.md` §1) y
`docs/estado/clf-medicion-completa-a0-b-t.md`.

- [x] 0. Rescate de 12 módulos sin seguimiento | `f14a63a` | `python -m ruff check <los 12>` + `PYTHONPATH=src pytest tests/test_tone_augment_lab.py tests/test_tone_augment_fst.py tests/test_tone_normalization.py tests/test_pretrain_tone_augment_pairs.py tests/test_paired_factor_gate.py tests/test_seg_hair_removal.py tests/test_seg_field_roi.py` → 140 passed
- [x] 1. Escalera de 4 brazos sobre S-SYNTH (n=600) | sin commit (artefactos en UNABIA) | `~/work/ita-ladder-20261104/join4.py`
- [x] 2. Escalera de 4 brazos sobre ISIC 2018 Task 1 (n=2.594) | sin commit (artefactos en UNABIA) | `~/work/ita-ladder-20261104/final_isic.py`
- [x] 3. Causa del 28 % sin ITA | sin commit (artefactos en UNABIA) | `~/work/ita-ladder-20261104/diag_b.py`
- [x] 4. Tono contra contraste como conductor del sesgo | sin commit (artefactos en UNABIA) | `~/work/ita-ladder-20261104/contrast.py`
- [x] 5. `tone_augment_lab_v2` con Corbin eq. 7 en grados ITA | `6997206` | `PYTHONPATH=src pytest -q tests/test_tone_augment_lab_v2.py tests/test_tone_augment_lab.py` → 46 passed (recorrido alcanzable por imagen + `brentq`; sin cap de politica por defecto — ver docstring del modulo sobre la inconsistencia Groh/Corbin en `MAX_ABS_ITA_DELTA`; corregida la agregacion del ITA de la curva — angulo de las medianas, no mediana de los angulos — causa dominante del residuo de la via B; abstencion nueva `border_population_shift_too_large` para la causa secundaria)
- [x] 5d. Unificar las dos definiciones de piel (Tarea 1A) | `99bad30` |
  `PYTHONPATH=src pytest -q tests/test_tone_augment_lab_v2.py tests/test_tone_augment_lab.py`
  → 48 passed. `_unified_skin_fov` pliega la exclusion de borde casi-negro en
  el `valid_fov` que se pasa a `augment_skin_tone`, sin tocar `_skin_selector`
  ni `tone_normalization.py` ni `tone_augment_lab.py` (editar `_skin_selector`
  cambiaria el comportamiento efectivo del modulo congelado para todo
  llamador). Test de regresion con vineteado sintetico — S-SYNTH no alcanza
  para probarlo, las dos definiciones coinciden ahi al 0,0000 %.
- [x] 5e. Hoja de contactos antes/despues (Tarea 1B) | `addc50a` |
  `PYTHONPATH=src pytest -q tests/test_make_tone_contact_sheet.py` → 8 passed.
  `scripts/make_tone_contact_sheet.py`, toma un manifiesto JSONL de
  imagen/mascara, estratifica por delta LOGRADO (no por melanina), fuerza los
  dos extremos absolutos. Pendiente: correrlo sobre S-SYNTH real (manifiesto
  no generado en esta sesion) y que Maria Paula juzgue la plausibilidad
  visual — eso desbloquea la Tarea 1C (calibrar `MAX_POPULATION_SHIFT_FRACTION`,
  bloqueada aparte, esperando la corrida `via_b3` de maestro).
- [~] 6. Validación de la transformación contra melanina verdadera en S-SYNTH |
  sin commit (artefactos en UNABIA) | `~/work/ita-ladder-20261104/via_b2.py` →
  `via_b2_rows.jsonl`. **Causa raiz del residuo aislada y corregida en `6997206`
  (agregacion angulo-de-medianas vs mediana-de-angulos); pendiente de re-correr
  la via B sobre S-SYNTH real para confirmar en datos reales** — ver "La
  transformación" abajo. No integrar hasta esa confirmación.
- [x] 7. Sesgo en PH2 (por tono y por contraste lesión-piel) | sin commit
  todavía | `PYTHONPATH=src python scripts/audit_ph2_tone_vs_contrast_bias.py`
  → `resultados/segmentacion/v8/ph2_tone_vs_contrast_bias/summary.json`.
  La varianza del brazo B entre tonos queda **fuera de alcance** (2026-11-04):
  requeriría correr la rejilla de 18 combinaciones de B, protocolo no
  ejecutado (`docs/estado/clf-medicion-completa-a0-b-t.md`), trabajo mayor no
  asignado.
- [x] 7b. ¿El removedor de pelo (`hair.py`) se come el borde de la lesión? |
  `2026cc4` (overlap) | `PYTHONPATH=src python scripts/audit_hair_removal_lesion_overlap.py`
  → `resultados/segmentacion/v8/hair_removal_lesion_overlap_ph2/summary.json`
- [x] 7c. Efecto del solapamiento sobre el Dice del teacher 026, PH2 |
  `39d5e75` | corrido en UNABIA (`scripts/measure_ph2_hair_removal_shaped_dice.py`,
  checkpoint `f4fcee05...462f4` reverificado) → `resultados/segmentacion/v8/hair_removal_shaped_dice_ph2/summary.json`
- [ ] 8. Rejilla completa del brazo T

## Dónde se retoma (2026-11-04, fin de sesión)

Síntesis en `docs/63-instrumento-ita-alcance-y-limites.md`. Dos líneas abiertas.

### Línea 1 — Piel oscura: lo que bloquea el objetivo 3

**Ningún dataset de esta línea tiene piel oscura real.** El tercil "oscuro" de
ISIC 2018 cae en *intermediate/tan* de Del Bino; el pool tiene terciles de ITA
48,05 y 67,61; Benčević descarta ISIC explícitamente para evaluar sesgo y su
propio N de FST V–VI es **5 imágenes**. El objetivo de equidad del acta
**no es testeable en los datos elegidos**, y eso no lo arregla ninguna técnica.

Candidatos sin evaluar: **S-SYNTH** con fototipos sintéticos controlados — el
único con piel oscura real *y* melanina verdadera, ya descargado y verificado en
UNABIA (72 GB, sha256 coincidente con HuggingFace, máscara de lesión por
muestra) —, **DDI**, **Fitzpatrick17k** (ya usado en `docs/20c`), **Dermofit** y
**Waterloo** (los tres que usa Benčević).

Caveat de S-SYNTH, declarado antes de empezar: es render sintético, y en
aprendizaje contrastivo el encoder puede aprender «real contra sintético» como
la invariancia más fácil de romper.

### Línea 2 — Cerrar el instrumento

1. **Plausibilidad visual: nunca medida.** Corbin acota el desplazamiento por
   eso y nosotros solo medimos puntería numérica. Un desplazamiento puede
   aterrizar con error 0,00001° y producir una imagen que no parece piel; con
   recorrido alcanzable de 75° el riesgo es concreto. **Bloquea fijar el cap de
   política.** Son ~20 imágenes transformadas a ojo, no un experimento.
2. **Residuo de 2,33° de la vía B.** El solver es exacto; la salida mide
   sistemáticamente más claro que lo que la curva predice. Ocho causas
   descartadas. En `transformacion-tono`, que tiene los internos.
3. **Unificar las dos definiciones de «piel».** Higiene, no causa del residuo.

### Decisiones pendientes del usuario

- Política de abstención: hoy corre con el default «no transformar y registrar»,
  **sin confirmación**.
- Cap de política: bloqueado por la plausibilidad visual.
- Si vale correr la rejilla de 18 combinaciones para medir la varianza del brazo
  B entre tonos — la única vía que la literatura 2025 señala como prometedora y
  que se descartó sin medir.

## Resultados medidos

Error de ITA contra el techo de máscara experta, ISIC 2018, denominador común
n=1.813 (los casos donde los cuatro brazos resuelven):

| brazo | sesgo mediano | MAE | p95 |
|---|---:|---:|---:|
| sin máscara | −1,44 | 5,25° | 24,92 |
| Otsu (fallback actual) | −0,09 | 1,88° | 8,02 |
| teacher 026 | −0,03 | **0,91°** | 4,28 |

Por estrato de área de lesión, el mismo error:

| área | n | sin máscara | Otsu | 026 |
|---|---:|---:|---:|---:|
| 0,00–0,10 | 801 | 0,75 | 1,26 | 0,29 |
| 0,10–0,20 | 365 | 2,14 | 1,34 | 0,36 |
| 0,20–0,30 | 215 | 4,08 | 1,62 | 0,62 |
| 0,30–0,40 | 155 | 5,83 | 1,92 | 0,84 |
| 0,40–0,50 | 91 | 11,41 | 2,87 | 1,58 |
| >0,50 | 186 | **28,63** | 5,39 | 4,74 |

Lecturas que se sostienen:

- El segmentador reduce el error de ITA entre 2x y 6x según el estrato. No lo
  elimina: 4,74° en lesiones que ocupan más de la mitad del cuadro.
- Otsu depende del dominio. En dermatoscopia le gana a no enmascarar en todos
  los estratos ≥ 0,10; en los renders de S-SYNTH daba MAE 15,71 y era
  catastrófico. No afirmar "Otsu es dañino" sin nombrar el dominio.
- El Dice del teacher 026 medido aquí, 0,8319, reproduce el 0,8320 publicado.
- El sesgo del segmentador es de **contraste**, no de tono: Spearman con el
  Dice da contraste Lab +0,604 contra tono +0,193, y el área no participa
  (−0,014, p=0,71). El tono sobrevive como efecto propio pequeño (parcial
  +0,136, p=3e−04).

### Sesgo en PH2: no replica el patrón "contraste domina sobre tono" de ISIC2018

`scripts/audit_ph2_tone_vs_contrast_bias.py`, mismas definiciones que
`contrast.py` (ITA con máscara experta, contraste = gap de L*/Lab mediano
piel-lesión, Dice del teacher 026 `shaped=False` reusado de 7c). n=177/200
(88,5 % de cobertura; los 23 que no resuelven `estimate_fst` por
`MIN_B_FOR_ITA` tienen área mediana 0,538 contra 0,217 de los que sí resuelven
— el mismo sesgo hacia lesiones grandes que en ISIC2018, declarado antes de
usar la tabla).

Spearman contra el Dice del 026, n=177:

| variable | rho | p |
|---|---:|---:|
| ITA (tono) | +0,047 | 0,53 |
| contraste L* | +0,125 | 0,097 |
| contraste Lab | −0,003 | 0,97 |
| área de la lesión | **+0,498** | **1,7e−12** |

**No replica el hallazgo de ISIC2018** (contraste Lab +0,604, tono +0,193,
área −0,014 n.s.). En PH2 ni el tono ni el contraste predicen el Dice de
forma significativa (contraste L* p=0,097, al borde pero no cruza 0,05); lo
que sí predice, con fuerza, es el **área de la lesión** — lo opuesto de
ISIC2018. Brecha por tercil de ITA: T1 0,9573 | T2 0,9759 | T3 0,9713, brecha
+0,0140, ANOVA F=2,08 p=0,13, Kruskal-Wallis H=1,58 p=0,45 — ninguna
significativa. Contraste L*: brecha +0,0250, ANOVA p=0,0098 (significativa)
pero Kruskal-Wallis p=0,41 (no) — los dos tests discrepan, señal débil e
inconsistente entre pruebas.

Lectura, sin forzar el paralelismo con ISIC2018: PH2 tiene el Dice cerca del
techo (mediana 0,9825, ver 7c) y es un dataset mucho más chico (n=177) y
curado que ISIC2018 (n=1.813-2.594). La correlación con el área puede ser el
artefacto conocido de Dice en objetos chicos (la proporción borde/área crece
al reducirse la lesión, penalizando el Dice independientemente de tono o
contraste), no necesariamente una señal de sesgo de equidad nueva. **No se
afirma que el hallazgo de ISIC2018 sea robusto entre cohortes** — en PH2 no
se repite, y esa es la respuesta, no un hallazgo que haya que maquillar para
que coincida.

### El removedor de pelo repinta la lesión, no solo el pelo (PH2, n=200)

`scripts/audit_hair_removal_lesion_overlap.py`, resize 224×224 (igual que
`pipeline_segment_circle.segment_image`), fracción de la máscara de pelo que
cae DENTRO de la máscara experta de lesión:

| máscara | n con pelo detectado | mediana | media | p95 |
|---|---:|---:|---:|---:|
| `hair_mask` (producción, `shaped=False`) | 200/200 | 0,5171 | 0,5460 | 0,9377 |
| `elongated_hair_mask` (`shaped=True`, nadie la llama) | 198/200 | 0,2293 | 0,3061 | 0,8626 |

Más de la mitad del área que `shaped=False` marca como "pelo" y repinta con
Telea cae dentro de la lesión experta, en la imagen mediana. El filtro de
elongación reduce el solapamiento ~2,3× (no lo elimina). `hair_fraction`
también baja (mediana 0,0535 → 0,0148, 3,6×) — bastante menos que el 37× de
un cuadro sintético con líneas finas ideales; 3,6× es la cifra real a citar.
Los tres llamadores de producción (`pipeline_segment_circle.py:268`,
`segmentation_methods/common/data.py:243`, `pilot_adapters.py:174`) y la
receta pineada del teacher 026 (`generate_masks_026_pool.py` en UNABIA) dejan
`shaped=False` en su valor por defecto.

### Pero el efecto medido sobre el Dice en PH2 es minúsculo (y en la dirección contraria a la esperada)

`scripts/measure_ph2_hair_removal_shaped_dice.py`, misma receta pineada
(`cruda+hair_removal+resize224_bilinear+thr0.5`, checkpoint
`f4fcee05...462f4` reverificado), Dice contra la máscara experta de PH2
(n=200), pareado por imagen:

| máscara | Dice media | Dice mediana | sd |
|---|---:|---:|---:|
| `shaped=False` (producción) | 0,9680 | 0,9825 | 0,0557 |
| `shaped=True` (elongado) | 0,9666 | 0,9807 | 0,0558 |

Delta pareado (`shaped − unshaped`): media −0,0014, mediana −0,0008.
Estadísticamente significativo (Wilcoxon signed-rank p=1,19e-08, t pareado
p=0,0085) pero el tamaño del efecto es minúsculo — por debajo de 0,2 puntos
de Dice — y **en contra** de la hipótesis: `shaped=True` (menos solapamiento
con la lesión, medido en 7b) sale levemente PEOR, no mejor (mejora en
58/200 imágenes, empeora en 142/200). Rango: −0,0256 a +0,0623 por imagen.

Lectura honesta, sin forzarla a confirmar 7b: en PH2 el Dice ya está cerca
del techo (mediana >0,98) para ambas variantes, y la fracción de pelo
absoluta es chica (mediana 5,35 % del cuadro para `shaped=False`) — el
solapamiento relativo con la lesión que mide 7b (>50 % del área de la
máscara de pelo) no se traduce en un área absoluta de lesión repintada
suficiente para mover el Dice en PH2. **Esto NO descarta el problema**: PH2
son imágenes de 256×256 relativamente limpias; no mide qué pasa en
cohortes con más pelo real (ISIC2018, HAM10000), donde `hair_fraction`
puede ser mayor y el mismo mecanismo de solapamiento sí podría pesar. No
generalizar este resultado fuera de PH2.

## Caveats que viajan con estas cifras

- Cobertura 69,9 %. El 30 % restante se abstiene por `MIN_B_FOR_ITA = 5.0`, y
  esa abstención crece con el área de lesión (20,3 % → 39,0 %). Toda la tabla
  es condicional a los casos que resuelven.
- La abstención no es un defecto: de las imágenes que no producen ITA, el 24 %
  tiene b\* negativo, y sin la guarda se les asignaría un ITA negativo, es
  decir fototipo V–VI. Sin esa guarda, imágenes con dominante azulada se
  etiquetarían como piel oscura.
- Ningún dataset de esta línea tiene piel oscura real. El tercil "oscuro" de
  ISIC 2018 cae en *intermediate/tan* de Del Bino. Decir "brecha por tercil de
  ITA", nunca "brecha por fototipo".

## Desajustes con el protocolo, abiertos

1. El acta manda seguir el flujo de Corbin, que muestrea `ITA'` dentro del
   rango del FST objetivo. El propio `plan-segmentacion/README.md` declara que
   ITA→FST no está validado (22,9 % contra 51,3 % de la baseline de mayoría en
   PAD-UFES-20). Corbin se adapta a grados ITA continuos, no se copia.
2. El acta exige separar bien lesión y piel "también en piel oscura", pero ni
   el pool ni ISIC 2018 la tienen, y Benčević descarta ISIC para evaluar sesgo.
   El objetivo no es testeable en los datos elegidos.
3. El acta apunta al tono; la medición dice que el conductor es el contraste.

Bloqueos: ninguno para los pasos 5–8. El paso 0 los desbloqueó.

## No verificado

- `tests/test_pretrain_tone_view.py` no corre en el entorno local: falta
  `torchvision`. Queda sin ejecutar, no pasado ni fallado. Se comprueba con
  `CI_REQUIRE_FULL_DEPS=1` en un entorno con las dependencias completas.
- Los 12 módulos del paso 0 se versionaron **tal como estaban, sin revisión**,
  para frenar el riesgo de pérdida primero. Su contenido no está auditado.
