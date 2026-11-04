# 63. El instrumento ITA: qué mide, qué no, y qué cifras quedaron retiradas

Estado: vigente (2026-11-04). Reemplaza el uso informal del ITA que circulaba en
`docs/estado/*` y en el brazo tonal. Las cifras de aquí se midieron en la sesión
del 2026-11-04 sobre S-SYNTH (melanina verdadera del renderizador) e ISIC 2018
Task 1 (máscaras expertas); cada una indica su cohorte y su `n`.

Documentos relacionados: `plan-segmentacion/README.md` §1 (acta 2026-09-18),
`docs/progress/escalera-ita-mascara.md` (registro de ejecución),
`docs/estado/clf-medicion-completa-a0-b-t.md` (protocolo del contraste).

---

## 1. Qué es el instrumento

`ITA = arctan((L* − 50) / b*) · 180/π`, calculado sobre la **mediana** de L\* y
b\* de los píxeles de piel, con la lesión excluida por máscara y el vineteado
excluido por luminancia (`fitzpatrick.estimate_fst`).

Dos decisiones de implementación que ya estaban tomadas y que conviene no
reabrir:

- **`arctan` del cociente, no `atan2`.** La literatura sugiere `atan2` para
  evitar la singularidad. Se rechazó a propósito: `atan2` es consciente de
  cuadrante y envolvería las lecturas de b\* negativo en (90, 180], rompiendo el
  invariante «el ITA no puede salir de [−90, 90]». Además `atan2` corrige
  ambigüedad de cuadrante, **no** mal condicionamiento: con b\* → 0 el ángulo
  sigue siendo hipersensible al ruido con cualquiera de las dos fórmulas.
- **Mediana, no media.** El punto de ruptura de la mediana es 50 %, y de ahí
  sale buena parte del comportamiento de la sección 3.

## 2. El ITA se gana su lugar frente a la luminancia desnuda

Pregunta razonable: si `b*` por sí solo no correlaciona con la melanina
(Spearman **+0,044**, p=0,44, n=300 en S-SYNTH), ¿para qué dividir por él?

Medido sobre S-SYNTH, n=794, pareado sobre los casos donde el ITA está definido:

| | rho con melanina verdadera |
|---|---:|
| L\* solo | −0,7039 |
| **ITA** | **−0,7287** |
| rho(L\*, ITA) | +0,9822 |

Prueba de Steiger para correlaciones dependientes: **z = +5,327, p < 0,0001**.
El ITA es mejor de forma significativa.

Y la descomposición de varianza explica el mecanismo (19 iluminaciones,
80 modelos de piel):

| variable | melanina | **iluminación** | modelo de piel |
|---|---:|---:|---:|
| L\* | 0,563 | **0,272** | 0,297 |
| ITA | 0,582 | **0,177** | 0,316 |

`b*` no está ahí para **aportar** señal de melanina: está para **normalizar la
iluminación**. El ITA es un cociente, y un cociente de dos cantidades
contaminadas por el mismo factor de ganancia cancela parte de ese factor. Por
eso `b*` solo parece ruido —lo domina la luz— y el cociente queda **35 % menos
contaminado** que el numerador solo.

Dato de contexto: el estimador de referencia de Benčević mide `eta2(luz) = 0,224`
sobre su CSV publicado. El nuestro, enmascarado y sobre imágenes reales, mide
**0,177**.

## 3. El 28 % que no produce ITA: causa y alternativas agotadas

Sobre ISIC 2018 Task 1, **con máscara experta**, los estados de
`estimate_fst` son:

| estado | n | % |
|---|---:|---:|
| `ok` | 1.872 | 72,2 % |
| `ita_indeterminate` | 718 | **27,7 %** |
| `degenerate_mask` | 4 | 0,2 % |

**No es falta de piel.** En esos 718 casos la fracción de piel mediana es
**0,790**. Lo dispara `MIN_B_FOR_ITA = 5.0`: la mediana de b\* de la piel cae
bajo el umbral y el ángulo saturaría.

Perfil del grupo que falla contra el que resuelve (n=300 cada uno):

| | con ITA | sin ITA |
|---|---:|---:|
| L\* | 67,58 | **70,98** |
| a\* | +8,56 | +4,71 |
| b\* | +11,53 | **+2,03** |
| C\* (croma) | 15,15 | 6,17 |

Es piel **desaturada y azulada**, y además **más clara**, no más oscura.

### La abstención es una guarda, no una pérdida

De las imágenes que no producen ITA, el **24 % tiene b\* negativo**. Sin la
guarda, `arctan((L−50)/b)` con b negativo da un **ITA negativo**, que en la
tabla de Groh es fototipo V–VI. **Sin `MIN_B_FOR_ITA`, imágenes con dominante
azulada se etiquetarían como piel oscura.** La guarda está tapando una fuente
grande de sesgo tonal espurio.

### Tres alternativas probadas y descartadas, con evidencia

1. **Constancia de color.** El repo ya tiene `pretrain/color_constancy.py` y
   `t4_sog.py`, sin cablear a la cadena de tono. Probado: **120/120 siguen
   `ita_indeterminate`**; b\* pasa de −0,85 a −0,38 contra un umbral de 5,0.
   Gray-world tampoco recupera ninguna. La dominante azulada **no** es un
   artefacto global de balance de blancos.
2. **Parametrización L\*–h\*.** El 59 % de las abstenciones conserva croma
   usable (C\* > 5), así que *técnicamente* recuperaría ~16 puntos de cobertura
   (72 % → 88 %). Pero el sustituto es peor instrumento: h\* da rho −0,490
   contra −0,650 del ITA, y L\* arrastra 0,272 de `eta2` de luz contra 0,177.
   Si alguna vez se adopta, **declarar que ese 16 % se mide con un instrumento
   más débil y no mezclarlo en la misma columna**.
3. **Sustituir el ITA por L\*.** Refutado en la sección 2.

**Recomendación vigente: abstenerse y declarar la cobertura.** Formulación
sugerida: «la intervención aplica sobre el 72 % de la cohorte; el 28 % restante
tiene b\* de piel bajo el umbral donde el ángulo ITA satura, y se abstiene por
diseño».

### El sesgo del condicionamiento, que hay que declarar siempre

La tasa de abstención **crece con el área de la lesión**: 20,3 % (área 0–0,10)
→ 28,4 → 31,6 → 38,1 → 39,0 → 35,4 (>0,50). Los casos que no resuelven tienen
área mediana 0,200 contra 0,121 de los que sí. **Toda cifra calculada sobre el
subconjunto que resuelve es condicional, y el condicionamiento aprieta justo en
el estrato de lesiones grandes.**

## 4. El ITA no clasifica fototipo, y eso ya se midió

Contra el campo `fitspatrick` de PAD-UFES-20 (n=637):

| configuración | acierto |
|---|---:|
| cortes de Groh importados | 19,4 % |
| cortes por cuantiles | 38,6 % |
| cortes optimizados | 53,3 % |
| **base trivial (predictor constante)** | **55,8 %** |

Recalibrar recupera ~34 puntos y **aun así no supera la constante**. Los cortes
optimizados salen `[68,3; −44,5; −47,5; −65; −75]`: el optimizador **convergió
al predictor constante**. Spearman entre ITA y FST real: **+0,027** (p=0,50).
No hay señal de orden contra esa etiqueta.

**Pero eso es un problema de la etiqueta, no del estimador.** Contra la melanina
controlada de S-SYNTH el mismo código da Spearman **−0,758**. El campo
`fitspatrick` de PAD es un atributo del **paciente** (autorreporte o asignación
clínica), no del tono visible en la foto.

Consecuencia operativa, ya adoptada: **se trabaja en grados ITA continuos, nunca
en categorías FST por imagen.** El IQR dentro de una melanina fija es ~20 grados
y las categorías de Groh miden 12–23, así que la incertidumbre de una imagen es
comparable al ancho de una categoría.

## 5. Desajuste abierto: el cap de desplazamiento

`tone_augment_lab.MAX_ABS_ITA_DELTA = 25.0` se deriva, según su propio
docstring, de los rangos de **Corbin** (T1>55 … T6≤10): anchos 14/13/9/9, dos
categorías abarcan 18–27 grados.

Pero el estimador clasifica con **Groh**, `FST_THRESHOLDS = (40, 23, 12, 0, −25)`:
anchos 17/11/12/25, dos categorías abarcan **23–37 grados**, mediana 28.

**El cap usa una tabla y el estimador otra.** Y la incoherencia es doble: el cap
se justifica como «dos categorías FST», una categorización que la sección 4 ya
retiró. Cambiar 25 por 28 arreglaría la tabla, no el fundamento.

Lo que Corbin acota de verdad no son categorías: es **plausibilidad visual**
(«results beyond this maximum value did not yield satisfactory visual
outcomes»). El reemplazo principiado es el **recorrido alcanzable por imagen**
—cota física, medida por caso— pero eso deja abierta la plausibilidad visual,
que **este proyecto nunca midió**.

El recorrido depende fuerte de la cohorte, así que la cifra hay que citarla con
su procedencia:

| cohorte | recorrido alcanzable, mediana | n |
|---|---:|---:|
| S-SYNTH | 75,4° | — |
| **pool real (ISIC 2019, teacher 026)** | **70,1°** (p05 50,9, p25 59,5) | 792 |

La del pool real se midió el 2026-11-04 con
`scripts/audit_pool_tone_eligibility.py` sobre las 792 imágenes medibles y
alcanzables de una muestra de 1.500
(`resultados/segmentacion/v8/pool_tone_eligibility/`). Es la **primera**
medición del recorrido sobre dermatoscopia real; la de 75,4° venía de S-SYNTH y
no se debe presentar como propiedad del pool de trabajo.

## 6. Cifras retiradas

No reutilizar ninguna de estas. Cada una con lo que la reemplaza.

| cifra retirada | dónde aparecía | reemplazo |
|---|---|---|
| delta del brazo B **−0,047** | `seg-mascaras-026-pipeline.md` | **−0,0934**; el pareado era cruzado (MoCo-42 contra A0 de SimSiam-42) |
| **1.917** coincidencias ISIC2018 | `docs/46` | **583**; la vieja comparaba solo el sufijo numérico, con 1.334 falsos positivos |
| «el ITA **ordena** bien» | afirmación previa | retirada: 70,9 % dentro de ±1 nunca se comparó contra la base trivial de ±1, que es 95,1 % |
| «la luz mueve el instrumento **más que la biología entera**» | análisis del CSV de Benčević | falso con sus propios datos: comparaba rango de **medias** contra **max−min** crudo. Medias contra medias: melanina 34,9° contra luz 19,3° |
| `b*` **es ruido**, L\* basta | hipótesis propia | refutada: Steiger p<0,0001 a favor del ITA (sección 2) |
| el 026 **elimina** la dependencia del área | medición en S-SYNTH | artefacto del rango de área limitado de S-SYNTH (tope 0,260). En ISIC2018 el MAE va de 0,29 a 4,74, factor 16x |
| reducción **37x** del filtro de forma de pelo | cuadro sintético propio | **3,6x** sobre PH2 real (mediana `hair_fraction` 0,0535 → 0,0148) |
| «Otsu es dañino» | medición en S-SYNTH | depende del dominio: MAE 15,71 en S-SYNTH, **1,88** en dermatoscopia, donde le gana a no enmascarar |

## 7. Hueco de la literatura que esta línea llena

Una búsqueda del 2026-11-04 **no encontró ninguna fuente que reporte un
desplazamiento de ITA realizado y validado en grados** para augmentación tonal.
La mayoría de los pipelines de equidad en dermatología usa augmentación
fotométrica genérica (brillo 0,8–1,1, rotaciones, volteos), que no es una
transformación tonal validada; un trabajo de 2025 sobre generativos admite que
los estudios «demonstrate technical feasibility but lack rigorous validation of
morphological, textural, and pigmentation properties».

Y la recomendación explícita del campo es exactamente el procedimiento de la vía
B de esta línea: *«measuring pre/post ITA on segmented healthy skin and reporting
the achieved Δ°ITA distribution, rather than reporting a brightness multiplier»*.

Medición propia, n=33 transformadas sobre S-SYNTH, con el módulo v2:

- razón logrado/pedido: **mediana 1,0889** (antes de corregir: **0,37**)
- desplazamiento **logrado**: mediana −9,55°, p10 −28,67, p90 +14,92,
  rango [−50,7; +33,0]; oscurece 36 %, aclara 64 %
- b\* de la piel: 27,65 → 19,93, **nunca negativo (0 %)**

## 8. No verificado

- **Plausibilidad visual de la transformación.** Corbin acota por eso y nosotros
  solo medimos puntería numérica. Un desplazamiento puede aterrizar con error
  0,00001° y aun así producir una imagen que no parece piel. Con recorrido
  alcanzable de 75°, el riesgo es concreto. **Bloquea fijar el cap de política.**
  - **Artefacto construido (2026-11-04), falta el juicio humano.** Hoja de
    contactos de 20 pares antes/después sobre S-SYNTH real, estratificada por
    delta **logrado** (no por melanina: los extremos son donde una imagen deja
    de parecer piel, y estratificar por melanina promedia justo esa cola).
    Generador: `scripts/make_tone_contact_sheet.py`. Salida en
    `resultados/segmentacion/v8/tone_contact_sheet/`. El recorrido cubierto va
    de −58,91° a +58,88°. Lo que falta sigue siendo irreductiblemente humano:
    mirarla.
  - **Hallazgo medible que acota el cap sin esperar al ojo.** Sobre los 745
    casos aplicados, 129 (17,3 %) pierden medibilidad del ITA *después* de
    transformar (`ita_indeterminate`), y el mecanismo es un corte limpio:
    `b*` de salida máximo 4,973 en los no medibles contra mínimo 5,035 en los
    medibles — exactamente `MIN_B_FOR_ITA = 5,0`. La pérdida es
    **completamente asimétrica**:

    | sentido | n | pierden medibilidad |
    |---|---|---|
    | oscurecer | 215 | **0 (0,0 %)** |
    | aclarar | 530 | **129 (24,3 %)** |

    Y en el lado de aclarado crece con la magnitud: 4,0 % hasta 20°, 8,9 %
    hasta 30°, 14,2 % hasta 40°, 24,3 % sin cap.

    **La causa está en las ecuaciones del paper, no en nuestro código.** Los
    signos de las eqs. 8–9 de Corbin están **acoplados**:
    `delta_b = +t·0,50` y `delta_l = −t·0,12`. Aclarar exige `t < 0` para que
    `delta_l > 0` suba L\*, y ese mismo `t < 0` hace `delta_b < 0`, que empuja
    b\* hacia cero **en la misma operación**. Oscurecer (`t > 0`) mueve los dos
    canales al revés: sube b\*, lo aleja de cero. No depende de nuestra curva,
    ni del feather, ni del submuestreo. Verificado por los dos lados: por
    `transformacion-tono` llamando a `augment_skin_tone` directo, y de forma
    independiente aplicando sólo las eqs. 8–9 a mano.

    De ahí sale un **umbral analítico** cerrado, que es la forma útil del
    hallazgo: aclarar `|t|` grados cruza `MIN_B_FOR_ITA` cuando
    **`b*_inicial < 5 + 0,50·|t|`**. Un cap de 25° en aclarado exige
    `b*_inicial ≥ 17,5`.

    **Y acá hay un riesgo de generalización que S-SYNTH esconde.** El b\* de
    piel de S-SYNTH tiene mediana **29,26** (p05 12,89). El grupo `ok` de
    ISIC 2018 tiene b\* mediana **11,53** (sección 3) — *por debajo del p05 de
    S-SYNTH*. Es decir que en dermatoscopia real la mayoría de las imágenes no
    llega al `b*_inicial ≥ 17,5` que un aclarado de 25° necesita. **El 24,3 %
    medido en S-SYNTH es un piso optimista, no una estimación.** Sobre
    S-SYNTH el mismo umbral analítico predice 10,2 % de fallas a 25° y 33,2 %
    a 40°.

    Un detalle que conviene no exagerar: el escenario de b\* **negativo** —que
    sin la guarda daría ITA negativo, o sea fototipo V–VI en Groh, es decir
    *fabricar* una etiqueta de piel oscura al aclarar— es analíticamente real
    (con `b*_inicial = 8` y `t = −25` da b\* = −4,5, ITA = −78,9°) pero **no
    ocurrió en S-SYNTH**: 0 de 129 no medibles tienen b\* negativo, el mínimo
    fue +0,237. La guarda los atrapa antes. Queda anotado como el modo de falla
    a vigilar cuando esto corra sobre dermatoscopia real de b\* bajo, no como
    algo observado.

    **Esto no coincide con la literatura que usamos.** El ±25 de Corbin es
    **simétrico**; medido en nuestro pipeline, oscurecer no degrada nada y
    aclarar sí, de forma monótona. Un cap simétrico o paga de más en el lado
    que no falla, o deja pasar el que sí. La consecuencia de diseño es que el
    cap debería ser **asimétrico**, y que el criterio no tiene por qué ser
    estético: «la salida sigue siendo medible por el mismo instrumento que
    fijó el objetivo» es una condición verificable. **No se fija el número
    todavía** — la hoja de contactos puede mover el límite del lado de
    aclarado hacia abajo, y ese juicio es de María Paula.
- **El residuo de 2,33° de la vía B: causa aislada y corregida, falta
  re-medirlo en S-SYNTH real.** El solver acertaba exacto contra su propia curva
  (0,00000) pero la salida medía sistemáticamente **más claro**: +0,30 a +25,24,
  siempre positivo, con los dos signos de `t`. Ocho causas se habían descartado
  por medición (feather, submuestreo por stride, ITA inicial, fidelidad de b,
  fidelidad de L, recorte, desajuste de conjuntos, signo invertido). La causa
  real era una **agregación distinta**: `_skin_state_at_t` calculaba
  `median(arctan por píxel)` — mediana de los ángulos — mientras
  `fitzpatrick._skin_ita`, y por lo tanto todo lo que mide la salida, calcula
  `arctan(median(L), median(b))` — ángulo de las medianas. `arctan` es no
  lineal, así que son estadísticos distintos en cuanto hay varianza entre
  píxeles, es decir siempre en una imagen real. Corregido en `6997206`
  (ángulo-de-medianas en los dos lados). Sobre un lote **sintético con textura**
  de 150 imágenes, el error mediana contra `estimate_fst` sobre la salida real
  bajó de 3,62° a 0,21°, y el máximo de 10,54° a 1,87°.
  - **CERRADO sobre S-SYNTH real (2026-11-04).** 2000 muestras sobre 50 grupos
    de melanina verdadera, misma semilla 42, mismo índice, `feather_sigma=0,0`,
    sin cap; única variable que se mueve, el módulo. 745 casos aplicados, 616
    con ITA medible después. Datos y scripts en
    `resultados/segmentacion/v8/via_b_cierre/`.

    | \|error\| contra `ita_objetivo` | antes (n=33) | después (n=616) |
    |---|---|---|
    | mediana | 2,3520 | **0,1148** |
    | p95 | 20,7190 | **0,5542** |
    | máximo | 25,2846 | **1,3275** |
    | positivos | 29/33 (87,9 %) | 189/616 (30,7 %) |

    Dos correcciones al relato anterior. El residuo **no era «siempre
    positivo»**: eran 29 de 33, con 4 negativos. Y el sesgo direccional
    desaparece tras el fix (30,7 %), que es lo que se espera si la causa era
    agregación no lineal y no un mecanismo de signo fijo. Por tercil de
    melanina verdadera el error sube monótono pero poco: 0,0934 (bajo),
    0,1209 (medio), 0,1420 (alto).
  - Hallazgo separado del mismo pase: el mecanismo de borde que era la primera
    hipótesis **sí es real**, solo que chico. Con población oscura suficiente
    (15 % de sombra) un desplazamiento grande hace que hasta 11 % de los píxeles
    cruce el umbral de borde *después* de transformar, moviendo la mediana
    ~1,85° en ese caso. No se corrigió moviendo la curva — excluir esos píxeles
    por paso de `t` rompía la monotonicidad. Se agregó `_population_shift_fraction`
    y la abstención `border_population_shift_too_large` sobre
    `MAX_POPULATION_SHIFT_FRACTION = 0,01`, **constante declarada, no medida**.
    Intentamos calibrarla y **no se puede con este dataset**: sobre los 745
    casos aplicados de S-SYNTH real, `population_shift_fraction` vale
    **exactamente 0,000000 en los 745** (p100 = 0). La guarda nunca se dispara
    acá, así que el umbral es infalsificable sobre S-SYNTH y el 0,01 sigue
    siendo declarado. Calibrarlo exige dermatoscopia real con vineteado o
    sombra; anotado como lo que es, no como cifra validada.
    Y ojo con atribuirle el cierre: los casos que superaban ese umbral tenían
    *menos* error final (1,03° contra 3,62°). El dominante es la agregación.
- **Dos definiciones de «piel» en el pipeline — RESUELTO (commit `99bad30`).**
  `tone_normalization._skin_selector` (`~lesión & valid_fov`) decide qué se
  transforma; `fitzpatrick._resolve_border` (que además excluye `gray < 0,02`)
  decide qué se mide. En S-SYNTH coincidían (0,0000 % de diferencia) pero en
  dermatoscopia real con vineteado no tenían por qué. No se editó
  `_skin_selector`: eso habría cambiado el comportamiento efectivo de
  `tone_augment_lab.augment_skin_tone` (congelado) y de los resultados ya
  reportados de `tone_normalization.py` para todo llamador. Se agregó
  `_unified_skin_fov` en `tone_augment_lab_v2.py`, que pliega la exclusión de
  borde casi-negro dentro del `valid_fov` que se pasa a la función congelada
  — `_skin_selector(lesión, fov_unificado)` queda matemáticamente igual a
  `~lesión & ~borde`, sin tocar `_skin_selector`, `tone_normalization.py`,
  `tone_augment_lab.py` ni `fitzpatrick.py`. Test de regresión con un
  vineteado sintético: S-SYNTH no puede probar esta unificación (coincide al
  0,0000 % ahí), mismo límite del dataset que bloquea calibrar
  `MAX_POPULATION_SHIFT_FRACTION` (Tarea 1C).
- **El sesgo por contraste no replica entre cohortes.** En ISIC 2018 el contraste
  lesión-piel predice el Dice (rho +0,604) mucho mejor que el tono (+0,193), y el
  área no participa (−0,014, p=0,71). En PH2 (n=177) ni tono ni contraste son
  significativos y domina el **área** (+0,498, p=1,7e−12). PH2 tiene el Dice en el
  techo (mediana 0,9825) y n es diez veces menor, pero **eso no convierte un
  no-replica en una replica**. Formulación defendible: «medido en ISIC 2018, no
  replicado en PH2», con las dos cifras a la vista.
