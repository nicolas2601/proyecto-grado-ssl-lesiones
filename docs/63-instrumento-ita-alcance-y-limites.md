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
—cota física, medida por caso, mediana **75,4 grados**— pero eso deja abierta la
plausibilidad visual, que **este proyecto nunca midió**.

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
- **El residuo de 2,33° de la vía B.** El solver acierta exacto (curva en `t`
  contra objetivo: 0,00000), pero la imagen de salida mide sistemáticamente
  **más claro** que lo que la curva predice: +0,30 a +25,24, siempre positivo,
  con los dos signos de `t`. Ocho causas descartadas por medición (feather,
  submuestreo por stride, ITA inicial, fidelidad de b, fidelidad de L, recorte,
  desajuste de conjuntos, signo invertido). Causa sin aislar.
- **Dos definiciones de «piel» en el pipeline.** `tone_normalization._skin_selector`
  (`~lesión & valid_fov`) decide qué se transforma; `fitzpatrick._resolve_border`
  (que además excluye `gray < 0,02`) decide qué se mide. En S-SYNTH coinciden
  (0,0000 % de diferencia) pero en dermatoscopia real con vineteado no tienen
  por qué. Hay que unificarlas por higiene, aunque no sean la causa del residuo.
- **El sesgo por contraste no replica entre cohortes.** En ISIC 2018 el contraste
  lesión-piel predice el Dice (rho +0,604) mucho mejor que el tono (+0,193), y el
  área no participa (−0,014, p=0,71). En PH2 (n=177) ni tono ni contraste son
  significativos y domina el **área** (+0,498, p=1,7e−12). PH2 tiene el Dice en el
  techo (mediana 0,9825) y n es diez veces menor, pero **eso no convierte un
  no-replica en una replica**. Formulación defendible: «medido en ISIC 2018, no
  replicado en PH2», con las dos cifras a la vista.
