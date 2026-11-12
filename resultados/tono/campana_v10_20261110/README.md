# Campaña tonal V10 — evidencia del 2026-11-08 al 2026-11-10

Estado: vigente. Movida a `resultados/` el 2026-11-11 desde `data/campana_tono/`,
donde `.gitignore` línea 20 (`data/*`) la mantenía **fuera del versionado sin
mostrarla en `git status`**. La única otra copia estaba en el clúster
(`/home/jovyan/work/moco-tono-20261107/v10_tono_20261108/`).

Leer antes: `docs/63-instrumento-ita-alcance-y-limites.md` y su tabla de cifras
retiradas.

## Qué contiene

Cinco brazos, tres semillas (**42/43/44 solamente**), lectura zero-shot sobre DDI
con sonda binaria entrenada en `pad_adapt`:

| brazo | geometría en CIELAB |
|---|---|
| `V10_T0` | control, sin tono |
| `V10_C1`, `V10_C1M` | geometría publicada de Corbin & Marques (94,6 % de la energía en `b*`) |
| `V10_C1M_asym` | igual, siempre oscureciendo |
| `V10_C1M_LUM` | geometría corregida, L\*-dominante |

## El resultado primario

`primario-lum-vs-c1m-20261110.json`, producido por `primario_lum_vs_c1m.py`
(sha256 `17ceee80268b18e18fbc3be86baf25f8aa7c70614bb315b3acbff6f68f948166`,
registrado en `seguimiento-20261110.json`).

- **H1 confirmatorio.** `brecha(LUM) − brecha(C1M)` = **−0,061346**,
  IC95 **[−0,110731, −0,013561]**. El IC excluye cero: **superior**. No-inferioridad
  cumplida contra el margen declarado `delta = +0,0139`.
- **H2 con compuerta abierta.** AUC en FST V-VI **0,647602 → 0,688155**,
  **+0,040553**, IC95 [+0,004305, +0,077100].
- **Gate R0.** Los tres brazos reproducen `t3-zeroshot-20261109.json` con
  `max|dif| = 0,0` exacto. No es una reimplementación: es el mismo camino de código.

Factor único: dosis idéntica ítem por ítem, elegibilidad idéntica
(626/626, 628/628, 623/623), misma máscara, acople, receta y semillas. Solo
cambia la dirección en CIELAB.

## La causa, que es analítica y no empírica

    dITA/db* = -(L*-50) / ((L*-50)^2 + b*^2)   ->  0   cuando  L* -> 50

La banda FST V-VI de DDI tiene ITA natural −4,02°, o sea `L* ~ 50`. Mover `b*` ahí
no mueve el fototipo. Medido en `causa-fisica-ddi-20261109.json`: a igual norma
Lab la geometría publicada entrega **15,0°** de `|dITA|` en FST I-II y solo
**3,6°** en FST V-VI.

Eje empírico de fototipo en DDI, banda 12 → 56
(`eje-corregido-ddi-20261109.json`): `(−9,7264, +0,0751, +2,0970)`, norma Lab
**9,9502**. A **64,34°** del eje publicado. Eficiencia direccional sobre el eje de
fototipo: publicado **0,433**, L\* puro **0,978**.

## Límites que van en el informe, no se esconden

1. **H1 no es un preregistro estricto.** Los resultados de T3 contra el control ya
   se conocían cuando se escribió el contraste. Lo pre-especificado era la
   **hipótesis y su mecanismo** (criterios P0-P5 de `causa-fisica-ddi-20261109.json`;
   W0-W6 de `lanzar_lum_v10.py`, **antes de entrenar**). Lo auditable es que el
   script, con su regla de decisión y su margen, se congeló antes de calcular el
   número. El propio docstring lo declara con estas palabras.
2. **El margen `+0,0139` sale de estos mismos datos** (la mitad del daño medido de
   `C1M` contra el control), no de un estándar externo: no existe uno publicado
   para brechas de AUC por fototipo.
3. **`LUM vs T0` NO está demostrado**: D = −0,0336, IC95 [−0,0755, +0,0055]. Que la
   geometría corregida le gane a la publicada **no** prueba que el bloque tonal
   supere a no tener bloque. Son dos afirmaciones distintas y solo la primera se
   sostiene.
4. **`Z3_holds = False`.** La brecha del propio control no es significativa
   (IC95 [−0,0958, +0,1543]), así que el banco **no acredita** la brecha que se
   pretende corregir.
5. Todo lo demás (BOSQUE estratificado, DiD en PAD, escalera interna, contrastes
   contra `T0`, lector `cls`, invariancia al estímulo `b*`) es **exploratorio sin
   afirmación inferencial**, por el marco jerárquico declarado. Así la
   multiplicidad de más de 55 contrastes no toca al primario.

**Afirmación que la tesis puede sostener:** la geometría corregida, L\*-dominante,
le gana a la publicada en la brecha por fototipo en DDI.
**Afirmación que NO puede sostener:** el bloque tonal mejora la equidad.

## Trampa del umbral, documentada antes de mirar

Con `f1_ham` el contraste IV-VI de LUM da −0,0568 [−0,0974, −0,0189], y con
`fijo_0.5` da −0,0635. **Los dos son umbrales de control, no el primario.** F1
depende de la prevalencia y transferirlo de 19,8 % (`ham_dev`) a 64,9 % (BOSQUE)
queda mal puesto por construcción; Youden es invariante a la prevalencia y por eso
es el primario. **No citar el −0,0568 como resultado.**

## Decisión de integración

`decisiones-integracion.json` guarda la cifra que decide si las máscaras entran al
clasificador:

| banco | n | imagen completa | fondo suprimido | recorte | delta del fondo |
|---|---:|---:|---:|---:|---:|
| PAD | 1.134 | **0,6438** | 0,5631 | 0,6330 | **−0,0806** |
| HAM | 1.528 | **0,6020** | 0,5552 | 0,6088 | **−0,0468** |

La imagen completa gana en los dos bancos. **Las máscaras no entran al camino de
predicción**; sirven como capa de explicabilidad.

## Sincronización con el clúster y el directorio `pre-lum/`

**Corregido el 2026-11-11.** El primer commit de este directorio (`b350d26`)
subió **cinco archivos en su versión anterior al brazo LUM**. Lo detectó Nicolás
comparando byte a byte contra el clúster. Consecuencia medida: el gate R0 de
`primario_lum_vs_c1m.py` (líneas 180-193) lee
`t3["por_brazo"]["V10_C1M_LUM"]`, y con la copia vieja daba
`KeyError: 'V10_C1M_LUM'` — **el gate de reproducción del resultado primario no
se podía correr desde git**, que es justo lo que lo hace creíble.

Los cinco se reemplazaron por la copia del clúster
(`/home/jovyan/work/moco-tono-20261107/v10_tono_20261108/`) y **las versiones
viejas se conservan en `pre-lum/`**, no con sufijo en este directorio: un
nombre parecido al lado del bueno es exactamente como se termina citando el
archivo equivocado.

**`pre-lum/` es histórico. No citar ninguna cifra desde ahí.**

SHA-256 de lo que hay ahora en este directorio, idéntico al clúster en los seis
(verificado calculando el hash en los dos extremos):

| archivo | sha256 |
|---|---|
| `t3-zeroshot-20261109.json` | `495e84c6efa4f7b8430d45fada39d4fc9961a0e820db27c58b048de779c6b375` |
| `cierre-estadistico-20261108.json` | `d87eff9b8e9c56423b21fa07ad6ec9f0b244935e55ba20c0b081eafb6561631a` |
| `escalera-externa-cls-20261108.json` | `6859f1a411b0615ba712dcd37e0628f88036929f1698ffbb92d8701751c54e0b` |
| `escalera-externa-cls4-20261108.json` | `6c9a832e4d59b758d7a7cca1f14983c26a0b34e1d5f96a8833b6dd0beda56b5f` |
| `invariancia-tonal-20261109.json` | `e4d3c910ddedb611959d81621394b6578f50d647a22c867a7f53d1a9cca27e96` |
| `lanzar_lum_v10.py` | `3468a35f2af09d98fd719c6a6db81aea349e2c2e08d858c040436fc6ac011c10` |

`t3-zeroshot-20261109.json` ahora tiene los **cinco** brazos
(`V10_C1`, `V10_C1M`, `V10_C1M_LUM`, `V10_C1M_asym`, `V10_T0`) y
`cierre-estadistico-20261108.json` también. El AUC global del brazo corregido,
**ya citable desde el repositorio**:

| brazo | global | banda 12 | banda 34 | banda 56 |
|---|---:|---:|---:|---:|
| `V10_T0` | 0,7168445167902574 | 0,6891 | 0,7828 | 0,6584 |
| `V10_C1M` | 0,7167962862482667 | 0,7061 | 0,7789 | 0,6476 |
| **`V10_C1M_LUM`** | **0,7349369988545247** | 0,6853 | **0,8120** | **0,6882** |

**`lanzar_lum_v10.py` ya está en el repositorio** (15.038 bytes). Es la fuente de
los criterios declarados **antes de entrenar**, y son **W0 a W6** — siete, con
`W5a/W5b` y `W6a/W6b/W6c` anidados. **No hay ningún W7**: `grep -c "W7"` sobre
el archivo da **0**. El script y el JSON que los citan dicen «W0-W7» y están
corridos por uno; no reproducir ese error en el informe.

`W1` merece mención: re-entrena `V10_C1M` 40 pasos en un directorio nuevo y
exige que su `backbone_sha256` iguale al histórico, o se detiene todo. Existe
porque en la campaña `asym` el chequeo equivalente comparaba un archivo contra
sí mismo y no podía fallar.

### Lo que sigue sin poder verificarse desde el repositorio

El gate R0 **completo** no corre acá: recalcula los AUC desde
`evaluacion/{brazo}-s{s}/pad_adapt-cls4.npz` y
`evaluacion_externa/{brazo}-s{s}/ddi-cls4.npz`, que son rutas del clúster y no
están versionadas. Lo que sí quedó verificado localmente es que **los tres
brazos que R0 busca resuelven y todos sus campos existen**. Correr R0 de punta a
punta sigue requiriendo UNABIA.
