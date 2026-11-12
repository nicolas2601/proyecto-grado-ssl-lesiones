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
   W0-W7 de `lanzar_lum_v10.py`, **antes de entrenar**). Lo auditable es que el
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

## Dependencias externas al repositorio

- **`lanzar_lum_v10.py` no está en el repo.** Es la única fuente de los criterios
  W0-W7 declarados antes de entrenar, y vive solo en UNABIA. Traerlo es condición
  para poder citar esa pre-especificación en el informe.
- La copia de `t3-zeroshot-20261109.json` de este directorio tiene **cuatro
  brazos** y no incluye `V10_C1M_LUM`; la del clúster sí. No citar un AUC global
  del brazo LUM desde este archivo.
