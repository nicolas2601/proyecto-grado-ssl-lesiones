# Aporte propio: la medibilidad del ITA depende del diagnóstico, y el desplazamiento logrado en grados

Estado: vigente. Escrito el 2026-11-05.

Leer antes: `63-instrumento-ita-alcance-y-limites.md`, que fija qué mide el ITA y
qué no, y trae la tabla de cifras retiradas.

Este documento reúne **dos resultados que no dependen de que el contraste de
tono salga positivo o negativo**. Los dos se sostienen solos, los dos están
medidos sobre artefactos que ya estaban en disco, y ninguno de los dos aparece
reportado en la literatura que se revisó.

Reproducible con `scripts/measure_ita_measurability_vs_diagnosis.py`, que deriva
cada cifra de los JSON y no copia ninguna a mano.

---

## 1. Aporte uno: la abstención del ITA no es aleatoria respecto al diagnóstico

### El problema que esto expone

El ITA (*Individual Typology Angle*) no se puede calcular sobre cualquier imagen.
Cuando el `b*` de la piel sana es demasiado bajo, el ángulo queda indeterminado y
el caso se abstiene. Eso es correcto y es lo que hace cualquier implementación
responsable.

**Lo que nadie reporta es a quién se le abstiene.** Toda métrica de equidad por
tono de piel que descarte lo no medible publica su resultado sobre el
subconjunto que queda. Si ese subconjunto no es una muestra aleatoria de los
diagnósticos, la métrica describe una población distinta de la que el lector
supone, y nada en el reporte habitual lo hace visible.

### Lo medido, cohorte A: HAM10000

Fuente: `resultados/segmentacion/v8/tone_stratified_eval/summary.json`.
2.004 predicciones, cobertura de manifiesto 1,0, **855 medibles (42,66 %)**. El
estrato no medible es el complemento exacto, no una tercera lista.

| | en el split completo | en el subconjunto medible |
|---|---:|---:|
| melanoma (`mel`) | 11,08 % | **4,44 %** |
| nevus (`nv`) | 66,92 % | **85,15 %** |
| malignas (`mel`+`bcc`+`akiec`) | 19,51 % | **9,94 %** |

**El subconjunto medible tiene la mitad de casos malignos que el split del que
sale, y es cuatro quintos nevus.**

Prueba exacta de Fisher sobre medible × maligno, con la agrupación canónica del
proyecto (`dermssl.ddi.HAM_MALIGNANT`, importada y no reescrita):

| lectura | tasa de malignas, medible | tasa, no medible | odds ratio | p |
|---|---:|---:|---:|---:|
| primaria, incluye `akiec` | 9,94 % | 26,63 % | **0,3041** | **9,25e-22** |
| estricta, excluye `akiec` | 7,72 % | 22,54 % | 0,2874 | 3,44e-20 |

La lectura estricta existe porque `akiec` es premaligna o in situ por
definición, no inequívocamente maligna. **El resultado sobrevive a excluirla**,
así que no es un artefacto de dónde se traza esa frontera.

### Lo medido, cohorte B: pool SSL de ISIC2019

Fuente:
`resultados/segmentacion/v8/pool_tone_eligibility/summary_pool_real_isic2019.json`.
1.500 imágenes, 53,53 % medibles.

| | medible | no medible | odds ratio | p |
|---|---:|---:|---:|---:|
| tasa de positivos | 13,84 % | 22,70 % | **0,5469** | **0,0099** |

**Caveat que el propio reporte declara y que hay que citar con la cifra:** esto
es condicional al **39,6 %** del pool que trae etiqueta (594 de 1.500), no al
pool completo.

Esta cohorte es independiente de la A en dataset, en ruta de enmascarado y en
etiquetado, así que la coincidencia de dirección es **replicación**, no la misma
medición reportada dos veces.

### El tamaño de la lesión no lo explica

La objeción obvia es que una lesión grande deja menos piel sana visible, así que
el efecto sería un proxy del área. Se controló por tercil de área de lesión:

| tercil de área | positivos, medible | positivos, no medible |
|---|---:|---:|
| 0 (menores) | 6,35 % | 12,50 % |
| 1 | 14,47 % | 19,67 % |
| 2 (mayores) | **18,82 %** | **38,39 %** |

El efecto **va en la misma dirección en los tres terciles**, así que el área no
lo explica. Pero sí lo modula: la brecha es mayor en el tercil superior, lo que
es consistente con que parte del mecanismo sea la piel sana disponible.

### Qué NO afirma este resultado

- **No es causal.** Se mide una asociación entre medibilidad y clase, no un
  mecanismo demostrado. La hipótesis plausible —menos piel sana utilizable en
  lesiones malignas, más grandes o en sitios distintos— es consistente con el
  gradiente por tercil, y no está probada.
- **No dice que el ITA esté mal implementado.** La abstención es correcta. El
  hallazgo es sobre **la composición de lo que queda**, no sobre el cálculo.
- **No se apoya en el ITA como instrumento de medición de fototipo.** Ese uso
  está **descartado** en este proyecto (ver `docs/63` sección 1: 22 % de la
  varianza es iluminación), y Alipour et al. 2026 lo reporta en 22,9 % de
  exactitud contra una base trivial de 51,3 %. Acá el ITA se usa solo como
  **criterio binario de si el ángulo se puede calcular**, que es la operación
  cuya consecuencia poblacional se está midiendo.
- **No reemplaza una verificación sobre el pool completo** en la cohorte B,
  donde la cobertura de etiqueta es del 39,6 %.

### Por qué es un aporte

La consecuencia es metodológica y aplica fuera de este proyecto: **cualquier
auditoría de equidad por tono que abstenga sobre lo no medible debe reportar la
composición diagnóstica del subconjunto que analiza**, o su métrica no es
interpretable. Con un odds ratio de 0,30 y p del orden de 1e-22 en una cohorte y
replicación independiente en otra, no es un detalle de borde.

---

## 2. Aporte dos: el desplazamiento tonal logrado, en grados de ITA

### El hueco

`docs/63` sección 7 registra que una búsqueda del 2026-11-04 **no encontró
ninguna fuente que reporte un desplazamiento de ITA realizado y validado en
grados** para augmentación tonal. La mayoría de los pipelines de equidad en
dermatología usa augmentación fotométrica genérica —brillo 0,8 a 1,1,
rotaciones, volteos— que no es una transformación tonal validada.

### La cifra

Medida sobre los manifiestos de los espejos ya materializados, 33.199 imágenes
transformadas por semilla, tres semillas:

| semilla | media con signo | **media absoluta** | mediana absoluta | máx absoluto |
|---|---:|---:|---:|---:|
| s42 | 0,0201 | **12,4757** | 12,4427 | 24,9999 |
| s43 | −0,1601 | **12,4775** | 12,4928 | 24,9972 |
| s44 | −0,0751 | **12,5248** | 12,5358 | 24,9998 |

Las dos columnas hacen falta y dicen cosas distintas:

- la **media absoluta de 12,5°** es la dosis: el desplazamiento típico aplicado;
- la **media con signo cercana a cero** dice que el muestreador no está sesgado
  hacia aclarar ni hacia oscurecer, que es una propiedad deseada del diseño;
- mediana ≈ media ≈ 12,5 con máximo en 25,0 es la firma de una **distribución
  uniforme sobre ±25**, consistente con `max_abs_ita_delta = 25.0`;
- las tres semillas coinciden en la tercera cifra significativa, así que no es
  un artefacto de una corrida.

**Advertencia de procedencia:** la cifra `0,0201` circuló como si fuera la media
absoluta, por un campo mal nombrado que calculaba una media con signo. Está en
la tabla de cifras retiradas de `docs/63` y no se reutiliza. Corregido el
2026-11-04 en `src/dermssl/tone_augment_lab.py`, que ahora reporta `_mean`,
`_abs_mean` y `_abs_max` por separado.

### Contexto que la vuelve interpretable

El **recorrido alcanzable por imagen** tiene mediana **70,117°** (p05 50,9, p95
87,6, n=792), medido en `pool_tone_eligibility`. Así que la dosis típica es
alrededor del **18 %** de lo que las imágenes soportan físicamente, y el tope de
25° es alrededor del 36 %.

Y la fidelidad del canal `b*` —cuánto del desplazamiento pedido sobrevive al
gamut— es **plana en 0,976 hasta el borde del rango**, así que el tope de 25° no
está donde está por un límite físico. De dónde sale ese tope, y por qué su
derivación es doblemente problemática, está en
`docs/estado/contraste-tono-pareado-estado-20261105.md` sección 3.6.

### Qué NO afirma

- **No es el ITA usado como instrumento de medición.** La transformación
  **nunca estima el ITA de una imagen**: aplica `delta_b` y `delta_l` en Lab
  derivados del grado pedido. El ITA acá es **unidad de parametrización**, que
  es el único uso que sobrevive al análisis de `docs/63`.
- **No afirma que 12,5° sea la dosis correcta.** Afirma que es la dosis
  aplicada, medida, que es lo que nadie reporta. Si es la adecuada es una
  pregunta abierta, y está registrada como tal.

---

## 3. Cómo reproducir

```sh
PYTHONPATH=src python scripts/measure_ita_measurability_vs_diagnosis.py
```

Escribe `resultados/segmentacion/v8/ita_measurability_vs_diagnosis/summary.json`
y verifica que su recomputación del lado ISIC coincida con la del análisis
original, de modo que una divergencia de lectura se detecte en vez de pasar.

Las cifras del desplazamiento salen de los
`tone_transform_manifest.pool34754.json` de cada espejo, campo
`summary.ita_delta_applied_abs_mean`, válido solo en manifiestos posteriores al
2026-11-04 (los anteriores traen el valor con signo bajo ese nombre).

## 4. Deuda conocida

`src/dermssl/ddi.py`, que contiene la agrupación canónica `HAM_MALIGNANT` que
este análisis importa, **no estaba versionado** al momento de escribir este
documento. Es un ejemplo del problema que motivó el trabajo de versionado: la
definición de referencia de un resultado vivía en un archivo que un
`git clean -fd` borraba sin aviso.
