# La medibilidad del ITA depende de la clase diagnóstica

Estado: vigente (2026-11-04). Medición sobre el split de validación congelado de
v7. Acota qué puede afirmar cualquier lectura de sesgo estratificada por tono,
incluida la que este mismo documento habilita.

Instrumento: `docs/63-instrumento-ita-alcance-y-limites.md`.
Datos: `resultados/clasificacion/v7/ita_stratified/`.

---

## 1. Procedencia del split, verificada por tres vías

El `npz` de v7 guarda `y_true` y probabilidades para 2.004 filas, pero **no
guarda los IDs de imagen** — sólo un `eval_ids_sha256`. El split se reconstruyó
con `scripts/c1_build_val_index.rebuild_split` sobre
`~/work/datos/ham10000` en UNABIA y se verificó así:

| prueba | resultado |
|---|---|
| `n` de `val_ids` | 2.004, coincide con `n_eval` |
| `c1_split_identity.ids_sha256(val_ids)` | `6ff4c17f04a9…a2fc6`, **idéntico** al `eval_ids_sha256` de `final_eval.json` |
| `id2label` sobre `val_ids` en orden contra `y_true` del `npz` | **iguales elemento por elemento**, 2.004/2.004 |
| IDs sin imagen en disco | 0 |

Las dos pruebas del medio son independientes entre sí, así que el orden de las
filas queda confirmado por doble vía, no por una.

**Nota de ruta:** el docstring de `c1_build_val_index.py` apunta a
`~/work/proyectonp/datos/ham10000`. Esa ruta **no existe**; el dataset está en
`~/work/datos/ham10000`. El docstring está obsoleto.

**Nota de nombre:** el módulo del hash es `dermssl.c1_split_identity`, no
`split_integrity`. El alias `si` de los imports confunde.

## 2. El hallazgo

Sobre las 2.004 imágenes, `estimate_fst` produce ITA en **855 (42,7 %)**. Y la
tasa **depende fuerte de la clase**:

| clase | n | medibles |
|---|---:|---:|
| nv | 1.341 | **54,3 %** |
| akiec | 66 | 28,8 % |
| bcc | 103 | 27,2 % |
| vasc | 29 | 24,1 % |
| df | 23 | 17,4 % |
| **mel** | 222 | **17,1 %** |
| bkl | 220 | 14,1 % |

Agrupado, y con la prueba:

| | medibilidad | |
|---|---:|---|
| maligno/premaligno (mel, bcc, akiec) | **21,7 %** | n=391 |
| benigno (nv, bkl, df, vasc) | **47,7 %** | n=1.613 |

**Fisher exacto: odds ratio 0,3041, p = 9,2·10⁻²².**
Para el par que más importa, mel contra nv: **odds ratio 0,1739,
p = 2,3·10⁻²⁶**.

### Lo que le hace a la cohorte

| | en el split | en el subconjunto medible |
|---|---:|---:|
| nv | 66,9 % | **85,1 %** |
| mel | 11,1 % | **4,4 %** |
| malignos (total) | 19,5 % | **9,9 %** |

El subconjunto medible **reduce los malignos a la mitad** y lleva los nevos al
85 % de la cohorte.

## 3. Consecuencia, incluida la que afecta a la recomendación vigente

`docs/estado/tono-cableado-real-vs-instrumento-v2.md` §7 recomienda reconvertir
el instrumento de intervención a **evaluación**, argumentando que estratificar
no pasa por el criterio 7 porque no hay brazo tratado. Eso sigue siendo cierto,
pero **esta medición debilita esa recomendación y hay que decirlo**: el
subconjunto donde se puede estratificar no es una muestra del split, es una
submuestra sesgada contra la clase de interés.

Concretamente, una lectura de macro-F1 por estrato de ITA sobre v7:

- se computa sobre 855 de 2.004 imágenes (42,7 %);
- en una cohorte con 85 % de nevos y 4,4 % de melanomas;
- y por lo tanto **no es comparable** con el macro-F1 publicado de v7, que se
  computó sobre las 2.004.

No es que la lectura sea imposible. Es que **no puede presentarse como «el
sesgo tonal de v7»**. Lo que mide es el sesgo tonal sobre la fracción medible,
con la composición de clases declarada al lado. Si se omite esa declaración, la
cifra dice algo que no se midió.

Esto **replica** el mismo mecanismo ya medido sobre el pool de ISIC 2019
(odds ratio 0,5469, p = 0,0099 — `tono-cableado-real-vs-instrumento-v2.md` §6)
en un dataset distinto, con etiqueta de 7 clases y con significancia veinte
órdenes de magnitud mayor. Dos cohortes, dos etiquetados, mismo signo: no es
una casualidad de muestreo.

## 4. Por origen de máscara

1.268 de las 2.004 tienen máscara del teacher 026 (porque los IDs de HAM10000
aparecen en el pool de ISIC 2019 — `docs/02-datasets.md` línea 37, composición
conocida, **no** un solapamiento descubierto acá). Las 736 restantes caen al
fallback de Otsu.

| origen | n | medibles | b\* mediana |
|---|---:|---:|---:|
| teacher 026 | 1.268 | 41,4 % | 2,49 |
| Otsu (fallback) | 736 | 44,8 % | 3,75 |

La máscara real **no** mejora la tasa de medibilidad acá; la empeora levemente.
Eso **no contradice** el 2x–6x de `docs/63`: esa cifra es de **exactitud** del
estimador contra melanina verdadera, no de **tasa de medibilidad**. Son métricas
distintas, y es esperable que excluir bien la lesión deje piel más desaturada,
que es justo lo que baja el b\*.

Y el dato que ordena todo lo demás: el b\* de piel de este split tiene mediana
**2,49** con máscara real. El pool de ISIC 2019 daba 5,53; S-SYNTH, 29,26.

## 5. No verificado

- **Si el protocolo de v7 aceptó explícitamente preentrenar SSL sobre imágenes
  que luego se usan como validación supervisada.** El solapamiento de datasets
  está documentado (`docs/02` línea 37); que el protocolo de v7 lo haya
  asumido como condición, no se verificó. No se afirma en ninguna dirección.
- **Si existe un estrato de ITA con `n` suficiente por clase** para que la
  lectura estratificada diga algo por clase y no sólo en agregado. Con 4,4 % de
  melanomas sobre 855 imágenes son ~38 casos repartidos en los estratos que se
  elijan.
