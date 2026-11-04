# Qué está cableado del brazo de tono, y qué no

Estado: vigente (2026-11-04). Inventario de solo lectura, sin cambios de código.
Responde a la pregunta «¿se puede probar el pipeline completo con el
clasificador?» antes de intentarlo.

Instrumento y sus límites: `docs/63-instrumento-ita-alcance-y-limites.md`.

---

## 1. El hallazgo principal

**Todo el trabajo de esta sesión vive en `tone_augment_lab_v2.py`, y la cadena
de producción no lo usa.**

El único consumidor de `tone_augment_lab_v2` en todo el repositorio es
`scripts/make_tone_contact_sheet.py` (más su test). Verificado:

```
grep -rn "tone_augment_lab_v2" --include=*.py src scripts tests
```

La cadena que construye el pool y alimenta el preentrenamiento importa
`tone_augment_lab` — la **v1**, el módulo congelado.

Lo que queda del lado de v2, fuera del pipeline:

- recorrido alcanzable por imagen y solver `brentq` (error 0,0000°)
- la corrección de agregación ángulo-de-medianas (`6997206`)
- la unificación de las dos definiciones de piel (`99bad30`)
- la guarda de desplazamiento de población (`MAX_POPULATION_SHIFT_FRACTION`)
- el test que fija la asimetría de `MIN_B_FOR_ITA` (`ab9a69f`)

Lo que v1 **no** tiene, verificado por conteo de coincidencias igual a cero:
`brentq`, `reachable`, `_skin_state_at_t`. v1 sortea un delta por hash
(`sample_ita_delta`) y lo aplica acotado a `MAX_ABS_ITA_DELTA = 25,0` —
simétrico, derivado de la tabla de Corbin, con el estimador clasificando por
Groh (el desajuste de `docs/63` §5). Sin recorrido alcanzable por imagen, es
el módulo con la razón logrado/nominal de **0,37** que originó toda esta línea.

## 2. La cadena real, de punta a punta

```
teacher 026 (máscaras)
  -> seg_masks_026_pool_20261103/full/masks.source_fit.jsonl   33.620 filas
  -> scripts/tone_mirror_pool_v1.py
       usa: dermssl.pretrain.tone_augment_pairs
            dermssl.tone_augment_lab            <-- v1
  -> espejo PNG del pool + copias tonales de dos canales
  -> scripts/tone_paired_launch.py  /  p2bis_build_paired_configs.py
       gate: dermssl.pretrain.paired_factor_gate
  -> preentrenamiento SSL pareado (control / tratado)
  -> encoder
  -> src/dermssl/finetune/  -> clasificador
```

Dos cosas que este trazado confirma, y conviene no perder:

- **El brazo de segmentación ya alimenta la cadena de tono.** El pool se
  construyó sobre las máscaras del teacher 026, no sobre Otsu. La pregunta
  «¿sirve el brazo de segmentación para la transformación tonal?» no solo está
  respondida en la medición; está cableada.
- **El tono no entra por el clasificador, entra por el preentrenamiento.** El
  contraste pareado control/tratado es sobre el *encoder*. El clasificador
  consume el encoder resultante. Por eso «probar el pipeline completo con el
  clasificador» no es correr el clasificador con aumento tonal: es correr dos
  preentrenamientos que difieren en un solo factor y después comparar sus
  clasificadores.

## 3. Estado de ejecución

Del `receipt.json` de `resultados/preparacion_v8/tone_mirror_pool_20261103/`:

| gate | valor |
|---|---|
| `execution_enabled` | `false` |
| `launch_ready` | `false` |
| `training_started` | `false` |

**Ningún preentrenamiento pareado corrió.** El espejo del pool está
materializado; el contraste no se lanzó.

Procedencia verificada: el `sha256` de `tone_augment_lab.py` que consumió el
pool (`75eaf26d7daf2eafbb504db981652c8c0709fee11cc5ab2e0e34c932d7bf5478`)
coincide exacto con el de `HEAD` hoy. El módulo congelado no se movió, y el
pool es trazable al código que lo produjo.

Roles nunca resueltos en ese receipt, que siguen sin resolverse:
`pad_test`, `ham_holdout`, `pad_adapt`, `ham_dev`.

## 4. Qué significa para la pregunta original

No se puede «probar el pipeline completo con el clasificador» hoy, y el
bloqueo no es de infraestructura. Son tres decisiones, en este orden:

1. **Qué módulo entra.** Si el contraste pareado se corre con v1, se corre con
   el cap simétrico de 25° derivado de la tabla equivocada y sin recorrido
   alcanzable por imagen — es decir, se mide un tratamiento cuyo
   desplazamiento nominal se realiza al 37 %. Si entra v2, hay que portar la
   cadena (`tone_augment_pairs` importa v1 explícitamente) y re-materializar
   el espejo del pool, que son 33.620 registros.
2. **El cap.** Abierto, asimétrico según lo medido, y bloqueado por la
   plausibilidad visual (`docs/63` §8). Es decisión de María Paula.
3. **El gate.** `execution_enabled` no se marca sin evidencia real, por
   convención dura del repo.

## 5. Medido: v2 no puede cumplir el criterio 7 sobre este pool

Se corrieron **los dos módulos sobre las mismas 1.500 imágenes** del pool real
(`masks.source_fit.jsonl`, muestra aleatoria con semilla 42 de los 33.620
registros elegibles del teacher 026, ISIC 2019, `feather_sigma=0,0`).

| | tasa efectiva de intervención |
|---|---:|
| v1 (en producción) | **100,0 %** |
| v2 (instrumento de esta sesión) | **25,7 %** |
| piso del criterio 7 | **80 %** |

El criterio 7 de `docs/estado/clf-trainer-pareado-tono.md` (línea 140) exige que
la tasa efectiva de intervención se mida y se publique, y que si cae por debajo
del **80 %** se detenga. Descompuesto, el techo de v2 no depende del sorteo:

| etapa | n | % |
|---|---:|---:|
| ITA medible antes de transformar | 803 / 1.500 | **53,5 %** |
| de esos, seleccionados por la mezcla | 392 / 803 | 48,8 % |
| de esos, aplicados | 385 / 392 | **98,2 %** |

Es decir: **cuando v2 puede medir y le toca actuar, actúa en el 98,2 % de los
casos.** El limitante no es el instrumento, es el pool. Y aun si la mezcla se
pusiera en 100 %, el techo sería **52,6 %** — sigue por debajo del piso.

Abstenciones de v2 sobre el pool real: 46,5 % `ita_unmeasured_ita_indeterminate`,
27,4 % `not_selected_by_mix`, 0,4 % `no_reachable_ita_range`, 0,1 %
`border_population_shift_too_large` — **primera vez que esa guarda se dispara
sobre datos reales**; en S-SYNTH valía 0 en los 745 casos.

### El b\* del pool real, que es la causa

| cohorte | b\* de piel, mediana | p05 |
|---|---:|---:|
| S-SYNTH | 29,26 | 12,89 |
| ISIC 2018 Task 1, grupo `ok` | 11,53 | — |
| **pool real (ISIC 2019, teacher 026)** | **5,53** | **−5,67** |

El p25 del pool es **0,75**, y el **46,5 %** cae bajo `MIN_B_FOR_ITA = 5,0`.
El pool sobre el que esto tiene que correr es **mucho** más desaturado que
cualquier cohorte donde se midió el instrumento. La advertencia de §4 era
correcta en dirección y conservadora en magnitud.

### Lo que el número obliga a decir

v1 cumple el criterio 7 **porque no mide**. Su desplazamiento es un sorteo por
hash (mediana −0,17°, rango ±25°) aplicado sin consultar el tono de partida,
incluido el 46,5 % de imágenes donde el ITA no es medible siquiera. v2 falla el
criterio **porque mide y se abstiene**.

Así que el criterio 7, tal como está escrito, **premia al módulo ciego**. No es
un empate entre dos opciones: es un umbral de cobertura que no distingue
«intervino» de «intervino sobre algo que entiende». Antes de elegir módulo hay
que decidir si el 80 % se mide sobre todo el pool o sobre el subconjunto
medible, y esa reformulación es una decisión de protocolo, no de código.

Datos: `resultados/segmentacion/v8/via_b_cierre/pool_v2_coverage.jsonl`.

**Nota de procedencia.** Una primera corrida de este script pasó
`source_revision` como hash por imagen; ese campo es una constante del dataset
(1 valor distinto en 500 filas, contra 500 de `mask_sha256`), así que las
1.500 imágenes recibieron el mismo sorteo y el mismo delta. Se detectó por dos
señales —`v1_delta` idéntico en todas, y `not_selected_by_mix` exactamente
igual al conteo de ITA medible— y se descartó. Las cifras de arriba son de la
corrida corregida. **No reutilizar ninguna cifra de esa primera pasada.**
