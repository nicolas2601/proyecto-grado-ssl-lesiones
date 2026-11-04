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

**No verificado:** si re-materializar el espejo con v2 cambia la cobertura del
pool. v2 abstiene por motivos que v1 no tiene (`no_reachable_ita_range`,
`border_population_shift_too_large`, y la unificación de piel que excluye
vineteado). Sobre S-SYNTH v2 aplicó en 745 de 2000 muestras (37,2 %), con
48,3 % no seleccionado por el sorteo de mezcla y 13,8 % sin recorrido
alcanzable. Si esa tasa se traslada a ISIC, el brazo tratado del contraste
pareado vería bastante menos imágenes transformadas de lo que vio el pool de
v1 — y el diseño pareado exige que los dos brazos difieran en **un** factor.
Medirlo requiere correr v2 sobre el pool real, que no se hizo.
