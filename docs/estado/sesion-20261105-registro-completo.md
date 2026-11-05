# Sesión 2026-11-05: registro completo

Estado: vigente. Escrito al cierre de la sesión para que nada quede solo en la
conversación.

Este documento es el índice de **todo** lo que se hizo, se encontró, se arregló
y se retiró. Los detalles viven en los documentos que se enlazan; acá está la
lista completa y el porqué de cada cosa.

Documentos que esta sesión produjo o modificó:
`64-aporte-medibilidad-ita-y-desplazamiento-logrado.md`,
`65-resultado-contraste-tono-y-que-prueba.md`,
`63-instrumento-ita-alcance-y-limites.md` (secciones 6 y 7),
`estado/contraste-tono-pareado-estado-20261105.md` (secciones 3.3 a 3.12),
`estado/envoltorio-cambios-para-contraste-tono.md`,
`cluster/runtime-snapshot-20261104/README.md`,
`cluster/tone-contrast-runtime/README.md`, `CLAUDE.md` sección 11,
`docs/README.md`.

---

## 1. El resultado científico

**El contraste pareado de tono corrió por primera vez de punta a punta
—segmentador 026, transformación tonal, SSL, clasificador— y salió negativo en
los dos objetivos.**

| objetivo | resultado |
|---|---|
| macro-F1 agregado | **negativo**: PAD −0,012928 IC95 [−0,0249; −0,0015]; HAM −0,011202 IC95 [−0,0315; +0,0064] |
| **brecha entre fototipos** (DDI) | **negativo**: la brecha se mueve +0,0619 hacia favorecer piel clara; **0 de 2.000 remuestreos la cierran** |
| CIRCLe fiel, nivel de cabeza | **cero**: −0,000253 |

Detalle, protocolo y lectura predeclarada en `docs/65`.

### Y la explicación de por qué, que es el hallazgo más profundo

**La línea base ya contenía la intervención.** El control corre con
`gray_world` y `shades_of_gray` al 60 %, que son normalizadores de iluminante
—la novedad 2 de la tesis— y la línea tonal ataca el mismo eje. Medido sobre
100 imágenes: **la constancia de color destruye entre el 91 % y el 95 % del
desplazamiento tonal**, así que el tratamiento se entrega al ~44 % de su fuerza.

Eso explica de una sola vez por qué los rasgos casi no se mueven (coseno mediana
0,990), por qué el regularizador con contrapeso da cero, y por qué en SSL hace
daño: empuja un eje ya saturado. `docs/65` sección 5-bis.

---

## 2. Lo que se construyó y quedó funcionando

| pieza | qué es | commit |
|---|---|---|
| `tone_overlay_ssl_runtime.py` | envoltorio del contraste con cinco cambios sobre el original, el crítico siendo parchear el colector pareado | `fa8f606`, `2ddbc0f`, `9024b49` |
| Espejo completo | 34.754 PNG por brazo (33.620 `source_fit` + 1.134 `pad_adapt`), 34.315 transformadas, 98,74 % | — |
| Revisión de datos del pool | `60ae7343...`, definida sobre los bytes que lee el entrenador | — |
| Manifiestos de rol con rutas del espejo | `content-exclusion-overlay-v1-mirror/`, cero faltantes, originales intactos | — |
| `measure_ita_measurability_vs_diagnosis.py` | deriva el aporte de `docs/64` desde los JSON, sin copiar cifras | `eb009f0` |
| `export_ddi_features.py` | rasgos CLS224 sobre DDI con la extracción del banco, verificando cada sha de imagen | `69c8e2f` |
| `evaluate_ddi_tone_gap.py` | brecha por fototipo, con guardia que rechaza un lector que no transfiere | `69c8e2f` |
| DDI en el clúster | 656 imágenes, hash contra el valor pineado, conteos por fototipo verificados | — |
| Arnés de UNABIA rescatado | 120 archivos, 11.993 líneas que no estaban en ningún repo | `0e0bd2f` |

---

## 3. Defectos de código encontrados y arreglados

Todos salieron **al ejecutar**, no al leer el plan.

1. **`fallback_reason` aplanaba la causa de abstención** (`2abe4c5`). Estaba fijo
   en `ineligible_mask`. Ahora propaga `quarantine_reason`, así que las 421
   máscaras degeneradas de `source_fit` no se confunden con imágenes sin máscara.
   Tres tests.
2. **`ita_delta_applied_abs_mean` calculaba una media CON SIGNO** (`7b6c2f9`).
   Valía 0,0201 y sugería que la transformación no hace nada; la dosis real es
   **12,48 grados**, factor 620. Ahora reporta `_mean`, `_abs_mean` y `_abs_max`
   por separado. Cifra retirada en `docs/63`.
3. **Los generadores exigían el archivo de máscara antes de comprobar
   elegibilidad** (`56addf8`). La rama de copia identidad nunca la usa, así que
   una imagen sin máscara reventaba antes de llegar ahí. Dos tests.
4. **El gate comparaba `world_size` y `effective_lr`**, que son **salidas** que
   una corrida reporta, no entradas (`2ddbc0f`). Más
   `checkpoint_every_steps` y `checkpoint_fractions` movidos a
   `OPERATIONAL_FIELDS`. Tres tests.
5. **El parche de `collect_images` era una constante** (`9024b49`). Devolvía las
   rutas del original también al barrer el árbol transformado. **El más grave:**
   el brazo tratado habría leído el árbol equivocado sin que nada levantara.
6. **`b*` no se acota** (`src/dermssl/tone_augment_lab.py:238`). `L*` sí. Queda
   **sin arreglar a propósito**: se verificó que no es la causa del negativo
   (ver sección 5), así que arreglarlo es rigor del método y no urgencia.

Y tres de configuración, no de código: `image_root` tenía que ser la raíz del
espejo; `color_constancy_probs` se serializa como cadena y el entrenador espera
tuplas; `budget_gpu_hours` quedaba sin declarar en el brazo tratado.

---

## 4. Las cuatro verificaciones que hacen el resultado atribuible

1. **Un solo factor, por código.** `paired_factor_gate: passed`, con los campos
   que difieren siendo exactamente los cinco del tratamiento.
2. **El control reprodujo el original BIT A BIT.** Mismo `backbone_sha256`
   (`09c15345...`) y mismo `coverage.counts` —el conteo de accesos de cada una de
   las 34.754 imágenes— a pesar de leer PNG donde el viejo leía JPG, declarar
   otra revisión y correr bajo otro envoltorio. Prueba a la vez que el espejo es
   pixel-idéntico sobre 96.000 extracciones, que el envoltorio nuevo no altera
   nada, y que el determinismo se sostiene.
3. **Sin colapso de representación.** Al mismo paso, `embedding_std` 0,022555
   contra 0,022542.
4. **Dilución declarada**, separada por causa: 421 `degenerate_area` + 18 de
   `pad_adapt` = 439 de 34.754 (1,26 %).

---

## 5. Afirmaciones propias retiradas, con lo que las tumbó

Seis. Todas se retiraron **antes** de llegar a un informe.

| afirmación | por qué es falsa |
|---|---|
| `f4fcee05` es una revisión de datos | es el **sha256 del checkpoint del teacher 026**; cinco confirmaciones independientes |
| `pad_adapt` no tiene máscaras del 026 | **sí tiene**, 1.116 de 1.134 elegibles; el teacher corrió el 2026-11-02. Costó 1.134 copias identidad inútiles, en cuarentena |
| hay un segundo factor por resolución | las fuentes `source_fit` **ya son 224×224** y el espejo no reescala |
| recortar el control a `source_fit` | la lista blanca del envoltorio rechaza `clinical_batch_quota = 0` por código |
| el daño viene de colores implausibles (`b*` cruzando el eje) | **refutado**: Spearman(coseno, `b*` invertido) = −0,12 contra **−0,84** de la magnitud de la dosis |
| las dos vistas reciben constancia de color independiente | **falso**: `_with_tone_view` rebobina el estado aleatorio a propósito. El efecto es dilución, no contaminación |

Y una corrección de **atribución**, que es de otra clase y más peligrosa:
justificar el peso 1.0 con el rango `lambda in [0,1]` de CIRCLe **no es
citable**. `docs/estado/clf-paridad-p2bis.md` ya lo advertía: el rango está
calibrado contra una entropía cruzada y una distancia distinta. Cuatro
diferencias medibles contra el paper, en `docs/65` sección 3.

**Dos falsas alarmas atrapadas antes de reportarlas:** `wrapper_sha256` no es el
hash del envoltorio sino un sello de la configuración (`config_seal`); y el
criterio 7 que "no sostiene" en `pool_tone_eligibility` pertenece al pipeline
**v2 descartado**, no al actual, que lo sostiene con 98,75 % contra un piso del
80 %.

---

## 6. Hallazgos nuevos que no estaban buscados

1. **La medibilidad del ITA depende del diagnóstico.** HAM: malignas 9,94 % en el
   subconjunto medible contra 26,63 % en el no medible, OR 0,3041,
   **p = 9,25e-22**. Sobrevive a excluir `akiec` (OR 0,2874) y replica en ISIC
   (OR 0,5469, p = 0,0099). **No lo explica el área de la lesión**: va en la misma
   dirección en los tres terciles. `docs/64`.
2. **La mapeabilidad a HAM depende del fototipo.** DDI: mapean 61 % de FST I-II,
   70 % de FST III-IV y solo **35 % de FST V-VI**, y la tasa de malignidad del
   grupo oscuro cae de 23,2 % a 13,7 %. La causa son malignidades que la
   ontología de HAM no contiene (`mycosis-fungoides`, `squamous-cell-carcinoma`,
   `subcutaneous-t-cell-lymphoma`), más frecuentes en piel oscura.
3. **Juntas generalizan el aporte.** Son **dos filtros independientes** que
   sesgan la población en direcciones distintas. La regla es que **cualquier
   filtro previo a un análisis de equidad debe auditarse por su interacción con
   el atributo protegido y con la etiqueta**.
4. **El desplazamiento tonal logrado, en grados**: 12,4757 / 12,4775 / 12,5248 de
   media absoluta en tres semillas, contra un recorrido alcanzable de 70,1°.
   Llena el hueco que `docs/63` sección 7 declaraba.
5. **El teacher 026 está mal calibrado.** Reporta confianza mediana 0,8963 sobre
   máscaras degeneradas. El umbral de confianza **no detecta ninguna** de las
   421; el criterio de área detecta todas.
6. **En el control, la brecha de sensibilidad es NEGATIVA**: 0,6458 en piel
   oscura contra 0,4898 en clara. Lo **opuesto** a la preocupación habitual de la
   literatura. No había brecha que cerrar en la dirección esperada, y el
   tratamiento le quitó 6,25 puntos de sensibilidad al grupo oscuro.
7. **Hay seis espejos transformados, no tres** (`tone_s42` a `tone_s47`), los
   seis completos. Y `capped: 0` en los seis no significa "se probó más y no hizo
   falta": el muestreador sortea **dentro** de ±25, así que el tope define el
   rango y es vinculante por construcción.

---

## 7. Los tres cortes, con su criterio de reingreso

Fijados a pedido de María Paula: un recorte sin criterio de reingreso es un
olvido con fecha. `estado/contraste-tono-pareado-estado-20261105.md` sección 3.8.

- **Re-correr el control**: entró y salió. Se cumplió su criterio 2 (el
  envoltorio rechazó los `resolved-params` viejos), se corrió, y **cambió
  exactamente nada** — lo que confirma que la razón científica que se había
  retirado estaba bien retirada.
- **Sonda de dosis**: entró. El brazo `w=0,1` está corriendo.
- **Ajustar el teacher 026**: sigue afuera. La dilución es 1,26 %, lejos del
  umbral del 5 % que lo traería de vuelta.

---

## 8. Lo que está corriendo al cierre

| brazo | qué prueba | estado |
|---|---|---|
| `tono_026_w01_s42_6000` | curva dosis-respuesta a peso 0,1 | 3.100/6.000 |
| `ctrl_nocc_s42_6000` | control **sin** constancia de color | 350/6.000 |
| `tono_nocc_w1_s42_6000` | tratado sin constancia de color: **el tratamiento entregado completo por primera vez** | encadenado, arranca al liberarse la GPU |

**Costo declarado del par sin constancia de color:** le saca a la línea base la
novedad 2, así que es otro control y más débil. El delta sigue siendo válido
internamente, pero **sus números absolutos no son comparables con v7**.

---

## 9. Deuda de repositorio, medida

| | |
|---|---:|
| commits sin pushear en `experiment/modular-audit-v8` | **88** |
| adelante de `origin/main` | 398 |
| **archivos sin seguimiento** | **423** |
| modificados sin commitear | 88 |

`src/dermssl/ddi.py` **no estaba versionado** —191 líneas, la agrupación canónica
`HAM_MALIGNANT` que el análisis de `docs/64` importa— y se rescató en `eb009f0`
junto con `scripts/ddi_manifest.py`. Es el caso de los doce módulos de `f14a63a`
otra vez.

**Y la causa del problema que María Paula señaló** —"siempre consultamos a
UNABIA cosas que ya habíamos hecho"— quedó identificada: `docs/README.md` se
detenía en el documento **58** y la numeración llega al **65**. Las siete filas
faltantes se agregaron, con la nota de que un documento fuera del índice no
existe en la práctica.

Trabajo repartido: `impl` tiene el versionado y el push; `transformacion-tono`
tiene **verificación independiente** de las cifras, recalculando desde el JSON
crudo y con permiso explícito de marcar errores propios.

---

## 10. Reglas de método que esta sesión fijó

`CLAUDE.md` sección 11, commit `42ff574`.

- **Contrato de respuesta.** La recomendación va adelante, derivada de lo
  observado. Prohibido partir un paso en varias preguntas. Una pregunta por
  respuesta como máximo. Cuando se cae una recomendación propia, se dice en la
  misma respuesta con la evidencia que la tumbó. **Nunca recomendar rendirse.**
- **El repositorio no es la fuente de verdad.** El repo y el código congelado
  divergieron en las dos direcciones; un grep local negativo no prueba que algo
  no exista.
- **Leer el campo, no su nombre.** Por `wrapper_sha256` y por
  `ita_delta_applied_abs_mean`.

La falla raíz detrás de las seis afirmaciones retiradas es una sola: **usar un
dato heredado sin re-verificarlo**. Las seis se cayeron por verificar, no por
suerte.
