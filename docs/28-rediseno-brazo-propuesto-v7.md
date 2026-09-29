# Rediseño del brazo propuesto (v7) — SimSiam anclado sobre DINOv2 ViT-B

**Fecha:** 2026-09-27
**Estado:** especificación registrada antes de correr. Decisión de Nicolás: solo ViT-B, se mantiene SimSiam sobre DINOv2.
**Base:** resultados de C1 ([resultados/clasificacion/c1](../resultados/clasificacion/c1/README.md)), diagnóstico del pretrain y revisión de literatura (sección 6).

---

## 1. Problema

Con probe lineal (`logreg`), DINOv2 público ViT-B le gana al propuesto en macro-F1 (0,614 contra 0,526; Δ +0,088, IC [+0,032; +0,139]). El propuesto gana en ECE en 7 de 7 contrastes y en AUC en 5 de 7. La segmentación en la entrada no cambia ese orden (C1).

Diagnóstico (hechos, 2026-09-27):

- Los dos brazos SSL convergieron (pérdida final −0,92 los dos, sin colapso). El −0,047 del plain en `run.json` es un error del promedio al reanudar, no del entrenamiento.
- `consistency_weight = 1.0` es la identidad: el propuesto no tuvo término de tono propio. Lo único que lo distingue es que la vista 2 lleva el tono desplazado.
- La receta actual no protege al backbone: sin capas congeladas, un solo LR para todas las capas, weight decay hacia cero y ningún ancla hacia DINOv2.
- Los IDs de validación estaban en el pool de preentrenamiento, y los brazos vieron cantidades distintas de muestras (docs/27 §0).

Inferencia: la adaptación completa reescribe capas tempranas que DINOv2 ya tenía bien, lo que baja la separabilidad lineal por clase. La vista de tono regulariza la confianza (mejor ECE) sin aportar a la separación por clase.

## 2. Qué cambia (solo SSL, sin etiquetas en el backbone)

| Palanca | Valor registrado | Fuente |
|---|---|---|
| Bloques congelados | patch embed + bloques 0-7 congelados; se entrenan 8-11 y todas las LayerNorm | ExPLoRA (Khanna et al., ICML 2025) |
| LR por capa | layer decay γ = 0,75 sobre los bloques entrenables | práctica BEiT/DINO/iBOT |
| Ancla de pesos | L2-SP hacia los pesos de DINOv2 público, λ ∈ {1e-4, 1e-3} (se elige en el piloto) | Li et al., ICML 2018 |
| Ancla de features | coseno entre el CLS entrenado y el de DINOv2 público congelado, λ_d ∈ {0, 0,1} (se elige en el piloto) | DINOv3, Gram anchoring (Siméoni et al., 2025), versión liviana |
| Color jitter | hue baja de 0,1 a 0,02 en los dos brazos; brillo, contraste y saturación se mantienen | Planckian Jitter (2022): el jitter RGB agresivo destruye color diagnóstico |
| Vista de tono (solo propuesto) | SimSiam sigue con sus dos vistas independientes (original y desplazada, cada una con su augmentación completa). Aparte, un término de consistencia REAL con peso w ∈ {0,1, 0,5}, sumado a SimSiam, entre el crop de la vista 1 y **ese mismo crop** sobre la imagen desplazada (mismo recorte, mismos volteos, mismas tiradas fotométricas; solo cambia el tono), con la original detached | CIRCLe (Pakzad et al., 2022); Tian et al. (2021) sobre vistas informativas |

**Por qué el término de tono usa el mismo crop (revisión Opus P2, 2026-09-27).** La primera versión comparaba el original y el desplazado con crops independientes. Sobre las features del backbone y sin predictor, eso pide invarianza a encuadre y a tono a la vez, y la invarianza al encuadre SimSiam ya la entrena. Además, un término sin predictor que empuja features de crops distintos a coincidir es justamente la ablación que colapsa en Chen y He (2021). CIRCLe compara la misma imagen con el tono cambiado, así que ahora la tercera vista repite exactamente la geometría y la fotometría de la vista 1 (`SimSiamPairDataset(tone_view=True)`) y el término mide solo el tono. Cuesta un forward extra del backbone sobre la vista de tono. Se loguea `lever_backbone_std` para vigilar el colapso.

## 3. Confusores que se arreglan

- Los IDs de validación (2004, `c1_val_ids_sha256`) quedan FUERA del pool de preentrenamiento, en los dos brazos.
- Mismo número de pasos y de muestras vistas en los dos brazos (presupuesto por pasos, no por tiempo).
- `code_revision`, `split_hash` del pool y todos los hiperparámetros se registran en `run.json`. Se arregla el promedio de `ssl_loss` al reanudar.

## 4. Protocolo (evita elegir la variante después de ver el dato)

1. **Piloto de selección** (≈2 GPU-h por variante, 4 variantes, las dos GPU): grilla λ_L2SP × λ_d para el plain anclado, y w para el propuesto. La selección se hace con un probe sobre un **holdout interno del train** (20 % de los IDs de train, agrupado por `lesion_id`), nunca sobre la validación.
2. **Corrida final**: la mejor variante de cada brazo, con el presupuesto completo, en paralelo en las dos GPU.
3. **Evaluación única** en la validación fold-0, con el mismo banco, probe `logreg` y bootstrap pareado que la v6 y C1. Se reportan todas las variantes del piloto, no solo la ganadora.

### 4.1 Variantes y regla de selección (fijadas antes de lanzar la oleada 1)

**Base común a las cuatro:** ViT-B/14 desde DINOv2 público, `--freeze-blocks 8`, `--layer-decay 0.75`, `--hue-jitter 0.02`, `--exclude-ids-file pretrain_exclude_ids.txt` (las 2004 de validación + el holdout interno), `--max-steps 950` (≈2 GPU-h), semilla 42. La única diferencia entre brazos es la vista de tono.

**Holdout interno:** 20 % de los IDs de train de C1, fold 0 de `StratifiedGroupKFold(5, shuffle=True, random_state=42)` agrupado por `lesion_id` y estratificado por `dx` (`scripts/c1_selection_holdout.py`), excluido del pool de pretraining igual que la validación. El probe de selección (`logreg`, el mismo de C1) se entrena en el 80 % restante de train y se evalúa en ese 20 %. DINOv2 público se evalúa en el mismo holdout como referencia.

| Oleada | Variante | Brazo | Palancas propias |
|---|---|---|---|
| 1 | A1 | plain | `--anchor-l2sp 1e-4 --anchor-distill 0` |
| 1 | A2 | plain | `--anchor-l2sp 1e-3 --anchor-distill 0.1` |
| 2 | B1 | propuesto | config ganadora de la oleada 1 + `--tone-consistency-weight 0.1` |
| 2 | B2 | propuesto | config ganadora de la oleada 1 + `--tone-consistency-weight 0.5` |

**Oleada 1b (registrada antes de ver la oleada 1).** El checkpoint del smoke (3 pasos, config A2) ya bajó el macro-F1 del holdout de 0,6108 (DINOv2 público) a 0,5861. Eso es lo que describen Kumar et al. (2022) para LP-FT: el gradiente de cabezas inicializadas al azar distorsiona el backbone antes de que las cabezas aprendan algo. Por eso se agrega A3 = ganadora de la oleada 1 + `--head-warmup-steps 150`: los primeros 150 pasos entrenan solo el proyector y el predictor, con el backbone entero congelado (LayerNorm incluidas), y después se habilita la configuración normal (bloques 8-11 + LayerNorm). A3 se compara con A1 y A2 con la misma regla de selección. Si A3 gana, el propuesto (B1, B2) hereda el head-warmup.

**B1 especulativo (registrado el 2026-09-27, con A1 ganadora de la oleada 1 y A3 corriendo).** B1-spec = A1 + `--tone-consistency-weight 0.1`, sin head-warmup, en la GPU que libera la selección. B1-spec se lanza antes de conocer A3. Si A3 gana por la regla, B1-spec se reporta pero no compite; B1 y B2 se relanzan con head-warmup. Si A3 pierde, B1-spec ES B1.

B1-spec se detuvo con SIGTERM a las 17:58:10 UTC, en el paso ~270 (el último logueado en W&B es el 200, run 22732c19750d), cuando A3 ganó la oleada 1b. Ya no hace falta: el warmup no tuvo efecto medible en el plain (A3 − A1, los cuatro IC incluyen el cero). Se conservan su run.json y sus curvas en W&B. No hay checkpoint parcial: en los pilotos no se escribe ninguno antes del final (tampoco en A1-A3, cuyas carpetas solo tienen `run.json` y el backbone final), y el SIGTERM no guarda. No se evalúa ni compite. B1 se relanzó con head-warmup 150 sobre la misma revisión que B2 (1453b06).

**Memoria y revisión de código (registrado el 2026-09-27).** B1-spec se quedó sin memoria al arrancar (CUDA OOM, 23,35 GiB en una GPU de 24): el tercer forward, el de la vista de tono con gradiente, no cabe con batch 128. Por eso B corre con `--grad-checkpoint`, que recalcula los 12 bloques en el backward. La matemática es la misma, porque no hay dropout ni drop-path y un test fija gradientes iguales con y sin checkpoint. Cambia el tiempo por paso, así que el flag es parte de la config de B. Batch 64 o acumular 2×64 quedan descartados: cambian el batch que ve el BatchNorm del proyector y rompen la comparación con A1. **Regla para las finales:** A\*, B\* y, si se reevalúa, el público corren sobre un mismo `code_revision`. En los pilotos las revisiones difieren (A1/A2 en 9021931; A3 y B después) y eso se documenta en el reporte.

**Réplica de semilla (opcional, registrada antes de las finales).** Las finales usan semilla 42. El IC del bootstrap cubre la variabilidad del conjunto de evaluación, NO la del entrenamiento. Si hay GPU libre al terminar las finales, se repiten A\* y B\* con semilla 43, en la misma revisión y con el mismo exclude, y se reportan AMBAS semillas. No se elige semilla: la conclusión se reporta con las dos. Si no hay tiempo, esto va como limitación explícita. Programación (2026-09-27): A\* con semilla 43 en cuda:0 apenas termine A\* (~23:10 UTC) y B\* con semilla 43 en cuda:1 apenas termine B\* (~01:50 UTC), las dos desde la revisión f1c2053 y con el exclude 95d21013….

**Evaluación de la réplica s43 (registrada el 2026-09-28, antes de correrla).** La evaluación de semilla 42 ya está hecha (`resultados/clasificacion/v7/`, commit f562ea9). La réplica es otra lectura de val, pero sobre modelos distintos, y se fija así antes de ver un solo número:

- Mismo comando que la evaluación única, con `--plain` y `--proposed` apuntando a `final_A_s43` y `final_B_s43`. La salida va a `salidas/v7/eval_final_s43/`. El código es el de f562ea9, no el de 7fd6245. Lo único que cambia es el bloque secundario del IC basic de ECE; las métricas, el probe, el bootstrap y sus semillas son los mismos. El sha256 del código de evaluación se vuelve a fijar en f562ea9 y se verifica justo antes de correr.
- Misma regla de veredicto: P1, P2 y P3 como primarias, y P1-NI, P3-IC y el IC basic de ECE como secundarias (§5).
- La conclusión se reporta con AMBAS semillas: cada una por separado, y además la media de las dos junto con la diferencia entre ellas (s43 − s42). No se elige semilla.
- Si s43 contradice a s42 en P3-IC, la frase "le gana al público" no se sostiene y se reporta como "no robusto a la semilla".

**Regla de pausa del trabajo de CPU (S2, casi-duplicados; 2026-09-27):** se pausa si cualquier entrenamiento cae más de 15 % bajo su ritmo propio medido solo (A ≈ 22-24 pasos/min, umbral ~19; B ≈ 14,4, umbral ~12,2).

El término de tono de B1 y B2 es el del mismo crop (§2), no el de crops independientes que existía cuando se lanzó la oleada 1; la oleada 1 no lo usa, así que no la afecta.

**Regla de selección, por oleada:** gana la variante con mayor macro-F1 en el holdout interno. Si `|Δ macro-F1| < 0,005`, gana la de menor ECE. La oleada 2 hereda las anclas de la ganadora de la oleada 1, así que los dos brazos difieren solo en el tono.

**Finales:** A* y B* con `--max-steps 5700` (≈12 GPU-h, el presupuesto de la v6), una GPU cada una, en paralelo, y la evaluación única del punto 3.

## 5. Predicciones registradas

- **P1 (anclaje):** el plain anclado v7 no pierde contra DINOv2 público en macro-F1: Δ ≥ −0,01. Si el intervalo excluye el cero a favor, la adaptación anclada mejora a DINOv2.
- **P2 (tono):** el propuesto v7 contra el plain anclado v7: ECE mejor (IC excluye el cero) y macro-F1 no peor (Δ ≥ −0,01).
- **P3 (objetivo de la tesis):** el propuesto v7 contra DINOv2 público: macro-F1 Δ ≥ 0 manteniendo la ventaja en ECE. Probabilidad estimada antes del dato: ≈35 %.

**Lecturas (registradas el 2026-09-27, antes de que termine A\*).** P1, P2 y P3 como están arriba son las lecturas PRIMARIAS y deciden el veredicto. Se reportan además dos lecturas SECUNDARIAS, rotuladas así en el JSON y en la tabla, que no cambian los veredictos primarios:

- **P1-NI** (no inferioridad estricta): el límite inferior del IC de Δ macro-F1 A\* − público es ≥ −0,01.
- **P3-IC**: el límite inferior del IC de Δ macro-F1 B\* − público es > 0. Solo si P3-IC se cumple, la tesis dice que el propuesto le gana al público. Si se cumple solo P3 primaria, dice que no pierde.

**Protocolo de la evaluación única:** el mismo de C1/v6. El probe `logreg` se entrena sobre el train COMPLETO de C1 (8.011 ids, fingerprint `d347edd52d0d7648`) y se evalúa sobre val (2.004 ids, `6ff4c17f04a9eae2`). `scripts/c1_final_eval.py` verifica los dos fingerprints y que los `run.json` de A\* y B\* compartan `code_revision` y exclude antes de extraer nada.

Criterio: se reporta el resultado sea cual sea. Si P3 no se cumple, la conclusión honesta es que la adaptación SSL mejora calibración y ranking pero no la separación por clase frente a DINOv2 público.

### 5.1 Resultado final (2026-09-28)

Las cifras están en la [tabla combinada de las dos semillas](../resultados/clasificacion/v7/SEMILLAS.md), generada desde los dos `final_eval.json`. Acá solo va la lectura. Las dos evaluaciones usan val del fold 0 (n = 2.004), el probe `logreg` y el bootstrap pareado de 1.000 remuestras. La de semilla 42 corrió con el código de 7fd6245 y la réplica de semilla 43 con el de f562ea9, como se registró en §4.1.

| Predicción | Semilla 42 | Semilla 43 |
|---|---|---|
| P1 (primaria) | cumplida | cumplida |
| P2 (primaria) | no cumplida | no cumplida |
| P3 (primaria) | cumplida | cumplida |
| P1-NI (secundaria) | no cumplida | cumplida |
| P3-IC (secundaria) | cumplida | cumplida |

- **Resultado principal:** P3-IC se cumple con las dos semillas, así que por la regla de §4.1 la conclusión es robusta a la semilla. El propuesto B\* le gana al DINOv2 público en macro-F1, con los dos intervalos por encima de cero. También lo mejora en AUC y en calibración con las dos semillas.
- **El término de tono no aporta nada medible.** B\* y A\* no se distinguen en macro-F1 con ninguna semilla, y el signo de la diferencia cambia de una a otra. La peor calibración de B\* que apareció con la semilla 42 no se repite con la 43. La mejora sobre el público viene de la adaptación SSL anclada a DINOv2, no del término de tono. Nota: en AUC, A\* ordena apenas mejor que B\* con la semilla 43 (el IC excluye el cero por poco). Con la semilla 42 eso no pasa.
- **A\* frente al público:** no pierde con ninguna semilla. Con la 43 además le gana (el intervalo excluye el cero); con la 42 no llegaba. Por eso P1-NI cambia de semilla a semilla.
- **Variación entre semillas:** A\* se movió unos 0,04 de macro-F1 de una semilla a la otra, lo mismo que el efecto de B\* contra el público. El intervalo del bootstrap cubre la variación del conjunto de evaluación, no la del entrenamiento. Con una sola semilla no se habría podido sostener la conclusión. Con dos ya no depende de la semilla, pero la magnitud del efecto sigue siendo incierta.

### 5.2 Evaluación secundaria C1-v7: ¿la máscara ayuda al clasificador final? (registrada el 2026-09-28, antes de correr)

Es otra lectura de val, sobre modelos ya fijados, y se declara como tal: no reemplaza el resultado primario de §5.1.

- **Qué:** la misma grilla de C1 (docs/27): sin máscara, recorte a la caja, fondo gris y pooling en la máscara. Usa las mismas máscaras de student_r3 (el mismo archivo fijado por sha en `c1_mask_provenance`). Los backbones son A\* y B\* con las semillas 42 y 43, más el DINOv2 público. Solo hay extracción y probe `logreg`, sin entrenar backbones.
- **Contraste:** bootstrap pareado (1.000 remuestras) de cada condición contra "sin máscara" del mismo backbone, en las cuatro métricas.
- **Lectura:** la máscara ayuda si alguna condición le gana a "sin máscara" en macro-F1, con el IC por encima de cero, en las dos semillas del mismo brazo. Si pasa en una sola semilla, se reporta como "no robusto a la semilla", con la misma regla de §4.1.
- **Secundario agregado el 2026-09-28, antes de correr:** la misma grilla con las máscaras de A2 (S4-UNet, docs/27), que pasó a segmentador candidato. Es solo extracción y probe. El resultado primario de esta sección sigue siendo el de las máscaras de student_r3, como estaba registrado. A2 se describe así: "mejora pequeña pero consistente en Dice sobre student_r3 (+0,004 en ISIC 2018 y +0,010 en HAM10000, IC pareado sin cero); HD95 igual en ISIC y mejor en HAM".

**Resultado (2026-09-28).** Datos en `resultados/clasificacion/c1_v7/` (primario en `student_r3/`, secundario en `a2/`, comparación entre segmentadores en `a2_vs_student_cells.json`).
- **Primario, máscaras de student_r3: la máscara no ayuda.** Ninguna de las 15 celdas le gana a "sin máscara" con el IC del macro-F1 por encima de cero, así que la regla de las dos semillas no llega a aplicarse. El fondo gris empeora en el público (−0,092 [−0,141; −0,041]) y en A\* con las dos semillas (s42 −0,052 [−0,095; −0,011]; s43 −0,058 [−0,101; −0,018]); el pooling en la máscara empeora en B\* s42 (−0,034 [−0,068; −0,004]). El resto no cambia; el mayor Δ puntual es +0,016 (B\* s42, recorte a la caja) con IC que cruza el cero.
- **Secundario, máscaras de A2: mismo patrón.** 0/15 mejoran; el fondo gris empeora en el público (−0,085 [−0,133; −0,034]) y en A\* s43 (−0,044 [−0,089; −0,002]); el resto no cambia.
- **A2 contra student_r3 en la misma celda** (bootstrap pareado del macro-F1 sobre las mismas predicciones de val): indistinguibles en 15/15. Como dato descriptivo, no como tendencia: 13 de los 15 Δ puntuales favorecen a A2. No estaba pre-registrado y las celdas no son independientes, porque comparten val y backbone.

#### 5.2.1 EXPLORATORIO: C1-v7 estratificado por tono (registrado el 2026-09-28, antes de calcular nada)

La pregunta es de Nicolás: ¿la máscara ayuda a clasificar mejor la piel más oscura? Hasta ahora C1-v7 solo se midió en global, y ahí la máscara no ayuda (§5.2). Este análisis es post-hoc y exploratorio: no reemplaza a §5.2 ni a §5.1, y ningún resultado suyo se cita como confirmatorio.

**Condición previa (bloqueante).** El estimador FST no se usa hasta validarlo. Primero va la validación T1: el estimador del proyecto se compara contra el FST autorreportado de PAD-UFES-20. Hay evidencia publicada en contra: en PAD-UFES-20, el ITA con umbrales fijos acierta el 22,9 %, por debajo de la baseline de mayoría (PMC13459198).
- Si T1 muestra que el estimador sirve, se calculan los bins FST de abajo sobre val de HAM10000.
- Si T1 no lo muestra, los bins FST de HAM10000 no se calculan. El análisis por tono se hace solo en PAD-UFES-20, con su FST autorreportado.
- En los dos casos los terciles de color se reportan, pero como descripción del color de la imagen, no como tono de piel validado.

**Qué se calcula (si se cumple la condición).**
- **Entradas:** las probabilidades por imagen de val (n = 2.004) que C1-v7 ya guardó por celda (`<celda>.proba.npy` más su `manifest.json`). Son 5 backbones × 4 condiciones (`none`, `bbox_crop`, `neutral_bg`, `mask_pool`) para cada uno de los dos segmentadores, student_r3 (primario) y A2 (secundario). Las etiquetas salen del mismo cache del banco que usó la grilla. No se reentrena ni se re-extrae nada. Antes de calcular se verifica que el sha de las probabilidades coincida con el manifiesto de la celda y que el orden de val coincida con `val_order_sha256`.
- **Tono por imagen:** el manifiesto publicado `fst_pseudo_manifest.csv`, que usa el estimador corregido (`c272bdb`) y el estrato `tone_tercile` (`bbd9bab`) sobre las pseudo-máscaras `pseudo_r2`. Supuesto heredado: `pseudo_r2` no registra `region_mode` y se toma `central` (docs/27, «FST con las máscaras de A2»). Se reporta la cobertura: cuántas imágenes de val tienen bucket FST y cuántas tienen tercil. Las que no tienen bucket (`ita_indeterminate`, es decir mediana de b\* < 5, y las que están en cuarentena) quedan fuera del bin, con su n a la vista.
- **Bins:** FST I, FST II y FST III-VI agrupado; terciles `tone_tercile` claro, medio y oscuro. Aclaración: `tone_tercile` es el tercil de la mediana de L\* de la piel sobre el pool, no un tercil de ITA. Se usa ese porque el ITA solo está definido cuando b\* ≥ 5 (README de camino-b-cnn, «Estimador corregido»). Los terciles son relativos al pool: «oscuro» significa lo más oscuro de un dataset de piel mayormente clara, no un fototipo oscuro.
- **Métricas por bin:** macro-F1, balanced accuracy y sensibilidad de melanoma (recall de `mel` con argmax). Cada bin informa su n y su número de melanomas.
- **IC por bin:** bootstrap percentil de 1.000 remuestras con semilla 20260923. Se remuestrea estratificando por clase dentro del bin, es decir, cada clase conserva su n en cada remuestra.
- **Contraste:** cada condición contra `none` del mismo backbone y segmentador, dentro del mismo bin. Es un bootstrap pareado sobre imágenes: los mismos índices para los dos lados, estratificados por clase como el IC. El código del contraste y la dirección del veredicto se reusan de `dermssl.paired_bootstrap`, no se duplican.
- **Regla de potencia:** un bin con n < 100 se reporta como «sin potencia», con su n y el valor puntual, sin IC de veredicto y sin veredicto. Si un bin no tiene melanomas, la sensibilidad de melanoma queda como «no definida».

**Reglas de lectura.**
- «Mejora» solo si el IC pareado excluye el cero a favor de la condición con máscara. «Empeora» solo si lo excluye en contra.
- Si el IC toca el cero, el resultado es «indistinguible». En ese caso nunca se dice «gana» ni «tendencia», aunque el Δ puntual sea positivo.
- **Multiplicidad:** son 5 backbones × 3 condiciones × 6 bins × 3 métricas × 2 segmentadores = 540 contrastes, sin corrección. Con el 5 % habría unos 27 IC que excluyen el cero por azar. Una celda aislada que excluye el cero es una hipótesis, no un hallazgo. Para leerla como señal tiene que repetirse con las dos semillas del mismo brazo (regla de §4.1) y con los dos segmentadores.
- **Límite de lo que responde:** comparar el Δ de un bin con el de otro («la máscara ayuda más en oscuro que en claro») es una interacción. Este análisis no la contrasta, así que no se concluye.

**Salida.** `resultados/clasificacion/c1_v7/tone_stratified/` con un JSON y una tabla markdown generada por script, con su test de frescura, igual que `build_closing_table`. Ninguna cifra se transcribe a mano en docs.

**Caveats que viajan con el resultado.**
- El tono sale de una estimación automática desde la imagen. No lo etiquetó un dermatólogo.
- El bucket FST existe solo para la submuestra con b\* ≥ 5, que no es aleatoria y tiene menos melanoma (README de camino-b-cnn).
- HAM10000 casi no tiene FST III-VI, así que ese bin probablemente quede «sin potencia».
- El tercil está confundido con el diagnóstico: la tasa de melanoma y de queratosis benigna cambia entre terciles. Por eso un Δ por bin mezcla tono y composición de clases.

**La ablación Plain vs Proposed-oracle-mask de docs/26 §7.3 queda reemplazada por este rediseño (anotado el 2026-09-28).** Esa ablación existía para aislar la causa de la caída del propuesto en la v6, y pedía steps emparejados, IDs de val excluidos del pool y revisión controlada. La v7 ya arregló esos confusores: pasos iguales (5.700), un solo `code_revision` por final, los IDs de val y del holdout excluidos del pretraining, y dos semillas. La variante con la máscara oráculo queda como fase 2 si sobra GPU.

### 5.3 Observación, no resultado registrado: las finales degradan las features densas (2026-09-28)

S-diag (docs/27) segmenta sin entrenar usando los tokens de parche del último bloque. En su método primario, las finales v7 quedaron por debajo del DINOv2 público, con intervalos que no se solapan:
- **ISIC 2018 (Dice [IC 95 %]):** público 0,452 [0,441; 0,464], A\* s42 0,338 [0,328; 0,348], B\* s42 0,364 [0,354; 0,374].
- **HAM10000:** público 0,563 [0,558; 0,569], A\* 0,398 [0,393; 0,402], B\* 0,437 [0,433; 0,442].

La predicción de S-diag no comparaba pesos entre sí, así que esto se reporta como observación. Es coherente con cómo se entrenó la v7: SimSiam y el ancla actúan sobre el CLS, y nada restringe los tokens de parche. Es la misma degradación de las features densas que motiva el Gram anchoring de DINOv3 (Siméoni et al., 2025, arXiv:2508.10104); acá se usó solo la versión liviana sobre el CLS (§2).

Implicancia para v7.1 (docs/29 §2.2 y §2.7): las features locales (media o GeM de los parches, attention pooling) podrían rendir menos en A\*/B\* que en el público. Se reporta lo que salga, sin ajustar nada por esta observación.

## 6. Referencias

- Khanna et al. *ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts*. ICML 2025. arXiv:2406.10973.
- Li, Grandvalet, Davoine. *Explicit Inductive Bias for Transfer Learning with Convolutional Networks* (L2-SP). ICML 2018.
- Siméoni et al. *DINOv3*. 2025. arXiv:2508.10104.
- Chaves et al. *An Evaluation of Self-Supervised Pre-Training for Skin-Lesion Analysis*. ISIC Workshop 2022.
- Zini et al. *Planckian Jitter*. 2022. arXiv:2202.07993.
- Kumar, Raghunathan, Jones, Ma, Liang. *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution* (LP-FT). ICLR 2022.
- Pakzad, Abhishek, Hamarneh. *CIRCLe: Color Invariant Representation Learning for Unbiased Classification of Skin Lesions*. ISIC Workshop 2022.
- Tian, Chen, Ganguli. *Understanding Self-Supervised Learning Dynamics without Contrastive Pairs*. ICML 2021. arXiv:2102.06810.
- Chen, He. *Exploring Simple Siamese Representation Learning* (SimSiam). CVPR 2021.
