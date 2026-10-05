# 31. T4 (exploratorio): constancia de color Shades of Gray antes de DINOv2

**Registrado el 2026-09-28, antes de escribir el código que lo corre y antes de ver cualquier número con Shades of Gray.** El bloque entre marcadores no se edita después del commit de registro: `scripts/t4_sog_run.py` compara, contra git, que el bloque en HEAD sea idéntico al del commit de registro y que el script se haya agregado después (mismo mecanismo que docs/30, `dermssl/pad_ufes_guard.py`). Las erratas, si hacen falta, van después del bloque.

<!-- registro-t4-sog:inicio -->

## 1. Pregunta

¿Normalizar el iluminante de cada imagen con Shades of Gray (SoG) antes del encoder cambia el macro-F1 global del probe lineal y la brecha de macro-F1 entre los terciles de luminancia oscuro y claro? Es un bloque **exploratorio**: el primario v7 (docs/28 §5.1) y la lectura v7.1 (docs/29 §2.4) no cambian con ningún resultado de acá.

## 2. La transformación

- **Método.** Shades of Gray: Finlayson y Trezzi, "Shades of Gray and Colour Constancy", 12th Color Imaging Conference, IS&T, 2004, pp. 37–41, doi 10.2352/CIC.2004.12.1.art00008. El iluminante de cada canal se estima con la norma de Minkowski de orden p de sus píxeles; p = 1 es gray-world y p → ∞ es max-RGB. En su base de datos calibrada, p = 6 fue el mejor en conjunto.
- **p = 6, fijo a priori, no se ajusta.** Es el valor estándar en dermatoscopía: Barata, Celebi y Marques, "Improving dermoscopy image classification using color constancy", *IEEE J Biomed Health Inform* 19(3):1146–1152, 2015, doi 10.1109/JBHI.2014.2336473; y Gessert et al., "Skin lesion classification using ensembles of multi-resolution EfficientNets with meta data", *MethodsX*, 2020 (ScienceDirect PII S2215016120300832) (primer puesto en las dos tareas de ISIC 2019), que aplican SoG con p = 6 siguiendo al ganador del año anterior. Las tres citas se verificaron contra su ficha bibliográfica y su resumen o sección de métodos públicos, no contra el texto completo.
- **Implementación.** `dermssl.pretrain.color_constancy.shades_of_gray`, la MISMA función que el preentrenamiento de A\* y B\* aplicó de forma estocástica (SoG en el 30 % de las vistas y gray-world en otro 30 %, `pretrain/config.py`). Ganancia del canal c: g_c = media_k(n_k) / n_c, con n_c = (media de x_c^p)^(1/p) y x en [0, 1]; después se recorta a [0, 1]. Barata usa d_c = 1/(√3·e_c) con e = n/‖n‖₂, es decir RMS(n)/n_c: la cromaticidad corregida es idéntica y solo cambia un factor global (RMS/media ≥ 1), que acá no se usa para no mover el brillo medio.
- **Dónde se aplica.** Sobre la imagen ya redimensionada y recortada al centro (la geometría del banco, `torch_v71.resize_crop_uint8`), pasada a float en [0, 1], y vuelta a uint8 por truncamiento: la misma composición que `pretrain/torch_data._finish_view`. Se aplica igual en fit, holdout, train y val. El recorte central quita parte del viñeteado negro de las esquinas; lo que queda del viñeteado entra en la estimación del iluminante (no se enmascara, igual que en Barata).

## 3. Pesos y protocolo de cada brazo

- **Pesos:** público; A\* = `A_soup_a1.0` (sopa de las finales s42 y s43 de A); B\* = `B_s42_a1.0`. Son los pesos de los ganadores v7.1 commiteados en `resultados/clasificacion/v71/selection.json`.
- **Protocolo primario, fijo (sin selección):** el ganador v7.1 commiteado de cada brazo.
  - A\*: `A_soup_a1.0@224|cls_attn|la0.5|tta`
  - B\*: `B_s42_a1.0@224|cls4|balanced|tta`
  - público: `public@224|cls4|la0.5|tta`
- **Por qué este protocolo y no el primario v7 (CLS, sin pesos, sin TTA).** Fijar el protocolo aísla el efecto de SoG: el único factor que cambia es la entrada. El lado sin SoG es exactamente la lectura v7.1 ya publicada, así que sirve de chequeo de reproducción. Y es la mejor lectura vigente, que es la que SoG tendría que mejorar. El costo: el efecto se mide en un protocolo elegido en el holdout sin SoG; si SoG favorece otro protocolo, eso lo mide la lectura secundaria de §6.
- **Probe:** `logreg` de C1 (C = 1, estandarizado, `probes.fit_logreg`), fit/holdout de docs/28 §4.1, val del fold 0 (n = 2.004), probe de val reentrenado en el train completo (8.011), como en v7.1.

## 4. Métricas

- **Primaria:** macro-F1 global en val. Contraste Δ = (con SoG) − (sin SoG), mismos pesos y mismo protocolo, bootstrap pareado de 1.000 remuestras con semilla 20260923.
- **Brecha de luminancia:** columna `tone_tercile` del manifiesto `fst_pseudo_manifest.csv` (sha256 `57df1e65e88d3b54ba530d6e026010bfc8ce25d1437e50b4ffe37cfd7d4f87da`, 58.457 filas; cubre las 10.015 imágenes de HAM10000: oscuro 4.082, medio 4.876, claro 864, sin dato 193). Son terciles de la mediana de L\* de la piel, medidos sobre las imágenes ORIGINALES; no se recalculan después de SoG. Son terciles de luminancia y **no son tono de piel**; la estimación ITA por imagen **no es FST** (docs/20). brecha = |macro-F1(oscuro) − macro-F1(claro)|; Δbrecha = brecha con SoG − brecha sin SoG, con el mismo bootstrap pareado (el tercil viaja con la imagen en cada remuestra). Secundaria: máximo − mínimo entre oscuro, medio y claro. `unavailable` no entra en ninguna brecha.
- **Secundarias, sin predicción:** balanced accuracy, AUC, ECE, sensibilidad de mel y AUC de mel.

## 5. Predicción registrada

Razones: A\* y B\* ya vieron SoG y gray-world en el 60 % de las vistas del preentrenamiento, así que deberían ser casi invariantes al iluminante; SoG conserva la media de las normas de canal, es decir, corrige cromaticidad y no brillo, y los terciles son de brillo; y HAM10000 está en distribución, mientras que la ganancia de SoG en la literatura es sobre todo entre fuentes.

- **P1 (A\* y B\*):** |Δ macro-F1| ≤ 0,01 en el punto, y el IC pareado incluye el 0. Se falsea si |Δ| > 0,01 o si el IC excluye el 0, en cualquiera de los dos. Probabilidad estimada ~70 %.
- **P2 (público):** |Δ macro-F1 del público| > máx(|Δ A\*|, |Δ B\*|), sin dirección: el público nunca vio SoG y se mueve más. Probabilidad ~55 %.
- **P3 (brecha):** en los tres pesos, el IC de Δbrecha incluye el 0 y |Δbrecha| ≤ 0,03 en el punto. Se falsea si algún IC excluye el 0 o algún punto pasa de 0,03. Probabilidad ~75 %.
- **Potencia.** El tercil claro tiene ~183 imágenes en val. El Δ pareado dentro de ese estrato tiene un error estándar del orden de 0,03, así que cambios de la brecha menores que ~0,06 no se detectan. Un IC que incluye el 0 se lee "no se detecta", no "no hay efecto".

## 6. Holdout, selección y una sola lectura de val

1. **`grid` (holdout).** Para cada peso (público, A\*, B\*) y cada entrada (sin SoG, SoG p = 6): las 60 celdas v7.1 (5 features × 6 ponderaciones × TTA sí/no) a 224, con las mismas métricas e IC. 448 no entra.
   - **Guard de reproducción (antes de todo):** las celdas sin SoG de los tres protocolos fijos reproducen las métricas puntuales de `resultados/clasificacion/v71/holdout_grid.json` con tolerancia 1e-6. Si no, se aborta y no se sigue.
2. **`select` (holdout).** Por peso, la regla de docs/28 §4.1 (máximo macro-F1; si |Δ| < 0,005, menor ECE) sobre sus 60 celdas CON SoG: es el "ganador SoG", una lectura secundaria. Se reportan también los contrastes pareados en el holdout (SoG contra sin SoG en el protocolo fijo), que tienen optimismo de selección cero en el protocolo pero el holdout ya se usó en v7.1.
3. **`val`: una sola lectura**, `--split val --single-evaluation`, con los guards de v7.1 (árbol limpio, sha del exclude, corridas consistentes) más el guard de este registro. Por peso se leen: sin SoG en el protocolo fijo, con SoG en el protocolo fijo y el ganador SoG. Contrastes: **primarios**, SoG contra sin SoG en el protocolo fijo (tres, en macro-F1 y en brecha); **secundarios**, ganador SoG contra sin SoG en el protocolo fijo. Las entradas sin SoG se comparan con `resultados/clasificacion/v71/v71_final_eval.json` y se reporta la diferencia.
4. **Multiplicidad.** Tres contrastes primarios × dos métricas, sin corrección. Es exploratorio y se declara así.

## 7. Qué cuenta como mejora

- **Mejora global:** el IC pareado de Δ macro-F1 excluye el 0 y Δ > 0.
- **Reducción de la brecha:** el IC pareado de Δbrecha excluye el 0 y Δbrecha < 0.
- Nunca "gana" ni "mejora" si el IC toca el 0. Una mejora acá no cambia el pipeline publicado: adoptar SoG sería una decisión nueva, con su propia lectura.

## 8. Costo proyectado

- **GPU (extracción, sin backprop, ~720 img/s a 224 medido en el pod):** grid 2 entradas × 3 pesos × (6.409 fit + 8 × 1.602 holdout) = 115.350 forwards; val como máximo 6 unidades × (8.011 + 8 × 2.004) = 144.258. Total ≈ 259.600 forwards ≈ **0,10 GPU-h**.
- **CPU:** probes de la grilla 6 unidades ≈ 0,4 CPU-h × 6/21 ≈ 0,11 (medido en v7.1); bootstrap de la grilla 360 celdas × ~2,5 s ≈ 0,25; decodificación y SoG de 18.026 imágenes × 2 entradas a ~60 img/s por núcleo ≈ 0,17; probes y contrastes de val < 0,1. Total ≈ **0,6 CPU-h**. `--dry-run` imprime la proyección con los tamaños reales antes de lanzar.

## 9. Pre-mortem

- *El efecto de SoG se confunde con un cambio de protocolo:* el primario fija el protocolo.
- *Se mezclan arrays o features con y sin SoG:* el nombre de cada array en caché y la huella de la corrida llevan la entrada; una unidad de otra entrada se rechaza, no se reusa.
- *La reproducción sin SoG no da 1e-6:* se aborta antes de extraer con SoG y se consulta.
- *El redondeo:* se trunca al volver a uint8, igual que en el preentrenamiento.

<!-- registro-t4-sog:fin -->

## 10. Notas posteriores al registro (no cambian el bloque)

- **Costo, con el código.** `t4_sog.cost_projection` (lo que imprime `--dry-run`) da 0,10 GPU-h y 0,75 CPU-h con los tamaños reales (fit 6.409, holdout 1.602, train 8.011, val 2.004): probes de la grilla 0,11, probes de val 0,17 (cota: la feature del ganador SoG no se conoce antes de `select` y se acota por la más ancha), bootstrap 0,30 (incluye los contrastes de val y las dos brechas), decodificación 0,17. El 0,6 de §8 subestimaba los probes y el bootstrap de val.
- **Macro-F1 dentro de un tercil.** Se calcula con `metrics.macro_f1` sobre las 7 clases, como en los estratos del banco: una clase ausente en el tercil cuenta con F1 = 0. Es la convención del repo y pesa más en el tercil claro (~183 imágenes).
- **Comandos del pod.** En el docstring de `scripts/t4_sog_run.py` (grid con `--dry-run` primero, grid, select y la lectura única de val).

## 11. Limitación: sensibilidad del probe a los hilos BLAS

<!-- limitacion-hilos-blas:inicio -->
Limitación (no errata): el probe lineal tiene una sensibilidad de implementación de hasta 0,014 de macro-F1 según el número de hilos BLAS por proceso (4 vs 8 hilos; probe_jobs 10 vs 6; cambian 48 de 60 celdas en cada una de las 3 unidades, con hasta 18 imágenes que cambian de clase predicha por celda); con la configuración registrada (4 hilos) reproduce la grilla publicada bit a bit (máx. |Δ| 0). El efecto SSL (A* v7.1 − público con el mismo protocolo, macro-F1 +0,091 [+0,050; +0,132]) es ~6,5× esa sensibilidad. No es falta de convergencia: los 30 ajustes publicados convergen por tolerancia (0/30 alcanzan max_iter=2000; n_iter 165–862); la sensibilidad es de redondeo en punto flotante dentro de la tolerancia del optimizador. En T4 el umbral de P1 (≤ 0,01) está por debajo de esa sensibilidad, así que P1 solo se lee pareado dentro de la misma corrida con hilos fijos. Evidencia: commits 0a691eb (reproducción) y 278efd2 (convergencia).

Generado desde `resultados/clasificacion/t4/diagnostico_reproduccion/identity_j10_vs_j10.json`, `identity_j10_vs_j6.json`, `t4/diagnostico_convergencia/probe_convergence.json`, el contraste pareado `A71 − public_A71` de `v71_final_eval.json` y `t4_sog.P1_MAX_ABS_DELTA`; no editar a mano.
<!-- limitacion-hilos-blas:fin -->
