# 29. Plan v7.1: sacar más de las finales existentes

**Registrado el 2026-09-28, antes de correr cualquier cosa.** Parte de la revisión de literatura; los papers citados están verificados.

## 1. Regla central

Toda selección (α, protocolo del probe, resolución) se hace en el **holdout interno**: el 20 % del train de C1, agrupado por `lesion_id` y excluido del pool de pretraining (docs/28 §4.1). La validación se toca **una sola vez al final**, con la configuración elegida.

Límites que se declaran desde ya:
- El holdout ya se usó para elegir A3 y B1 en los pilotos, así que elegir de nuevo sobre él suma optimismo de selección en el holdout. La lectura en val no lo tiene.
- Val ya se leyó con las finales de la semilla 42 y de la 43, y se va a leer con C1-v7 (docs/28 §5.2). La lectura v7.1 es una más, sobre modelos derivados de los mismos; se reporta como tal y el resultado primario de docs/28 §5.1 no cambia.

## 2. Bloques, en el orden de GPU

Antes van S-diag (docs/27), C1-v7 (docs/28 §5.2) y S4-UNet (docs/27). El costo de cada bloque se envía antes de lanzarlo.

### 2.1 WiSE-FT y seed soup (Wortsman et al., arXiv:2109.01903 y arXiv:2203.05482)

Todo se hace sobre las finales existentes, sin entrenar.
- θ = (1 − α)·público + α·adaptado, con α ∈ {0,2; 0,4; 0,6; 0,8; 1,0}, para A\* y B\*.
- **Soup:** promedio de los pesos de las semillas 42 y 43 del mismo brazo. También se evalúa el soup combinado con WiSE (mismos α).
- **Test previo, obligatorio:** en el holdout, α = 0 reproduce las métricas del público y α = 1 las de la final, con tolerancia 1e-6. Si no, no se sigue.

### 2.2 Protocolo del probe (DINOv2, Oquab et al., arXiv:2304.07193)

- **Features:** CLS | CLS + media de los parches | concatenación del CLS de las 4 últimas capas.
- **Pesos de clase:** ninguno | balanced.
- Todo en el holdout. La ganadora se reporta solo como SECUNDARIA en val. El protocolo primario (CLS, `logreg`, C = 1) no cambia.

### 2.3 Resolución

Extracción a 224 contra 448 con los mismos pesos, en el holdout. Es un diagnóstico secundario.

### 2.3b Métricas clínicas secundarias (registradas el 2026-09-28, antes de correr)

Para cada configuración evaluada en el holdout (α, soup, protocolo, pesos de clase, resolución) se reportan también la sensibilidad de melanoma (recall de `mel`) y el AUC de melanoma contra el resto, ambas con IC bootstrap. Con `class_weight = balanced` se miran en particular: es la palanca más directa para bajar la confusión de melanoma con nevus, que en la v7 es el error dominante (sensibilidad de melanoma entre ~0,43 y 0,50 en val). Son secundarias y no cambian la regla de selección, que sigue siendo el macro-F1 de §2.4.

### 2.4 Regla de combinación

En el holdout se elige la mejor configuración de cada brazo por macro-F1; si |Δ| < 0,005, gana la de menor ECE (igual que docs/28 §4.1). Después se corre una sola evaluación en val de esa configuración contra el público, con bootstrap pareado. Se reportan siempre las dos cosas: el primario v7 intacto y la lectura v7.1.

### 2.5 Semilla 44

A\* y B\* con la semilla 44, una GPU cada una, con la misma receta, la revisión f1c2053 y el exclude 95d21013. Es una réplica adicional y se evalúa con la regla de docs/28 §4.1 extendida a tres semillas: se reporta cada semilla, la media y el rango, sin elegir ninguna. La varianza entre semillas (~0,04 en macro-F1 de A\*) es del tamaño del efecto; más semillas le dan peso a la conclusión y alimentan el soup.

**Enmienda (registrada el 2026-09-28, antes de lanzar s44 y antes de ver cualquier número suyo).**

- **"Alimentan el soup" queda sin efecto para v7.1.** La selección de v7.1 se commiteó en 56e7670 con el soup de las semillas 42 y 43, y no se toca. s44 no entra en ningún soup ni en ninguna configuración de v7.1 (`scripts/v71_io.py` fija esas dos semillas aparte del registro de semillas).
- **Propósito.** s44 es robustez de semilla de la v7 (A\* y B\* contra el público), con el mismo protocolo que s42 y s43: `scripts/c1_final_eval.py --split val --single-evaluation`, val del fold 0 (n = 2.004), probe `logreg`, bootstrap pareado de 1.000 remuestras, código de evaluación de f562ea9 (el de s43), salida en `resultados/clasificacion/v7/s44/`. Responde a la fila P1-NI de la tabla de cierre, que hoy es "no robusto a la semilla" (s42 no cumplida, s43 cumplida).
- **Lectura, fijada antes del dato.** Se reporta cada semilla, la media y el rango, sin elegir ninguna. Una predicción es robusta a la semilla solo si se cumple en las tres. P1-NI queda robusta solo si s44 la cumple; si no, sigue como "no robusto a la semilla". P3-IC, hoy cumplida en s42 y s43, deja de ser robusta si s44 no la cumple, y entonces "le gana al público" no se sostiene (docs/28 §4.1). Sin predicción direccional para P1-NI: con s42 y s43 en desacuerdo, se estima ~50 %.
- **Desviaciones de la receta de s42/s43** (la receta se leyó de la config de W&B de las cuatro finales: 41f0e8d164a3, 8f1684ceea84, 4672e3ff0604 y b8e85ebc8ece):
  1. **`num_workers` 4 en vez de 0.** Las finales de s42 y s43 corrieron con `num_workers = 0`, no con 8. s44 usa 4 por brazo, a pedido de Paula, porque el pod es compartido. Con 0 workers el recorte, los volteos y el ColorJitter consumen el RNG global de torch del proceso principal; con 4, cada worker siembra torch con `base_seed + worker_id` (`torch/utils/data/_utils/worker.py`), y el loader saca `base_seed` de su propio generador. Esas tiradas no son las de una corrida con 0 workers. La constancia de color no cambia, porque usa `np.random.default_rng((seed, idx))` por muestra, igual que el orden de las muestras (`ResumableRandomSampler(seed)`). La distribución de las aumentaciones es la misma, pero la secuencia es otra. El consumo de CPU de s43 según W&B (métrica `cpu` del proceso): ~35 % en A y ~31 % en B, sobre 48 CPU lógicas. Los workers corren con un hilo de torch cada uno. **Riesgo:** si 4 workers no alcanzan a alimentar la GPU, baja el ritmo. Aplica la regla de pausa de docs/28 §4.1 (A ≥ ~19 pasos/min, B ≥ ~12,2): si en los primeros ~200 pasos el ritmo cae por debajo, se para y se consulta antes de gastar el presupuesto.
  2. **Revisión de código.** s44 no corre desde f1c2053, sino desde la revisión del clon de lanzamiento: ahí viven los guards (`GIT_COMMIT` obligatorio, `--resume` explícito). El camino de entrenamiento no cambió. Entre f1c2053 y HEAD solo cambió `collect_images` (`os.walk` en vez de `rglob`, el mismo conjunto ordenado), y `tests/test_pretrain_finals_characterization.py` compara HEAD contra una referencia generada en f1c2053 con las palancas de A\* y B\*. La evaluación exige la misma revisión y la misma semilla entre A y B de s44, no contra s42 ni s43.
  3. **Sin desviación en el presupuesto.** `--budget-gpu-hours` es 8 en A y 12 en B, como en s42 y s43. Lo fija `scripts/v7_final_launch.py`. Costo medido en s43 (W&B, `gpu_hours`): A 4,05 y B 6,76 GPU-h.
  4. **Reanudación.** Un `--output-dir` que ya tiene `resume.pt` aborta, salvo que se pase `--resume`. La matemática es la misma; un relanzamiento tiene que declarar que reanuda.

### 2.6 Prueba externa sin entrenar: PAD-UFES-20 (Pacheco et al., arXiv:2007.00478; CC BY 4.0; Mendeley zr7vgbcyr2)

- Se baja al pod y se verifica el conteo (2.298 imágenes, 1.373 pacientes) y la columna de paciente antes de usarlo.
- El probe se entrena en el train de HAM (como en C1) y se evalúa en PAD solo en la intersección registrada:
  - **primaria:** bcc, mel y nv (NEV en PAD);
  - **secundaria:** además, akiec contra ACK + SCC.
- Público contra A\* contra B\*, con bootstrap agrupado por paciente.
- Es robustez fuera de dominio (PAD son fotos clínicas de smartphone, no dermatoscopia) y va a la tesis tal cual, sin ajustar nada.
- Protocolo completo, licencia y predicción registrada: [docs/30](30-pad-ufes-zero-shot.md).

### 2.7 Bloque v7.1-mel (registrado el 2026-09-28, antes de correr)

Viene de una segunda revisión de literatura. Corre dentro de v7.1, sobre las mismas finales y el mismo holdout, y con la misma regla: todo se elige en el holdout y val se lee una sola vez, junto con la lectura v7.1 de §2.4. Casi todo es CPU; la única extracción nueva (TTA y features locales) es un forward sin backprop.

1. **Punto de operación de melanoma.** Umbral sobre la probabilidad de `mel` (mel contra el resto) fijado en el holdout para especificidad del 80 % y del 90 %; se reporta la sensibilidad resultante en val, y la especificidad real en val para mostrar cuánto se corre el umbral. Se agrega un tercer punto, fijado en el holdout para sensibilidad del 95 %, con la especificidad resultante en val: es la convención que usa de verdad PROVE-AI (Marchetti et al., *npj Digit Med* 2023, doi 10.1038/s41746-023-00872-1; umbral predefinido de sensibilidad 95 %, verificado en el resumen). Los puntos de especificidad fija son la lectura principal de este bloque; el de sensibilidad 95 % es para comparar con ese estudio. Ninguno cambia la predicción multiclase ni el macro-F1.

   **Protocolo en val (decisión de Paula, 2026-09-28, registrada antes de leer val).** Los umbrales se fijan en el holdout con el probe entrenado en el 80 % del fit. Reentrenar el probe cambia la escala de los scores, así que un umbral no se transfiere a otro probe:
   - Las métricas con umbral (sensibilidad y especificidad en los umbrales de especificidad 80 % y 90 %, y en el de sensibilidad 95 %) se evalúan en val con el MISMO probe entrenado en el 80 % del fit.
   - Las métricas sin umbral (macro-F1 por argmax, AUC, balanced accuracy, ECE y sensibilidad de mel por argmax) se evalúan en val con el probe reentrenado en el train completo, como en la v7.
   - Los puntos de operación re-ajustados sobre val no se reportan.

   El costo sube en un ajuste del probe por configuración leída en val (sobre el 80 % del fit). Las features son las mismas, porque el fit es un subconjunto del train. `v71_final_eval.json` lo describe en `note_operating_points`.
2. **Logit adjustment post-hoc** (Menon et al., ICLR 2021, arXiv:2007.07314, verificado): logit_y − τ·log π_y sobre la salida de la regresión logística, con π_y las frecuencias de clase del train del probe y τ ∈ {0; 0,5; 1; 1,5; 2} elegido en el holdout. τ = 0 es el probe plano. Se compara contra `class_weight = balanced` y contra el probe plano.
3. **Features locales.** Se suman a la grilla de features de §2.2:
   - CLS + GeM de los tokens de parche (p = 3 fijo, no se ajusta);
   - CLS + attention pooling: media de los tokens de parche ponderada por la atención CLS→parche del último bloque, promediada sobre las cabezas.
   (CLS + media de los parches ya está en §2.2.)
4. **TTA D4.** Las 8 transformaciones del grupo diedral (identidad, rotaciones de 90°, 180° y 270°, y sus volteos), que preservan la etiqueta en dermatoscopía. Se promedia el softmax del probe sobre las 8. El probe se entrena sin TTA. Nada de crops agresivos.
5. **Métricas.** Para cada configuración, con IC bootstrap: balanced accuracy (la métrica de ISIC 2018), accuracy, macro-F1, sensibilidad de mel a especificidad 80 % y 90 %, y AUC mel-vs-rest.
6. **Grilla y selección.** Pesos de clase ∈ {ninguno; balanced; logit adjustment τ}, cruzados con las features de §2.2 + ítem 3 y con TTA sí/no. Los puntos de operación del ítem 1 son post-hoc y no son un factor de selección. La regla no cambia: macro-F1 en el holdout, desempate por ECE si |Δ| < 0,005 (§2.4). La sensibilidad de mel es secundaria y se reporta siempre. La grilla es más grande que la de §2.2, así que el optimismo de selección en el holdout crece; se declara, y la lectura que vale es la de val.
7. **Condición para la fase 2 de este bloque** (ver §3): se evalúa en el **holdout**, no en val, con la configuración elegida. Si la sensibilidad de mel a especificidad 80 % no pasa de 0,70 en el holdout para ningún brazo, queda habilitado proponer la fase 2. No se lanza sin el OK de Paula.

Citas que llegaron en el informe y **no** se usan acá: arXiv:2609.30613 (MedTokenBudget) y arXiv:2607.22687 (DINOv3-MIL) no se pudieron verificar. Si alguna vez se mencionan, van como analogía no verificada.

## 3. Fase 2 (fuera por ahora, anotada)

- **v7.1-mel, fase 2 (solo con la condición de §2.7 ítem 7 y con OK de Paula):** continued pretraining corto con muestreo guiado por pérdida, adaptado de MAK (Jiang et al., NeurIPS 2021, arXiv:2111.01004, verificado). MAK original elige muestras de un pool externo con la esperanza de la pérdida contrastiva sobre aumentaciones más un criterio de diversidad (K-center). La adaptación de acá sobremuestrea, dentro del mismo pool sin etiquetas, las imágenes con pérdida SimSiam alta promediada sobre aumentaciones. Es una adaptación, no el método tal cual, y se reporta así.

- ExPLoRA / LoRA de pretraining extendido (arXiv:2406.10973).
- Sumar PAD al pool SSL.
- Fine-tune supervisado en PAD: solo demo, fuera de la tesis.

## 4. Trazabilidad

Todo va a W&B, a Engram y a las tablas generadas (`resultados/clasificacion/TABLA_CLASIFICADOR.md`). Se hace push al cerrar cada bloque.
