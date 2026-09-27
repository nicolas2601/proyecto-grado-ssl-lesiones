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

**Réplica de semilla (opcional, registrada antes de las finales).** Las finales usan semilla 42. El IC del bootstrap cubre la variabilidad del conjunto de evaluación, NO la del entrenamiento. Si hay GPU libre al terminar las finales, se repiten A\* y B\* con semilla 43, en la misma revisión y con el mismo exclude, y se reportan AMBAS semillas. No se elige semilla: la conclusión se reporta con las dos. Si no hay tiempo, esto va como limitación explícita.

**Regla de pausa del trabajo de CPU (S2, casi-duplicados; 2026-09-27):** se pausa si cualquier entrenamiento cae más de 15 % bajo su ritmo propio medido solo (A ≈ 22-24 pasos/min, umbral ~19; B ≈ 14,4, umbral ~12,2).

El término de tono de B1 y B2 es el del mismo crop (§2), no el de crops independientes que existía cuando se lanzó la oleada 1; la oleada 1 no lo usa, así que no la afecta.

**Regla de selección, por oleada:** gana la variante con mayor macro-F1 en el holdout interno. Si `|Δ macro-F1| < 0,005`, gana la de menor ECE. La oleada 2 hereda las anclas de la ganadora de la oleada 1, así que los dos brazos difieren solo en el tono.

**Finales:** A* y B* con `--max-steps 5700` (≈12 GPU-h, el presupuesto de la v6), una GPU cada una, en paralelo, y la evaluación única del punto 3.

## 5. Predicciones registradas

- **P1 (anclaje):** el plain anclado v7 no pierde contra DINOv2 público en macro-F1: Δ ≥ −0,01. Si el intervalo excluye el cero a favor, la adaptación anclada mejora a DINOv2.
- **P2 (tono):** el propuesto v7 contra el plain anclado v7: ECE mejor (IC excluye el cero) y macro-F1 no peor (Δ ≥ −0,01).
- **P3 (objetivo de la tesis):** el propuesto v7 contra DINOv2 público: macro-F1 Δ ≥ 0 manteniendo la ventaja en ECE. Probabilidad estimada antes del dato: ≈35 %.

Criterio: se reporta el resultado sea cual sea. Si P3 no se cumple, la conclusión honesta es que la adaptación SSL mejora calibración y ranking pero no la separación por clase frente a DINOv2 público.

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
