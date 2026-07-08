# 04 — Decisiones técnicas (ADR-style)

Registro de las decisiones clave, con contexto y justificación.

---

## D-01 — Núcleo metodológico del anteproyecto: DINO / DINOv2

**Decisión (documento):** el método SSL de referencia en el anteproyecto es **DINO/DINOv2**
(alternativa: Barlow Twins), backbone **ViT-B/16** si GPU ≥ 16 GB, o **ResNet-50** si no.
Basado en deep research con 141 referencias verificadas.

**Estado:** vigente como marco teórico / objetivo ideal.

---

## D-02 — Implementación pragmática en Kaggle: SimSiam sobre ResNet-50

**Decisión (implementación real):** el pipeline `v2` que corre en Kaggle usa **SimSiam**
(Chen & He, CVPR 2021) sobre **ResNet-50**, domain-adaptive desde ImageNet.

**Por qué la divergencia con D-01:**
- SimSiam es **negative-free** (no necesita batch de miles ni cola de memoria) → corre en **1 GPU T4**.
- DINO/ViT-B con el presupuesto de estudiante (T4 x2) es inviable de entrenar bien.
- La **Práctica 1** demostró empíricamente que entrenar SSL *desde cero* pierde contra un backbone
  random con cómputo de estudiante → la estrategia correcta es **continuar** un backbone
  pre-entrenado (ImageNet), no partir de cero.

**Consecuencia honesta:** cuando se consiga cómputo real (GPUs UNAB), se podrá escalar a
DINOv2/ViT-B como plantea el anteproyecto. SimSiam/ResNet-50 es la versión **defendible y ejecutable
hoy**, no el techo del método.

---

## D-03 — Split lesión-disjunto (StratifiedGroupKFold por `lesion_id`)

**Decisión:** partir por `lesion_id`, no por `image_id`, con `StratifiedGroupKFold(5)` + `assert`
de cero solapamiento train/val.

**Por qué:** partir por imagen mete la misma lesión en train y val → leakage que infla la métrica.
Con split honesto el número baja (~0.74) pero es **real y defendible**.

---

## D-04 — Ablación ImageNet vs SSL + linear probe

**Decisión:** el aporte del SSL se mide con dos evidencias:
1. **Ablación full fine-tune**: mismo split/hiperparámetros/semilla, solo cambia el punto de partida.
2. **Linear probe** (backbone congelado + clasificador lineal) en regímenes de pocas etiquetas
   (10/20/50/100%). El linear probe aísla la **calidad de la representación** (el full fine-tune la tapa).

---

## D-05 — SSL a 224px (no 128px)

**Decisión:** `SSL_RES = 224`, igual que el fine-tune y la evaluación.

**Por qué:** correr SSL a 128px y evaluar a 224px provoca **BN drift** (las estadísticas de
BatchNorm no coinciden) → posible colapso aparente del linear probe. Igualar resolución elimina esa
variable de confusión (ver [06 — Problemáticas](06-problematicas.md)).

**Estado:** ✅ **confirmado en la corrida 2026-07-07** — el linear probe SSL subió de 0.264 a 0.4509;
el colapso desapareció.

---

## D-08 — Backbone = ResNet-50 (por qué esta arquitectura)

**Decisión:** ResNet-50 pre-entrenada en ImageNet como backbone único, tanto para el baseline como
para el SSL domain-adaptive.

**Por qué (resumen):** es el estándar de la literatura dermatológica (comparabilidad), cabe en una
T4, tiene pesos ImageNet probados, es el punto dulce tamaño/costo, y soporta Grad-CAM directo.
Explicación completa y alternativas descartadas (ViT, EfficientNet, ResNet-18/101) en
[13 — ¿Por qué estas decisiones?](13-por-que-estas-decisiones.md).

---

## D-09 — Modelo desplegado = fine-tune SSL (no el baseline ImageNet)

**Decisión:** el checkpoint que se sirve (`best_resnet50_ssl.pth`) es el fine-tune construido sobre
el **backbone SSL domain-adaptive**, no el baseline ImageNet.

**Por qué:** la ablación dio empate en macro-F1 (Δ 0.0007, ruido) y **ventaja del SSL en AUC**
(+0.0087). Ante empate, se despliega el modelo SSL por ser el **artefacto central de la tesis** y el
de mejor ranking. La inicialización ImageNet queda documentada como línea base.

**Naturaleza:** ⚠️ modelo **preliminar** — el techo con recursos de estudiante (T4). El definitivo se
entrenará con GPUs de la UNAB.

---

## D-06 — KPIs clínicos como criterio de aceptación

**Decisión:** el modelo se juzga contra KPIs, no contra una sola cifra:

| KPI | Meta |
|---|---|
| macro-F1 | ≥ 0.65 |
| AUC-ROC macro | ≥ 0.85 |
| Sensibilidad melanoma (screening) | ≥ 0.80 |
| ECE (calibrado) | ≤ 0.10 |
| Brecha de fairness (Fitzpatrick) | ≤ 0.15 (objetivo, requiere datos) |

---

## D-07 — Multi-GPU + robustez operativa

**Decisión:** `DataParallel` cuando hay 2 GPUs (guardando pesos con llaves planas), carga de imagen
robusta, modo `SMOKE` de validación rápida, checkpoint parcial del SSL y ETA por época.

**Por qué:** hacer el run de Kaggle **reproducible y a prueba de caídas de sesión** (ver
[05 — Pipeline](05-pipeline-objetivo2.md)).
