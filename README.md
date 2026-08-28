# Proyecto de Grado UNAB — Clasificación de Lesiones Cutáneas con Aprendizaje Autosupervisado (SSL)

> Repositorio privado de trabajo. Documentación, investigación, decisiones técnicas, el pipeline
> que estamos corriendo y las problemáticas abiertas del proyecto.

## Resumen en una línea

Sistema de **apoyo** al tamizaje de lesiones cutáneas basado en **aprendizaje autosupervisado
(self-supervised learning)** sobre imágenes dermatoscópicas, con foco en fiabilidad clínica,
control de sesgo y honestidad científica. Contexto: epidemiología de Santander (Colombia).

## Equipo

| Rol | Persona |
|---|---|
| Autor | Nicolás Santiago Moreno Monroy |
| autora | María Paula Saavedra |
| Director | Andrés Jerez |
| Asesora | Karen Sánchez (KAUST) |
| Institución | Universidad Autónoma de Bucaramanga (UNAB) — Ing. de Sistemas |

## Estado actual (rápido)

- **Objetivo 1 (EDA):** ✅ Completado (notebook + 13 figuras 300 DPI + Tabla I).
- **Objetivo 2 (algoritmo SSL):** ✅ **Corrida definitiva v4 ejecutada** (Kaggle T4 x2). **El SSL
  domain-adaptive SUPERA a ImageNet** en macro-F1 (0.6907 ± 0.006 vs 0.6614 ± 0.004, Δ +0.0294, fuera
  del ruido) y en AUC → **valida la hipótesis del anteproyecto**. Modelo SSL entregable con **4/4 KPIs
  clínicos cumplidos**. Ver [**Informe**](INFORME.md) y [11 — Resultados finales](docs/11-resultados-finales.md).
- **Objetivo 2 · v5 (SSL puro):** ✅ **corrido en Kaggle — el SSL gana 6/6.** Auditoría del v4:
  su "SSL" arrancaba de pesos ImageNet supervisados, así que **no era auto-supervisado puro**.
  v5 evalúa con el **backbone congelado** (DINOv2, cero etiquetas en el extractor) + probes
  kNN/prototipos/linear: las features de DINOv2 superan a las supervisadas de ImageNet en las
  6 comparaciones. Ver [15 — Auditoría SSL puro y plan v5](docs/15-auditoria-ssl-puro-y-plan-v5.md).
- **Objetivo 2 · v6 (pre-entrenamiento propio en GPU):** 🔧 **construido, pendiente de cluster.**
  Dos brazos 100 % sin etiquetas: DINOv2 continual (principal) + SimSiam desde cero (purista),
  a igual presupuesto de GPU-horas, con MLflow. Ver [16 — Plan v6](docs/16-plan-v6-pretraining-gpu.md).
  ⚠️ Fixes DDP pendientes antes de correr multi-GPU (documentados en Engram #1328).
- **Banco de baselines (tarea de la dirección):** ✅ **CORRIDO en Kaggle (2026-08-27).** **8 modelos** evaluados
  con backbone congelado (incluido DINOv2 ViT-B/14, par justo de tamaño vs ViT-B/16).
  **Mejor baseline: ViT-B/16 (bal_acc 0.629)**; DINOv2 ViT-B/14 (SSL, igual tamaño) 2º con 0.618;
  SimSiam-ImageNet último → el SSL genérico débil no transfiere a piel, lo que motiva el v6.
  Tabla completa abajo y en
  [`notebooks/baselines/`](notebooks/baselines/README.md); JSONs en `results/baselines/`.
- **Siguiente hito:** fixes DDP → pre-entrenamiento v6 en el clúster de la asesora.

> ⚠️ **Resultados preliminares.** Obtenidos con los **recursos actuales** (GPU T4 de estudiante,
> dataset acotado) — no el modelo final. El entrenamiento definitivo (más cómputo, DINOv2/ViT,
> validación externa) se hará con las GPUs de la UNAB. Aun así, esta corrida **valida la hipótesis**.

## Arquitectura del sistema

![Arquitectura completa del sistema](assets/figuras/fig11_infografia_arquitectura.png)

**Cómo leer la figura:**

1. **Datos** — el modelo aprende de ~48-70k imágenes de piel **sin diagnóstico**
   (HAM10000 train + ISIC 2020) tras una etapa de transformación (skin tone Fitzpatrick,
   constancia de color, crops por lesión, augmentation). Las 10 015 imágenes etiquetadas de
   HAM10000 quedan reservadas **únicamente para el examen final**, con split anti-fuga por
   `lesion_id` (cero lesiones compartidas entre train y val).
2. **Pre-entrenamiento SSL (v6)** — dos estrategias compiten con el mismo presupuesto de
   GPU: **DINO continual** (refinar el DINOv2 público con estudiante/profesor EMA y
   multi-crop) vs. **SimSiam desde cero** (torres siamesas ResNet-50 con stop-gradient y
   detector de colapso). Ninguna ve una sola etiqueta; tracking con MLflow.
3. **Evaluación (protocolo v5)** — el backbone se **congela** (`requires_grad=False` +
   `model.eval()` + `assert trainable == 0`) y se examina con 3 clasificadores ligeros
   (kNN coseno, regresión logística, prototipos) y métricas completas, contra los mismos
   8 baselines con idéntico split, probes y métricas: **la diferencia es atribuible solo
   al backbone**.

*Glosario: SSL = aprendizaje auto-supervisado (sin etiquetas) · EMA = promedio móvil
exponencial · probe = clasificador ligero sobre features congeladas · ECE = error de
calibración esperado.*

> Figuras técnicas de detalle (pipeline del banco de baselines, protocolo de test por
> modelo y arquitecturas v6 por brazo) en
> [18 — Estructura visual y tests](docs/18-estructura-visual-y-tests-baseline.md).

## Resultados del banco de baselines (Kaggle, 2026-08-27)

Corrida real (SMOKE=False) sobre HAM10000 (8.001 train / 2.014 val, resolución 224,
split anti-fuga `StratifiedGroupKFold` por `lesion_id` con 0 lesiones compartidas),
seed 42. Mejor probe = regresión logística en los 8. Ancla: Gessert, ganador
ISIC 2019 = **0.636**. JSONs completos en `results/baselines/`.

### Métricas completas (mejor probe: regresión logística)

| # | Modelo | Tipo | dims | acc | bal_acc | F1 macro | Prec macro | Recall macro | AUC | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ViT-B/16 | Supervisado | 768 | 0.761 | **0.629** | **0.605** | 0.601 | 0.629 | **0.933** | 0.115 |
| 2 | DINOv2 ViT-B/14 | SSL | 768 | 0.747 | 0.618 | 0.595 | 0.598 | 0.618 | 0.907 | 0.162 |
| 3 | AlexNet | Supervisado | 256 | 0.666 | 0.589 | 0.493 | 0.450 | 0.589 | 0.896 | 0.079 |
| 4 | EfficientNet-B0 | Supervisado | 1280 | 0.734 | 0.586 | 0.546 | 0.519 | 0.586 | 0.917 | 0.125 |
| 5 | DINOv2 ViT-S/14 | SSL | 384 | 0.711 | 0.575 | 0.547 | 0.543 | 0.575 | 0.905 | 0.149 |
| 6 | ResNet-50 | Supervisado | 2048 | 0.731 | 0.572 | 0.528 | 0.498 | 0.572 | 0.908 | 0.097 |
| 7 | VGG19 | Supervisado | 512 | 0.618 | 0.515 | 0.415 | 0.377 | 0.515 | 0.870 | 0.071 |
| 8 | SimSiam R50 | SSL | 2048 | 0.490 | 0.494 | 0.317 | 0.335 | 0.494 | 0.829 | 0.247 |

### Balanced accuracy por probe (los tres evaluadores congelados)

| Modelo | kNN | LogReg | Prototipos |
|---|---|---|---|
| ViT-B/16 | 0.312 | 0.629 | 0.514 |
| DINOv2 ViT-B/14 | 0.311 | 0.618 | 0.458 |
| AlexNet | 0.402 | 0.589 | 0.482 |
| EfficientNet-B0 | 0.328 | 0.586 | 0.475 |
| DINOv2 ViT-S/14 | 0.318 | 0.575 | 0.499 |
| ResNet-50 | 0.328 | 0.572 | 0.437 |
| VGG19 | 0.265 | 0.515 | 0.405 |
| SimSiam R50 | 0.338 | 0.494 | 0.491 |

### Recall por clase (regresión logística)

| Modelo | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| ViT-B/16 | 0.618 | 0.700 | 0.629 | 0.318 | 0.608 | 0.826 | 0.706 |
| DINOv2 ViT-B/14 | 0.574 | 0.770 | 0.615 | 0.273 | 0.545 | 0.814 | 0.735 |
| AlexNet | 0.588 | 0.710 | 0.556 | 0.273 | 0.590 | 0.701 | 0.706 |
| EfficientNet-B0 | 0.603 | 0.760 | 0.580 | 0.136 | 0.505 | 0.810 | 0.706 |
| DINOv2 ViT-S/14 | 0.515 | 0.640 | 0.580 | 0.273 | 0.532 | 0.782 | 0.706 |
| ResNet-50 | 0.412 | 0.670 | 0.571 | 0.182 | 0.631 | 0.801 | 0.735 |
| VGG19 | 0.471 | 0.520 | 0.478 | 0.182 | 0.514 | 0.674 | 0.765 |
| SimSiam R50 | 0.647 | 0.610 | 0.151 | 0.273 | 0.459 | 0.525 | 0.794 |

Lectura: ningún backbone genérico congelado alcanza el estado del arte. **A igual
tamaño (86M params), el ViT supervisado aún gana al SSL genérico por solo 0.011 de
balanced accuracy** — DINOv2 ViT-B/14 es 2º del banco y supera a todas las CNN
supervisadas. SimSiam-ImageNet, último, muestra que el SSL genérico débil no
transfiere a dermatoscopia. **La pregunta del proyecto queda afilada: ¿puede el SSL
específico de piel (v6) cerrar esos 0.011 y superar al supervisado sin etiquetas?**

Lecturas adicionales de las tablas completas:

- **kNN colapsa en todos los modelos** (bal_acc 0.26–0.40): con desbalance 58:1
  el vecindario lo domina `nv`. La regresión logística con pesos de clase es el
  probe informativo; los prototipos quedan en el medio.
- **`df` es la clase débil transversal** (115 imágenes): recall ≤ 0.32 en todos.
  `mel` (melanoma, la clase clínicamente crítica) queda entre 0.46 y 0.63 —
  ResNet-50 curiosamente es el mejor en melanoma (0.631) pese a ser 6º global.
- **Calibración (ECE)**: las CNN supervisadas están mejor calibradas
  (VGG19 0.071, AlexNet 0.079) que los ViT (0.115–0.162); SimSiam R50 es el
  peor también en calibración (0.247). Relevante para la lectura clínica: un
  modelo mal calibrado sobreestima su confianza.
