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
- **Bancos de baselines congelados (tarea de la dirección):** ✅ **CORRIDOS en Kaggle.**
  **8 modelos** con backbone congelado sobre **HAM10000** (7 clases), **ISIC 2019**
  (8 clases, la única tarea comparable con el ancla de Gessert) e **ISIC 2020**
  (binario, antecedente: ningún backbone congelado sirve para tamizaje).
  **Lidera DINOv2 ViT-B/14 (SSL) en los dos**: 0.618 en HAM10000 y 0.620 en ISIC 2019,
  por encima de su par exacto de tamaño ViT-B/16 supervisado (0.567 y 0.579).
  SimSiam-ImageNet último en ambos → el SSL genérico débil no transfiere a piel, lo que
  motiva el v6. Tablas completas abajo y en
  [`notebooks/baselines/`](notebooks/baselines/README.md); JSONs en `results/baselines/`,
  `results/baselines_isic2019/` y `results/baselines_isic2020/`.
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
> Lectura de los resultados y la leccion de medicion:
> [19 — Resultados de los bancos congelados](docs/19-resultados-bancos-congelados.md).

> Banco de fine-tuning de los baselines en el cluster (un solo comando para la
> asesora): [cluster/finetune/README.md](cluster/finetune/README.md).

## Resultados de los bancos de baselines congelados (Kaggle, 2026-09-03)

Corridas reales (`SMOKE=False`, seed 42, backbone congelado, resolución 224) sobre
los tres datasets: 24 corridas en total. Las tablas se generan con
`python scripts/build_results_tables.py --bank <banco>`; no se escriben a mano.

### HAM10000 (7 clases, 8.001 train / 2.014 val, split por `lesion_id`)

Ancla de literatura: Gessert, ganador del ISIC 2019 = **0.636** de balanced accuracy.
JSONs en `results/baselines/`.

#### Métricas completas (mejor probe de cada modelo)

| # | Modelo | Tipo | dims | acc | bal_acc | F1 macro | Prec macro | Recall macro | AUC | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | DINOv2 ViT-B/14 | SSL | 768 | **0.747** | **0.618** | **0.595** | **0.598** | **0.618** | 0.907 | 0.162 |
| 2 | AlexNet | Supervisado | 256 | 0.666 | 0.589 | 0.493 | 0.450 | 0.589 | 0.896 | 0.079 |
| 3 | EfficientNet-B0 | Supervisado | 1280 | 0.734 | 0.586 | 0.546 | 0.519 | 0.586 | 0.917 | 0.125 |
| 4 | DINOv2 ViT-S/14 | SSL | 384 | 0.711 | 0.575 | 0.547 | 0.543 | 0.575 | 0.905 | 0.149 |
| 5 | ResNet-50 | Supervisado | 2048 | 0.731 | 0.572 | 0.528 | 0.498 | 0.572 | 0.908 | 0.097 |
| 6 | ViT-B/16 | Supervisado | 768 | 0.723 | 0.567 | 0.540 | 0.530 | 0.567 | **0.920** | 0.168 |
| 7 | VGG19 | Supervisado | 512 | 0.618 | 0.515 | 0.415 | 0.377 | 0.515 | 0.870 | **0.071** |
| 8 | SimSiam R50 | SSL | 2048 | 0.490 | 0.494 | 0.317 | 0.335 | 0.494 | 0.829 | 0.247 |

#### Intervalos de confianza (bootstrap 95%)

| Modelo | bal_acc | IC95 bal_acc | F1 macro | IC95 F1 macro | Separado del líder |
|---|---|---|---|---|---|
| DINOv2 ViT-B/14 | 0.618 | [0.576, 0.660] | 0.595 | [0.551, 0.635] | - |
| AlexNet | 0.589 | [0.548, 0.631] | 0.493 | [0.458, 0.521] | no |
| EfficientNet-B0 | 0.586 | [0.547, 0.623] | 0.546 | [0.509, 0.579] | no |
| DINOv2 ViT-S/14 | 0.575 | [0.531, 0.617] | 0.547 | [0.504, 0.584] | no |
| ResNet-50 | 0.572 | [0.533, 0.615] | 0.528 | [0.492, 0.563] | no |
| ViT-B/16 | 0.567 | [0.523, 0.612] | 0.540 | [0.497, 0.579] | no |
| VGG19 | 0.515 | [0.472, 0.555] | 0.415 | [0.386, 0.440] | sí |
| SimSiam R50 | 0.494 | [0.453, 0.539] | 0.317 | [0.294, 0.341] | sí |

#### Balanced accuracy por probe

| Modelo | kNN | LogReg | Prototipos |
|---|---|---|---|
| DINOv2 ViT-B/14 | 0.311 | 0.618 | 0.458 |
| AlexNet | 0.402 | 0.589 | 0.482 |
| EfficientNet-B0 | 0.328 | 0.586 | 0.475 |
| DINOv2 ViT-S/14 | 0.318 | 0.575 | 0.499 |
| ResNet-50 | 0.328 | 0.572 | 0.437 |
| ViT-B/16 | 0.307 | 0.567 | 0.453 |
| VGG19 | 0.265 | 0.515 | 0.405 |
| SimSiam R50 | 0.338 | 0.494 | 0.491 |

#### Recall por clase (mejor probe)

| Modelo | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| DINOv2 ViT-B/14 | 0.574 | 0.770 | 0.615 | 0.273 | 0.545 | 0.814 | 0.735 |
| AlexNet | 0.588 | 0.710 | 0.556 | 0.273 | 0.590 | 0.701 | 0.706 |
| EfficientNet-B0 | 0.603 | 0.760 | 0.580 | 0.136 | 0.505 | 0.810 | 0.706 |
| DINOv2 ViT-S/14 | 0.515 | 0.640 | 0.580 | 0.273 | 0.532 | 0.782 | 0.706 |
| ResNet-50 | 0.412 | 0.670 | 0.571 | 0.182 | 0.631 | 0.801 | 0.735 |
| ViT-B/16 | 0.529 | 0.590 | 0.600 | 0.318 | 0.581 | 0.795 | 0.559 |
| VGG19 | 0.471 | 0.520 | 0.478 | 0.182 | 0.514 | 0.674 | 0.765 |
| SimSiam R50 | 0.647 | 0.610 | 0.151 | 0.273 | 0.459 | 0.525 | 0.794 |

#### Precisión por clase (mejor probe)

| Modelo | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| DINOv2 ViT-B/14 | 0.453 | 0.621 | 0.432 | 0.545 | 0.412 | 0.945 | 0.781 |
| AlexNet | 0.301 | 0.518 | 0.409 | 0.077 | 0.388 | 0.955 | 0.500 |
| EfficientNet-B0 | 0.451 | 0.571 | 0.476 | 0.143 | 0.380 | 0.929 | 0.686 |
| DINOv2 ViT-S/14 | 0.315 | 0.552 | 0.442 | 0.429 | 0.348 | 0.940 | 0.774 |
| ResNet-50 | 0.315 | 0.493 | 0.494 | 0.154 | 0.440 | 0.933 | 0.658 |
| ViT-B/16 | 0.400 | 0.440 | 0.504 | 0.318 | 0.389 | 0.929 | 0.731 |
| VGG19 | 0.227 | 0.311 | 0.408 | 0.053 | 0.339 | 0.934 | 0.366 |
| SimSiam R50 | 0.205 | 0.299 | 0.463 | 0.053 | 0.295 | 0.948 | 0.086 |

#### F1 por clase (mejor probe)

| Modelo | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| DINOv2 ViT-B/14 | 0.506 | 0.688 | 0.507 | 0.364 | 0.469 | 0.875 | 0.758 |
| AlexNet | 0.398 | 0.599 | 0.471 | 0.120 | 0.468 | 0.809 | 0.585 |
| EfficientNet-B0 | 0.516 | 0.652 | 0.523 | 0.140 | 0.433 | 0.865 | 0.696 |
| DINOv2 ViT-S/14 | 0.391 | 0.593 | 0.502 | 0.333 | 0.421 | 0.854 | 0.738 |
| ResNet-50 | 0.357 | 0.568 | 0.529 | 0.167 | 0.519 | 0.862 | 0.694 |
| ViT-B/16 | 0.456 | 0.504 | 0.548 | 0.318 | 0.466 | 0.856 | 0.633 |
| VGG19 | 0.306 | 0.390 | 0.440 | 0.082 | 0.409 | 0.783 | 0.495 |
| SimSiam R50 | 0.311 | 0.401 | 0.228 | 0.088 | 0.359 | 0.675 | 0.155 |

### ISIC 2019 (multiclase, 8 clases del challenge oficial, 5.067 val)

JSONs en `results/baselines_isic2019/`. 20.264 train / 5.067 val, semilla 42.

> **Dos limitaciones que viajan con esta tabla.** (1) El release no trae columna
> de paciente, asi que el split es **estratificado simple**: dos imagenes del
> mismo paciente pueden caer en train y en val, y las metricas quedan con sesgo
> optimista. (2) **HAM10000 esta contenido entero en ISIC 2019** (10.015 de
> 10.015 imagenes, el 39,5 % de este banco), asi que este banco y el de HAM10000
> **no son dos evaluaciones independientes**.

#### Métricas completas (mejor probe de cada modelo)

| # | Modelo | Tipo | dims | acc | bal_acc | F1 macro | Prec macro | Recall macro | AUC | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | DINOv2 ViT-B/14 | SSL | 768 | **0.681** | **0.620** | **0.563** | **0.528** | **0.620** | **0.920** | 0.106 |
| 2 | DINOv2 ViT-S/14 | SSL | 384 | 0.653 | 0.597 | 0.518 | 0.481 | 0.597 | 0.910 | 0.067 |
| 3 | ViT-B/16 | Supervisado | 768 | 0.654 | 0.579 | 0.528 | 0.498 | 0.579 | 0.906 | 0.116 |
| 4 | EfficientNet-B0 | Supervisado | 1280 | 0.661 | 0.575 | 0.525 | 0.495 | 0.575 | 0.911 | 0.106 |
| 5 | ResNet-50 | Supervisado | 2048 | 0.638 | 0.569 | 0.499 | 0.463 | 0.569 | 0.906 | 0.096 |
| 6 | AlexNet | Supervisado | 256 | 0.574 | 0.533 | 0.421 | 0.398 | 0.533 | 0.873 | **0.050** |
| 7 | VGG19 | Supervisado | 512 | 0.558 | 0.497 | 0.391 | 0.373 | 0.497 | 0.860 | 0.054 |
| 8 | SimSiam R50 | SSL | 2048 | 0.478 | 0.494 | 0.329 | 0.336 | 0.494 | 0.839 | 0.221 |

#### Intervalos de confianza (bootstrap 95%)

| Modelo | bal_acc | IC95 bal_acc | F1 macro | IC95 F1 macro | Separado del líder |
|---|---|---|---|---|---|
| DINOv2 ViT-B/14 | 0.620 | [0.591, 0.649] | 0.563 | [0.538, 0.587] | - |
| DINOv2 ViT-S/14 | 0.597 | [0.568, 0.626] | 0.518 | [0.494, 0.540] | no |
| ViT-B/16 | 0.579 | [0.551, 0.606] | 0.528 | [0.502, 0.550] | no |
| EfficientNet-B0 | 0.575 | [0.546, 0.603] | 0.525 | [0.500, 0.548] | no |
| ResNet-50 | 0.569 | [0.539, 0.598] | 0.499 | [0.474, 0.523] | no |
| AlexNet | 0.533 | [0.503, 0.562] | 0.421 | [0.401, 0.440] | sí |
| VGG19 | 0.497 | [0.468, 0.525] | 0.391 | [0.373, 0.407] | sí |
| SimSiam R50 | 0.494 | [0.466, 0.522] | 0.329 | [0.313, 0.343] | sí |

#### Balanced accuracy por probe

| Modelo | kNN | LogReg | Prototipos |
|---|---|---|---|
| DINOv2 ViT-B/14 | 0.390 | 0.620 | 0.388 |
| DINOv2 ViT-S/14 | 0.417 | 0.597 | 0.418 |
| ViT-B/16 | 0.393 | 0.579 | 0.387 |
| EfficientNet-B0 | 0.407 | 0.575 | 0.378 |
| ResNet-50 | 0.393 | 0.569 | 0.344 |
| AlexNet | 0.436 | 0.533 | 0.394 |
| VGG19 | 0.350 | 0.497 | 0.341 |
| SimSiam R50 | 0.406 | 0.494 | 0.430 |

#### Recall por clase (mejor probe)

| Modelo | MEL | NV | BCC | AK | BKL | DF | VASC | SCC |
|---|---|---|---|---|---|---|---|---|
| DINOv2 ViT-B/14 | 0.602 | 0.757 | 0.663 | 0.511 | 0.570 | 0.596 | 0.760 | 0.500 |
| DINOv2 ViT-S/14 | 0.566 | 0.737 | 0.585 | 0.569 | 0.539 | 0.553 | 0.720 | 0.508 |
| ViT-B/16 | 0.565 | 0.731 | 0.595 | 0.563 | 0.600 | 0.404 | 0.740 | 0.437 |
| EfficientNet-B0 | 0.579 | 0.732 | 0.668 | 0.523 | 0.558 | 0.383 | 0.700 | 0.460 |
| ResNet-50 | 0.571 | 0.711 | 0.579 | 0.529 | 0.556 | 0.489 | 0.660 | 0.452 |
| AlexNet | 0.524 | 0.662 | 0.475 | 0.529 | 0.402 | 0.532 | 0.760 | 0.381 |
| VGG19 | 0.484 | 0.645 | 0.489 | 0.425 | 0.425 | 0.468 | 0.600 | 0.437 |
| SimSiam R50 | 0.484 | 0.561 | 0.305 | 0.534 | 0.232 | 0.638 | 0.760 | 0.437 |

#### Precisión por clase (mejor probe)

| Modelo | MEL | NV | BCC | AK | BKL | DF | VASC | SCC |
|---|---|---|---|---|---|---|---|---|
| DINOv2 ViT-B/14 | 0.561 | 0.902 | 0.696 | 0.303 | 0.446 | 0.400 | 0.613 | 0.306 |
| DINOv2 ViT-S/14 | 0.538 | 0.896 | 0.662 | 0.299 | 0.442 | 0.329 | 0.450 | 0.229 |
| ViT-B/16 | 0.538 | 0.887 | 0.616 | 0.341 | 0.438 | 0.288 | 0.627 | 0.249 |
| EfficientNet-B0 | 0.535 | 0.897 | 0.658 | 0.333 | 0.418 | 0.250 | 0.583 | 0.283 |
| ResNet-50 | 0.545 | 0.884 | 0.583 | 0.290 | 0.429 | 0.235 | 0.485 | 0.256 |
| AlexNet | 0.536 | 0.888 | 0.558 | 0.249 | 0.350 | 0.082 | 0.365 | 0.150 |
| VGG19 | 0.523 | 0.877 | 0.568 | 0.212 | 0.345 | 0.075 | 0.224 | 0.162 |
| SimSiam R50 | 0.468 | 0.865 | 0.501 | 0.179 | 0.324 | 0.098 | 0.069 | 0.183 |

#### F1 por clase (mejor probe)

| Modelo | MEL | NV | BCC | AK | BKL | DF | VASC | SCC |
|---|---|---|---|---|---|---|---|---|
| DINOv2 ViT-B/14 | 0.581 | 0.823 | 0.679 | 0.380 | 0.500 | 0.479 | 0.679 | 0.380 |
| DINOv2 ViT-S/14 | 0.552 | 0.809 | 0.621 | 0.392 | 0.486 | 0.413 | 0.554 | 0.316 |
| ViT-B/16 | 0.551 | 0.802 | 0.606 | 0.425 | 0.506 | 0.336 | 0.679 | 0.317 |
| EfficientNet-B0 | 0.556 | 0.806 | 0.663 | 0.407 | 0.478 | 0.303 | 0.636 | 0.350 |
| ResNet-50 | 0.558 | 0.788 | 0.581 | 0.375 | 0.485 | 0.317 | 0.559 | 0.327 |
| AlexNet | 0.530 | 0.759 | 0.513 | 0.339 | 0.374 | 0.142 | 0.494 | 0.216 |
| VGG19 | 0.503 | 0.744 | 0.525 | 0.283 | 0.381 | 0.129 | 0.326 | 0.236 |
| SimSiam R50 | 0.476 | 0.681 | 0.379 | 0.268 | 0.271 | 0.170 | 0.126 | 0.258 |

### ISIC 2020 (binario benigno/maligno, 6.660 val, split por `patient_id`)

JSONs en `results/baselines_isic2020/`. 26.466 train / 6.660 val, semilla 42, mismo protocolo y misma
resolución que los otros dos bancos. Es el único de los tres con agrupación
anti-fuga por paciente real.

> **Cómo leer esta tabla.** Es una tarea binaria con desbalance cercano a 50:1
> (unos 133 malignos en 6.660 de validación). La balanced accuracy de ~0.77 es
> cosmética: la precisión de la clase maligna va de 0.045 a 0.076 en los ocho
> modelos, es decir unos trece falsos positivos por cada acierto. **Ningún backbone
> congelado sirve para tamizaje binario**; ese es el hallazgo, no el ranking. Los
> ocho intervalos se solapan entre sí, así que el orden no es interpretable. Este
> banco se retiró del README el 2026-09-04 cuando el proyecto pasó a ISIC 2019 y
> se restituyó el 2026-09-11 como antecedente.

#### Métricas completas (mejor probe de cada modelo)

| # | Modelo | Tipo | dims | acc | bal_acc | F1 macro | Prec macro | Recall macro | AUC | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | AlexNet | Supervisado | 256 | 0.794 | **0.774** | 0.501 | 0.529 | **0.774** | 0.857 | **0.044** |
| 2 | SimSiam R50 | SSL | 2048 | 0.752 | 0.765 | 0.480 | 0.525 | 0.765 | 0.847 | 0.085 |
| 3 | ViT-B/16 | Supervisado | 768 | 0.751 | 0.761 | 0.479 | 0.524 | 0.761 | 0.843 | 0.145 |
| 4 | DINOv2 ViT-S/14 | SSL | 384 | **0.847** | 0.757 | **0.527** | **0.535** | 0.757 | **0.858** | 0.056 |
| 5 | VGG19 | Supervisado | 512 | 0.719 | 0.748 | 0.463 | 0.522 | 0.748 | 0.824 | 0.133 |
| 6 | EfficientNet-B0 | Supervisado | 1280 | 0.691 | 0.742 | 0.450 | 0.520 | 0.742 | 0.819 | 0.084 |
| 7 | ResNet-50 | Supervisado | 2048 | 0.743 | 0.740 | 0.473 | 0.522 | 0.740 | 0.811 | 0.158 |
| 8 | DINOv2 ViT-B/14 | SSL | 768 | 0.731 | 0.734 | 0.467 | 0.521 | 0.734 | 0.807 | 0.138 |

#### Intervalos de confianza (bootstrap 95%)

| Modelo | bal_acc | IC95 bal_acc | F1 macro | IC95 F1 macro | Separado del líder |
|---|---|---|---|---|---|
| AlexNet | 0.774 | [0.735, 0.813] | 0.501 | [0.489, 0.512] | - |
| SimSiam R50 | 0.765 | [0.730, 0.800] | 0.480 | [0.469, 0.491] | no |
| ViT-B/16 | 0.761 | [0.723, 0.796] | 0.479 | [0.468, 0.489] | no |
| DINOv2 ViT-S/14 | 0.757 | [0.713, 0.798] | 0.527 | [0.512, 0.539] | no |
| VGG19 | 0.748 | [0.712, 0.786] | 0.463 | [0.454, 0.472] | no |
| EfficientNet-B0 | 0.742 | [0.706, 0.778] | 0.450 | [0.441, 0.459] | no |
| ResNet-50 | 0.740 | [0.704, 0.782] | 0.473 | [0.463, 0.482] | no |
| DINOv2 ViT-B/14 | 0.734 | [0.696, 0.773] | 0.467 | [0.457, 0.476] | no |

#### Balanced accuracy por probe

| Modelo | kNN | LogReg | Prototipos |
|---|---|---|---|
| AlexNet | 0.500 | 0.774 | 0.761 |
| SimSiam R50 | 0.500 | 0.765 | 0.739 |
| ViT-B/16 | 0.500 | 0.734 | 0.761 |
| DINOv2 ViT-S/14 | 0.500 | 0.757 | 0.723 |
| VGG19 | 0.500 | 0.737 | 0.748 |
| EfficientNet-B0 | 0.500 | 0.637 | 0.742 |
| ResNet-50 | 0.500 | 0.688 | 0.740 |
| DINOv2 ViT-B/14 | 0.500 | 0.675 | 0.734 |

#### Recall por clase (mejor probe)

| Modelo | benign | malignant |
|---|---|---|
| AlexNet | 0.795 | 0.754 |
| SimSiam R50 | 0.751 | 0.779 |
| ViT-B/16 | 0.751 | 0.770 |
| DINOv2 ViT-S/14 | 0.850 | 0.664 |
| VGG19 | 0.717 | 0.779 |
| EfficientNet-B0 | 0.689 | 0.795 |
| ResNet-50 | 0.743 | 0.738 |
| DINOv2 ViT-B/14 | 0.731 | 0.738 |

#### Precisión por clase (mejor probe)

| Modelo | benign | malignant |
|---|---|---|
| AlexNet | 0.994 | 0.064 |
| SimSiam R50 | 0.995 | 0.055 |
| ViT-B/16 | 0.994 | 0.054 |
| DINOv2 ViT-S/14 | 0.993 | 0.076 |
| VGG19 | 0.994 | 0.049 |
| EfficientNet-B0 | 0.994 | 0.045 |
| ResNet-50 | 0.993 | 0.051 |
| DINOv2 ViT-B/14 | 0.993 | 0.049 |

#### F1 por clase (mejor probe)

| Modelo | benign | malignant |
|---|---|---|
| AlexNet | 0.883 | 0.118 |
| SimSiam R50 | 0.856 | 0.103 |
| ViT-B/16 | 0.855 | 0.102 |
| DINOv2 ViT-S/14 | 0.916 | 0.137 |
| VGG19 | 0.834 | 0.092 |
| EfficientNet-B0 | 0.814 | 0.086 |
| ResNet-50 | 0.850 | 0.095 |
| DINOv2 ViT-B/14 | 0.842 | 0.091 |

### Lectura

**Lo que se sostiene.**

- **Ningún backbone genérico congelado alcanza el ancla clínica.** El mejor de
  HAM10000 (DINOv2 ViT-B/14, 0.618) queda por debajo de 0.636 y su intervalo
  incluye ese valor: ni siquiera se puede afirmar la distancia con certeza.
- **Los dos DINOv2 barren el top-2 en ISIC 2019**, la única tarea comparable con la
  literatura (8 clases del challenge oficial). Ambos son SSL.
- **El SSL genérico de ImageNet no transfiere a dermatoscopia.** SimSiam R50 es
  último en los dos bancos multiclase y su intervalo no toca al del líder en ninguno,
  así que esa diferencia sí es real. Es el argumento que motiva la fase v6: pre-entrenar
  en piel.
- **ISIC 2019 separa mejor que HAM10000.** Acá tres modelos quedan estadísticamente
  separados del líder (AlexNet, VGG19, SimSiam), contra dos en HAM10000: el conjunto
  de validación es 2,5 veces más grande y los intervalos se angostan.
- **Ningún backbone congelado sirve para tamizaje en ISIC 2020.** Precisión de la
  clase maligna entre 0.045 y 0.076 en los ocho; la balanced accuracy de 0.77 viene
  del desbalance 50:1, no de discriminar. Es el banco con menos resolución: los ocho
  intervalos se solapan y AlexNet (2012, 256 dims) queda primero, señal de que ahí
  no se distingue nada.
- **El kNN colapsa en los tres bancos** (0.26–0.44 en multiclase, exactamente 0.500
  en ISIC 2020): con desbalance fuerte el vecindario lo domina la clase mayoritaria. La regresión logística con pesos de
  clase es el probe informativo, y en ISIC 2019 es el mejor probe **en los ocho
  modelos**.
- **Las clases raras son la debilidad transversal.** En ISIC 2019, `SCC` y `AK`
  quedan con precisión ≤ 0.34 en todos los modelos; `MEL`, la clínicamente crítica,
  llega a 0.602 de recall en el mejor.

**Lo que NO se puede afirmar.**

**El orden de ISIC 2020 no es un ranking.** Con ocho intervalos solapados y ~133
positivos, AlexNet primero y DINOv2 ViT-B/14 último no dicen nada de los modelos.
Se probó una hipótesis propia (que la regresión logística sobre features sin
normalizar favorecía a los backbones chicos); predecía AlexNet primero en ISIC 2019
y quedó sexto, así que la hipótesis no explica el fenómeno. Detalle en
[docs/19](docs/19-resultados-bancos-congelados.md#41-una-hipotesis-propia-que-el-dato-descarto).

**Ninguno alcanza el ancla de Gessert (0.636) en estimación puntual**: el mejor
—DINOv2 ViT-B/14 con 0.620— queda a 0.016. Pero su intervalo llega a 0.649, así que
**cubre el ancla**. La redacción defendible es que *con estos datos no se distingue
de la marca ganadora del challenge*, **no** que la iguale. Y conviene subrayar que
es con **backbone congelado**, sin fine-tuning y sin haber visto dermatoscopia nunca.

**El SSL no gana probado, aventaja.** Contra su par exacto de tamaño (86M params),
DINOv2 ViT-B/14 le saca +0.041 al ViT-B/16 supervisado (0.620 vs 0.579), pero los
intervalos todavía se tocan: el techo del supervisado es 0.606 y el piso del SSL
0.591. Es una ventaja consistente en los dos bancos, no una victoria estadística.

**Y los dos bancos no son independientes.** HAM10000 está contenido entero en
ISIC 2019, así que la coincidencia del ranking entre ambos **no es replicación en
datos nuevos**. La validación externa honesta exigiría el complemento
`ISIC 2019 \ HAM10000` (15.316 imágenes), que está pendiente de decidir.

> Cambio respecto de la versión anterior de esta tabla: `vit_b16` pasó de los pesos
> de torchvision a `augreg_in1k` de timm, el comparador entrenado solo con
> ImageNet-1k, y cayó de 0.629 a 0.567. Además se corrigió el punto de extracción de
> VGG19 (`timm` con `num_classes=0` devolvía las pre-logits de 4096 dims en lugar
> del GAP convolucional). Los otros seis modelos reprodujeron sus números al cuarto
> decimal, lo que confirma que el pipeline es determinista.
