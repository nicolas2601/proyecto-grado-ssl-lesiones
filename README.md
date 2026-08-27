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

Corrida real (SMOKE=False) sobre HAM10000, split anti-fuga por `lesion_id`, seed 42.
Mejor probe = regresión logística en los 8. Ancla: Gessert, ganador ISIC 2019 = **0.636**.

| # | Modelo | Tipo | bal_acc | F1 macro | AUC |
|---|---|---|---|---|---|
| 1 | ViT-B/16 | Supervisado | **0.629** | **0.605** | **0.933** |
| 2 | DINOv2 ViT-B/14 | SSL | 0.618 | 0.595 | 0.907 |
| 3 | AlexNet | Supervisado | 0.589 | 0.493 | 0.896 |
| 4 | EfficientNet-B0 | Supervisado | 0.586 | 0.546 | 0.917 |
| 5 | DINOv2 ViT-S/14 | SSL | 0.575 | 0.547 | 0.905 |
| 6 | ResNet-50 | Supervisado | 0.572 | 0.528 | 0.908 |
| 7 | VGG19 | Supervisado | 0.515 | 0.415 | 0.870 |
| 8 | SimSiam R50 | SSL | 0.494 | 0.317 | 0.829 |

Lectura: ningún backbone genérico congelado alcanza el estado del arte. **A igual
tamaño (86M params), el ViT supervisado aún gana al SSL genérico por solo 0.011 de
balanced accuracy** — DINOv2 ViT-B/14 es 2º del banco y supera a todas las CNN
supervisadas. SimSiam-ImageNet, último, muestra que el SSL genérico débil no
transfiere a dermatoscopia. **La pregunta del proyecto queda afilada: ¿puede el SSL
específico de piel (v6) cerrar esos 0.011 y superar al supervisado sin etiquetas?**
