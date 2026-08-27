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
- **Banco de baselines (tarea de la dirección):** ✅ **CORRIDO en Kaggle (2026-08-27).** 7 modelos
  evaluados con backbone congelado + un 8º test generado (DINOv2 ViT-B/14, par justo de tamaño).
  **Mejor baseline: ViT-B/16 (bal_acc 0.629)**, a 0.007 del ancla de Gessert (0.636); SimSiam-ImageNet
  último → el SSL genérico no transfiere a piel, lo que motiva el v6. Tabla completa abajo y en
  [`notebooks/baselines/`](notebooks/baselines/README.md); JSONs en `results/baselines/`.
- **Siguiente hito:** correr `dinov2_vitb14` en Kaggle → fixes DDP →
  pre-entrenamiento v6 en el clúster de la asesora.

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
   7 baselines con idéntico split, probes y métricas: **la diferencia es atribuible solo
   al backbone**.

*Glosario: SSL = aprendizaje auto-supervisado (sin etiquetas) · EMA = promedio móvil
exponencial · probe = clasificador ligero sobre features congeladas · ECE = error de
calibración esperado.*

> Figuras técnicas de detalle (pipeline del banco de baselines, protocolo de test por
> modelo y arquitecturas v6 por brazo) en
> [18 — Estructura visual y tests](docs/18-estructura-visual-y-tests-baseline.md).

## Resultados del banco de baselines (Kaggle, 2026-08-27)

Corrida real (SMOKE=False) sobre HAM10000, split anti-fuga por `lesion_id`, seed 42.
Mejor probe = regresión logística en los 7. Ancla: Gessert, ganador ISIC 2019 = **0.636**.

| # | Modelo | Tipo | bal_acc | F1 macro | AUC |
|---|---|---|---|---|---|
| 1 | ViT-B/16 | Supervisado | **0.629** | **0.605** | **0.933** |
| 2 | AlexNet | Supervisado | 0.589 | 0.493 | 0.896 |
| 3 | EfficientNet-B0 | Supervisado | 0.586 | 0.546 | 0.917 |
| 4 | DINOv2 ViT-S/14 | SSL | 0.575 | 0.547 | 0.905 |
| 5 | ResNet-50 | Supervisado | 0.572 | 0.528 | 0.908 |
| 6 | VGG19 | Supervisado | 0.515 | 0.415 | 0.870 |
| 7 | SimSiam R50 | SSL | 0.494 | 0.317 | 0.829 |

Lectura: ningún backbone genérico congelado alcanza el estado del arte. DINOv2 ViT-S/14
iguala en F1 a los mejores supervisados siendo 4× más pequeño (21M params, 384 dims);
la comparación justa de tamaño (DINOv2 **ViT-B/14** vs ViT-B/16) está generada en
`notebooks/baselines/dinov2_vitb14/` y pendiente de correr. SimSiam-ImageNet, último del
banco, muestra que el SSL genérico no transfiere a dermatoscopia: **la pregunta del
proyecto es si el SSL específico de piel (v6) cierra la brecha sin etiquetas.**

## Índice de documentación

| Doc | Contenido |
|---|---|
| [01 — Contexto y objetivos](docs/01-contexto-y-objetivos.md) | Problema clínico, objetivos, alcance |
| [02 — Datasets](docs/02-datasets.md) | HAM10000, BCN20000, CO2Wounds-V2, ISIC2019 |
| [03 — Metodología CRISP-ML(Q)](docs/03-metodologia-crisp-mlq.md) | Marco metodológico y por qué |
| [04 — Decisiones técnicas](docs/04-decisiones-tecnicas.md) | Núcleo SSL, SimSiam vs DINO, backbone |
| [05 — Pipeline Objetivo 2](docs/05-pipeline-objetivo2.md) | El notebook que corre, paso a paso |
| [06 — Problemáticas abiertas](docs/06-problematicas.md) | Contradicción linear probe, colapso, leakage, etc. |
| [07 — Fiabilidad y sesgo](docs/07-fiabilidad-y-sesgo.md) | KPIs, calibración, fairness, abstención |
| [08 — Estado actual y próximos pasos](docs/08-estado-actual-y-next-steps.md) | Resultados, roadmap |
| [09 — Referencias](docs/09-referencias.md) | Papers, incl. evaluación de MambaMIM |
| [10 — Glosario ELI5](docs/10-glosario-eli5.md) | Conceptos explicados sencillo |
| [11 — Resultados finales](docs/11-resultados-finales.md) | **Informe técnico completo + figuras** |
| [12 — Model card](docs/12-model-card.md) | Ficha del modelo + cómo servirlo |
| [13 — ¿Por qué estas decisiones?](docs/13-por-que-estas-decisiones.md) | **Por qué ResNet-50, SimSiam, etc.** |
| [14 — Mejoras del pipeline](docs/14-mejoras-pipeline.md) | **LP-FT + LLRD + más datos → que el SSL gane** |
| [15 — Auditoría SSL puro y plan v5](docs/15-auditoria-ssl-puro-y-plan-v5.md) | **Por qué v4 no es SSL puro, los cambios de v5, hardware, novedad** |
| [16 — Plan v6 pre-entrenamiento GPU](docs/16-plan-v6-pretraining-gpu.md) | DINOv2 continual + SimSiam desde cero, MLflow, presupuesto GPU |
| [17 — Modelos baseline](docs/17-modelos-baseline.docx) | Fichas de los 7 modelos del banco de comparación |
| [18 — Estructura visual y tests baseline](docs/18-estructura-visual-y-tests-baseline.md) | **Figuras del pipeline + protocolo de los tests por modelo** |
| [Informe](INFORME.md) | Resumen ejecutivo del avance |

## Estructura

```
proyecto-grado-ssl-lesiones/
├── README.md
├── INFORME.md                   # Resumen ejecutivo del avance
├── docs/                        # Documentación 01..13
├── data/
│   └── RESULTS_SUMMARY.json     # Datos crudos de la corrida
├── assets/figuras/              # 12 figuras (PNG 300 DPI): ablación, linear probe, KPIs, abstención,
│                                #   recall, calibración, Grad-CAM, reliability, pipeline del banco
│                                #   de baselines, baseline vs propuesta, arquitecturas v6
├── results/
│   └── generar_figuras.py       # Script reproducible de las figuras
├── src/dermssl/                 # Núcleo probado del pipeline v5 (solo numpy + stdlib)
├── tests/                       # 101 pruebas: sin GPU, sin internet, < 1 s
├── scripts/
│   ├── notebook_cells_v5.py     # Celdas que dependen de torch
│   └── build_notebook_v5.py     # Genera el notebook v5 (compila cada celda)
└── notebooks/
    ├── proyecto_ssl_lesiones_v4.ipynb   # Pipeline Objetivo 2 (SSL domain-adaptive + fine-tune)
    └── proyecto_ssl_lesiones_v5.ipynb   # Evaluación SSL pura, backbone congelado (GENERADO)
```

> `proyecto_ssl_lesiones_v5.ipynb` es un **artefacto generado**: no se edita a mano. Se cambia el
> módulo en `src/dermssl/`, se corre `pytest`, y se regenera con
> `python scripts/build_notebook_v5.py`. Así el código que corre en Kaggle es exactamente el que
> está cubierto por las pruebas.

> El modelo entrenado (`best_resnet50_ssl.pth`, ~90 MB) no va en el repo (`.gitignore` no versiona
> pesos): está publicado como asset en el
> [**Release `modelo-v4-definitiva`**](https://github.com/nicolas2601/proyecto-grado-ssl-lesiones/releases/tag/modelo-v4-definitiva).

## Regla de oro del proyecto

**Honestidad científica > número bonito.** Se reportan los límites, no se esconden.
Es una herramienta de *apoyo*, nunca de diagnóstico final.
