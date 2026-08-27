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
- **Banco de baselines (tarea de la dirección):** ✅ 7 modelos (AlexNet, VGG19, EfficientNet-B0,
  ResNet-50, ViT-B/16, SimSiam, DINOv2) con **un notebook de test autocontenido por modelo**,
  listos para Kaggle: [`notebooks/baselines/`](notebooks/baselines/README.md). Fichas y protocolo
  en [17 — Modelos baseline](docs/17-modelos-baseline.docx) y
  [18 — Estructura visual y tests](docs/18-estructura-visual-y-tests-baseline.md).
- **Siguiente hito:** correr los 7 tests de baseline en Kaggle (`SMOKE` → real) → fixes DDP →
  pre-entrenamiento v6 en el clúster de la asesora.

> ⚠️ **Resultados preliminares.** Obtenidos con los **recursos actuales** (GPU T4 de estudiante,
> dataset acotado) — no el modelo final. El entrenamiento definitivo (más cómputo, DINOv2/ViT,
> validación externa) se hará con las GPUs de la UNAB. Aun así, esta corrida **valida la hipótesis**.

## Pipeline visual

**Banco de baselines** — 7 modelos evaluados con el mismo protocolo: backbone **congelado**
(cero fine-tuning), 3 probes (kNN coseno, regresión logística balanceada, prototipos) sobre
HAM10000 con split anti-fuga por `lesion_id`:

![Pipeline del banco de baselines](assets/figuras/fig7_pipeline_banco.png)

**Baseline vs. propuesta** — los supervisados son solo comparadores; la propuesta es
**SSL puro**: las etiquetas entran únicamente en la evaluación, nunca al extractor:

![Baseline vs propuesta SSL](assets/figuras/fig8_baseline_vs_proposal.png)

**Arquitecturas v6** (pre-entrenamiento propio, pendiente de clúster):

| DINOv2 continual (brazo principal) | SimSiam desde cero (brazo purista) |
|---|---|
| ![v6 DINOv2 continual](assets/figuras/fig9_v6_dino_continual.png) | ![v6 SimSiam](assets/figuras/fig10_v6_simsiam.png) |

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
