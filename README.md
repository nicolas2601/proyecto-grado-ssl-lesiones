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
- **Siguiente hito:** validación externa (ISIC 2019 multiclase, sobre el modelo exportado) + 3.ª
  semilla → envío al director → acceso a GPUs de la UNAB.

> ⚠️ **Resultados preliminares.** Obtenidos con los **recursos actuales** (GPU T4 de estudiante,
> dataset acotado) — no el modelo final. El entrenamiento definitivo (más cómputo, DINOv2/ViT,
> validación externa) se hará con las GPUs de la UNAB. Aun así, esta corrida **valida la hipótesis**.

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
| [Informe](INFORME.md) | Resumen ejecutivo del avance |

## Estructura

```
proyecto-grado-ssl-lesiones/
├── README.md
├── INFORME.md                   # Resumen ejecutivo del avance
├── docs/                        # Documentación 01..13
├── data/
│   └── RESULTS_SUMMARY.json     # Datos crudos de la corrida
├── assets/figuras/              # 8 figuras explicativas (PNG): ablación, linear probe, KPIs,
│                                #   abstención, recall, calibración, Grad-CAM, reliability diagram
├── results/
│   └── generar_figuras.py       # Script reproducible de las figuras
└── notebooks/
    └── proyecto_ssl_lesiones_v4.ipynb   # Pipeline Objetivo 2 definitivo (Kaggle)
```

> El modelo entrenado (`best_resnet50_ssl.pth`, ~90 MB) no va en el repo (`.gitignore` no versiona
> pesos): está publicado como asset en el
> [**Release `modelo-v4-definitiva`**](https://github.com/nicolas2601/proyecto-grado-ssl-lesiones/releases/tag/modelo-v4-definitiva).

## Regla de oro del proyecto

**Honestidad científica > número bonito.** Se reportan los límites, no se esconden.
Es una herramienta de *apoyo*, nunca de diagnóstico final.
