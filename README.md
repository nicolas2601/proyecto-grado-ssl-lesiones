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
- **Objetivo 2 (algoritmo SSL):** ✅ **Corrida de referencia ejecutada** (Kaggle T4 x2). Modelo
  entrenado, demo de inferencia OK, **4/4 KPIs clínicos cumplidos**. Ver
  [**Informe**](INFORME.md) y [11 — Resultados finales](docs/11-resultados-finales.md).
- **Siguiente hito:** validación externa cross-dataset + envío al director → gestionar acceso a GPUs
  de la UNAB para el entrenamiento definitivo.

> ⚠️ **Resultados preliminares.** Son pruebas y evidencia con los **recursos actuales** (GPU T4 de
> estudiante, dataset pequeño) — el **techo alcanzable así**, no el modelo final. El entrenamiento
> definitivo (más cómputo, más datos, validación externa) se hará con las GPUs de la UNAB.

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
| [Informe](INFORME.md) | Resumen ejecutivo del avance |

## Estructura

```
proyecto-grado-ssl-lesiones/
├── README.md
├── INFORME.md                   # Resumen ejecutivo del avance
├── docs/                        # Documentación 01..13
├── data/
│   └── RESULTS_SUMMARY.json     # Datos crudos de la corrida
├── assets/figuras/              # 6 figuras explicativas (PNG)
├── results/
│   └── generar_figuras.py       # Script reproducible de las figuras
└── notebooks/
    └── proyecto_ssl_lesiones_v2.ipynb   # Pipeline Objetivo 2 (Kaggle)
```

> El modelo entrenado (`best_resnet50_ssl.pth`, ~90 MB) no va en el repo (límite de 100 MB de
> GitHub): está publicado como asset en el
> [**Release `modelo-v1-preliminar`**](https://github.com/nicolas2601/proyecto-grado-ssl-lesiones/releases/tag/modelo-v1-preliminar).

## Regla de oro del proyecto

**Honestidad científica > número bonito.** Se reportan los límites, no se esconden.
Es una herramienta de *apoyo*, nunca de diagnóstico final.
