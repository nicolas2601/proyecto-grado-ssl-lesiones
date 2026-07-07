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
- **Objetivo 2 (algoritmo SSL):** 🟡 En curso. Pipeline `v2` corriendo en Kaggle (T4 x2).
  Corrida de referencia de ~6 h en marcha (SimSiam domain-adaptive + fine-tune + ablación + fiabilidad).
- **Siguiente hito:** demo de inferencia + envío al director → gestionar acceso a GPUs de la UNAB
  para el entrenamiento definitivo.

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
| [08 — Estado actual y próximos pasos](docs/08-estado-actual-y-next-steps.md) | Run de 6 h, roadmap |
| [09 — Referencias](docs/09-referencias.md) | Papers, incl. evaluación de MambaMIM |
| [10 — Glosario ELI5](docs/10-glosario-eli5.md) | Conceptos explicados sencillo |

## Estructura

```
proyecto-grado-ssl-lesiones/
├── README.md
├── docs/                 # Toda la documentación
└── notebooks/
    └── proyecto_ssl_lesiones_v2.ipynb   # Pipeline Objetivo 2 (Kaggle)
```

## Regla de oro del proyecto

**Honestidad científica > número bonito.** Se reportan los límites, no se esconden.
Es una herramienta de *apoyo*, nunca de diagnóstico final.
