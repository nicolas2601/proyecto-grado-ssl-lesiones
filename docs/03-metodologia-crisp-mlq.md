# 03 — Metodología: CRISP-ML(Q)

## Qué es y por qué

El proyecto migró formalmente de **CRISP-DM** (Wirth & Hipp, 2000) a **CRISP-ML(Q)**
(Studer et al., 2021 — arXiv:2003.05155). Motivo: la asesora Karen Sánchez (KAUST) marcó que
CRISP-DM es *legacy* para proyectos de machine learning; CRISP-ML(Q) es su evolución específica
para ML, con **aseguramiento de calidad (Q)** en cada fase.

## Fases y cómo las mapeamos

| Fase CRISP-ML(Q) | En este proyecto |
|---|---|
| 1. Business & Data Understanding | Objetivo 1 (EDA), epidemiología Santander × datasets |
| 2. Data Preparation | Split lesión-disjunto, pesos de clase, augmentations, pool SSL |
| 3. Modeling | Objetivo 2: SSL domain-adaptive (SimSiam) + fine-tune + ablación |
| 4. Evaluation | KPIs clínicos, calibración, fiabilidad, validación externa |
| 5. Deployment | Demo de inferencia (`predict_path`) → futura API |
| 6. Monitoring & Maintenance | Trabajo futuro (drift, re-entrenamiento con datos UNAB) |

## El "(Q)" — aseguramiento de calidad

Cada fase declara sus **riesgos y controles**. Ejemplos aplicados en el pipeline:

- **Riesgo de leakage** → control: split por `lesion_id` + `assert` de cero solapamiento.
- **Riesgo de métrica engañosa** (accuracy con desbalance) → control: macro-F1 + recall por clase.
- **Riesgo de sobreconfianza** → control: calibración ECE + temperature scaling + reliability diagram.
- **Riesgo de shortcut learning** → control: Grad-CAM sobre melanomas.
- **Riesgo de sesgo por fototipo** → control: declararlo + plan de validación en piel oscura (BOSQUE).
- **Riesgo de contribución decorativa** (SSL que no aporta) → control: ablación ImageNet vs SSL + linear probe.

## Principio rector

Honestidad científica > número bonito. La calidad (Q) se demuestra reportando límites, intervalos
de confianza y ablaciones honestas, no maximizando una sola cifra.
