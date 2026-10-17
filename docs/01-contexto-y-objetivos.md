# 01 — Contexto y objetivos

## Problema clínico

El cáncer de piel y las lesiones cutáneas tienen incidencia relevante en Santander / Bucaramanga.
El diagnóstico temprano de melanoma (`mel`) es crítico: la sensibilidad (no dejar pasar un
melanoma) importa más que la exactitud global. En zonas con acceso limitado a dermatología
especializada, una herramienta de **apoyo al tamizaje** puede ayudar a priorizar quién debe ser
derivado a especialista.

## Por qué aprendizaje autosupervisado (SSL)

Las etiquetas dermatológicas de calidad son caras y escasas. El SSL permite **pre-entrenar sobre
imágenes sin etiquetar** (aprender la estructura del dominio "piel") y luego afinar con las pocas
etiquetas disponibles. Es el enfoque adecuado cuando hay muchas imágenes pero pocas anotaciones.

## Objetivos

- **Objetivo 1 — EDA (✅ completado).** Análisis exploratorio cruzado entre la epidemiología de
  Santander y tres datasets dermatológicos (HAM10000, BCN20000, CO2Wounds-V2). Entregado como
  notebook reproducible con 13 figuras a 300 DPI + Tabla I. Definió el núcleo de trabajo:
  **7 clases de HAM10000**.
- **Objetivo 2 — Diseño del algoritmo SSL (🟡 en curso).** Diseñar y validar un algoritmo de
  clasificación de lesiones cutáneas basado en SSL domain-adaptive, con una **ablación** que
  demuestre el aporte del SSL frente a un baseline supervisado (ImageNet). Es el objetivo del
  pipeline `v2` de este repo.
- **Objetivo 3 — Evaluación / validación (posterior).** Evaluar el modelo contra KPIs clínicos y,
  de ser posible, validación externa cross-dataset y análisis de sesgo por fototipo.

> Nota: la redacción formal de los objetivos vive en el anteproyecto (Google Doc). Aquí se resume
> el estado operativo, no reemplaza el documento oficial.

## Alcance y límites declarados

- Es una herramienta de **apoyo**, no da diagnóstico final.
- HAM10000 es mayoritariamente **piel clara (Fitzpatrick I–III)** → no se puede afirmar rendimiento
  en fototipos IV–VI sin datos regionales. Se declara como limitación + trabajo futuro
  (ver [07 — Fiabilidad y sesgo](07-fiabilidad-y-sesgo.md)).
- Cómputo actual = GPU de estudiante (Kaggle T4 x2). El entrenamiento definitivo depende de
  conseguir acceso a las GPUs de la UNAB.

## Las 7 clases (HAM10000)

| Código | Significado |
|---|---|
| `akiec` | Queratosis actínica / carcinoma intraepitelial |
| `bcc` | Carcinoma basocelular |
| `bkl` | Lesiones benignas tipo queratosis |
| `df` | Dermatofibroma |
| `mel` | **Melanoma** (la clase crítica del tamizaje) |
| `nv` | Nevus melanocítico (lunar común) |
| `vasc` | Lesiones vasculares |

Fuerte **desbalance**: `nv` ≈ 6705 imágenes vs `df` ≈ 115 (razón ~58×). Por eso la métrica honesta
es **macro-F1**, no accuracy.
