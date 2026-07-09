# Informe de avance — Objetivo 2

**Proyecto de grado:** Clasificación de lesiones cutáneas con aprendizaje autosupervisado (SSL)
**Autores:** Nicolás Santiago Moreno Monroy · María Paula Saavedra
**Director:** Andrés Jerez — **Asesora:** Karen Sánchez (KAUST)
**Institución:** Universidad Autónoma de Bucaramanga (UNAB) — Ingeniería de Sistemas
**Fecha:** 2026-07-07

---

## 1. Resumen ejecutivo

Completamos la **corrida de referencia del Objetivo 2** (diseño y evaluación del algoritmo SSL) en
Kaggle (2× GPU T4): pretrain SSL domain-adaptive + fine-tune supervisado + ablación + evaluación
clínica de fiabilidad, con protocolo **anti-leakage** (split por lesión, no por imagen).

**Resultado principal, con honestidad científica:**

- El modelo final **cumple los 4 KPIs clínicos medibles** (macro-F1, AUC, sensibilidad de melanoma,
  calibración).
- El **SSL domain-adaptive NO superó** a la inicialización ImageNet en macro-F1: quedaron en
  **empate técnico** (0.7454 vs 0.7447). El SSL **sí gana en AUC** (mejor ranking de casos).
- Corregimos el problema de resolución (SSL a 224 px) que en la corrida previa producía un colapso
  aparente del *linear probe*; ahora los números son coherentes y la contradicción quedó **cerrada y
  documentada** como límite honesto, no como bug.

En una frase: **el aporte de la tesis no es "el SSL gana", sino un protocolo de evaluación honesto y
un pipeline de fiabilidad clínica riguroso**, con un hallazgo matizado sobre dónde ayuda el SSL en
régimen de bajos recursos.

> **Modelo entregable = SSL.** El pipeline se corrigió para que el modelo que se **evalúa, calibra y
> exporta** sea el **SSL domain-adaptive** (el núcleo del proyecto), no el ganador numérico de la
> ablación. Las métricas de la §2 corresponden a la corrida de referencia (brazo evaluado); la
> corrida SSL-céntrica definitiva se ejecuta en la próxima ventana de GPU (reset de cuota Kaggle) y
> refrescará esos números con el SSL como entregable. La ablación (empate en macro-F1, SSL gana en
> AUC) se mantiene como comparación honesta contra el baseline.

---

## 2. Resultados clave (val, split honesto por lesión)

| Métrica | Resultado | Meta (KPI) | Estado |
|---|---|---|---|
| macro-F1 | 0.7454 · IC95 % [0.7005, 0.7807] | ≥ 0.65 | ✅ |
| AUC-ROC macro | 0.9343 | ≥ 0.85 | ✅ |
| Sensibilidad melanoma (screening) | 0.8018 (esp. 0.8873) | ≥ 0.80 | ✅ |
| ECE calibrado (T = 0.676) | 0.0598 | ≤ 0.10 | ✅ |
| Accuracy | 0.8659 | — | — |

**Ablación (aporte del SSL):** macro-F1 ImageNet 0.7454 vs SSL 0.7447 (Δ −0.0007, empate);
AUC ImageNet 0.9343 vs SSL 0.9430 (Δ **+0.0087**, SSL gana). El SSL también gana el AUC del linear
probe al 100 % (+0.0132).

**Fiabilidad:** calibración por temperature scaling (ECE 0.167 → 0.060); intervalos bootstrap;
mecanismo de **abstención** (a confianza ≥ 0.8 responde el 83.7 % con 91.8 % de exactitud);
**Grad-CAM confirma que el modelo mira la lesión, no los artefactos** (sin *shortcut learning*).

**Demo funcional:** clasificación correcta de un melanoma real con confianza 0.957 y alerta de
melanoma disparada. Modelo servible vía `predict_path()`.

---

## 3. Decisiones metodológicas (qué usamos, qué no, y por qué)

| Decisión | Qué elegimos | Por qué |
|---|---|---|
| Método SSL | **SimSiam** (no DINO/ViT en esta fase) | negative-free → corre en 1 GPU T4; DINO/ViT es inviable con cómputo de estudiante. DINOv2 queda para la fase con GPUs UNAB. |
| Inicialización | **domain-adaptive desde ImageNet** (nunca desde cero) | probamos SSL desde cero: pierde contra un backbone aleatorio. Requiere millones de imágenes. |
| Split | por **`lesion_id`** (StratifiedGroupKFold) | partir por imagen mete la misma lesión en train/val → métrica inflada (el error del prototipo previo, que daba 0.873 falso). |
| Evaluación del SSL | **ablación + linear probe** multi-régimen | el linear probe aísla la calidad de la representación; el fine-tune la tapa. |
| Resolución | **224 px en SSL, fine-tune y eval** | igualar resolución elimina el *BN drift* que causaba el colapso del probe. |
| Aceptación | **KPIs clínicos**, no una sola cifra | macro-F1 + AUC + sensibilidad mel + ECE. |
| Metodología | **CRISP-ML(Q)** (migrado desde CRISP-DM) | por recomendación de la asesora: CRISP-DM es legacy para proyectos ML. |

Descarte relevante: **MambaMIM** (sugerido como referencia) es 3D/CT/segmentación con backbone Mamba
→ no encaja con nuestro caso 2D de clasificación dermatoscópica. Se conserva como referencia del
estado del arte; la vía viable de *masked image modeling* en 2D sería MAE/SimMIM.

---

## 4. Limitaciones (declaradas, no escondidas)

1. **Validación externa cross-dataset pendiente** — es la prueba de generalización más fuerte;
   quedó omitida por falta del ground truth externo mapeado. **Prioridad #1.**
2. **Sesgo por fototipo** — dataset de piel clara (Fitzpatrick I–III); sin base para afirmar
   rendimiento en piel oscura. Plan: test set BOSQUE (Bogotá, con fototipo anotado).
3. **Cómputo limitado** — T4 de estudiante. El entrenamiento definitivo (más épocas, DINOv2/ViT,
   más datos) requiere **acceso a las GPUs de la UNAB**.
4. Clase `df` (dermatofibroma) poco fiable (recall 0.41, pocas muestras).

---

## 5. Petición y próximos pasos

1. **Feedback** sobre el enfoque (reposicionar el aporte hacia protocolo + fiabilidad, dado el
   empate SSL vs ImageNet).
2. **Acceso a las GPUs de la UNAB** para el entrenamiento definitivo.
3. Ejecutar la **validación externa** (ISIC 2019 / BCN20000) — ya está el dataset cargado.
4. Evaluación de **sesgo por fototipo** (BOSQUE) como trabajo del Objetivo 3.

---

## 6. Material de respaldo (este repositorio)

- **Informe técnico completo:** [`docs/11-resultados-finales.md`](docs/11-resultados-finales.md)
- **Datos crudos de la corrida:** [`data/RESULTS_SUMMARY.json`](data/RESULTS_SUMMARY.json)
- **Decisiones técnicas (ADR):** [`docs/04-decisiones-tecnicas.md`](docs/04-decisiones-tecnicas.md)
- **Ficha del modelo:** [`docs/12-model-card.md`](docs/12-model-card.md)
- **Notebook reproducible:** [`notebooks/proyecto_ssl_lesiones_v2.ipynb`](notebooks/proyecto_ssl_lesiones_v2.ipynb)
- **Modelo entrenado** (`.pth`): [Release `modelo-v1-preliminar`](https://github.com/nicolas2601/proyecto-grado-ssl-lesiones/releases/tag/modelo-v1-preliminar).
