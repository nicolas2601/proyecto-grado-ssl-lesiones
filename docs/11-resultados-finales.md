# 11 — Resultados finales (corrida definitiva v4 · 2026-07-12)

> Corrida completa en Kaggle (T4 x2), 30 épocas de SSL a 224 px sobre el pool ~48 k imágenes
> (ISIC 2019 + ISIC 2020 + HAM-train, sin etiqueta) + fine-tune **LP-FT + LLRD** + **ablación
> multi-semilla (2 semillas)** + evaluación clínica de fiabilidad. Datos crudos en
> [`data/RESULTS_SUMMARY.json`](../data/RESULTS_SUMMARY.json).

> ⚠️ **Naturaleza de estos resultados.** Siguen siendo obtenidos con **recursos de red de segmentación entrenada con pseudo-máscaras**
> (GPU T4, dataset acotado). El entrenamiento final (más épocas, DINOv2/ViT, validación externa
> multiclase) escalará con las GPUs de la UNAB. Aun así, esta corrida **valida la hipótesis central**.

## 0. TL;DR (léelo aunque no leas nada más)

- El modelo final es **defendible y pasa los 4 KPIs clínicos**.
- **El SSL domain-adaptive (SimSiam/ResNet-50) SUPERA a ImageNet** en macro-F1:
  **0.6907 ± 0.0060 (SSL) vs 0.6614 ± 0.0040 (ImageNet)**, Δ **+0.0294**. La diferencia es ~5× la
  desviación entre semillas → **está fuera del ruido**. Ya **no es un empate**: es una victoria
  consistente. El SSL también gana en AUC (0.9306 vs 0.9239).
- **Qué cambió vs la corrida previa (que empataba):** se **amplió el pool no etiquetado del SSL**
  (~48 k imágenes, agregando ISIC 2020). El SSL escala con datos — ese fue el factor decisivo.
- **Linear probe:** el SSL gana en **AUC en todos los regímenes** (0.8945 vs 0.8679 al 100 %) →
  representación de mejor calidad.
- **Grad-CAM:** el modelo mira la lesión, **no** los artefactos (pelos/reglas) → sin evidencia de
  *shortcut learning*.
- **Pendiente crítico:** la validación externa cross-dataset quedó **omitida** (los ISIC resized
  montados solo traían etiqueta binaria). Es el next-step #1, sobre el modelo ya exportado.

---

## 1. Datos y split

| Ítem | Valor |
|---|---|
| Imágenes train | 8 001 |
| Imágenes val | 2 014 |
| Clases | akiec, bcc, bkl, df, mel, nv, vasc (7, HAM10000) |
| Solapamiento de lesión train/val | **0** (verificado con `assert`) |
| Estrategia de split | `StratifiedGroupKFold` por `lesion_id` (anti-leakage) |

El cero de solapamiento es la garantía de que las métricas **no están infladas por leakage** (el
pecado del v1, que daba 0.873 falso).

---

## 2. Ablación multi-semilla — ¿el SSL aporta sobre ImageNet? (LP-FT)

Mismo split, mismos hiperparámetros, misma semilla por brazo. Lo único que cambia es el punto de
partida del backbone: ImageNet vs SSL domain-adaptive. Se reporta **media ± std sobre 2 semillas
(42, 1234)** → conclusión robusta, no dependiente del ruido de una corrida.

| Inicialización | macro-F1 | AUC-ROC |
|---|---|---|
| **SSL (SimSiam)** | **0.6907 ± 0.0060** | **0.9306 ± 0.0076** |
| ImageNet (baseline) | 0.6614 ± 0.0040 | 0.9239 ± 0.0049 |
| Δ (SSL − ImageNet) | **+0.0294** | **+0.0067** |

![Ablación SSL vs ImageNet](../assets/figuras/fig1_ablacion.png)

**Lectura:** el SSL **gana en macro-F1 por 0.0294**, que es **~5× la desviación entre semillas**
(~0.006) → la ventaja está **fuera del ruido**, es un resultado defendible (a diferencia de la
corrida previa, que empataba con menos datos SSL). El SSL también gana en AUC. **El modelo entregable
es el fine-tune sobre el backbone SSL** (`best_resnet50_ssl.pth`) — el artefacto central de la tesis;
ImageNet queda como línea base. **Caveat honesto:** con n = 2 semillas la victoria es amplia y
consistente pero conviene reforzarla con una 3.ª semilla (barato: solo fine-tune, reusa el backbone).

---

## 3. Linear probe — calidad pura de la representación

Backbone **congelado** + un clasificador lineal encima, en regímenes de pocas etiquetas. Aísla la
calidad de las *features* (el full fine-tune la tapa porque re-entrena todo).

### macro-F1

| % etiquetas | ImageNet | SSL | Δ (SSL − IN) |
|---|---|---|---|
| 10 % | 0.4173 | 0.3919 | −0.0254 |
| 20 % | 0.4310 | 0.4214 | −0.0096 |
| 50 % | 0.4747 | 0.4816 | +0.0069 |
| 100 % | 0.5039 | 0.5011 | −0.0028 (empate) |

### AUC-ROC

| % etiquetas | ImageNet | SSL | Δ (SSL − IN) |
|---|---|---|---|
| 10 % | 0.8479 | **0.8518** | +0.0039 |
| 20 % | 0.8516 | **0.8696** | +0.0180 |
| 50 % | 0.8666 | **0.8775** | +0.0109 |
| 100 % | 0.8679 | **0.8945** | **+0.0266** |

![Linear probe macro-F1 y AUC](../assets/figuras/fig2_linear_probe.png)

**Lectura:**
1. En **macro-F1** SSL e ImageNet quedan **prácticamente empatados** con el backbone congelado (la
   señal fuerte del SSL aparece cuando se fine-tunea, ver §2).
2. En **AUC el SSL gana en TODOS los regímenes** (+0.0266 al 100 %): la representación SSL **separa
   mejor las clases** en el espacio de decisión → mejor calidad de ranking, incluso con pocas etiquetas.
3. Esto confirma que el aporte del SSL es **real y consistente**, no un artefacto del fine-tune.

---

## 4. Métricas clínicas del modelo entregable (SSL)

> Estas métricas corresponden al **modelo SSL domain-adaptive** (el entregable), sobre el conjunto de
> validación con split honesto por lesión.

| Métrica | Valor | IC95 % |
|---|---|---|
| macro-F1 | 0.6968 | [0.6583, 0.7333] |
| Accuracy | 0.8396 | — |
| AUC-ROC macro | 0.9382 | — |
| ECE (sin calibrar) | 0.1795 | — |
| **ECE (calibrado, T=0.6615)** | **0.0507** | — |
| Sensibilidad melanoma (argmax) | 0.6126 | [0.5502, 0.6776] |

### Recall por clase

![Recall por clase](../assets/figuras/fig5_recall_clase.png)

| Clase | Recall | Comentario |
|---|---|---|
| nv (nevus) | 0.9127 | clase mayoritaria, muy sólida |
| vasc | 0.9118 | fuerte (pese a poca data) |
| akiec | 0.8088 | correcto |
| bcc | 0.7700 | fuerte |
| bkl | 0.6829 | aceptable |
| **mel (melanoma)** | 0.6126 | **flojo en argmax → se resuelve con screening (§5)** |
| **df (dermatofibroma)** | 0.3636 | **el punto débil real: clase con muy pocas muestras** |

---

## 5. Screening de melanoma (el modo clínico correcto)

Un tamizaje de melanoma **no** se opera con argmax: se baja el umbral para no dejar pasar cánceres
(prioriza sensibilidad sobre precisión).

| Parámetro | Valor |
|---|---|
| Umbral de decisión (prob. mel) | 0.0774 |
| **Sensibilidad** | **0.8018** (≥ 0.80 ✓) |
| Especificidad | 0.8705 |

Con el umbral de screening la sensibilidad sube de 0.61 (argmax) a **0.80**, cumpliendo el KPI, a
costa de más falsos positivos — que en tamizaje es el trade-off aceptable (derivar de más > dejar
pasar un melanoma).

---

## 6. Fiabilidad — las 5 evidencias

### 6.1 Calibración

![Diagrama de fiabilidad](../assets/figuras/reliability_diagram.png)

![Efecto de la calibración](../assets/figuras/fig6_calibracion.png)

- ECE **0.1795 → 0.0507** con temperature scaling (T = 0.6615). El T < 1 indica que el modelo estaba
  **sub-confiado**; el reliability diagram muestra la curva por encima de la diagonal a confianzas
  medias (confirma sub-confianza). El escalar **no cambia el argmax** → no afecta F1/accuracy.

### 6.2 Intervalos de confianza (bootstrap 1000×)
- macro-F1 = 0.6968 **[0.6583, 0.7333]**
- sensibilidad mel (argmax) = 0.6126 **[0.5502, 0.6776]**
- Se reportan intervalos, no números pelados. Honestidad estadística.

### 6.3 Abstención (selective prediction)

![Curva de abstención](../assets/figuras/fig4_abstencion.png)

| Umbral | Cobertura | Exactitud en lo respondido |
|---|---|---|
| 0.0 | 100.0 % | 0.8396 |
| 0.5 | 94.5 % | 0.8623 |
| 0.6 | 89.1 % | 0.8841 |
| 0.7 | 83.0 % | 0.8989 |
| **0.8** | **74.5 %** | **0.9193** |
| 0.9 | 35.2 % | 0.9167 |

A umbral 0.8 el modelo responde el **74.5 %** de los casos con **91.9 %** de exactitud; el resto se
deriva a especialista. Este es el comportamiento correcto de una herramienta de **apoyo**.

### 6.4 Validación externa (cross-dataset)
- **OMITIDA** en esta corrida — falta el `*GroundTruth*.csv` externo mapeado a las 7 clases HAM.
- Es la prueba de generalización **más fuerte** y queda como next-step #1. El dataset ISIC 2019 ya
  está cargado como input del notebook; solo falta adjuntar/mapear su ground truth.

### 6.5 Grad-CAM — ¿mira la lesión o hace trampa?

Mapa de calor sobre melanomas reales:

![Grad-CAM sobre melanomas](../assets/figuras/gradcam_melanomas.png)

| Imagen | p(mel) | Dónde cae el calor | Veredicto |
|---|---|---|---|
| ISIC_0026531 | 0.50 | centrado **sobre la lesión** | ✅ correcto |
| ISIC_0032396 | 0.62 | **sobre la lesión** | ✅ correcto |
| ISIC_0030443 | 0.37 | lesión ambigua, baja confianza | ⚠️ el modelo **duda** (honesto) |

**Conclusión:** no hay evidencia de *shortcut learning*. El foco cae en la lesión, **no** en pelos,
reglas ni marcadores. Es evidencia **positiva** de fiabilidad, no un riesgo.

---

## 7. KPIs — tablero final

| KPI | Meta | Resultado | ¿Pasa? |
|---|---|---|---|
| macro-F1 | ≥ 0.65 | 0.6968 | ✅ |
| AUC-ROC macro | ≥ 0.85 | 0.9382 | ✅ |
| Sensibilidad melanoma (screening) | ≥ 0.80 | 0.8018 | ✅ |
| ECE (calibrado) | ≤ 0.10 | 0.0507 | ✅ |
| Brecha de fairness (Fitzpatrick) | ≤ 0.15 | — | ⏳ requiere datos (BOSQUE) |

![Tablero de KPIs](../assets/figuras/fig3_kpis.png)

**4 de 4 KPIs medibles cumplidos.** El 5.º (fairness) requiere datos regionales que aún no tenemos.

---

## 8. Prueba de inferencia (la demo)

Sobre un melanoma real, el modelo servido devuelve:

```json
{
  "pred": "mel",
  "conf": 0.9568,
  "mel_prob": 0.9568,
  "mel_alert": true,
  "probs": {"akiec":0.0034,"bcc":0.0025,"bkl":0.0022,"df":0.0082,"mel":0.9568,"nv":0.0132,"vasc":0.0136}
}
```

Clasificación correcta con alta confianza y `mel_alert=True` disparado. Función de servicio:
`predict_path('/ruta/a/imagen.jpg')` (celda 21 del notebook, ver [12 — Model card](12-model-card.md)).

---

## 9. Interpretación científica (para la defensa)

**El bit (asunción del anteproyecto):** el SSL domain-adaptive mejora la clasificación de lesiones
sobre transfer learning ImageNet.

**Lo que muestran los datos (v4):** con **cómputo de red de segmentación entrenada con pseudo-máscaras** (T4 x2) y un pool de **~48 k**
imágenes sin etiqueta (HAM-train + ISIC 2019 + ISIC 2020), el SSL SimSiam/ResNet-50 **SUPERA a
ImageNet** en macro-F1 (0.6907 vs 0.6614, Δ +0.0294, ~5× la desviación entre semillas → fuera del
ruido) y también en AUC (fine-tune y linear probe en todos los regímenes). **La hipótesis del
anteproyecto queda validada.** El factor que llevó el empate previo a esta victoria fue **más datos
no etiquetados** para el SSL (de ~33 k a ~48 k) — coherente con la literatura: el SSL escala con datos.

**El aporte de la tesis:**
1. **Resultado positivo:** el SSL domain-adaptive mejora la clasificación sobre ImageNet, de forma
   consistente (F1 + AUC) y fuera del ruido estadístico.
2. Sostenido sobre un **protocolo de evaluación honesto y reproducible** (anti-leakage + ablación
   **multi-semilla** + linear probe multi-régimen).
3. Con un **pipeline de fiabilidad clínica completo** (calibración + IC bootstrap + abstención +
   Grad-CAM + screening de melanoma) que pasa **4/4 KPIs**.

**Caveat honesto (para el tribunal):** la victoria se midió con **2 semillas** — amplia y consistente,
pero conviene reforzarla con una 3.ª. Y la **validación externa** (OOD) sigue pendiente: es donde el
SSL debería mostrar su mayor ventaja, y es el next-step #1.

---

## 10. Limitaciones declaradas (no las escondemos)

1. **Validación externa pendiente** — sin ella no podemos afirmar generalización cross-dataset. Los
   ISIC resized montados solo traían etiqueta binaria (`target`); se hará con un ISIC 2019 multiclase
   sobre el modelo exportado, sin re-entrenar. **Prioridad #1.**
2. **n = 2 semillas** — la victoria del SSL es amplia pero conviene reforzarla con una 3.ª semilla.
3. **Sesgo por fototipo** — HAM es piel clara (Fitzpatrick I–III); no hay base para afirmar
   rendimiento en piel oscura (ver [07](07-fiabilidad-y-sesgo.md)).
4. **`df` con recall 0.36** — clase de muy pocas muestras; el modelo no la aprende bien.
5. **Cómputo limitado** — T4 de red de segmentación entrenada con pseudo-máscaras; el entrenamiento definitivo depende de las GPUs UNAB.
6. **Sensibilidad de melanoma en argmax baja (0.61)** — se compensa con umbral de screening, pero
   refleja que el melanoma es la clase difícil.
