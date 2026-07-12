# 07 — Fiabilidad y sesgo

Una herramienta médica de **apoyo** (no da diagnóstico final) se defiende con 5 evidencias, todas
implementadas en la sección 6c del notebook.

> **Resultados reales de la corrida definitiva v4 (2026-07-12)** (detalle en [11](11-resultados-finales.md) §6):
> - Calibración: ECE **0.1795 → 0.0507** (temperature scaling, T = 0.6615). ✅ meta ≤ 0.10.
> - IC95 % (bootstrap): macro-F1 **[0.6583, 0.7333]**; sens. mel **[0.5502, 0.6776]**.
> - Abstención: a confianza ≥ 0.8 → cubre **74.5 %** con **91.9 %** de exactitud.
> - Validación externa: ⚠️ **omitida** (los ISIC resized solo traen etiqueta binaria) → pendiente #1.
> - Grad-CAM: ✅ el calor cae **sobre la lesión**, no en pelos/reglas → **sin shortcut learning**.

## 1. Calibración (¿la confianza es honesta?)

- **Reliability diagram:** cuando el modelo dice "80% seguro", debería acertar ~80%.
- **ECE (Expected Calibration Error):** mide la brecha entre confianza y exactitud. Meta ≤ 0.10.
- **Temperature scaling:** ajusta un escalar T sobre los logits → baja el ECE **sin** cambiar el
  argmax (no afecta accuracy/F1). Se guarda la T en el checkpoint.

## 2. Intervalos de confianza (bootstrap)

No se reporta un número pelado. Se reporta `macro-F1 = 0.697 [0.658, 0.733]` y
`sensibilidad mel = 0.613 [0.550, 0.678]` con 1000 remuestreos bootstrap. Honestidad estadística.

## 3. Abstención (selective prediction)

Si el modelo no está seguro, responde **"indeterminado → derivar a especialista"**. Se reporta la
curva cobertura vs exactitud por umbral de confianza (0.5–0.9). Una herramienta de tamizaje puede
preferir abstenerse antes que equivocarse en un caso dudoso.

## 4. Validación externa (cross-dataset)

Testear en OTRO dataset que el modelo nunca vio (ISIC2019 / BCN20000) es la prueba **más fuerte** de
generalización. El pipeline auto-detecta un `*GroundTruth*.csv` y mapea ISIC → 7 clases HAM. Si no
hay dataset externo adjunto, se omite con aviso (pendiente adjuntarlo).

## 5. Grad-CAM (¿mira la lesión?)

Mapa de calor sobre melanomas. El calor debe caer **sobre la lesión**, no en pelos, reglas o
marcadores. Si mira artefactos → *shortcut learning* → se reporta como riesgo.

---

## Sesgo por fototipo (lo que NO podemos esconder)

- HAM10000 = piel clara (**Fitzpatrick I–III**). No hay base para afirmar rendimiento en fototipos
  IV–VI.
- Evidencia de la literatura del problema: sensibilidad **0.69 (piel clara) vs 0.23 (piel oscura)**
  en un modelo comparable → el sesgo racial en dermatología IA es real y grave.
- **Plan (trabajo futuro):** evaluar en el test set **BOSQUE** (165 imgs dermatoscópicas de Bogotá
  con fototipo Fitzpatrick anotado — arXiv 2507.14176) para medir la brecha en piel oscura.
- Meta declarada: brecha de fairness ≤ 0.15 (requiere datos regionales; hoy es objetivo, no logro).

## Mejores prácticas médicas aplicadas (checklist)

1. Split por lesión/paciente (no por imagen).
2. Validación externa cross-dataset.
3. macro-F1 + sensibilidad + especificidad (no accuracy).
4. Calibración: ECE + reliability diagram.
5. Grad-CAM con verificación de que mira la lesión.
6. Manejo de artefactos (pelos bajan rendimiento).
7. Equidad Fitzpatrick declarada como límite.
8. Tests estadísticos (bootstrap; McNemar/DeLong como extensión).
9. Abstención / selective prediction.
10. SSL nunca desde cero.
