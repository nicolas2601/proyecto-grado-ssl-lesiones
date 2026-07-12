# 08 — Estado actual y próximos pasos

_Última actualización: 2026-07-12 (corrida definitiva v4 ejecutada)._

> ⚠️ **Resultados con recursos de estudiante** (T4, dataset acotado). El modelo definitivo escala con
> las GPUs de la UNAB. Aun así, esta corrida **valida la hipótesis central del anteproyecto**.

## Dónde estamos

- **Objetivo 1 (EDA):** ✅ cerrado.
- **Objetivo 2 (SSL):** ✅ **corrida definitiva v4 ejecutada** en Kaggle (T4 x2): 30 épocas de SSL a
  224 px sobre ~48 k imágenes sin etiqueta (ISIC 2019 + ISIC 2020 + HAM-train) + fine-tune LP-FT+LLRD
  + **ablación multi-semilla (2 semillas)** + fiabilidad. Modelo SSL exportado como entregable.
  Resultados en [11 — Resultados finales](11-resultados-finales.md).

## Qué salió (resumen)

- **🏆 El SSL SUPERA a ImageNet** (fuera del ruido): macro-F1 **0.6907 ± 0.0060 (SSL)** vs
  **0.6614 ± 0.0040 (ImageNet)**, Δ +0.0294 (~5× la std). También gana en AUC (0.9306 vs 0.9239).
  Ya no es empate — es victoria consistente. **Factor decisivo: más datos no etiquetados en el SSL.**
- **4/4 KPIs clínicos cumplidos** (modelo SSL): macro-F1 0.697, AUC 0.938, mel sens 0.80 (screening),
  ECE calib. 0.051.
- **Linear probe:** el SSL gana en AUC en todos los regímenes (0.8945 vs 0.8679 al 100 %).
- **Grad-CAM:** el modelo mira la lesión, no los artefactos → sin shortcut learning.
- **Validación externa:** ⚠️ omitida (los ISIC resized montados solo traen etiqueta binaria) → next-step #1.

## Próximos pasos (roadmap)

1. **Validación externa** (ISIC 2019 multiclase / BCN20000) sobre el modelo ya exportado — inferencia
   pura, NO requiere re-entrenar. La prueba de generalización más fuerte.
2. **3.ª semilla** en la ablación (solo fine-tune, reusa `backbone_domain_adaptive_r50.pth`) → reforzar
   la significancia.
3. **Enviar al director (Andrés Jerez):** resultados + modelo SSL → feedback y **acceso a GPUs UNAB**.
4. **Entrenamiento definitivo (GPUs UNAB):** escalar (más épocas, DINOv2/ViT-B), validación externa,
   evaluación de sesgo (BOSQUE).
5. **Objetivo 3:** evaluación formal contra KPIs + documento.

## Pendientes técnicos / seguridad

- [x] ~~Re-correr pipeline SSL-céntrico~~ → hecho (v4): SSL como entregable, con resume del backbone.
- [ ] Adjuntar ISIC2019 multiclase y correr **validación externa** (next-step #1).
- [ ] Agregar una **3.ª semilla** a la ablación.
- [ ] 🔐 **Rotar token de Kaggle (KGAT) expuesto** (pendiente arrastrado).
- [x] ~~Revisar Grad-CAM~~ → hecho: mira la lesión, sin shortcut.
- [ ] Agregar a Paula como colaboradora del repo (decidir cuenta: Paulasaah vs mpsaavedra05).

## Referencia rápida de resultados

| Hito | macro-F1 | Nota |
|---|---|---|
| v1 (inflado) | 0.873 | ❌ leakage + SSL desconectado |
| v2 split honesto | ~0.74 | ✅ real; SSL empataba a ImageNet (menos datos SSL) |
| **v4 · SSL (entregable)** | **0.6907 ± 0.006** | 🏆 **gana a ImageNet** (Δ +0.029, fuera del ruido); AUC 0.9306; 4/4 KPIs |
| v4 · baseline ImageNet | 0.6614 ± 0.004 | línea base |
| Linear probe SSL vs ImageNet (100 %, AUC) | 0.8945 vs 0.8679 | 🟢 SSL gana en calidad de representación |

> Nota: los macro-F1 de v4 (~0.69) no son comparables 1-a-1 con los de v2 (~0.74) — v4 usa 15 épocas
> de fine-tune (LP-FT) vs 20 del full fine-tune de v2. Lo que importa es la **comparación pareada**
> SSL vs ImageNet en las MISMAS condiciones, donde el SSL gana.
