# 08 — Estado actual y próximos pasos

_Última actualización: 2026-08-24 (banco de baselines + tests por modelo)._

> **Actualización 2026-08-24.** (1) v5 SSL-puro ya corrió en Kaggle: DINOv2 congelado gana 6/6
> comparaciones contra ImageNet supervisado. (2) v6 (pre-entrenamiento propio, dos brazos SSL)
> está construido con 190 tests en verde; pendiente de cluster y de los fixes DDP (Engram #1328).
> (3) Banco de baselines completo: 7 modelos con un notebook de test autocontenido por modelo en
> `notebooks/baselines/` (ver docs/17 y docs/18). Métricas ampliadas a todas las de literatura
> (balanced accuracy, precision/recall macro, además de F1, AUC, accuracy, ECE).
> Lo que sigue abajo es el estado histórico de la corrida v4.

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

## Auditoría v5 (2026-07-29): el v4 no es SSL puro

El SSL del v4 arranca de `ResNet50_Weights.IMAGENET1K_V2` (celda 7) y el fine-tune descongela
todo el backbone con etiquetas. Es **adaptación de dominio**, no auto-supervisión pura: el
Δ +0.0294 mide cuánto aporta adaptar al dominio, no SSL contra supervisado.

Se construyó el pipeline **v5** (`notebooks/proyecto_ssl_lesiones_v5.ipynb`, generado): backbone
**congelado** (DINOv2, cero etiquetas en el extractor) + probes kNN/prototipos/linear en
regímenes 1/5/10/20/50/100 %, con control de arquitectura (ViT supervisado) y 101 pruebas
automáticas que corren sin GPU. Detalle completo en
[15 — Auditoría SSL puro y plan v5](15-auditoria-ssl-puro-y-plan-v5.md).

## Próximos pasos (roadmap)

0. **Correr v5 en Kaggle**: `SMOKE = True` → Run All (~5 min, valida el pipeline y las compuertas
   G0–G2) → `SMOKE = False` → Save & Run All. No necesita el clúster: el backbone va congelado.
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
