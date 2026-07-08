# 08 — Estado actual y próximos pasos

_Última actualización: 2026-07-07 (corrida ejecutada)._

> ⚠️ **Los resultados son preliminares** — el techo alcanzable con los recursos actuales (T4 de
> estudiante, dataset pequeño). No son el modelo definitivo; ese llega con las GPUs de la UNAB.

## Dónde estamos

- **Objetivo 1 (EDA):** ✅ cerrado.
- **Objetivo 2 (SSL):** ✅ **corrida de referencia ejecutada** en Kaggle (T4 x2), 30 épocas de SSL a
  224 px sobre el pool ~33 k + fine-tune + ablación + fiabilidad. Modelo exportado y demo de
  inferencia funcionando. Resultados completos en [11 — Resultados finales](11-resultados-finales.md).

## Qué salió (resumen)

- **4/4 KPIs clínicos cumplidos** (macro-F1 0.7454, AUC 0.9343, mel sens 0.8018, ECE calib. 0.0598).
- **Ablación:** empate SSL vs ImageNet en macro-F1; SSL gana en AUC. Desplegamos el modelo SSL.
- **Linear probe:** el fix de 224 px eliminó el colapso (0.264 → 0.4509); SSL sigue por debajo en F1,
  gana en AUC al 100 %. Contradicción **cerrada** como límite honesto.
- **Grad-CAM:** el modelo mira la lesión, no los artefactos → sin shortcut learning.
- **Validación externa:** ⚠️ omitida (falta ground truth externo mapeado) → next-step #1.

## Próximos pasos (roadmap)

1. **Leer resultados** de la corrida de 6 h y decidir sobre la contradicción del linear probe
   (rescatada vs límite honesto — ver [06](06-problematicas.md)).
2. **Demo de inferencia:** empaquetar `predict_path()` + el `.pth` exportado en algo probable
   (script o API mínima) para mostrarle al director cómo clasifica una imagen nueva.
3. **Enviar al director (Andrés Jerez):** resultados + demo → pedir feedback y **acceso a las GPUs
   de la UNAB** para el entrenamiento definitivo.
4. **Entrenamiento definitivo (con GPUs UNAB):** escalar (más épocas, posible DINOv2/ViT-B como
   plantea el anteproyecto), validación externa con dataset etiquetado, y evaluación de sesgo (BOSQUE).
5. **Objetivo 3:** evaluación formal contra KPIs + documento.

## Pendientes técnicos / seguridad

- [ ] **Descargar de Kaggle** `best_resnet50_ssl.pth` + `RESULTS_SUMMARY.json` + PNGs → subir modelo
      a un **GitHub Release** (el `.pth` no cabe en el repo normal, >100 MB).
- [ ] 🔐 **Rotar token de Kaggle (KGAT) expuesto** (pendiente arrastrado).
- [ ] Borrar `~/Descargas/math.so` corrupto.
- [ ] Adjuntar ISIC2019 `GroundTruth.csv` y correr **validación externa** (next-step #1).
- [x] ~~Revisar Grad-CAM~~ → hecho: mira la lesión, sin shortcut.
- [ ] Agregar a Paula como colaboradora del repo (decidir cuenta: Paulasaah vs mpsaavedra05).
- [ ] Para futuras corridas: `SSL_EPOCHS=15` + `num_workers=4` → ~3 h en vez de ~6 h.

## Referencia rápida de resultados

| Hito | macro-F1 | Nota |
|---|---|---|
| v1 (inflado) | 0.873 | ❌ leakage + SSL desconectado |
| v2 split honesto | ~0.74 | ✅ real y defendible |
| **Corrida final · modelo SSL** | **0.7447** | ✅ AUC 0.9430; 4/4 KPIs |
| Corrida final · baseline ImageNet | 0.7454 | empate (Δ 0.0007, ruido) |
| Linear probe SSL vs ImageNet (100 %) | 0.4509 vs 0.5123 | 🟢 colapso resuelto (224 px); SSL gana AUC |
