# 08 — Estado actual y próximos pasos

_Última actualización: 2026-07-07 (corrida ejecutada)._

> ⚠️ **Los resultados son preliminares** — el techo alcanzable con los recursos actuales (T4 de
> estudiante, dataset pequeño). No son el modelo definitivo; ese llega con las GPUs de la UNAB.

## Dónde estamos

- **Objetivo 1 (EDA):** ✅ cerrado.
- **Objetivo 2 (SSL):** ✅ **corrida de referencia ejecutada** en Kaggle (T4 x2), 30 épocas de SSL a
  224 px sobre el pool ~33 k + fine-tune + ablación + fiabilidad. **Pipeline corregido** para exportar
  y evaluar el **modelo SSL** como entregable. Resultados en [11 — Resultados finales](11-resultados-finales.md).

## Qué salió (resumen)

- **4/4 KPIs clínicos cumplidos** (macro-F1 0.7454, AUC 0.9343, mel sens 0.8018, ECE calib. 0.0598).
- **Ablación:** empate SSL vs ImageNet en macro-F1; SSL gana en AUC. Pipeline corregido para
  desplegar y evaluar el **modelo SSL** como entregable (la corrida 1 había exportado el brazo ImageNet).
- **Linear probe:** el fix de 224 px eliminó el colapso (0.264 → 0.4509); SSL sigue por debajo en F1,
  gana en AUC al 100 %. Contradicción **cerrada** como límite honesto.
- **Grad-CAM:** el modelo mira la lesión, no los artefactos → sin shortcut learning.
- **Validación externa:** ⚠️ omitida (falta ground truth externo mapeado) → next-step #1.

## Próximos pasos (roadmap)

1. **Re-correr el pipeline SSL-céntrico** (próxima semana, al resetear la cuota GPU de Kaggle):
   exporta el **modelo SSL** como entregable + su `RESULTS_SUMMARY.json` y figuras. No hace falta
   repetir el pretrain SSL si se reusa `backbone_domain_adaptive_r50.pth`.
2. **Validación externa** (ISIC2019 `GroundTruth.csv`) — la prueba de generalización más fuerte.
3. **Enviar al director (Andrés Jerez):** resultados + modelo SSL → pedir feedback y **acceso a las
   GPUs de la UNAB** para el entrenamiento definitivo.
4. **Entrenamiento definitivo (con GPUs UNAB):** escalar (más épocas, posible DINOv2/ViT-B como
   plantea el anteproyecto), validación externa con dataset etiquetado, y evaluación de sesgo (BOSQUE).
5. **Objetivo 3:** evaluación formal contra KPIs + documento.

## Pendientes técnicos / seguridad

- [ ] **Próxima semana (reset GPU Kaggle):** correr el pipeline corregido → exportar el **modelo SSL**
      → reemplazar el artefacto del Release + refrescar los números en los docs.
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
