# 08 — Estado actual y próximos pasos

_Última actualización: 2026-07-07._

## Dónde estamos

- **Objetivo 1 (EDA):** ✅ cerrado.
- **Objetivo 2 (SSL):** 🟡 pipeline `v2` completo y validado (sintaxis OK, 13 celdas de código).
  Corrida de **referencia de ~6 h** en marcha en Kaggle (T4 x2), 30 épocas de SSL sobre el pool
  completo (~33.332 imgs) + fine-tune ablación + fiabilidad.
- Es la **última corrida de referencia** en Kaggle antes de armar la demo y contactar al director.

## Qué esperamos de esta corrida

1. Tabla de **ablación** (ImageNet vs SSL, macro-F1 / AUC / ECE).
2. Tabla de **linear probe** en pocas etiquetas (10/20/50/100%) → ¿el SSL a 224px recupera?
3. Líneas `out_std` del SSL por época (¿colapsó o no?).
4. `RESULTS_SUMMARY.json` con todo consolidado (KPIs, IC95%, recall por clase, screening, abstención).

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

- [ ] Rotar token de Kaggle (KGAT) expuesto.
- [ ] Borrar `~/Descargas/math.so` corrupto.
- [ ] Adjuntar ISIC2019 `GroundTruth.csv` para validación externa.
- [ ] Revisar Grad-CAM (¿mira lesión o artefactos?).
- [ ] Para futuras corridas: `SSL_EPOCHS=15` + `num_workers=4` → ~3 h en vez de ~6 h.

## Referencia rápida de resultados conocidos

| Hito | macro-F1 | Nota |
|---|---|---|
| v1 (inflado) | 0.873 | ❌ leakage + SSL desconectado |
| v2 split honesto | ~0.74 | ✅ real y defendible |
| Full fine-tune SSL (último) | 0.771 | +0.035 vs baseline, pero es ruido |
| Linear probe SSL vs ImageNet | 0.264 vs 0.506 | 🔴 contradicción a resolver (fix 224px) |
