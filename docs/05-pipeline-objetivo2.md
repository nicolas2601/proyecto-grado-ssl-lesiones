# 05 — Pipeline del Objetivo 2 (notebook `v2`)

Archivo: [`notebooks/proyecto_ssl_lesiones_v2.ipynb`](../notebooks/proyecto_ssl_lesiones_v2.ipynb)
Entorno: Kaggle, GPU **T4 x2**, 24 celdas.

## Diagrama de flujo

```
Datasets (HAM10000 + ISIC2019)
        │
        ▼
[1] Config + carga robusta + SMOKE/multi-GPU
        │
        ▼
[2] Split lesión-disjunto (StratifiedGroupKFold por lesion_id) + pesos de clase
        │
        ▼
[3] SSL domain-adaptive  ── SimSiam(ResNet50) sobre ISIC + HAM-train SIN etiquetas
        │                   (parte de ImageNet, 224px, diagnóstico de colapso out_std)
        ▼
[4-5] Fine-tune supervisado + ABLACIÓN  ──  Brazo A: ImageNet  vs  Brazo B: SSL
        │                                    (mismo split/hiperparámetros/semilla)
        ▼
[6] Reporte clínico + calibración (ECE + temperature scaling) + screening melanoma
        │
        ▼
[6b] Linear probe (pocas etiquetas 10/20/50/100%)  ── prueba REAL de la contribución SSL
        │
        ▼
[6c] Fiabilidad: reliability diagram, bootstrap IC95%, abstención, validación externa, Grad-CAM
        │
        ▼
[7-8] Export del mejor modelo (.pth) + predict_path() + RESULTS_SUMMARY.json
```

## Componentes clave

- **SSL = SimSiam** (negative-free, stop-gradient + predictor). Diagnóstico de colapso: `out_std`
  de la representación L2-normalizada por dimensión (sano ≈ 0.022; colapso si → 0).
- **Fine-tune:** ResNet-50, 224px, AdamW, cosine LR, label smoothing, RandomErasing, rotación/flips,
  pesos de clase suavizados, TTA (hflip), guarda el **mejor epoch por macro-F1**.
- **`build_model('ssl')`** carga de verdad el backbone domain-adaptive (bug del v1 resuelto).
- **Calibración:** temperature scaling (LBFGS) → baja ECE sin cambiar el argmax.
- **Screening melanoma:** umbral movido con `roc_curve` para garantizar sensibilidad ≥ 0.80.
- **Export:** checkpoint con clases, temperatura, umbral de screening, norm stats → listo para API.

## Cómo correrlo (recomendado)

1. Kaggle → Settings → Accelerator **GPU T4 x2**.
2. Add Input: `Skin Cancer MNIST: HAM10000` (kmader) + `ISIC 2019 JPG 224x224 RESIZED` (nischaydnk).
3. **Primero `SMOKE = True`** → Run All (~3–5 min). Si llega al final sin error, el pipeline sirve.
4. **`SMOKE = False`** → **Save & Run All (Commit)** para la corrida real (headless, robusto a
   caídas de sesión).

## Presupuesto de cómputo (perillas)

| Perilla | Valor actual | Nota |
|---|---|---|
| `SSL_EPOCHS` | 30 | La loss se aplana rápido (~ép. 4). 12–15 basta; 30 es exceso. |
| `SSL_BATCH` / `SSL_RES` | 128 / 224 | 224 igual al fine-tune (evita BN drift). |
| `SUP_EPOCHS` | 20 (× 2 brazos) | Ablación completa. |
| Pool SSL | ~33.332 imgs | ISIC (~25k) + HAM-train. |

> En la corrida de referencia (30 épocas, pool completo) el tiempo total estimado es **~6 h**.
> Para futuras corridas: bajar `SSL_EPOCHS` a 15 y subir `num_workers` del dataloader SSL a 4
> reduce el total a ~3 h sin dañar la ciencia (la loss ya converge).

## Robustez añadida (D-07)

- `SMOKE` para validar el pipeline entero en minutos.
- `DataParallel` en SSL y fine-tune (pesos guardados con llaves planas).
- `load_rgb()` + `LOAD_TRUNCATED_IMAGES` → una imagen corrupta no tumba el run.
- Checkpoint parcial del backbone SSL cada 10 épocas + ETA por época.
- `predict_path()` para inferencia sobre cualquier ruta (base de la API/demo).
