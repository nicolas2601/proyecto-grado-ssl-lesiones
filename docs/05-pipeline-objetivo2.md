# 05 — Pipeline del Objetivo 2 (notebook `v4`)

> **Documento histórico.** Describe una etapa anterior del proyecto y fue superado. Los resultados vigentes están en el [README](../README.md) y en los documentos 27 a 33. Se conserva para mostrar cómo se llegó al diseño actual.

Archivo: [`notebooks/proyecto_ssl_lesiones_v4.ipynb`](../notebooks/proyecto_ssl_lesiones_v4.ipynb)
Entorno: Kaggle, GPU **T4 x2**, 28 celdas.

## Diagrama de flujo

```
Datasets (HAM10000 + ISIC 2019 + ISIC 2020)
        │
        ▼
[1] Config + carga robusta + SMOKE + resume-aware
        │
        ▼
[2] Roles anti-leakage: HAM(train/val) · ISIC2020(pool SSL) · ISIC2019(val externa)
        │  Split lesión-disjunto (StratifiedGroupKFold por lesion_id) + pesos de clase
        ▼
[3] SSL domain-adaptive  ── SimSiam(ResNet50) sobre ~48k imgs SIN etiqueta
        │                   (parte de ImageNet, 224px, diagnóstico de colapso out_std)
        │                   ⚡ RESUME: si el backbone ya existe (Input), se salta el pretrain (~6h)
        ▼
[4-5] Fine-tune LP-FT + LLRD + ABLACIÓN MULTI-SEMILLA  ── Brazo SSL (primero) vs ImageNet
        │                                                 (2 semillas → media ± std)
        ▼
[6] Reporte clínico + calibración (ECE + temperature scaling) + screening melanoma
        │
        ▼
[6b] Linear probe (pocas etiquetas 10/20/50/100%)  ── prueba REAL de la contribución SSL
        │
        ▼
[6c-d] Fiabilidad: reliability diagram, bootstrap IC95%, abstención, Grad-CAM, validación externa
        │
        ▼
[7-8] Export del modelo SSL (.pth) + predict_path() + RESULTS_SUMMARY.json
```

## Componentes clave

- **SSL = SimSiam** (negative-free, stop-gradient + predictor). Diagnóstico de colapso: `out_std`
  de la representación L2-normalizada por dimensión (sano ≈ 0.022; colapso si → 0).
- **RESUME del backbone SSL:** si `backbone_domain_adaptive_r50.pth` está disponible (montado como
  Input Model/Dataset, o en `/kaggle/working`), se reusa y se **salta el pretrain de ~6h**. Robusto a
  que venga como `.pth`, `.zip` o archivos sueltos extraídos.
- **Fine-tune = LP-FT + LLRD** (Kumar 2022): fase 1 linear probe (backbone congelado), fase 2
  fine-tune con layer-wise LR decay (0.75/bloque) → **preserva la representación SSL**.
- **Ablación multi-semilla:** 2 semillas (42, 1234) por brazo → **media ± std** (rigor estadístico).
  El brazo **SSL se entrena primero** para exportar el entregable temprano.
- **`build_model('ssl')`** carga de verdad el backbone domain-adaptive (bug del v1 resuelto).
- **Calibración:** temperature scaling (LBFGS, T acotada a [0.5, 5]) → baja ECE sin cambiar el argmax;
  fallback a T=1 si no mejora.
- **Screening melanoma:** umbral movido con `roc_curve` para garantizar sensibilidad ≥ 0.80.
- **Export:** checkpoint con clases, temperatura, umbral de screening, norm stats → listo para API.

## Cómo correrlo (recomendado)

1. Kaggle → Settings → Accelerator **GPU T4 x2**.
2. Add Input:
   - `Skin Cancer MNIST: HAM10000` (kmader) — train/val etiquetado.
   - `ISIC 2020 JPG 256x256 RESIZED` (nischaydnk) — pool SSL no etiquetado.
   - `ISIC 2019 JPG 224x224 RESIZED` (nischaydnk) — imágenes de val externa (necesita un GT multiclase).
   - **(resume)** el Model/output con `backbone_domain_adaptive_r50.pth` → salta el pretrain SSL.
3. **Primero `SMOKE = True`** → Run All (~3–5 min). Si llega al final sin error, el pipeline sirve.
4. **`SMOKE = False`** → **Save & Run All (Commit)** para la corrida real (headless, robusto a
   caídas de sesión).

## Presupuesto de cómputo (perillas)

| Perilla | Valor actual | Nota |
|---|---|---|
| `SSL_EPOCHS` | 30 | La loss se aplana rápido; con resume no se re-entrena. |
| `SSL_BATCH` / `SSL_RES` | 128 / 224 | 224 igual al fine-tune (evita BN drift). |
| `LP_EPOCHS` / `FT_EPOCHS` | 10 / 15 | LP-FT por brazo. |
| `N_SEEDS` | 2 | Ablación multi-semilla (media ± std). |
| Pool SSL | ~48 k imgs | ISIC 2019 + ISIC 2020 + HAM-train, sin etiqueta. |

> Con el **backbone SSL resumido** (skip del pretrain), la corrida completa (fine-tune 2 semillas × 2
> brazos + eval) toma **~2–4 h**. Sin resume, el pretrain SSL agrega ~6 h.

## Robustez añadida (v3/v4)

- `SMOKE` para validar el pipeline entero en minutos (incl. pool SSL recortado).
- **RESUME** del backbone SSL (skip pretrain) + checkpoint parcial cada 10 épocas.
- **Fix OOM:** `persistent_workers=False` + `gc.collect()` + `torch.cuda.empty_cache()` entre brazos.
- **SSL-primero:** el modelo entregable se exporta temprano; Kaggle commitea outputs parciales.
- Todo bloque secundario (linear probe, Grad-CAM, val externa) es **non-fatal** (try/except).
- `load_rgb()` + `LOAD_TRUNCATED_IMAGES` → una imagen corrupta no tumba el run.
- `predict_path()` para inferencia sobre cualquier ruta (base de la API/demo).
