# Evaluación

Estado: primera etapa Camino B -> ITA/FST implementada mediante CLI standalone.

Único lugar donde se pueden leer las máscaras de referencia (ISIC 2018, PH2, Dermofit), y solo para medir.

Métricas previstas:

- Segmentación: Dice, IoU, accuracy de píxel, sensibilidad, especificidad, Hausdorff-95 y F-score de borde.
- Objetivo real: ΔITA (ITA con máscara predicha menos ITA con máscara real), error de FST, tasa de cambio de bucket y correlación de Spearman.
- Estratificación por ITA continuo o escala Monk además del FST, Dice del peor grupo y brecha entre el mejor y el peor.
- Wilcoxon de rangos con signo y corrección de Bonferroni. Barrido de Dice contra umbral, con el número principal a 0,5.

Protocolo de ITA (Benčević 2024): excluir lesión, pelo y pigmentación, pasar a CIELAB y aplicar k-means a la piel restante. ITA = arctan((L* - 50) / b*) x 180 / pi.

Candidatos del repositorio a revisar: `src/dermssl/fitzpatrick.py`, `scripts/diagnostico_fst_comparativa.py`, `scripts/validar_fitzpatrick17k.py` y `src/dermssl/metrics.py`.

## Pseudo-mascaras -> ITA/FST

> **Nota: estimación ITA no validada (no es FST).** Lo que aquí aparece como «FST», «fototipo» o «bucket Fitzpatrick» por imagen es el ITA con los umbrales fijos de Groh (`src/dermssl/fitzpatrick.py`), y no está validado. Contra FST asignado por expertos acierta 28,5–30,8 % frente a 32,9 % de la baseline trivial en Fitzpatrick17k-C (11.098 imágenes, docs/20-validacion-externa-fst-fitzpatrick17k.md); en PAD-UFES-20, el ITA con umbrales fijos acierta 22,9 % frente a 51,3 % de la baseline de mayoría (Alipour et al., *Skin Research and Technology* 2026, PMC13459198; FST asignado por dermatólogo según ese artículo). Los terciles `tone_tercile` son terciles de luminancia (mediana de L\* de la piel), no tono de piel.

El adaptador lee el `manifest.csv` de la cache, `pseudo_masks.npy` y el
`pseudo_masks_stats.json` opcional, y escribe el manifiesto canonico JSON por
imagen. La mascara de lesion es estrictamente `pseudo == 1`; los pixeles
`255` se excluyen del estimador y se contabilizan en `ignore_fraction`.

Desde la raiz del repositorio:

```bash
python plan-segmentacion/comun/evaluacion/fst_pseudo_manifest.py \
  --manifest salidas/cache/manifest.csv \
  --pseudo-masks salidas/pseudo_r0/pseudo_masks.npy \
  --output-json salidas/fst/fst_manifest.json \
  --source-revision pseudo-r0
```

Si existe, el CLI busca automaticamente `pseudo_masks_stats.json` junto a
`pseudo_masks.npy`. Se puede indicar otra ubicacion con `--stats-json`.

Cada registro conserva indice, dataset, imagen, split, ruta local, origen y
ronda de mascara, cobertura, fraccion ignorada, ITA, bucket de la estimación ITA no validada (no es FST), confianza,
calidad, estado y el registro de provenance de `mask_manifest`. Las filas con
cobertura insuficiente o un estado de estimacion distinto de `ok` quedan en
`quarantine` y no inventan ITA/FST. El resumen incluye `n`, estados,
estadisticos ITA y distribucion de buckets de la
estimación ITA por dataset.

Esta etapa no modifica `run_pipeline.py` y todavia no conecta DINOv2 ni
CIRCLe.
