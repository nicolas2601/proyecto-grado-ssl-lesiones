# T1: la estimación ITA del proyecto contra el `fitspatrick` de PAD-UFES-20 (docs/30 §5)

Generado por `scripts/build_pad_ufes_fst_table.py` desde `pad_ufes_fst_validation.json`; no editar a mano (`--check` lo compara, y `tests/test_build_pad_ufes_fst_table.py` lo corre en CI). Exploratorio. `code_revision` 57a28d2f0ff9cd34dd821c1d32c3b29de812c825, registro `d351f36df4a525b27661eced45eefaca38c888b2`. 2.298 imágenes, 804 sin fototipo válido.

Nota: estimación ITA no validada (no es FST). Contra FST experto acierta 28,5–30,8 % frente a 32,9 % de la baseline trivial en Fitzpatrick17k-C (docs/20), y en PAD-UFES-20 el ITA con umbrales fijos acierta 22,9 % frente a 51,3 % de la baseline de mayoría (Alipour et al. 2026, PMC13459198).

## Métricas por condición

Las filas sin estimación (`status` distinto de `ok`) quedan fuera de las métricas. Los IC son bootstrap agrupado por paciente y son descriptivos.

| Condición | n evaluables | n sin estimación | Accuracy exacta [IC 95 %] | Off-by-one | Kappa lineal [IC 95 %] | Baseline de mayoría (fototipo · acc. · off-by-one) | ¿Supera la baseline? |
|---|---:|---:|---|---|---|---|---|
| Imagen completa | 1.265 | 229 | 0,2174 [0,1918; 0,2448] | 0,6213 | 0,0105 [-0,0232; 0,0405] | II · 0,5937 · 0,9510 | no |
| Fondo por Otsu (máscara) | 1.189 | 305 | 0,2288 [0,2012; 0,2588] | 0,7208 | 0,0256 [-0,0053; 0,0598] | II · 0,5980 · 0,9521 | no |

## Veredictos de las predicciones de docs/30 §5

- **FST-P1** (predicción primaria: en ninguna condición la accuracy exacta supera la baseline de mayoría en las mismas filas): **cumplida**.
- **FST-P2** (predicción secundaria: el kappa lineal queda por debajo de 0,20 en las dos condiciones): **cumplida**.

## Matriz de confusión (filas: fototipo de PAD; columnas: bucket ITA predicho)

### Imagen completa

| Real \ Predicho | I | II | III | IV | V | VI | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| I | 53 | 24 | 14 | 10 | 14 | 9 | 124 |
| II | 251 | 174 | 110 | 91 | 89 | 36 | 751 |
| III | 121 | 80 | 36 | 31 | 44 | 16 | 328 |
| IV | 18 | 9 | 6 | 10 | 7 | 2 | 52 |
| V | 4 | 1 | 1 | 1 | 2 | 0 | 9 |
| VI | 0 | 0 | 0 | 0 | 1 | 0 | 1 |

### Fondo por Otsu (máscara)

| Real \ Predicho | I | II | III | IV | V | VI | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| I | 76 | 21 | 9 | 6 | 5 | 1 | 118 |
| II | 419 | 161 | 63 | 36 | 26 | 6 | 711 |
| III | 188 | 56 | 29 | 16 | 13 | 1 | 303 |
| IV | 23 | 12 | 6 | 5 | 2 | 0 | 48 |
| V | 4 | 2 | 0 | 1 | 1 | 0 | 8 |
| VI | 0 | 0 | 0 | 0 | 1 | 0 | 1 |

## Nota sobre los fototipos

El grupo III-VI es 84 % tipo III (392/465), así que NO representa piel oscura. IV (62), V (10) y VI (1) van solo como descriptivo, sin veredicto. PAD-UFES-20 no permite concluir sobre FST V-VI.

## Datos y referencia

- Datos: Pacheco et al., *PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones*, Mendeley Data, v1, doi [10.17632/zr7vgbcyr2.1](https://doi.org/10.17632/zr7vgbcyr2.1); artículo en *Data in Brief* 32 (2020) 106221, doi [10.1016/j.dib.2020.106221](https://doi.org/10.1016/j.dib.2020.106221). Licencia CC BY 4.0 (<http://creativecommons.org/licenses/by/4.0>). Cambios: solo se usan las imágenes con `fitspatrick` válido; no se guarda ni se modifica ninguna imagen.
- Comparación: Alipour, Burke y Courtney, *Skin Research and Technology* 32(8):e70374, 2026, doi [10.1111/srt.70374](https://doi.org/10.1111/srt.70374), PMC13459198: con parches de piel, el ITA con umbrales fijos queda por debajo de la baseline de mayoría en PAD-UFES-20 (docs/30 §5).
