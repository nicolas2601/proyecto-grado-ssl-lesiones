# Banco completo: Fitzpatrick descriptivo (E5)

Generado por `scripts/build_fitzpatrick_table.py` desde los JSON de `resultados/banco_completo/e5/`. No editar a mano. Plan y reglas: `docs/33-banco-completo-multidataset.md` (E5).

> **Aviso.** FST estimado por ITA, NO validado (22,9 % en PAD), pseudo-máscaras automáticas. Descriptivo y NO está validado: los buckets I-VI son la salida del estimador ITA con umbrales fijos, no el fototipo Fitzpatrick. Ese estimador acierta 22,9 % frente a 51,3 % de la baseline de mayoría en PAD-UFES-20 (Alipour et al., Skin Research and Technology 32(8):e70374, 2026, PMC13459198; docs/20, docs/30) y 28,5-30,8 % frente a 32,9 % en Fitzpatrick17k-C. Los terciles de L* son luminancia (mediana de L* de la piel), no es tono de piel, y son relativos a cada dataset: «oscuro» en uno no equivale a «oscuro» en otro. Ningún conteo de este archivo cambia el resultado primario del banco (docs/33, sección 5).

## Glosario

- **ITA (ángulo de tipología individual):** ángulo calculado de la mediana de L\* y b\* de la piel en CIELAB, `arctan((L\* − 50) / b\*)`. Es una medida de color, no de fototipo.
- **Bucket I-VI:** el tipo Fitzpatrick que resulta de cortar el ITA con los umbrales fijos de Groh (`fitzpatrick.FST_THRESHOLDS`). **Es la salida de un estimador no validado, no el fototipo real.** Un valor justo en un umbral va al tipo más oscuro.
- **Abstención:** la imagen no recibe bucket (columna «abstención») cuando no hay piel suficiente, la máscara es degenerada, b\* queda bajo el mínimo o el archivo no se pudo leer. Nunca se fuerza un bucket.
- **Estimador de Otsu (referencia):** quita la lesión con un umbral de Otsu y lee la piel del resto; es el mismo en los tres datasets, así que es el comparable.
- **Estimador de imagen completa:** todos los píxeles cuentan, sin quitar la lesión.
- **Anillo perilesional (ROI de Pau, `perilesional_roi`):** banda justo fuera de la máscara de la lesión, sin pelo, reflejos, píxeles saturados ni borde negro, con abstención. Las máscaras son **pseudo-máscaras automáticas** de `student_r3`: nativas 450 × 600 en HAM10000 y del pool 224 × 224 en ISIC.
- **Estado de la máscara:** `c1_native` (HAM10000), `eligible` (máscara del pool usable), `quarantine` (el manifiesto la rechazó, p. ej. área degenerada) o `absent` (la imagen no está en el pool).
- **Borde sin máscara (`no_mask_border`):** las imágenes sin máscara elegible usan el marco de la imagen como muestra de piel, con las mismas exclusiones y abstención. Se cuenta aparte: asume que el borde es piel, lo que una lesión en el margen o una viñeta rompen.
- **L\*:** luminosidad CIELAB (0 negro, 100 blanco). **Tercil de L\*:** el dataset se divide en tres tercios por la mediana de L\* de su piel; es luminancia, **no es tono de piel**, y sus cortes son propios de cada dataset.
- **Descriptivo:** solo conteos. No hay exactitud, ni intervalos, ni contrastes: no se compara contra ningún fototipo real.
- **n/d:** no disponible o no aplicable.

## Cómo leer esta tabla

- Los conteos por bucket describen lo que el estimador ve, no la composición de fototipos del dataset: contra fototipos asignados por dermatólogos acierta 22,9 % frente a 51,3 % de la baseline de mayoría en PAD-UFES-20 (PMC13459198; docs/20, docs/30).
- Los terciles son relativos a cada dataset: «oscuro» en HAM10000 y «oscuro» en ISIC 2020 no son la misma luminancia, y ninguno es piel oscura.
- Que dos estimadores coincidan (cuadro Otsu × ROI) no los valida: dos estimadores no validados pueden coincidir y equivocarse juntos. Es acuerdo, no exactitud.
- Los splits son los que ya existen en cada dataset (no se unifican, docs/33, sección 1): HAM10000 la partición de C1 agrupada por `lesion_id`, ISIC 2019 `StratifiedKFold` y ISIC 2020 agrupada por `patient_id`; «val» es el primer fold de cada una.
- ISIC 2020: split reproducible solo con scikit-learn 1.7.2 (guard en el loader).
- E5 no cambia el resultado primario del banco (docs/33, sección 5).

## HAM10000

- Imágenes: 10.015 (train 8.001, val 2.014); ilegibles: 0.
- Máscaras: Experimento de segmentación 3 nativas, sha256 `8b356ce0fb04`.
- Revisión de código: `48d5f63bd6ef197d138bb44ebb7732294bef7243`.

### Cobertura de los estimadores

| Estimador | ¿Aplica? | Con bucket | Abstención | % abstención | Motivos (estado: n) |
|---|---:|---:|---:|---:|---|
| ITA, lesión por Otsu (referencia) | sí | 4.339 | 5.676 | 56,7 % | ita_indeterminate: 5.676 |
| ITA, imagen completa | sí | 4.802 | 5.213 | 52,1 % | ita_indeterminate: 5.213 |
| ITA, ROI (anillo perilesional o borde sin máscara) | sí | 4.539 | 5.476 | 54,7 % | degenerate_mask: 26; ita_indeterminate: 5.377; ring_too_small: 73 |

### Distribución de buckets por estimador (conteo y % del total)

| Estimador | I | II | III | IV | V | VI | abstención |
|---|---:|---:|---:|---:|---:|---:|---:|
| ITA, lesión por Otsu (referencia) | 3.723 (37,2 %) | 464 (4,6 %) | 113 (1,1 %) | 31 (0,3 %) | 7 (0,1 %) | 1 (0,0 %) | 5.676 (56,7 %) |
| ITA, imagen completa | 3.662 (36,6 %) | 683 (6,8 %) | 253 (2,5 %) | 112 (1,1 %) | 70 (0,7 %) | 22 (0,2 %) | 5.213 (52,1 %) |
| ITA, ROI (anillo perilesional o borde sin máscara) | 3.644 (36,4 %) | 643 (6,4 %) | 171 (1,7 %) | 56 (0,6 %) | 23 (0,2 %) | 2 (0,0 %) | 5.476 (54,7 %) |

### Clase × bucket (Otsu)

| Clase | I | II | III | IV | V | VI | abstención | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| akiec | 109 | 2 | 0 | 0 | 0 | 0 | 216 | 327 |
| bcc | 125 | 2 | 3 | 0 | 0 | 0 | 384 | 514 |
| bkl | 177 | 11 | 4 | 0 | 0 | 0 | 907 | 1.099 |
| df | 40 | 2 | 0 | 0 | 0 | 0 | 73 | 115 |
| mel | 219 | 11 | 3 | 1 | 0 | 0 | 879 | 1.113 |
| nv | 3.022 | 435 | 102 | 30 | 7 | 1 | 3.108 | 6.705 |
| vasc | 31 | 1 | 1 | 0 | 0 | 0 | 109 | 142 |

### Bucket × split (Otsu)

| Split | I | II | III | IV | V | VI | abstención | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 2.981 | 359 | 94 | 27 | 6 | 1 | 4.533 | 8.001 |
| val | 742 | 105 | 19 | 4 | 1 | 0 | 1.143 | 2.014 |

### Cobertura de las máscaras

| Split | nativa C1 | elegible | en cuarentena | ausente | Total |
|---|---:|---:|---:|---:|---:|
| train | 8.001 | 0 | 0 | 0 | 8.001 |
| val | 2.014 | 0 | 0 | 0 | 2.014 |
| Total | 10.015 | 0 | 0 | 0 | 10.015 |

**Origen de la imagen del ROI por estado de máscara**

| Estado | pool 224 (`pool_224`) | fuente a 224 (`source_224`) | Total |
|---|---:|---:|---:|
| c1_native | 0 | 0 | 10.015 |

`source_224` = el archivo fuente redimensionado a 224 × 224 con la función que construyó el pool (`cache_io.load_resized_rgb`); se usa cuando el pool no exportó el original (cuarentena) o la imagen no está en el pool. **No es idéntica** a la imagen del pool: no se sabe si la caché del pool se construyó con quitar-pelo ni si leyó la misma versión del archivo, por eso se cuenta aparte.

Estimador del ROI por imagen: perilesional_roi: 10.015 imágenes · no_mask_border: 0 imágenes.

### Clase × bucket (ROI: anillo o borde)

| Clase | I | II | III | IV | V | VI | abstención | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| akiec | 121 | 11 | 1 | 0 | 0 | 1 | 193 | 327 |
| bcc | 128 | 15 | 2 | 0 | 0 | 0 | 369 | 514 |
| bkl | 215 | 22 | 10 | 3 | 1 | 0 | 848 | 1.099 |
| df | 49 | 4 | 0 | 0 | 0 | 0 | 62 | 115 |
| mel | 207 | 20 | 1 | 1 | 3 | 0 | 881 | 1.113 |
| nv | 2.896 | 570 | 156 | 52 | 19 | 1 | 3.011 | 6.705 |
| vasc | 28 | 1 | 1 | 0 | 0 | 0 | 112 | 142 |

### Bucket × split (ROI: anillo o borde)

| Split | I | II | III | IV | V | VI | abstención | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 2.913 | 504 | 137 | 48 | 19 | 2 | 4.378 | 8.001 |
| val | 731 | 139 | 34 | 8 | 4 | 0 | 1.098 | 2.014 |

### Otsu × ROI (conteos; filas = Otsu, columnas = ROI)

| Otsu \ ROI | I | II | III | IV | V | VI | abstención | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| I | 3.291 | 250 | 24 | 3 | 1 | 0 | 154 | 3.723 |
| II | 26 | 359 | 66 | 8 | 4 | 0 | 1 | 464 |
| III | 1 | 15 | 74 | 20 | 3 | 0 | 0 | 113 |
| IV | 0 | 1 | 3 | 21 | 6 | 0 | 0 | 31 |
| V | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 7 |
| VI | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| abstención | 326 | 18 | 4 | 4 | 1 | 2 | 5.321 | 5.676 |

Con bucket en ambos estimadores: 4.184 imágenes; mismo bucket en 3.752 (89,7 %). Es acuerdo entre dos estimadores no validados, no exactitud.

### Terciles de L\* (luminancia)

Cortes de L\* propios de este dataset (n = 10.015 con L\*): oscuro L\* ≤ 66,02; medio hasta 71,78; claro L\* > 71,78. Son luminancia, no tono de piel.

**Clase × tercil**

| Clase | oscuro | medio | claro | sin L\* | Total |
|---|---:|---:|---:|---:|---:|
| akiec | 65 | 109 | 153 | 0 | 327 |
| bcc | 71 | 183 | 260 | 0 | 514 |
| bkl | 504 | 344 | 251 | 0 | 1.099 |
| df | 18 | 44 | 53 | 0 | 115 |
| mel | 441 | 323 | 349 | 0 | 1.113 |
| nv | 2.204 | 2.300 | 2.201 | 0 | 6.705 |
| vasc | 36 | 35 | 71 | 0 | 142 |

**Tercil × split**

| Split | oscuro | medio | claro | sin L\* | Total |
|---|---:|---:|---:|---:|---:|
| train | 2.647 | 2.662 | 2.692 | 0 | 8.001 |
| val | 692 | 676 | 646 | 0 | 2.014 |

## ISIC 2019

- Imágenes: 25.331 (train 20.264, val 5.067); ilegibles: 0.
- Máscaras: pseudo-máscaras automáticas (Experimento de segmentación 3, sin etiquetas expertas) del pool de `salidas/tone/inputs`; manifiesto sha256 `fbbaa27e89c1` (58.457 entradas).
- Revisión de código: `7a029a8abcbb577fcc8d68c2f47bcb1897e6e8f3`.

### Cobertura de los estimadores

| Estimador | ¿Aplica? | Con bucket | Abstención | % abstención | Motivos (estado: n) |
|---|---:|---:|---:|---:|---|
| ITA, lesión por Otsu (referencia) | sí | 11.266 | 14.065 | 55,5 % | degenerate_mask: 19; ita_indeterminate: 14.046 |
| ITA, imagen completa | sí | 12.146 | 13.185 | 52,1 % | ita_indeterminate: 13.185 |
| ITA, ROI (anillo perilesional o borde sin máscara) | sí | 12.558 | 12.773 | 50,4 % | ita_indeterminate: 12.393; ring_too_small: 380 |

### Distribución de buckets por estimador (conteo y % del total)

| Estimador | I | II | III | IV | V | VI | abstención |
|---|---:|---:|---:|---:|---:|---:|---:|
| ITA, lesión por Otsu (referencia) | 8.751 (34,5 %) | 1.571 (6,2 %) | 562 (2,2 %) | 262 (1,0 %) | 95 (0,4 %) | 25 (0,1 %) | 14.065 (55,5 %) |
| ITA, imagen completa | 7.763 (30,6 %) | 2.053 (8,1 %) | 976 (3,9 %) | 646 (2,6 %) | 448 (1,8 %) | 260 (1,0 %) | 13.185 (52,1 %) |
| ITA, ROI (anillo perilesional o borde sin máscara) | 9.165 (36,2 %) | 1.930 (7,6 %) | 696 (2,7 %) | 413 (1,6 %) | 269 (1,1 %) | 85 (0,3 %) | 12.773 (50,4 %) |

### Clase × bucket (Otsu)

| Clase | I | II | III | IV | V | VI | abstención | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MEL | 1.442 | 223 | 85 | 36 | 18 | 4 | 2.714 | 4.522 |
| NV | 5.233 | 905 | 261 | 112 | 39 | 11 | 6.314 | 12.875 |
| BCC | 832 | 215 | 111 | 63 | 13 | 8 | 2.081 | 3.323 |
| AK | 228 | 41 | 17 | 19 | 8 | 0 | 554 | 867 |
| BKL | 738 | 153 | 74 | 27 | 16 | 2 | 1.614 | 2.624 |
| DF | 62 | 8 | 2 | 1 | 0 | 0 | 166 | 239 |
| VASC | 55 | 1 | 2 | 0 | 0 | 0 | 195 | 253 |
| SCC | 161 | 25 | 10 | 4 | 1 | 0 | 427 | 628 |

### Bucket × split (Otsu)

| Split | I | II | III | IV | V | VI | abstención | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 7.012 | 1.264 | 447 | 203 | 69 | 19 | 11.250 | 20.264 |
| val | 1.739 | 307 | 115 | 59 | 26 | 6 | 2.815 | 5.067 |

### Cobertura de las máscaras

| Split | nativa C1 | elegible | en cuarentena | ausente | Total |
|---|---:|---:|---:|---:|---:|
| train | 0 | 18.863 | 1.401 | 0 | 20.264 |
| val | 0 | 4.741 | 326 | 0 | 5.067 |
| Total | 0 | 23.604 | 1.727 | 0 | 25.331 |

**Origen de la imagen del ROI por estado de máscara**

| Estado | pool 224 (`pool_224`) | fuente a 224 (`source_224`) | Total |
|---|---:|---:|---:|
| eligible | 23.604 | 0 | 23.604 |
| quarantine | 0 | 1.727 | 1.727 |

`source_224` = el archivo fuente redimensionado a 224 × 224 con la función que construyó el pool (`cache_io.load_resized_rgb`); se usa cuando el pool no exportó el original (cuarentena) o la imagen no está en el pool. **No es idéntica** a la imagen del pool: no se sabe si la caché del pool se construyó con quitar-pelo ni si leyó la misma versión del archivo, por eso se cuenta aparte.

Estimador del ROI por imagen: perilesional_roi: 23.604 imágenes · no_mask_border: 1.727 imágenes.

Hash de la máscara (el del manifiesto: sha256 de la matriz booleana y su forma, `mask_manifest.mask_sha256`): coincide: 23.604; difiere: 0; no verificada (sin máscara elegible): 1.727.

### Clase × bucket (ROI: anillo o borde)

| Clase | I | II | III | IV | V | VI | abstención | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MEL | 1.350 | 270 | 123 | 67 | 54 | 30 | 2.628 | 4.522 |
| NV | 5.320 | 1.077 | 345 | 162 | 83 | 21 | 5.867 | 12.875 |
| BCC | 1.068 | 254 | 100 | 100 | 66 | 16 | 1.719 | 3.323 |
| AK | 292 | 74 | 22 | 22 | 16 | 6 | 435 | 867 |
| BKL | 793 | 198 | 89 | 53 | 40 | 9 | 1.442 | 2.624 |
| DF | 96 | 8 | 3 | 4 | 0 | 0 | 128 | 239 |
| VASC | 56 | 6 | 2 | 0 | 0 | 0 | 189 | 253 |
| SCC | 190 | 43 | 12 | 5 | 10 | 3 | 365 | 628 |

### Bucket × split (ROI: anillo o borde)

| Split | I | II | III | IV | V | VI | abstención | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 7.387 | 1.560 | 532 | 312 | 210 | 69 | 10.194 | 20.264 |
| val | 1.778 | 370 | 164 | 101 | 59 | 16 | 2.579 | 5.067 |

### Otsu × ROI (conteos; filas = Otsu, columnas = ROI)

| Otsu \ ROI | I | II | III | IV | V | VI | abstención | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| I | 7.314 | 663 | 105 | 34 | 26 | 28 | 581 | 8.751 |
| II | 150 | 1.048 | 234 | 74 | 32 | 9 | 24 | 1.571 |
| III | 4 | 89 | 300 | 126 | 30 | 5 | 8 | 562 |
| IV | 0 | 5 | 24 | 151 | 74 | 4 | 4 | 262 |
| V | 0 | 0 | 0 | 5 | 74 | 11 | 5 | 95 |
| VI | 0 | 0 | 0 | 0 | 6 | 15 | 4 | 25 |
| abstención | 1.697 | 125 | 33 | 23 | 27 | 13 | 12.147 | 14.065 |

Con bucket en ambos estimadores: 10.640 imágenes; mismo bucket en 8.902 (83,7 %). Es acuerdo entre dos estimadores no validados, no exactitud.

### Terciles de L\* (luminancia)

Cortes de L\* propios de este dataset (n = 25.312 con L\*): oscuro L\* ≤ 65,92; medio hasta 73,24; claro L\* > 73,24. Son luminancia, no tono de piel.

**Clase × tercil**

| Clase | oscuro | medio | claro | sin L\* | Total |
|---|---:|---:|---:|---:|---:|
| MEL | 1.356 | 1.318 | 1.847 | 1 | 4.522 |
| NV | 3.822 | 4.422 | 4.618 | 13 | 12.875 |
| BCC | 1.417 | 1.122 | 783 | 1 | 3.323 |
| AK | 377 | 303 | 185 | 2 | 867 |
| BKL | 1.111 | 865 | 646 | 2 | 2.624 |
| DF | 62 | 94 | 83 | 0 | 239 |
| VASC | 77 | 78 | 98 | 0 | 253 |
| SCC | 216 | 235 | 177 | 0 | 628 |

**Tercil × split**

| Split | oscuro | medio | claro | sin L\* | Total |
|---|---:|---:|---:|---:|---:|
| train | 6.740 | 6.725 | 6.782 | 17 | 20.264 |
| val | 1.698 | 1.712 | 1.655 | 2 | 5.067 |

## ISIC 2020

- Imágenes: 33.126 (train 26.466, val 6.660); ilegibles: 0.
- Máscaras: pseudo-máscaras automáticas (Experimento de segmentación 3, sin etiquetas expertas) del pool de `salidas/tone/inputs`; manifiesto sha256 `fbbaa27e89c1` (58.457 entradas).
- Revisión de código: `7a029a8abcbb577fcc8d68c2f47bcb1897e6e8f3`.

### Cobertura de los estimadores

| Estimador | ¿Aplica? | Con bucket | Abstención | % abstención | Motivos (estado: n) |
|---|---:|---:|---:|---:|---|
| ITA, lesión por Otsu (referencia) | sí | 21.692 | 11.434 | 34,5 % | degenerate_mask: 88; ita_indeterminate: 11.346 |
| ITA, imagen completa | sí | 22.824 | 10.302 | 31,1 % | ita_indeterminate: 10.302 |
| ITA, ROI (anillo perilesional o borde sin máscara) | sí | 21.973 | 11.153 | 33,7 % | ita_indeterminate: 6.465; ring_too_small: 4.688 |

### Distribución de buckets por estimador (conteo y % del total)

| Estimador | I | II | III | IV | V | VI | abstención |
|---|---:|---:|---:|---:|---:|---:|---:|
| ITA, lesión por Otsu (referencia) | 15.699 (47,4 %) | 2.928 (8,8 %) | 1.226 (3,7 %) | 845 (2,6 %) | 748 (2,3 %) | 246 (0,7 %) | 11.434 (34,5 %) |
| ITA, imagen completa | 15.080 (45,5 %) | 2.990 (9,0 %) | 1.501 (4,5 %) | 1.141 (3,4 %) | 1.334 (4,0 %) | 778 (2,3 %) | 10.302 (31,1 %) |
| ITA, ROI (anillo perilesional o borde sin máscara) | 14.569 (44,0 %) | 3.266 (9,9 %) | 1.613 (4,9 %) | 1.131 (3,4 %) | 971 (2,9 %) | 423 (1,3 %) | 11.153 (33,7 %) |

### Clase × bucket (Otsu)

| Clase | I | II | III | IV | V | VI | abstención | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benign | 15.427 | 2.852 | 1.186 | 812 | 720 | 240 | 11.305 | 32.542 |
| malignant | 272 | 76 | 40 | 33 | 28 | 6 | 129 | 584 |

### Bucket × split (Otsu)

| Split | I | II | III | IV | V | VI | abstención | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 12.699 | 2.328 | 946 | 658 | 602 | 212 | 9.021 | 26.466 |
| val | 3.000 | 600 | 280 | 187 | 146 | 34 | 2.413 | 6.660 |

### Cobertura de las máscaras

| Split | nativa C1 | elegible | en cuarentena | ausente | Total |
|---|---:|---:|---:|---:|---:|
| train | 0 | 25.095 | 1.371 | 0 | 26.466 |
| val | 0 | 6.335 | 325 | 0 | 6.660 |
| Total | 0 | 31.430 | 1.696 | 0 | 33.126 |

**Origen de la imagen del ROI por estado de máscara**

| Estado | pool 224 (`pool_224`) | fuente a 224 (`source_224`) | Total |
|---|---:|---:|---:|
| eligible | 31.430 | 0 | 31.430 |
| quarantine | 0 | 1.696 | 1.696 |

`source_224` = el archivo fuente redimensionado a 224 × 224 con la función que construyó el pool (`cache_io.load_resized_rgb`); se usa cuando el pool no exportó el original (cuarentena) o la imagen no está en el pool. **No es idéntica** a la imagen del pool: no se sabe si la caché del pool se construyó con quitar-pelo ni si leyó la misma versión del archivo, por eso se cuenta aparte.

Estimador del ROI por imagen: perilesional_roi: 31.430 imágenes · no_mask_border: 1.696 imágenes.

Hash de la máscara (el del manifiesto: sha256 de la matriz booleana y su forma, `mask_manifest.mask_sha256`): coincide: 31.430; difiere: 0; no verificada (sin máscara elegible): 1.696.

### Clase × bucket (ROI: anillo o borde)

| Clase | I | II | III | IV | V | VI | abstención | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benign | 14.331 | 3.168 | 1.568 | 1.091 | 939 | 404 | 11.041 | 32.542 |
| malignant | 238 | 98 | 45 | 40 | 32 | 19 | 112 | 584 |

### Bucket × split (ROI: anillo o borde)

| Split | I | II | III | IV | V | VI | abstención | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 11.710 | 2.602 | 1.263 | 894 | 782 | 365 | 8.850 | 26.466 |
| val | 2.859 | 664 | 350 | 237 | 189 | 58 | 2.303 | 6.660 |

### Otsu × ROI (conteos; filas = Otsu, columnas = ROI)

| Otsu \ ROI | I | II | III | IV | V | VI | abstención | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| I | 11.926 | 1.183 | 112 | 41 | 23 | 15 | 2.399 | 15.699 |
| II | 116 | 1.964 | 656 | 142 | 40 | 6 | 4 | 2.928 |
| III | 0 | 58 | 775 | 341 | 42 | 9 | 1 | 1.226 |
| IV | 0 | 1 | 44 | 559 | 226 | 14 | 1 | 845 |
| V | 0 | 0 | 1 | 26 | 608 | 111 | 2 | 748 |
| VI | 0 | 0 | 0 | 0 | 6 | 239 | 1 | 246 |
| abstención | 2.527 | 60 | 25 | 22 | 26 | 29 | 8.745 | 11.434 |

Con bucket en ambos estimadores: 19.284 imágenes; mismo bucket en 16.071 (83,3 %). Es acuerdo entre dos estimadores no validados, no exactitud.

### Terciles de L\* (luminancia)

Cortes de L\* propios de este dataset (n = 33.038 con L\*): oscuro L\* ≤ 70,03; medio hasta 83,13; claro L\* > 83,13. Son luminancia, no tono de piel.

**Clase × tercil**

| Clase | oscuro | medio | claro | sin L\* | Total |
|---|---:|---:|---:|---:|---:|
| benign | 10.688 | 10.810 | 10.957 | 87 | 32.542 |
| malignant | 325 | 202 | 56 | 1 | 584 |

**Tercil × split**

| Split | oscuro | medio | claro | sin L\* | Total |
|---|---:|---:|---:|---:|---:|
| train | 8.732 | 8.964 | 8.705 | 65 | 26.466 |
| val | 2.281 | 2.048 | 2.308 | 23 | 6.660 |
