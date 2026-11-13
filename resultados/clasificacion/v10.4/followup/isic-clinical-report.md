# Dataset clínico ISIC (fotos no dermoscópicas) — reporte

## Totales
- Registros clínicos crudos (close-up + overview): **9748**
- Excluidos por reglas: **8017** (ver tabla de exclusiones)
- Conservados tras exclusiones y mapeo de etiquetas: **501**
- Descargados OK (filas en rows.csv): **501**
- Fallos de descarga: **0**
- Fitzpatrick V–VI en rows.csv: **0**
- Tamaño en disco de images/: **94.1 MiB** (98640802 bytes)

## Conteo por etiqueta (rows.csv)
| Etiqueta | N |
|---|---|
| BCC | 154 |
| NEV | 88 |
| ACK | 17 |
| SEK | 26 |
| SCC | 71 |
| MEL | 123 |
| DF | 22 |
| VASC | 0 |

## Etiqueta × Fitzpatrick (rows.csv)
| Etiqueta | I | II | III | IV | V | VI | (vacío) | Total |
|---|---|---|---|---|---|---|---|---|
| BCC | 4 | 91 | 11 | 0 | 0 | 0 | 48 | 154 |
| NEV | 3 | 52 | 7 | 4 | 0 | 0 | 22 | 88 |
| ACK | 0 | 14 | 3 | 0 | 0 | 0 | 0 | 17 |
| SEK | 2 | 22 | 2 | 0 | 0 | 0 | 0 | 26 |
| SCC | 1 | 46 | 1 | 0 | 0 | 0 | 23 | 71 |
| MEL | 10 | 59 | 2 | 0 | 0 | 0 | 52 | 123 |
| DF | 2 | 17 | 3 | 0 | 0 | 0 | 0 | 22 |
| VASC | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Atribución (top 15, rows.csv)
| Atribución | N |
|---|---|
| Hospital Italiano de Buenos Aires | 345 |
| Anonymous | 100 |
| Memorial Sloan Kettering Cancer Center | 56 |

## Exclusiones
Cada regla se aplica en orden; «solapamiento» cuenta todas las coincidencias sin importar el orden.

| Regla | Coincidencias (solapamiento) | Eliminadas (orden secuencial) |
|---|---|---|
| R1 MILK10k (424/425) | 5719 | 5719 |
| R2 PAD-UFES-20 (406) | 2298 | 2298 |
| R3 lesion_id en exclude_milk_lesions.txt | 5137 | 0 |

Desglose por colección de la regla 1 y 2 (clínicas crudas):
| Colección | ID | Clínicas en colección |
|---|---|---|
| PAD-UFES-20 | 406 | 2298 |
| MILK10k Benchmark | 424 | 479 |
| MILK10k | 425 | 5240 |

Colecciones «Challenge …» (regla 4: se conservan si son clínicas):
| Colección | Clínicas crudas | Conservadas |
|---|---|---|
| (ninguna clínica en colecciones Challenge) | 0 | 0 |

Verificación de intersecciones (deben ser 0):
| Conjunto | Intersección |
|---|---|
| kept ∩ MILK10k+Benchmark+PAD (ids) | 0 |
| kept ∩ exclude_milk_lesions (lesion_id) | 0 |

Registros sin `diagnosis_3`: 1227

## Descargas fallidas
Ninguna.

## diagnosis_3 no mapeados (descartados; NO mapeados, para revisión)
| diagnosis_3 | N |
|---|---|
| Scar | 2 |
| Atypical melanocytic neoplasm | 1 |
