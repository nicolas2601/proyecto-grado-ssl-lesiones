# Índice de la entrega

| Grupo | Figura | Pregunta | Estado | Archivo principal |
|---|---|---|---|---|
| 1_caracterizacion | fig_char_diagnosis_distribution | Cómo se reparten los diagnósticos en cada conjunto, con su taxonomía original. | Artefactos verificados | fig_char_diagnosis_distribution.svg |
| 1_caracterizacion | fig_char_ita_distribution | Cómo se distribuye el ITA estimado en cada conjunto y cuántas imágenes se abstienen. | Artefactos verificados | fig_char_ita_distribution.svg |
| 2_segmentacion | fig_s2_dice_ci | Qué tan cerca están student_r3 y S4-A2 en Dice medio, con su incertidumbre individual. | Artefactos verificados | fig_s2_dice_ci.svg |
| 2_segmentacion | fig_s3_dice_distribution | Cómo se reparte el Dice entre imágenes, más allá de la media. | Artefactos verificados | fig_s3_dice_distribution.svg |
| 3_tono | fig_t1_ita_geometry | Esquema conceptual del ángulo ITA a partir de las coordenadas CIELAB L* y b*. | Esquema conceptual (no es un resultado experimental) | fig_t1_ita_geometry.svg |
| 3_tono | tab_t2_ita_intervals | Reglas de pertenencia para grupos del estimador ITA, según los cortes de Groh. | Tabla derivada del código (cortes de Groh) | tab_t2_ita_intervals.tex |
| 3_tono | fig_t3_ita_vs_fitzpatrick | ¿El estimador ITA reproduce el fototipo registrado mejor que predecir siempre el fototipo mayoritario? | Artefactos verificados | fig_t3_ita_vs_fitzpatrick.svg |
| 3_tono | fig_t4_ita_violins | Distribución continua de ITA por imagen para HAM10000, ISIC 2019 e ISIC 2020, con el ROI perilesional de E5. | Artefactos verificados | fig_t4_ita_violins.svg |
| 4_clasificacion | fig_c1_f1_macro_ham10000 | F1 macro de B71 con TTA canónico frente a ocho baselines congelados re-evaluados en el mismo split v6/v7 (2.004 imágenes). | Artefactos verificados | fig_c1_f1_macro_ham10000.svg |
| 4_clasificacion | fig_c1_f1_macro_isic2019 | F1 macro del E1 canónico (batch 64 + autocast; n=5.067): B71 original transductivo y ocho baselines congelados con probe logreg; validación StratifiedKFold sin patient_id. | Artefactos verificados | fig_c1_f1_macro_isic2019.svg |
| 4_clasificacion | fig_c1_f1_macro_isic2020 | F1 macro del E1 canónico (batch 64 + autocast; n=6.660): B71 original transductivo y ocho baselines congelados con probe logreg; validación agrupada por patient_id. | Artefactos verificados | fig_c1_f1_macro_isic2020.svg |
| 4_clasificacion | fig_c2_paired_delta_f1 | C2: diferencias pareadas de F1 macro entre B71 y los controles con predicciones sobre las mismas imágenes. Panel (a), HAM10000, compara ocho baselines congelados; panel (b), ISIC, compara el control público B71. | Artefactos verificados | fig_c2_paired_delta_f1.svg |
| 4_clasificacion | fig_c3_confusion_ham10000 | Qué clases confunde B71 y con qué frecuencia, normalizado por clase real. | Artefactos verificados | fig_c3_confusion_ham10000.svg |
| 4_clasificacion | fig_c3_confusion_isic2019 | Qué clases confunde B71 y con qué frecuencia, normalizado por clase real. | Artefactos verificados | fig_c3_confusion_isic2019.svg |
| 4_clasificacion | fig_c3_confusion_isic2020 | Qué clases confunde B71 y con qué frecuencia, normalizado por clase real. | Artefactos verificados | fig_c3_confusion_isic2020.svg |
| 4_clasificacion | fig_c4_per_diagnosis_ham10000 | Precisión, sensibilidad y F1 por diagnóstico para B71 y A71 en la partición indicada. Con n = 2.004 en HAM10000 | Artefactos verificados | fig_c4_per_diagnosis_ham10000.svg |
| 4_clasificacion | fig_c4_per_diagnosis_isic2019 | Precisión, sensibilidad y F1 por diagnóstico para B71 y A71 en la partición indicada. Con n = 5.067 en ISIC 2019 | Artefactos verificados | fig_c4_per_diagnosis_isic2019.svg |
| 4_clasificacion | fig_c4_per_diagnosis_isic2020 | Precisión, sensibilidad y F1 por diagnóstico para B71 y A71 en la partición indicada. Con n = 6.660 en ISIC 2020 | Artefactos verificados | fig_c4_per_diagnosis_isic2020.svg |
| 4_clasificacion | fig_c5_pr_roc_ham10000 | Curvas de precisión–sensibilidad y ROC de melanoma frente al resto en HAM10000 · E3b canónico. | Cálculo adicional desde probabilidades por imagen | fig_c5_pr_roc_ham10000.svg |
| 4_clasificacion | fig_c5_pr_roc_isic2019 | Curvas de precisión–sensibilidad y ROC de melanoma frente al resto en ISIC 2019 · E1 canónico. | Cálculo adicional desde probabilidades por imagen | fig_c5_pr_roc_isic2019.svg |
| 4_clasificacion | fig_c5_pr_roc_isic2020 | Curvas de precisión–sensibilidad y ROC de melanoma frente al resto en ISIC 2020 · E1 canónico. | Cálculo adicional desde probabilidades por imagen | fig_c5_pr_roc_isic2020.svg |
| 4_clasificacion | fig_c5_pr_roc_isic2020_ft | Curvas de precisión–sensibilidad y ROC de melanoma frente al resto en ISIC 2020 · ajuste supervisado. | Cálculo adicional desde probabilidades por imagen | fig_c5_pr_roc_isic2020_ft.svg |
| 4_clasificacion | fig_c6_calibration_ham10000 | Confiabilidad de las probabilidades por clase predicha en HAM10000 · E3b canónico. | Cálculo adicional desde probabilidades por imagen | fig_c6_calibration_ham10000.svg |
| 4_clasificacion | fig_c6_calibration_ham10000_v71 | Confiabilidad de las probabilidades por clase predicha en HAM10000 · validación v7.1. | Cálculo adicional desde probabilidades por imagen | fig_c6_calibration_ham10000_v71.svg |
| 4_clasificacion | fig_c6_calibration_isic2019 | Confiabilidad de las probabilidades por clase predicha en ISIC 2019 · E1 canónico. | Cálculo adicional desde probabilidades por imagen | fig_c6_calibration_isic2019.svg |
| 4_clasificacion | fig_c6_calibration_isic2020 | Confiabilidad de las probabilidades por clase predicha en ISIC 2020 · E1 canónico. | Cálculo adicional desde probabilidades por imagen | fig_c6_calibration_isic2020.svg |
| 4_clasificacion | fig_c6_calibration_isic2020_ft | Confiabilidad de las probabilidades por clase predicha en ISIC 2020 · ajuste supervisado. | Cálculo adicional desde probabilidades por imagen | fig_c6_calibration_isic2020_ft.svg |
| 4_clasificacion | fig_c7_a71_vs_b71 | Comparación pareada A71 frente a B71 en HAM10000; extracción y sonda cambian junto con el protocolo. | Cálculo adicional desde contrastes por imagen | fig_c7_a71_vs_b71.svg |
| 4_clasificacion | fig_c7_b71_vs_b71_f4 | Contraste pareado de F1 macro B71+F4 frente a B71 y sensibilidad del resultado por semilla. | Cálculo adicional desde contrastes por imagen | fig_c7_b71_vs_b71_f4.svg |
| 5_dominio | fig_d1_pad_d0_d3 | Cómo se comportan D0–D3 en PAD-UFES-20 (fotos clínicas): F1 macro y sensibilidad por diagnóstico. | Artefactos verificados | fig_d1_pad_d0_d3.svg |
| 5_dominio | fig_d2_pad_contrasts | Si la adaptación mejora frente a su control (D3 − D2) y si supera al público (D3 − D0). | Artefactos verificados | fig_d2_pad_contrasts.svg |

## Pendientes (dependen del servidor de la UNAB)

- Figura A (CIELAB del anillo perilesional): cálculo en el servidor de la UNAB (HAM10000, ISIC 2019, ISIC 2020).
- T5 antes/después de la transformación de tono: faltan las imágenes originales/transformadas del servidor.
- S1 comparación cualitativa de máscaras: faltan imágenes y máscaras por caso del servidor.

## Requieren experimentos nuevos (sin gráfica)

- Ablaciones aisladas de F4 (recorte, descriptores, combinación): requieren ejecutar esas condiciones.
- Eficiencia de etiquetas (F1 frente a fracción etiquetada, con muestreos repetidos).
- Robustez a iluminación (perturbaciones controladas sobre las mismas imágenes).

## Limitaciones relevantes

- ITA no equivale al fototipo Fitzpatrick; no está validado como estimador de fototipo.
- La fila de VGG19 queda pendiente de confirmar con la Dra. Karen (posible medición anterior a 6bd96b7).
- C5/C6 no incluyen PAD-UFES-20: sus archivos de probabilidades no traen etiquetas por imagen.
- La transformación de tono no está demostrada como mejora de clasificación.
- ISIC 2018 es parcialmente transductivo; B71 original de ISIC 2019/2020 también (ver notas).
