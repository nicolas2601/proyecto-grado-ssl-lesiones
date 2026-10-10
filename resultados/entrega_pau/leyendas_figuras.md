# Leyendas de las figuras

## fig_char_diagnosis_distribution

Distribución diagnóstica de HAM10000, ISIC 2019, ISIC 2020 y PAD-UFES-20.

Nota. Cada panel conserva la taxonomía original de su conjunto; las categorías no son equivalentes entre paneles. El eje es el porcentaje de las imágenes del conjunto completo (denominador = N del título de cada panel) y el número junto a cada barra es el conteo de imágenes. Describe la composición de los conjuntos, no la prevalencia poblacional.

## fig_char_ita_distribution

Distribución del ITA estimado en HAM10000, ISIC 2019, ISIC 2020 y PAD-UFES-20.

Nota. Histogramas apilados con intervalos contiguos de 10°; los colores son las categorías del estimador ITA (umbrales de Groh fijados en el código: 40, 23, 12, 0, −25), no fototipos clínicos Fitzpatrick. HAM10000, ISIC 2019 e ISIC 2020 usan la ROI perilesional; PAD-UFES-20 usa el fondo por Otsu y solo representa las imágenes con fototipo registrado en los metadatos. Las abstenciones (sin ITA válido) no aparecen en el histograma y se informan en cada panel. Cada panel tiene su propio eje vertical.

## fig_s2_dice_ci

Dice medio con IC 95 % de student_r3 y S4-A2 en ISIC 2018 y HAM10000.

Nota. Puntos: Dice medio por imagen; barras: IC 95 % bootstrap individual de cada modelo, no de la diferencia. El eje horizontal está truncado (0,70–0,82) para hacer visibles los intervalos. ISIC 2018 es parcialmente transductivo: 722 de sus imágenes estuvieron en el conjunto de preentrenamiento sin etiquetas.

## fig_s3_dice_distribution

Distribución del Dice por imagen de student_r3 y S4-A2 en ISIC 2018 y HAM10000.

Nota. Cada violín resume el Dice por imagen (marca interna: mediana; líneas: cuartiles) calculado sobre todas las imágenes de la evaluación; no se reconstruye desde medias.

## fig_t1_ita_geometry

Esquema conceptual del ángulo ITA a partir de las coordenadas CIELAB L* y b*.

Nota. ITA = arctan((L*−50)/b*) expresado en grados. El estimador reporta ITA cuando la mediana b* ≥ 5; de lo contrario, se abstiene. Las flechas y ángulos son esquemáticos, no corresponden a observaciones; el esquema no constituye validación clínica.

## tab_t2_ita_intervals

Reglas de pertenencia para grupos del estimador ITA, según los cortes de Groh.

Nota. Los grupos I–VI son categorías del estimador ITA; no equivalen a fototipos clínicos validados ni a fototipos Fitzpatrick. Cada corte exacto se asigna al grupo más oscuro. La estimación requiere mediana b* ≥ 5; los casos en abstención no reciben grupo.

## fig_t3_ita_vs_fitzpatrick

Exactitud exacta del estimador ITA frente al fototipo Fitzpatrick registrado en PAD-UFES-20, con la línea base de predecir siempre el fototipo mayoritario.

Nota. Exploratorio. Solo imágenes de PAD-UFES-20 con fototipo registrado en los metadatos (el autor de la asignación no está verificado); se excluyen las que el estimador no pudo evaluar. Barras: IC 95 % bootstrap agrupado por paciente, descriptivo. La línea de mayoría siempre predice el fototipo II. Este resultado no valida las máscaras.

## fig_t4_ita_violins

Distribución continua de ITA por imagen para HAM10000, ISIC 2019 e ISIC 2020, con el ROI perilesional de E5.

Nota. Cada violín representa ITA por imagen con `roi_status=ok` del estimador ROI de E5; el anillo perilesional se calcula alrededor de la lesión y sus ITA se expresan en grados. I–VI son categorías del estimador ITA; no equivalen a fototipos clínicos validados. Las abstenciones no entran a la distribución y se reportan con n. ISIC incluye y reporta por separado los valores válidos del fallback `no_mask_border`. PAD-UFES-20 usa Otsu y fototipo registrado; queda fuera de este violín y se cubre en `fig_char_ita_distribution` y T3. HAM10000: n ITA válido=4.539, n total=10.015, abstenciones=5.476, fallback no_mask_border válido=0; ISIC 2019: n ITA válido=12.558, n total=25.331, abstenciones=12.773, fallback no_mask_border válido=753; ISIC 2020: n ITA válido=21.973, n total=33.126, abstenciones=11.153, fallback no_mask_border válido=1.269.

## fig_c1_f1_macro_ham10000

F1 macro de B71 con TTA canónico frente a ocho baselines congelados re-evaluados en el mismo split v6/v7 (2.004 imágenes).

Nota. Las barras muestran F1 macro puntual y los bigotes, IC 95 % individuales publicados por cada fuente. No son intervalos de diferencias ni pruebas pareadas. Fila VGG19: ΔF1 +0,3331 se conserva según E3b/CONTRASTES_SSL.md; pendiente confirmación con Karen por posible medición anterior a 6bd96b7.

## fig_c1_f1_macro_isic2019

F1 macro del E1 canónico (batch 64 + autocast; n=5.067): B71 original transductivo y ocho baselines congelados con probe logreg; validación StratifiedKFold sin patient_id.

Nota. Las barras muestran F1 macro puntual y los bigotes, IC 95 % individuales publicados por cada fuente. No son intervalos de diferencias ni pruebas pareadas.

## fig_c1_f1_macro_isic2020

F1 macro del E1 canónico (batch 64 + autocast; n=6.660): B71 original transductivo y ocho baselines congelados con probe logreg; validación agrupada por patient_id.

Nota. Las barras muestran F1 macro puntual y los bigotes, IC 95 % individuales publicados por cada fuente. No son intervalos de diferencias ni pruebas pareadas.

## fig_c2_paired_delta_f1

C2: diferencias pareadas de F1 macro entre B71 y los controles con predicciones sobre las mismas imágenes. Panel (a), HAM10000, compara ocho baselines congelados; panel (b), ISIC, compara el control público B71.

Nota. N: HAM10000 2.004; ISIC 2019 5.067; ISIC 2020 6.660. Bootstrap pareado, 1.000 remuestras, semilla 20260923. Unidades: HAM lesion_id; ISIC 2019 imagen; ISIC 2020 patient_id. Sin ajuste por multiplicidad. VGG19: ΔF1 +0,3331 se conserva según E3b/CONTRASTES_SSL.md; pendiente confirmar con Karen si la medición es anterior a 6bd96b7.

## fig_c3_confusion_ham10000

Matriz de confusión de B71 (propuesto) en HAM10000 (n = 2.004), evaluación congelada, normalizada por clase real.

Nota. Filas: etiqueta real; columnas: predicción. Cada fila suma 100 %. Predicciones por argmax de las probabilidades publicadas; los conteos entre paréntesis son imágenes; el color y el primer número de cada celda son el porcentaje de la fila.

## fig_c3_confusion_isic2019

Matriz de confusión de B71 original (canónico, transductivo) en ISIC 2019 (n = 5.067), evaluación congelada, normalizada por clase real.

Nota. Filas: etiqueta real; columnas: predicción. Cada fila suma 100 %. Predicciones por argmax de las probabilidades publicadas; los conteos entre paréntesis son imágenes; el color y el primer número de cada celda son el porcentaje de la fila.

## fig_c3_confusion_isic2020

Matriz de confusión de B71 original (canónico, transductivo) en ISIC 2020 (n = 6.660), evaluación congelada, normalizada por clase real.

Nota. Filas: etiqueta real; columnas: predicción. Cada fila suma 100 %. Predicciones por argmax de las probabilidades publicadas; los conteos entre paréntesis son imágenes; el color y el primer número de cada celda son el porcentaje de la fila.

## fig_c4_per_diagnosis_ham10000

Precisión, sensibilidad y F1 por diagnóstico para B71 y A71 en la partición indicada. Con n = 2.004 en HAM10000

Nota. B71 y A71 congelados, partición común v6/v7. Filas = clase real; columnas = clase predicha. Precisión = TP/(TP+FP), sensibilidad = TP/(TP+FN) y F1 = 2TP/(2TP+FP+FN); precisión se fija en 0 si no hubo predicciones de una clase. Son estimaciones puntuales derivadas de conteos; no se muestran intervalos por clase. HAM10000 excluye DINOv2 ViT-B/14: su matriz usa n=2.014, distinto del split B71/A71 (n=2.004).

## fig_c4_per_diagnosis_isic2019

Precisión, sensibilidad y F1 por diagnóstico para B71 y A71 en la partición indicada. Con n = 5.067 en ISIC 2019

Nota. B71 y A71 originales transductivos, E1 canónico. Filas = clase real; columnas = clase predicha. Precisión = TP/(TP+FP), sensibilidad = TP/(TP+FN) y F1 = 2TP/(2TP+FP+FN); precisión se fija en 0 si no hubo predicciones de una clase. Son estimaciones puntuales derivadas de conteos; no se muestran intervalos por clase. ISIC 2019: B71/A71 son originales transductivos del E1 canónico; no son evaluación independiente.

## fig_c4_per_diagnosis_isic2020

Precisión, sensibilidad y F1 por diagnóstico para B71 y A71 en la partición indicada. Con n = 6.660 en ISIC 2020

Nota. B71 y A71 originales transductivos, E1 canónico. Filas = clase real; columnas = clase predicha. Precisión = TP/(TP+FP), sensibilidad = TP/(TP+FN) y F1 = 2TP/(2TP+FP+FN); precisión se fija en 0 si no hubo predicciones de una clase. Son estimaciones puntuales derivadas de conteos; no se muestran intervalos por clase. ISIC 2020: B71/A71 son originales transductivos del E1 canónico; no son evaluación independiente.

## fig_c5_pr_roc_ham10000

Curvas de precisión–sensibilidad y ROC de melanoma frente al resto en HAM10000 · E3b canónico.

Nota. Positivo: melanoma; negativo: resto. Precision–recall: prevalencia de melanoma de la partición como baseline; AP se calcula como suma de incrementos de sensibilidad por precisión en cada umbral único. ROC: diagonal de azar y AUC como área trapezoidal de los puntos empíricos. No se seleccionan umbrales en esta partición.

## fig_c5_pr_roc_isic2019

Curvas de precisión–sensibilidad y ROC de melanoma frente al resto en ISIC 2019 · E1 canónico.

Nota. Positivo: melanoma; negativo: resto. Precision–recall: prevalencia de melanoma de la partición como baseline; AP se calcula como suma de incrementos de sensibilidad por precisión en cada umbral único. ROC: diagonal de azar y AUC como área trapezoidal de los puntos empíricos. No se seleccionan umbrales en esta partición.

## fig_c5_pr_roc_isic2020

Curvas de precisión–sensibilidad y ROC de melanoma frente al resto en ISIC 2020 · E1 canónico.

Nota. Positivo: melanoma; negativo: resto. Precision–recall: prevalencia de melanoma de la partición como baseline; AP se calcula como suma de incrementos de sensibilidad por precisión en cada umbral único. ROC: diagonal de azar y AUC como área trapezoidal de los puntos empíricos. No se seleccionan umbrales en esta partición.

## fig_c5_pr_roc_isic2020_ft

Curvas de precisión–sensibilidad y ROC de melanoma frente al resto en ISIC 2020 · ajuste supervisado.

Nota. Positivo: melanoma; negativo: resto. Precision–recall: prevalencia de melanoma de la partición como baseline; AP se calcula como suma de incrementos de sensibilidad por precisión en cada umbral único. ROC: diagonal de azar y AUC como área trapezoidal de los puntos empíricos. No se seleccionan umbrales en esta partición.

## fig_c6_calibration_ham10000

Confiabilidad de las probabilidades por clase predicha en HAM10000 · E3b canónico.

Nota. Confianza = probabilidad máxima entre clases; acierto = argmax coincide con etiqueta real. Se usan 15 bins equidistantes: (lo, hi], salvo el primero [0, hi]. La línea diagonal indica calibración perfecta. La franja inferior muestra n por bin en la misma escala de confianza; bins con n < 10 usan marcador hueco. Los soportes exactos constan en el sidecar de trazabilidad.

## fig_c6_calibration_ham10000_v71

Confiabilidad de las probabilidades por clase predicha en HAM10000 · validación v7.1.

Nota. Confianza = probabilidad máxima entre clases; acierto = argmax coincide con etiqueta real. Se usan 15 bins equidistantes: (lo, hi], salvo el primero [0, hi]. La línea diagonal indica calibración perfecta. La franja inferior muestra n por bin en la misma escala de confianza; bins con n < 10 usan marcador hueco. Los soportes exactos constan en el sidecar de trazabilidad. El NPZ V7.1 no contiene IDs por imagen; la correspondencia exacta con el hash de partición no puede verificarse independientemente.

## fig_c6_calibration_isic2019

Confiabilidad de las probabilidades por clase predicha en ISIC 2019 · E1 canónico.

Nota. Confianza = probabilidad máxima entre clases; acierto = argmax coincide con etiqueta real. Se usan 15 bins equidistantes: (lo, hi], salvo el primero [0, hi]. La línea diagonal indica calibración perfecta. La franja inferior muestra n por bin en la misma escala de confianza; bins con n < 10 usan marcador hueco. Los soportes exactos constan en el sidecar de trazabilidad.

## fig_c6_calibration_isic2020

Confiabilidad de las probabilidades por clase predicha en ISIC 2020 · E1 canónico.

Nota. Confianza = probabilidad máxima entre clases; acierto = argmax coincide con etiqueta real. Se usan 15 bins equidistantes: (lo, hi], salvo el primero [0, hi]. La línea diagonal indica calibración perfecta. La franja inferior muestra n por bin en la misma escala de confianza; bins con n < 10 usan marcador hueco. Los soportes exactos constan en el sidecar de trazabilidad.

## fig_c6_calibration_isic2020_ft

Confiabilidad de las probabilidades por clase predicha en ISIC 2020 · ajuste supervisado.

Nota. Confianza = probabilidad máxima entre clases; acierto = argmax coincide con etiqueta real. Se usan 15 bins equidistantes: (lo, hi], salvo el primero [0, hi]. La línea diagonal indica calibración perfecta. La franja inferior muestra n por bin en la misma escala de confianza; bins con n < 10 usan marcador hueco. Los soportes exactos constan en el sidecar de trazabilidad.

## fig_c7_a71_vs_b71

Comparación pareada A71 frente a B71 en HAM10000; extracción y sonda cambian junto con el protocolo.

Nota. B71−A71 es una comparación pareada de protocolos agrupados, no un efecto aislado del tono: A71 usa una sopa de pesos de semillas 42 y 43; B71 usa pesos de semilla 42. También difieren en extracción (cls_attn vs cls4) y sonda (la0.5 vs balanced); ambos usan TTA. Bootstrap pareado por imagen (fila), 1.000 remuestras, semilla 20260923.

## fig_c7_b71_vs_b71_f4

Contraste pareado de F1 macro B71+F4 frente a B71 y sensibilidad del resultado por semilla.

Nota. F4 combina recorte de bbox y descriptores de máscara; no equivale solo a segmentación. El contraste E3 registrado es positivo, pero el IC excluye cero solo en ese cómputo. La sensibilidad por semillas tiene 0/3 IC que excluyen cero (lectura registrada: none); por tanto no se afirma mejora robusta. Semilla 42 procede de S1 batch64+autocast; las diferencias de SHA posteriores corresponden a campos MCC añadidos, con F1/IC idénticos. No se aíslan recorte ni descriptores por separado.

## fig_d1_pad_d0_d3

F1 macro con IC 95 % y sensibilidad por diagnóstico de D0 a D3 en PAD-UFES-20 (target_test, BCC/MEL/NV).

Nota. Evaluación única en PAD-UFES-20 target_test, lectura primaria BCC/MEL/NV: 567 imágenes de 363 pacientes (25 melanomas). Barras del panel (a): IC 95 % bootstrap pareado por patient_id, 1.000 remuestras. El panel (b) muestra sensibilidad puntual por diagnóstico, sin IC (el artefacto no los publica). D2 y D3 provienen de adaptaciones de 2.000 pasos y no equivalen a B71.

## fig_d2_pad_contrasts

Contrastes pareados de F1 macro en PAD-UFES-20 con IC 95 %: D3 − D2, D3 − D1 y D3 − D0.

Nota. Δ de F1 macro con IC 95 % bootstrap pareado por patient_id (1.000 remuestras, semilla 20260923), sin ajuste por multiplicidad. D3 − D2 es la lectura registrada (mejora frente al control adaptado); D3 − D0 compara con el DINOv2 público: el IC queda por debajo de cero, de modo que el público sigue siendo mejor.

## fig_a_cielab_ring

Distribución CIELAB de los píxeles del anillo perilesional en HAM10000, ISIC 2019 e ISIC 2020.

Nota. Población: píxeles del anillo perilesional de las imágenes cuya estimación ITA por anillo es válida (4.539 HAM10000; 11.805 ISIC 2019 y 20.704 ISIC 2020, sin las que usaron el borde de la imagen como alternativa): fuera de la máscara de lesión, dilatación de Chebyshev del 25 % del diámetro equivalente (mínimo 3 px), sin pelo, reflejos, píxeles saturados ni borde negro. Se usan todos los píxeles, sin submuestreo. (a) porcentaje de los píxeles de cada conjunto por intervalo de 2 unidades de L*; (b) porcentaje de los píxeles por celda de 4 × 4 unidades en a*–b*, escala logarítmica común.

## fig_t5_tone_pairs

Pares original/transformada CIRCLe con la misma máscara de lesión y el tercil de luminancia registrado.

Nota. El campo shift_buckets es s. El cambio L_shift = −8s se deriva con FST_SHIFT_L_STEP=8 L* por nivel y se muestra antes del clipping L*′=clip(L*−8s,0,100); no es un delta medido de píxeles. La máscara usa semántica true=lesion: la transformación actúa fuera de la lesión, que queda protegida. Los hashes source_sha256, transformed_sha256 y mask_sha256 se verifican contra los archivos. El tercil se calculó con la media de L del original sobre una muestra de 605 imágenes candidatas del servidor; no es el tercil de luminancia del manifiesto de tono del conjunto completo.

