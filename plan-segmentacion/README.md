# Plan: segmentación de lesiones autosupervisada (sin máscaras ni etiquetas)

Estado: implementación y pruebas del plan CNN sin etiquetas (Camino B) preparadas y staged, 2026-09-20. La validación en el clúster y la evaluación experimental siguen pendientes. Este documento no afirma éxito científico final; se actualizará con evidencia reproducible de los pilotos.

## 1. Contexto y decisión

En la reunión con el profesor (2026-09-18) se acordó hacer más pruebas en el servidor con este enfoque:

- Entrenar una CNN tipo U-Net **sin usar las máscaras de ISIC 2018 y sin usar etiquetas** de diagnóstico. El aprendizaje debe ser autosupervisado o no supervisado.
- Entrenar sobre imágenes sin etiquetar de **HAM10000, ISIC 2019 e ISIC 2020**.
- Con la máscara obtenida, calcular el **ITA y el tipo Fitzpatrick (FST)** siguiendo el flujo del paper de referencia (Corbin y Marques, IEEE Access 2023).
- Después, aplicar la **transformación de color** de ese paper y la **adaptación de dominio con DINOv2** (pesos de Hugging Face).

Alcance de este documento: **solo la etapa de segmentación**. La etapa de clasificación y preentrenamiento (SimSiam sobre DINOv2) la desarrolla Nicolás por separado; su prueba de infraestructura está en https://nicolaspruebasweb.vercel.app/.

Por qué importa la máscara: se usa solo para excluir la lesión antes de calcular el ITA de la piel sana. Lo crítico es separar bien lesión y piel, también en piel oscura. La precisión de los bordes es secundaria.

## 2. Diseño experimental

Dos caminos, con el mismo estudiante, de modo que la **única variable sea la fuente de pseudo-etiquetas**.

| | Camino A | Camino B |
|---|---|---|
| Fuente de pseudo-máscaras | Maestro DINOv2 (métodos espectrales tipo TokenCut o Deep Spectral Methods) | Solo CNN: preentrenamiento contrastivo y saliencia por CAM al estilo USL-Net |
| Conocimiento externo | Sí: DINOv2 viene preentrenado con datos ajenos | No: aprende solo de nuestras ~68k imágenes |
| Estudiante | U-Net | U-Net |
| Se compara después | DoubleU-Net, si el resultado lo justifica | DoubleU-Net, si el resultado lo justifica |

Referencias de comparación (pisos y techo):

- Piso clásico: Otsu, y el método de umbral adaptativo con normalización de color (Thanh et al., ICEEE 2019).
- Piso zero-shot: SAM con prompt central automático.
- Techo supervisado (solo referencia, nunca como competidor): U-Net entrenada con máscaras.

### Receta del camino B (según USL-Net, texto completo leído)

1. Características de SimCLR, MoCo v1 y MoCo v2 (ResNet50) concatenadas.
2. Mapa de saliencia con CCAM (mapas de activación contrastivos, sin etiquetas).
3. Módulo de incertidumbre: saliencia menor a 0,35 es fondo, mayor a 0,65 es lesión, y lo intermedio se ignora en la pérdida. Se conserva la componente conexa más grande y más central.
4. Estudiante ResNet50+FCN a 224x224, con rondas de re-etiquetado (5 rondas de 300 épocas en el paper).
5. Umbral de inferencia 0,5.

Fallos declarados por los autores: manchas de yodo, pelo, y sensibilidad a los umbrales 0,35 y 0,65. No hay análisis por tono de piel.

### Receta del camino A (propuesta, sin evidencia publicada en dermoscopía)

1. DINOv2 ViT-S/14 a 448 px sobre cada imagen.
2. Grafo de afinidad entre parches y bipartición con el vector de Fiedler.
3. Elegir como lesión la componente más pequeña o más central.
4. Ignorar en la pérdida los píxeles de confianza intermedia.
5. Destilar en la U-Net. Se ejecuta una sola vez, offline, sobre todo el pool.

Ningún paper revisado prueba un maestro DINOv2 espectral en dermoscopía. Que funcione en piel es una hipótesis a validar. Un ablation de USL-Net indica que DINO y DINOv2 con Grad-CAM++ rinden peor que ResNet50, pero eso no equivale a un maestro espectral.

### 2.1 Estructura de carpetas

```
plan-segmentacion/
  README.md                 este plan
  .gitignore                excluye artefactos pesados (salidas/, checkpoints, .npy)
  comun/                    lo que debe ser idéntico en A y B
    datos/                  inventario, redimensionado, caché, splits
    estudiante/             U-Net y bucle de entrenamiento sin máscaras
    evaluacion/             métricas, ITA/FST, ΔITA, estratificación por tono
    configs/                argumentos del piloto y del final
  baselines/                pisos de comparación
    otsu/  umbral-adaptativo/  sam/
  camino-a-dinov2/          maestro DINOv2 destilado en la U-Net
    pseudo-mascaras/  entrenamiento/  resultados/
  camino-b-cnn/             solo CNN, contrastivo + saliencia CAM
    preentrenamiento-contrastivo/  pseudo-mascaras/  entrenamiento/  resultados/
```

Criterio: los caminos A y B están separados, y todo lo que deben compartir vive en `comun/` para que la única variable sea la fuente de pseudo-etiquetas. Cada carpeta tiene su README con el alcance y los pendientes. El Camino B ya tiene implementación y pruebas staged; la ejecución y validación en el clúster, junto con la evaluación científica, siguen pendientes.

## 3. Datos

| Uso | Datos | Notas |
|---|---|---|
| Entrenamiento (sin máscaras ni etiquetas) | HAM10000, ISIC 2019, ISIC 2020 | ~68k imágenes |
| Test principal | ISIC 2018 Task 1 (máscaras solo para evaluar) | Nunca entra al entrenamiento ni a la selección de umbral o checkpoint |
| Test fuera de dominio | PH2 (200 imágenes dermatoscópicas) y Dermofit (1.300 imágenes) | Usados por Benčević para medir sesgo por tono; confirmar el tipo de imagen de Dermofit antes de usarlo |

Preparación: redimensionar y cachear las imágenes una sola vez (224 a 256 px) para no decodificar JPEG grandes en cada época. En el clúster, `/dev/shm` es de 64 MB y obliga a `workers=0`, así que la CPU se vuelve el cuello de botella.

## 4. Entrenamiento

- Optimizador Adam, umbral inicial 0,5, hiperparámetros a ajustar durante el piloto.
- Propuesta inicial: lr 1e-4 con decaimiento coseno, batch 32, BCE más Dice suave, precisión mixta, augmentation de color fuerte para que la red no dependa del color absoluto de la piel.
- **Piloto: 10 épocas**, para validar el pipeline y comparar A contra B. El entrenamiento final lo corre Nicolás en el servidor con el mismo script, cambiando solo los argumentos.
- Todo parametrizado (épocas, rondas, resolución, batch). El presupuesto se define en **imágenes procesadas**, no en épocas: 5 rondas de 300 épocas sobre 68k imágenes son ~102 millones de pasadas, y hay que medir la velocidad real antes de fijar el final.
- Con 10 épocas, el camino B no reproduce a USL-Net. Si rinde peor, la causa puede ser el presupuesto y no la idea, y se reporta así.
- Sin self-training en el piloto. Se puede agregar después como mejora (Noisy Student, Dzieniszewska 2024, aporta cerca de +0,8 a 1 punto de mIoU).

## 5. Evaluación

Métricas de segmentación: Dice, IoU (Jaccard), accuracy de píxel, sensibilidad, especificidad, Hausdorff-95 y F-score de borde.

Métricas del objetivo real (excluir la lesión antes del ITA):

- **ΔITA**: ITA calculado con la máscara predicha menos ITA con la máscara real.
- **Error de FST** y tasa de cambio de bucket.
- Correlación de Spearman con el ITA obtenido con la máscara real.

Estratificación por tono de piel: por ITA continuo o escala Monk además del FST, con Dice del peor grupo y brecha entre el mejor y el peor. Motivo: Ulrich 2025 halló que la conversión de ITA a Fitzpatrick es poco fiable (exactitud de 20 % en caras y 0,5 % en brazos, en imágenes clínicas), mientras que con Monk llega a 89 a 92 %.

Protocolo de ITA (Benčević 2024): excluir lesión, pelo y pigmentación, pasar a CIELAB y aplicar k-means a los píxeles de piel restantes. Fórmula: ITA = arctan((L* - 50) / b*) x 180 / pi.

Pruebas estadísticas: Wilcoxon de rangos con signo y corrección de Bonferroni. Se reporta el barrido de Dice contra umbral, pero el número principal es con umbral 0,5. También se reporta si un umbral por tono cerraría la brecha (Stothers 2025 halló hasta +24 Dice en FST VI con umbral por grupo).

### Escala de referencia para juzgar un "buen porcentaje"

Son test sets distintos, no comparables uno a uno, solo sirven como escala.

| Referencia | Valor | Fuente |
|---|---|---|
| Otsu | IoU 0,477 | Nagaoka 2025, ISIC 2018 |
| SAM con punto central | IoU 0,578 | Nagaoka 2025, ISIC 2018 |
| SLED (no supervisado) | Dice 73,9 | USL-Net Tabla 2, ISIC-2017 |
| USL-Net (sin etiquetas) | Dice 80,5, IoU 68,5 | Li et al., ISIC-2017 |
| U-Net supervisada simple | Dice ~0,88 | EIU-Net, ISIC 2018 |

## 6. Hardware disponible (clúster UNABIA, medido por Nicolás el 2026-09-17)

- 2x NVIDIA RTX PRO 4000 Blackwell, 24 GB cada una (48 GB en total, PCIe sin NVLink).
- Xeon w7-2495X (24 núcleos, 48 hilos), 123 GB de RAM.
- CUDA 12.8, torch 2.11.0+cu128, Python 3.11.16.
- Home NFS de 40 TB. Pedir 2 GPUs reduce el cupo de los demás, así que se usan solo para entrenar.

## 7. Fases

1. Preparar datos: inventario, redimensionado y caché.
2. Generar pseudo-máscaras: maestro DINOv2 (A) y saliencia CAM (B), más los pisos clásicos y SAM.
3. Piloto de 10 épocas con A y B, midiendo tiempo por época.
4. Evaluación en ISIC 2018, PH2 y Dermofit, con las métricas de la sección 5.
5. Definir el presupuesto final y correrlo con Nicolás en el servidor.
6. Si hace falta, comparar con DoubleU-Net.
7. Calcular ITA y FST con las máscaras resultantes y continuar con la transformación de color y la adaptación de dominio.

## 8. Riesgos y decisiones abiertas

- **Selección de checkpoint sin máscaras.** El código actual (`src/dermssl/torch_unet.py`, `train_unet`) elige el mejor checkpoint por Dice de validación con máscaras de ISIC 2018. Eso viola la regla de no usar máscaras. Hay que definir otro criterio (pérdida sobre pseudo-etiquetas, acuerdo entre maestros o estabilidad del ITA).
- **Circularidad.** DINOv2 sería maestro en la segmentación y también base de la adaptación de dominio. La comparación A contra B mide justamente si el conocimiento externo ayuda.
- **Color como señal.** Las pseudo-etiquetas basadas en color pueden heredar el sesgo por contraste lesión-piel (Paxton 2026) y correlacionar con el propio ITA.
- **Presupuesto del camino B.** A confirmar con Nicolás cuántas horas de GPU hay disponibles para el final.
- **Dataset de entrenamiento de USL-Net.** Falta confirmar en el paper con cuál entrena, para dimensionar una eventual reproducción.
- **Dermofit.** El paper de Benčević lo describe como dermatoscópico con estándares de color, pero conviene confirmarlo.
- **Pre-procesado.** Evaluar quitar pelo por inpainting (MFSNet) y color constancy, que ya aparece en el repositorio.

## 9. Qué se puede reciclar del repositorio (por revisar)

Solo una lista de candidatos. No se movió ni se modificó nada todavía.

- `src/dermssl/torch_unet.py`: arquitectura U-Net de ~7,77 M de parámetros, reutilizable como estudiante. Su bucle de entrenamiento es supervisado y no sirve tal cual.
- `src/dermssl/pretrain/lesion_guided.py`: fallback de Otsu, candidato para el piso clásico.
- `src/dermssl/fitzpatrick.py`: estimación de FST a partir de una máscara.
- `src/dermssl/mask_manifest.py` y `scripts/kaggle_fst_manifest.py`: manifiesto de máscaras.
- `scripts/diagnostico_fst_comparativa.py` y `scripts/validar_fitzpatrick17k.py`: diagnóstico y validación de FST.
- `src/dermssl/pretrain/color_constancy.py`: constancia de color.

## 10. Referencias

Leídas en texto completo:

- Li et al., USL-Net, Biomedical Signal Processing and Control, 2024. DOI 10.1016/j.bspc.2023.105769. arXiv 2309.13289.
- Benčević et al., Understanding skin color bias in deep learning-based skin lesion segmentation, Computer Methods and Programs in Biomedicine, 2024. DOI 10.1016/j.cmpb.2024.108044.
- Corbin y Marques, Assessing Bias in Skin Lesion Classifiers, IEEE Access, 2023. DOI 10.1109/ACCESS.2023.3289320.
- Ulrich et al., npj Digital Medicine, 2025. DOI 10.1038/s41746-025-01770-4.
- Nagaoka et al., Clinical, Cosmetic and Investigational Dermatology, 2025. DOI 10.2147/ccid.s518751.
- Dzieniszewska et al., Improving Skin Lesion Segmentation with Self-Training, Cancers, 2024. DOI 10.3390/cancers16061120.
- Yu et al., EIU-Net, Computers in Biology and Medicine, 2023. DOI 10.1016/j.compbiomed.2023.107081.
- Thanh et al., adaptive thresholding with normalization of color models, ICEEE 2019. DOI 10.1109/ICEEE2019.2019.00030.
- Jha et al., DoubleU-Net, 2020. arXiv 2006.04868.

Solo resumen o abstract (verificar antes de citar cifras):

- Zeng et al., SLED, ICDM 2023. DOI 10.1109/icdm58522.2023.00086.
- Stothers et al., MICCAI ISIC Workshop 2025. arXiv 2510.06655.
- Paxton et al., 2026. arXiv 2603.29694.
- Wang et al., TokenCut, arXiv 2209.00383. Melas-Kyriazi et al., Deep Spectral Methods, arXiv 2205.07839. Oquab et al., DINOv2, arXiv 2304.07193.

Descartadas como referencia: Innani et al. (el GAN "no supervisado" se entrena con máscaras reales), MFSNet (Dice de 0,987 en ISIC 2017 es inconsistente con otros trabajos), y el paper de ResUNet con ACO (split aleatorio no comparable con el oficial).

Nota sobre el paper base: Corbin y Marques **no entrenan** la segmentación. Usan una DoubleU-Net con pesos preentrenados en ISIC 2018 (supervisada), por eso no aporta hiperparámetros ni métricas de segmentación, y no existe en el paper una comparación entre U-Net y DoubleU-Net.
