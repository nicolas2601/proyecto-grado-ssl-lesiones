# Plan ejecutivo de pruebas para segmentación no supervisada por tono

## Regla metodológica

Los experimentos propuestos deben entrenar únicamente con datos sin etiquetas humanas, usando aprendizaje no supervisado/autosupervisado y pseudoetiquetas generadas automáticamente. No se propone fine-tuning con máscaras expertas, anotación activa, reponderación basada en etiquetas clínicas ni selección de hiperparámetros con el test. Las máscaras de referencia ya disponibles se pueden usar exclusivamente para evaluar checkpoints congelados; no se incorporan al entrenamiento ni a la generación/corrección de pseudo-máscaras.

El estudiante actual aprende contra pseudo-máscaras generadas por el propio pipeline. Por eso describo el método como **entrenamiento sin anotación humana con pseudoetiquetas algorítmicas**, no como supervisión cero en sentido matemático estricto.

## Resumen ejecutivo

La corrida final del 21 de septiembre produce Dice medio de **0,7847 en HAM10000** y **0,7177 en ISIC 2018** con MoCo v2, CCAM, pseudo-máscaras y un estudiante U-Net. En HAM10000, el contraste descriptivo entre los terciles relativos claro y oscuro de `L*` es **0,1114 Dice** (0,8616 vs. 0,7503). Es una disparidad medida en esa evaluación, no una conclusión sobre fototipos clínicos ni sobre población externa: HAM10000 estuvo en el pool no etiquetado. En ISIC 2018, el contraste es menor (**0,0554**, n=260 vs. n=158), pero la partición de 1.872 imágenes sin tono no está atestiguada por ID y el control por tamaño es mixto.

La adaptación tone-paired de clasificación tampoco es una solución demostrada: frente a su control plain pierde macro-F1 y balanced accuracy en ViT-S y ViT-B. Su causalidad no está establecida: hay un entrenamiento efectivo por brazo, las réplicas bootstrap son imágenes del mismo holdout y no seeds independientes.

**Ruta recomendada:** no lanzar otro barrido de umbrales ni de arquitecturas. Primero reconciliar por imagen los manifiestos y verificar el split externo. Después ejecutar dos auditorías sin entrenamiento: reproducir la brecha con el grupo `L*` correctamente asignado y medir qué píxeles de lesión modifica la transformación por tono. Solo si la brecha sigue en imágenes realmente fuera del pool, comparar una vez el pipeline actual con **USL-Net**, un método de self-learning no supervisado identificado en el survey principal. La evaluación final puede usar las máscaras de referencia existentes, pero estas quedan fuera de todo entrenamiento y selección.

## Qué se ejecutó

> **Nota: estimación ITA no validada (no es FST).** Lo que aquí aparece como «FST», «fototipo» o «bucket Fitzpatrick» por imagen es el ITA con los umbrales fijos de Groh (`src/dermssl/fitzpatrick.py`), y no está validado. Contra FST asignado por expertos acierta 28,5–30,8 % frente a 32,9 % de la baseline trivial en Fitzpatrick17k-C (11.098 imágenes, docs/20-validacion-externa-fst-fitzpatrick17k.md); en PAD-UFES-20, el ITA con umbrales fijos acierta 22,9 % frente a 51,3 % de la baseline de mayoría (Alipour et al., *Skin Research and Technology* 2026, PMC13459198; FST asignado por dermatólogo según ese artículo). Los terciles `tone_tercile` son terciles de luminancia (mediana de L\* de la piel), no tono de piel.

1. **Segmentación sin anotaciones humanas:** se preentrenó MoCo v2 desde cero (ResNet-50, 30 épocas) con 58.457 imágenes sin etiquetas. CCAM produjo mapas de activación; un umbral generó pseudo-máscaras ternarias (lesión >0,65, fondo <0,35, banda intermedia ignorada); luego el estudiante U-Net se entrenó en tres rondas de 20 épocas, a 224 × 224. Las máscaras de HAM10000 e ISIC 2018 se usaron para medir el resultado, no para entrenar el pipeline final.
2. **Medición de tono para segmentación:** se retiró la primera medición ITA por la inestabilidad de `b*` cerca de cero. El análisis final usa terciles relativos de mediana `L*` de la piel obtenida automáticamente. Son grupos relativos de luminosidad de este pool, no fototipos clínicos.
3. **Clasificación relacionada, pero distinta de la segmentación:** se comparó DINOv2 público con cuatro adaptaciones SimSiam autosupervisadas de 12 horas-GPU por brazo. Los pares tone-paired se generaron usando la pseudo-máscara para proteger los píxeles que ésta identifica como lesión. El probe logístico posterior usa diagnósticos como lectura de evaluación; no cambia el hecho de que la adaptación del backbone fue autosupervisada. Esa adaptación de clasificación no se propone como intervención para la brecha de segmentación.

## Resultados medibles hasta ahora

| Hecho observado | Resultado medible | Qué permite concluir |
|---|---|---|
| Segmentación final, HAM10000 | Dice medio **0,784658** [0,781202; 0,788259], n=10.015. | Medición del checkpoint final en HAM10000; no es prueba independiente de dominio porque HAM10000 estuvo en el pool sin etiquetas. |
| Terciles relativos de `L*`, HAM10000 | Claro **0,861643** [0,8535; 0,8698], n=864; oscuro **0,750279** [0,7440; 0,7565], n=4.082; diferencia de medias **0,111365**. | Hay una señal descriptiva grande en esta cohorte. El JSON no incluye un intervalo directo pareado para la diferencia; los intervalos separados no reemplazan ese contraste. Los seis de siete grupos diagnósticos mayoritarios/clases examinables también muestran caída claro→oscuro, pero las celdas pequeñas no permiten la misma inferencia. |
| Segmentación final, ISIC 2018 | Dice medio **0,717677** [0,709009; 0,725988], n=2.594. Claro **0,820450**, n=260; oscuro **0,765094**, n=158; diferencia **0,055357**. | El contraste por tono es menor. Solo 722 imágenes tienen tono asignado y el código usa disponibilidad de tono para formar la separación; no comprueba la pertenencia al pool por `image_id`. Las 1.872 restantes tienen Dice 0,6856, pero no un grupo de tono asignado. |
| Confusión por tamaño, ISIC 2018 | Las imágenes oscuras contienen 2,5 veces más lesiones grandes. En cobertura de referencia 0,25–0,60, el gap reportado es 0,1023 (claro n=113; oscuro n=45); no hay intervalo directo del gap. | El ajuste por tamaño deja una señal en un estrato, pero no confirma ni descarta por sí solo un efecto general. |
| Cobertura de máscaras | ISIC 2018: cobertura predicha 0,17 vs. referencia 0,21; se observa subsegmentación. | Es un detonante medible para revisar el borde y el error de máscara. No identifica su causa ni demuestra que la subsegmentación dependa de tono. |
| Ablación del tope de área | Cambiar `max_area_fraction` de 0,6 a 0,75 cambia Dice −0,0015 en ISIC 2018 y −0,0021 en HAM10000. | Esa modificación ya se probó y no mejoró el resultado agregado; no se propone repetirla como solución. |
| Máscaras elegibles | `pseudo_r2`: 58.457 imágenes, 2.508 excluidas, 55.949 retenidas por resta. Estimación ITA no validada (no es FST): 54.978 `ok` y 3.479 en cuarentena. Exportador de tono: 55.034 elegibles y 3.423 en cuarentena. | Las etapas tienen universos/filtros distintos y no están reconciliadas por ID. La diferencia de 56 entre exportador y FST está documentada; los 971 entre retención implícita de r2 y FST tampoco se pueden explicar sin el manifiesto por imagen. No deben mezclarse los conteos ni iniciar otro entrenamiento antes de aclararlos. |
| Clasificación, tone-paired vs. plain | ViT-S: Δ macro-F1 **−0,0467** [−0,0878; −0,0036], Δ balanced accuracy **−0,0404** [−0,0769; −0,0060]. ViT-B: Δ macro-F1 **−0,0619** [−0,1131; −0,0109], Δ balanced accuracy **−0,0655** [−0,1119; −0,0193]. Los intervalos de AUC/ECE incluyen cero. | La intervención tone-paired pierde en las dos métricas principales publicadas. El resultado no demuestra que las máscaras causen la pérdida; el n efectivo de entrenamiento es 1 por brazo. |
| Clasificación por tono | Brecha max–min de macro-F1: plain ViT-S 0,048 vs. proposed ViT-S 0,097; plain ViT-B 0,108 vs. proposed ViT-B 0,034. Los intervalos de los extremos se solapan en los seis backbones; el grupo light es n=183 frente a dark n=820 y medium n=957. | La dirección no es consistente entre tamaños y no se demuestra ni brecha ni mitigación de fairness en clasificación. |

### Interpretación, separada de los hechos

- **Apoyado:** la U-Net actual tiene una diferencia de Dice descriptiva de 0,1114 entre grupos relativos de luminosidad dentro de HAM10000; la adaptación tone-paired pierde frente a plain en macro-F1 y balanced accuracy.
- **Plausible, aún no demostrado:** falsos negativos de pseudo-máscara pueden dejar píxeles lesionados en la región que la transformación por tono modifica. La transformación preserva los píxeles que la pseudo-máscara marca como lesión; eso establece el camino técnico posible, no la causalidad sobre macro-F1.
- **No establecido:** que el gap represente fototipos clínicos; que se generalice a IDs externos; que una arquitectura supervisada o un ajuste de color lo resuelva; que la adaptación por tono mejore la clasificación; que haya una causa única para los conteos de máscaras distintos.

## Discusión bibliográfica aplicada a esta decisión

### El survey principal: mapa de modelos, no tabla de ganadores

Sharma, Sharma y Ghosal (2025), *Skin lesion segmentation: A systematic review of computational techniques, tools, and future directions*, DOI [10.1016/j.compbiomed.2025.110842](https://doi.org/10.1016/j.compbiomed.2025.110842), selecciona **108 artículos publicados entre 2018 y mayo de 2025** y organiza métodos clásicos y deep learning: FCN, U-Net, atención, GAN, Transformer, Mamba y otros (pp. 3–5). Es un mapa amplio de la literatura, no literalmente un inventario completo de todo modelo existente. Señala color, tamaño, forma, pelo, ruido, bajo contraste y bordes difusos como dificultades (pp. 2, 5). También observa protocolos distintos por dataset, resolución, augmentations, pérdidas y métricas (Tabla 5, pp. 9–11). Por eso sus resultados agregados no permiten declarar un ganador para esta corrida ni una mejora por tono.

Dos errores internos exigen cautela adicional: la ecuación que el survey rotula `Specificity` es `TP/(TP+FP)`, es decir, precision (p. 24); y en la Tabla 15 (p. 29) EIU-Net figura con JI 94,70 y Dice 91,6 para PH2, una relación inconsistente para la misma evaluación. No reutilizar esas columnas/rankings sin verificar los papers originales. El propio survey deja la validación cross-dataset y fairness entre tipos de piel como trabajo futuro (p. 27); no ofrece evidencia de que atención, Transformer o Mamba eliminen esta brecha.

### Qué literatura sí respeta la regla metodológica

- **USL-Net, Li et al. (2024):** *USL-Net: Uncertainty self-learning network for unsupervised skin lesion segmentation*, Biomedical Signal Processing and Control, 89, 105769, DOI [10.1016/j.bspc.2023.105769](https://doi.org/10.1016/j.bspc.2023.105769). El survey lo describe como encoder–decoder sin etiquetas manuales: combina representación contrastiva, mapas de activación, regiones foreground/background/uncertain y refinamiento iterativo de pseudoetiquetas (p. 19). Es el candidato que mejor encaja para una única comparación label-free. El resultado que publicó no es una comparación pareada en este pipeline ni una prueba de fairness; se debe medir de nuevo bajo la misma cohorte y protocolo del proyecto.
- **Benčević et al. (2024):** *Understanding skin color bias in deep learning-based skin lesion segmentation*, DOI [10.1016/j.cmpb.2024.108044](https://doi.org/10.1016/j.cmpb.2024.108044). La revisión del paper encontró grupos de tono medidos con ITA y análisis de sesgo; la estratificación simple y CIELAB no eliminaron significativamente el sesgo fuera de muestra. Implicación: tratar otra conversión de color o reequilibrio como hipótesis a probar, no como corrección probada. Sus grupos no convierten los terciles `L*` de este proyecto en fototipos clínicos.
- **EIU-Net, Yu et al. (2023):** DOI [10.1016/j.compbiomed.2023.107081](https://doi.org/10.1016/j.compbiomed.2023.107081). Es una variante supervisada con máscaras de referencia y no analiza fairness por tono; queda fuera del plan de entrenamiento por la regla metodológica, aunque aparezca en el survey.
- **Ye (EMBC 2025), active domain adaptation:** DOI [10.1109/EMBC58623.2025.11253341](https://doi.org/10.1109/EMBC58623.2025.11253341). Evalúa clasificación melanoma/nevus entre dominios con selección de etiquetas expertas. Sirve para contextualizar la diferencia entre domain adaptation y el experimento actual, pero requiere etiquetas target y no es segmentación; queda fuera de los escenarios propuestos.
- **SM3, aprendizaje multimodal autosupervisado:** el paper requiere modalidades clínicas y dermatoscópicas pareadas del mismo caso. No se traslada al pool actual de una sola modalidad; no se recomienda como siguiente prueba.

## Tres escenarios de prueba, formulados como preguntas

### 1. ¿La brecha de 0,1114 persiste cuando el grupo de tono y el holdout se verifican imagen por imagen?

**Qué se sabe:** HAM10000 estuvo dentro del pool sin etiquetas; el gap se calculó con terciles relativos de `L*`; el informe no publica un intervalo directo pareado del gap. En ISIC 2018, el código infiere parte de la partición in/out-of-pool desde disponibilidad de tono.

**Prueba label-free:** reconciliar por `image_id` el manifiesto de los 58.457 inputs con los CSV de evaluación y el manifiesto de tono. Si están disponibles los RGBs y el procedimiento automático actual produce una región de piel válida, recalcular los grupos claros/medios/oscuros para los 1.872 IDs marcados como tono no disponible; no crear etiquetas clínicas ni imputar fototipo. Si esos inputs/procedimiento no están disponibles o fallan, conservar `tono no asignado` y no afirmar independencia por tono. Congelar el checkpoint, evaluar cada ID una sola vez contra las máscaras de referencia oficiales y calcular el intervalo del gap claro–oscuro emparejado por lesión cuando `lesion_id` exista. Hacer análisis de sensibilidad del proxy con `L*` e ITA solo en los casos elegibles `b* ≥ 5`.

**Pregunta que decide:** ¿el gap aparece también en IDs cuyo no-solapamiento con el pool se demuestra directamente y se mantiene bajo los proxies automáticos predefinidos?

**Lectura:** sin atestación ID-a-ID, el resultado se reporta como partición indicada por README, no como test externo. Si el gap desaparece fuera del pool o depende de un solo proxy/covariable, la conclusión queda limitada a la asociación observada en HAM10000.

### 2. ¿Las pseudo-máscaras dejan píxeles de lesión en la región que transforma el brazo tone-paired?

**Qué se sabe:** la transformación conserva los píxeles que la pseudo-máscara marca como lesión y modifica los restantes; el brazo tone-paired pierde frente a plain en F1 macro y balanced accuracy. No se ha medido el solapamiento entre píxel lesionado transformado y máscara de referencia por imagen.

**Prueba sin reentrenamiento:** regenerar determinísticamente las vistas desde imagen, pseudo-máscara, código y semilla registrados; calcular por imagen la fracción de píxeles transformados, su intersección con la máscara de lesión de referencia, falsos negativos de pseudo-máscara y grupo relativo `L*`. Si los inputs/hashes del run no permiten reproducir esa vista, declararlo y no presentarla como una auditoría de los píxeles exactos de la corrida. Usar la máscara de referencia solo para esta medición pixelwise; nunca para alterar entrenamiento, corregir la pseudo-máscara ni seleccionar umbrales. Comparar claro/medio/oscuro y estratos de Dice/tamaño.

**Pregunta que decide:** ¿la fracción de lesión real alterada aumenta cuando cae la calidad de la pseudo-máscara, y se concentra en los grupos con peor Dice?

**Lectura:** un solapamiento medible confirma que el mecanismo de corrupción de vistas ocurre, no que explique por sí solo la pérdida de F1. Si no ocurre o no se relaciona con calidad/grupo, descartar ese detonante y no ejecutar otra variante de tone-pair.

### 3. ¿Un método autosupervisado con incertidumbre reduce la brecha sin etiquetas humanas?

**Condición de entrada:** solo ejecutar si el Escenario 1 confirma una brecha en IDs fuera del pool y las discrepancias de manifests quedan reconciliadas o explicadas. El Escenario 2 diagnostica por separado la alteración de vistas del brazo de clasificación; que no encuentre píxeles lesionados transformados no refuta por sí solo una brecha de segmentación.

**Comparación única:** pipeline actual `MoCo v2 + CCAM + pseudo-máscaras + U-Net` frente a USL-Net. Ambos reciben los mismos IDs del pool sin etiquetas humanas, la misma partición por imagen/lesión y el mismo límite de cómputo predefinido; cada receta conserva sus componentes propios y se congela antes de abrir el test. No entrenar con máscaras expertas ni etiquetas clínicas; no elegir hiperparámetros mirando el test. Evaluar checkpoints congelados con las máscaras de referencia reservadas únicamente para métricas. El resultado compara dos pipelines label-free completos; no aislaría causalmente el módulo de incertidumbre de USL-Net.

**Pregunta que decide:** ¿el pipeline label-free de USL-Net mejora Dice/HD95 globales y el gap de tono frente a la línea base, en seeds independientes?

**Lectura:** antes de adoptarlo, comprobar en la fuente primaria/implementación que ninguna etiqueta humana ni máscara de referencia entra en la ruta de entrenamiento. Ejecutar una configuración y el número de seeds fijado antes del test, estimado con la varianza de resultados disponibles. Si la receta no es reproducible, requiere etiquetas humanas, o el intervalo queda inconcluso, detenerse; no sustituirlo por una búsqueda de Transformers/Mamba ni seguir agregando seeds/modelos.

## Ruta recomendada, en orden

1. **Cerrar denominadores y split, sin GPU.** Resolver por `image_id` `pseudo_r2` (2.508 excluidas), exportador de tono (55.034/3.423) y FST (54.978/3.479). Si el manifiesto fuente no está disponible, registrar ese límite y no atribuir la discrepancia.
2. **Reproducir resultados por imagen, sin entrenar.** Exportar checkpoint, predicción, referencia, `image_id`, `lesion_id`, proxy `L*`/ITA y pertenencia al pool verificada. Añadir un intervalo directo del gap, no inferirlo por solapamiento de intervalos separados.
3. **Ejecutar las auditorías de los escenarios 1 y 2 una sola vez.** La primera resuelve si hay evidencia externa por tono; la segunda cuantifica si las vistas tone-paired modifican lesión real. No hacer ajuste de umbrales con el test.
4. **Tomar la decisión de GPU.** Si no hay gap externo, o si los IDs y conteos no pueden verificarse, cerrar la tesis con el resultado descriptivo/inconcluso, sin entrenar. Si el gap externo persiste y la procedencia de datos está reconciliada, correr una sola comparación de USL-Net frente a la línea base bajo datos sin etiquetas. El resultado del Escenario 2 se reporta aparte: no convertirlo en condición causal para explicar el gap de segmentación.
5. **Cerrar con resultados honestos.** Informar diferencia por seed e imagen, incertidumbre, cohortes verificadas, costo, límites del proxy de tono y si el gap bajó sin deteriorar Dice/HD95 global. No usar “fairness resuelta” salvo evidencia externa y método de tono validado.

## Reglas de parada

- No añadir entrenamiento supervisado, máscaras expertas, anotación activa ni etiquetas target a los brazos de aprendizaje.
- No seguir si no se reconcilian IDs/denominadores o no se demuestra qué imágenes están fuera del pool.
- No repetir un cálculo idéntico sin nueva evidencia; cada pregunta tiene una ejecución definida y una salida terminal.
- No abrir un “model zoo”: USL-Net es la única challenger porque corresponde a la restricción no supervisada/autosupervisada y tiene mecanismo de incertidumbre relevante para el fallo medido.
- Si la conclusión sigue incierta tras el número de seeds predefinido, reportar incertidumbre; no ampliar el experimento después de ver el resultado.

## Fuentes

1. Sharma, A. L., Sharma, K., & Ghosal, P. (2025). *Skin lesion segmentation: A systematic review of computational techniques, tools, and future directions*. Computers in Biology and Medicine, 196, 110842. [DOI](https://doi.org/10.1016/j.compbiomed.2025.110842). Survey principal, 108 artículos; discusión de métodos en pp. 3–19, métricas y limitaciones en pp. 22–29.
2. Li, X., Peng, B., Hu, J., Ma, C., Yang, D., & Xie, Z. (2024). *USL-Net: Uncertainty self-learning network for unsupervised skin lesion segmentation*. Biomedical Signal Processing and Control, 89, 105769. [DOI](https://doi.org/10.1016/j.bspc.2023.105769).
3. Benčević, M., Habijan, M., Galić, I., Babin, D., & Pižurica, A. (2024). *Understanding skin color bias in deep learning-based skin lesion segmentation*. Computer Methods and Programs in Biomedicine, 245, 108044. [DOI](https://doi.org/10.1016/j.cmpb.2024.108044).
4. Ye, J. (2025). *Enhancing Skin Lesion Classification Generalization with Active Domain Adaptation*. 47th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC). [DOI](https://doi.org/10.1109/EMBC58623.2025.11253341). Incluido solo para delimitar: clasificación ADA con etiquetas expertas, fuera del método permitido aquí.
5. Wang, H., Ahn, E., Bi, L., & Kim, J. (2025). *Self-supervised multi-modality learning for multi-label skin lesion classification*. Computer Methods and Programs in Biomedicine, 265, 108729. [DOI](https://doi.org/10.1016/j.cmpb.2025.108729). Autosupervisado, pero requiere pares clínico/dermatoscópicos y es clasificación, no segmentación del pool actual.
