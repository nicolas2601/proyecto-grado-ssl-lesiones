# 10 — Glosario y nomenclatura del proyecto

Este documento explica, en lenguaje sencillo, qué es cada nombre que aparece en el repositorio. Está pensado para leerse en orden, de arriba hacia abajo, sin conocer el código. Al final hay una tabla de equivalencias con los códigos antiguos, que solo sirve para ubicar archivos y reproducir resultados.

## 1. El proyecto en seis frases

1. **Problema.** A partir de una fotografía de una lesión de piel, un programa ayuda a distinguir de qué tipo de lesión se trata (por ejemplo, nevo o melanoma). Es una herramienta de investigación; no es un dispositivo médico ni diagnostica.
2. **Idea.** En lugar de enseñarle al programa con miles de etiquetas hechas por médicos, primero se le deja "mirar" muchas imágenes sin etiquetas para que aprenda a reconocer patrones de la piel (aprendizaje autosupervisado, SSL). Después se le enseña a clasificar con pocas etiquetas.
3. **Punto de partida.** Se parte de un modelo público llamado DINOv2, que ya sabe "ver" imágenes en general, y se adapta a imágenes de piel.
4. **Propuesta.** Adaptar el modelo de forma cuidadosa (sin que olvide lo que ya sabía) y, además, enseñarle a ser coherente cuando cambia el tono de la imagen. Ese es el modelo que llamamos **Proposed**.
5. **Comparaciones.** Cada propuesta se mide contra un **modelo de control** (el mismo método sin una de las piezas) y contra modelos de la literatura, siempre sobre las mismas imágenes y con intervalos de confianza.
6. **Lo que se puede y no se puede afirmar.** El aporte demostrado es la adaptación autosupervisada. El ajuste de tono no mostró un aporte demostrable, y la versión con segmentación sigue siendo exploratoria. Las cifras de un conjunto de datos no se trasladan automáticamente a otro.

## 2. Los modelos, con sus nombres

| Nombre en los documentos | Qué es, en palabras simples | Código antiguo |
|---|---|---|
| **Proposed sin segmentación** | La propuesta del proyecto. Mira la fotografía completa y decide la clase. No usa ninguna máscara que marque dónde está la lesión. Incluye la enseñanza de coherencia frente al tono. | `B71` (modelo) y `B*` (la familia de entrenamientos) |
| **Proposed con segmentación** | La misma propuesta, pero además usa la zona de la lesión (un recorte) y datos de su forma y color. Es **experimental**: su ventaja no fue estable al repetir el cálculo, así que no se presenta como mejora confirmada. | `B71 + F4` |
| **Modelo de control** | El mismo método de adaptación pero **sin** la enseñanza de coherencia frente al tono. Sirve para saber si esa pieza aporta algo. No usa segmentación. | `A71` (modelo) y `A*` (la familia de entrenamientos) |
| **DINOv2 público** | El modelo original de la literatura, sin adaptar a piel. Es la referencia con la que se compara. | `public`, `PUBLIC_*` |
| **Pesos limpios** | Versión de los modelos anteriores entrenada **sin** las imágenes que luego se usan para evaluar en ISIC. Así la evaluación es más honesta. Los pesos "originales" sí vieron esas imágenes (sin etiquetas) y se marcan como transductivos. | `*-limpio`, `clean` |

Solo las dos primeras filas son versiones de la propuesta. El modelo de control no es una "tercera propuesta" ni una versión segmentada: es un punto de comparación.

## 3. Los experimentos de segmentación

Segmentar es marcar dónde está la lesión dentro de la imagen. Se probaron varias redes que lo hacen **sin usar máscaras dibujadas por personas**: aprenden de máscaras generadas automáticamente (pseudo-máscaras).

| Nombre en los documentos | Qué se probó | Código antiguo |
|---|---|---|
| **Experimento de segmentación 1** | Una red U-Net aprendiendo de pseudo-máscaras, primera ronda. | `student_r1` |
| **Experimento de segmentación 2** | La misma red, segunda ronda: se vuelve a entrenar con las máscaras que ella misma predijo en la primera. | `student_r2` |
| **Experimento de segmentación 3** | La misma red, tercera ronda. Es la que se publicó y la que se usa en Proposed con segmentación. | `student_r3` |
| **Experimento de segmentación 4** | DINOv2 congelado más una pieza final que decide la máscara, entrenada con máscaras iniciales sacadas de su propia atención. Resultó peor que el 3. | `S4-A1` |
| **Experimento de segmentación 5** | DINOv2 congelado más la misma pieza final, entrenada con máscaras iniciales de otro método (CCAM). Quedó como candidato a segmentador y es ligeramente mejor que el 3. | `S4-A2` |

Los experimentos 1, 2 y 3 son rondas de **la misma red**, no tres redes distintas. La palabra "estudiante" (student) en los documentos viejos se refería a esa red: aprende de las máscaras que otro paso del proceso le da. Los números 1 a 5 son etiquetas de esta documentación y no cambian los nombres internos de las corridas.

## 4. Las pruebas realizadas

Cada prueba responde una pregunta. Aquí están todas con su nombre y su pregunta.

### 4.1 Pruebas del banco completo

| Nombre | Pregunta que responde | Código antiguo |
|---|---|---|
| Prueba de pesos limpios | ¿Cambian los resultados si el entrenamiento previo no ve las imágenes de validación? | `E1` |
| Prueba de adaptación a fotos clínicas | Si se adapta el modelo, ¿mejora en fotos tomadas con celular (PAD-UFES-20)? Compara cuatro modelos: referencia, adaptación inicial, control de entrenamiento y adaptación al nuevo conjunto. | `E2` y `D0`–`D3` |
| Prueba con segmentación | ¿Ayuda dar al clasificador la zona de la lesión? Se probaron cinco variantes: imagen y recorte, recorte de resolución original, vistas de la lesión, forma y color de la máscara, y piel alrededor. | `E3` y `F1`–`F5` |
| Prueba con segmentación en los modelos de comparación | Si una variante ayudara, ¿ayudaría también a los modelos de la literatura? Quedó condicionada a que antes ayudara. | `E3b` |
| Prueba de matrices de confusión | ¿Qué clases se confunden entre sí? Incluye además MCC y AUPRC. | `E4` |
| Prueba descriptiva del tono de piel | ¿Cómo se reparte el tono (estimado por color) en cada conjunto de datos? Es solo descripción. | `E5` |

### 4.2 Pruebas de segmentación y tono

| Nombre | Pregunta que responde | Código antiguo |
|---|---|---|
| Prueba de segmentación como entrada del clasificador | ¿Mejora clasificar si se le da la máscara al clasificador? Seis modelos por cuatro condiciones: sin máscara, recorte alrededor de la lesión, fondo gris y solo la lesión. | `C1` |
| Comparación emparejada con una condición ideal | ¿Cuánto del resultado se debe al ajuste de tono? Compara el modelo de control con Proposed bajo una condición ideal ("oráculo"). | `C3` |
| Prueba de segmentación sin entrenamiento | ¿Puede DINOv2 marcar la lesión sin entrenar nada? | `S-diag` |
| Mejora de máscaras: unir regiones | ¿Mejoran las máscaras si se aceptan varias regiones y no solo una? | `S1` |
| Mejora de máscaras: quitar pelo | ¿Mejoran si se elimina el pelo de la imagen antes de segmentar? (y su variante con filtro de forma) | `S2`, `S2'` |
| Mejora de máscaras: usar la confianza del tono | ¿Sirve usar el nivel de confianza de la estimación de tono? | `S3` |
| Segmentación con DINO y cabezal entrenado | ¿Funciona DINOv2 con una pieza final entrenada? (son los experimentos 4 y 5) | `S4` |
| Segmentación con red entrenada con pseudo-máscaras | ¿Se puede pasar el conocimiento de DINO a una red pequeña? | `S5` |
| Prueba de segmentación por luminancia | ¿Segmentan peor las imágenes más oscuras? | `T2` |
| Prueba de clasificación por luminancia | ¿Ayuda la máscara al clasificar las imágenes más oscuras? | `T3` |
| Prueba de constancia de color | ¿Cambia algo si se corrige el color de la imagen antes de analizarla? | `T4` (método Shades of Gray) |
| Prueba de umbral de máscara por luminancia | ¿Mejora la máscara si cada grupo de luminancia usa su propio umbral? | `T6` |
| Prueba de oscurecimiento | ¿Mejora la red si se entrena con imágenes artificialmente oscurecidas y más imágenes oscuras? | `T7` |
| Prueba del estimador de tono | ¿Coincide el tono estimado por color con el fototipo registrado por médicos en PAD-UFES-20? No coincide bien, por eso no se usa como fototipo. | `T1` |

### 4.3 Predicciones registradas

Antes de mirar los resultados se escribieron predicciones que luego se contrastaron.

| Nombre | Qué afirmaba | Código antiguo |
|---|---|---|
| Predicción de anclaje | Adaptar con anclaje no empeora respecto a DINOv2 público. | `P1`, y `P1-NI` para la versión estricta |
| Predicción de tono | El ajuste de tono mejora la calibración sin empeorar la clasificación. | `P2` |
| Predicción de objetivo | Proposed no pierde frente a DINOv2 público; y si el intervalo lo respalda, le gana. | `P3`, y `P3-IC` para la versión con intervalo |

## 5. Condiciones en que se midió

| Nombre | Qué significa | Código antiguo |
|---|---|---|
| **Repetición 42, 43, 44** | Un mismo entrenamiento repetido con distinta semilla aleatoria. Sirve para saber cuánto cambia el resultado solo por azar. No se elige la repetición que salga mejor. | `s42`, `s43`, `s44` |
| **Lote 16**, **lote 64**, **lote 64 en precisión completa** | Cuántas imágenes se procesan a la vez al extraer características. Cambia el cálculo en la tercera cifra decimal. El **lote 64** es el que se reporta como oficial. | `S0`, `S1`, `S2` en las tablas del banco |
| **Versión 4, 5, 6, 7, 7.1, 8** | Etapas del proyecto. v4: modelo de ResNet-50 entrenado en Kaggle (superado); v5: evaluación con el modelo congelado; v6: preentrenamiento con GPU; v7: rediseño con anclaje y tono; v7.1: el clasificador final se elige con un conjunto separado; v8: pesos limpios. | `v4` … `v8` |
| **Partición base de HAM10000** | Cómo se reparten las imágenes: 8.011 para entrenar y 2.004 para validar, sin que la misma lesión quede en ambos lados. La definió la prueba `C1`, por eso comparte código con ella. | `C1` |
| **Ronda 0, 1, 2, 3** | En la segmentación, cuántas veces se reentrenó la red con sus propias máscaras. | `r0` … `r3` |

## 6. Palabras técnicas, en orden alfabético

- **Ablación:** experimento que quita o cambia una sola pieza para ver si esa pieza aporta.
- **Abstención:** que el modelo diga "no sé, mejor que lo vea un especialista" en lugar de arriesgar una respuesta.
- **AUC:** mide qué tan bien el modelo ordena los casos (pone lo sospechoso arriba), sin fijar un umbral. 0,5 es azar y 1 es perfecto.
- **AUPRC:** parecido al AUC, pero más fiable cuando hay muy pocos casos positivos, como el melanoma en ISIC 2020.
- **Backbone:** la parte del modelo que "entiende" la imagen y la convierte en números. El clasificador se pone encima.
- **Bootstrap:** técnica que repite el cálculo muchas veces con imágenes tomadas al azar para estimar cuánto puede variar el resultado.
- **Brazo:** una de las familias de entrenamiento que se comparan (por ejemplo, con y sin ajuste de tono).
- **Calibración (ECE):** que cuando el modelo diga "80 % seguro" acierte de verdad cerca del 80 % de las veces. El ECE mide cuánto se aleja; menor es mejor.
- **CCAM:** método que dibuja un mapa de atención de la imagen y sirve para crear máscaras iniciales sin humanos.
- **Checkpoint:** archivo con los pesos de un modelo guardado en un momento del entrenamiento.
- **Cómputo canónico:** la forma de cálculo que se declara oficial (lote 64) para que todas las cifras sean comparables.
- **Contraste pareado:** comparar dos modelos sobre las mismas imágenes y mirar la diferencia. Si el intervalo de la diferencia incluye el cero, no se puede afirmar que uno sea mejor.
- **Dice e IoU:** miden cuánto se parece una máscara predicha a la máscara de referencia. Van de 0 a 1; mayor es mejor.
- **DINOv2:** modelo público de visión por computador. Aquí es el punto de partida.
- **Domain-adaptive:** tomar un modelo que ya sabe ver imágenes normales y acostumbrarlo a un tipo concreto de imágenes, como las de piel.
- **Época:** una pasada completa por todas las imágenes de entrenamiento.
- **F1 macro:** promedio del rendimiento por clase, sin dar más peso a las clases con más fotos. Es la métrica principal porque hay clases muy grandes y otras muy pequeñas. Mayor es mejor.
- **Fine-tuning (afinado):** reentrenar el modelo completo con las etiquetas. Lo contrario de dejarlo congelado.
- **Fitzpatrick:** escala clínica de tonos de piel (I a VI) según cómo reacciona al sol. No es lo mismo que el color de una fotografía.
- **Grad-CAM:** mapa de calor que muestra dónde está mirando el modelo.
- **Fuga de datos (leakage):** hacer trampa sin querer, por ejemplo si la misma lesión aparece en entrenamiento y en evaluación.
- **HD95:** mide cuánto se aleja el borde de la máscara predicha del borde real. Menor es mejor.
- **Holdout:** subconjunto reservado para elegir configuraciones sin tocar el conjunto con el que se evalúa al final.
- **Intervalo de confianza (IC 95 %):** rango en el que probablemente cae el valor real. Si dos intervalos se solapan, no se puede afirmar que uno gane.
- **ITA:** fórmula que estima un grupo de tono a partir del color de la piel en la imagen. **No es** fototipo ni se debe presentar como tal.
- **Lote (batch):** cuántas imágenes se procesan a la vez.
- **MCC:** resume en un número la matriz de confusión; va de -1 a 1 y 0 es azar.
- **Melanoma / nevo:** el melanoma es un cáncer de piel; el nevo es un lunar benigno. Distinguirlos es la tarea más importante del proyecto.
- **MoCo:** otra receta de aprendizaje autosupervisado, usada para la segmentación.
- **Modelo congelado:** sus pesos no cambian mientras se ajusta el clasificador.
- **Pesos:** los números internos que definen lo que aprendió el modelo.
- **Probe (clasificador lineal):** un clasificador sencillo que se coloca sobre el modelo congelado. Mide qué tan buenas son las características que aprendió el modelo.
- **Pseudo-máscara:** máscara de lesión generada automáticamente, no dibujada por una persona.
- **Segmentación:** marcar dónde está la lesión. No dice qué es la lesión.
- **Semilla:** número que fija el azar de un entrenamiento. Cambiarla produce una repetición distinta.
- **Sensibilidad de melanoma:** de todos los melanomas reales, cuántos detecta el modelo. Mayor es mejor.
- **SimSiam:** receta de aprendizaje autosupervisado que compara dos versiones deformadas de la misma foto y pide al modelo que las vea parecidas.
- **Sopa de pesos:** promediar los pesos de dos entrenamientos para obtener un modelo único.
- **SSL (aprendizaje autosupervisado):** enseñar al modelo a entender imágenes sin darle la respuesta.
- **TTA:** al predecir, mirar la imagen de varias formas (volteada, girada) y promediar.
- **Luminancia (L*):** qué tan clara u oscura es la imagen, medida sobre la piel. Se usa para agrupar imágenes sin hablar de tono de piel.
- **Tercil de luminancia:** se ordenan las imágenes de un conjunto de la más oscura a la más clara y se dividen en tres grupos iguales. **No es** tono de piel.
- **Transductivo:** el modelo vio las imágenes de evaluación, sin etiquetas, durante el preentrenamiento. Debe declararse porque la evaluación deja de ser totalmente independiente.
- **U-Net:** tipo de red usada para segmentar imágenes.
- **Umbral de screening:** en tamizaje se baja el umbral de decisión para no dejar pasar cánceres, aceptando más falsas alarmas.

### Más términos que aparecen en los informes

- **Autocast (precisión mixta):** hacer parte de los cálculos con menos decimales para ir más rápido.
- **Camino A y Camino B (segmentación):** dos líneas de trabajo. El Camino A segmenta usando DINOv2; el Camino B usa una red convolucional propia entrenada con pseudo-máscaras.
- **Clases de HAM10000:** `nv` = nevo (lunar benigno); `mel` = melanoma; `bkl` = lesión benigna parecida a una queratosis; `bcc` = carcinoma basocelular; `akiec` = queratosis actínica o enfermedad de Bowen; `vasc` = lesión vascular; `df` = dermatofibroma.
- **Cobertura:** qué fracción de la imagen ocupa la máscara que predice el modelo.
- **Cuarentena:** imágenes apartadas de un cálculo porque presentan algún problema; se cuentan aparte para no esconderlas.
- **Equidad entre grupos (fairness):** revisar si el modelo rinde parecido en grupos distintos, por ejemplo por luminancia de la imagen. Un rendimiento parecido no demuestra por sí solo ausencia de sesgo.
- **Gate (compuerta):** condición que debe cumplirse para que el proceso continúe al paso siguiente.
- **Kaggle y UNABIA:** Kaggle es una plataforma pública de cómputo; UNABIA es el cluster de GPU de la universidad.
- **Otsu:** método clásico que separa fondo y objeto según el brillo; se usa como alternativa sencilla cuando no hay máscara.
- **Prueba rápida de funcionamiento (smoke):** corrida con muy pocos datos para comprobar que todo arranca. No es un resultado.
- **ROI perilesional (anillo):** franja de piel que rodea la lesión; se usa para estimar el tono de la piel sin que la lesión lo distorsione.
- **Regresión logística (logreg):** el tipo de clasificador lineal que se coloca sobre el modelo congelado.
- **USL-Net:** método de la literatura para segmentar lesiones sin máscaras dibujadas por personas, en el que se basa el Camino B.
- **Validación y prueba:** la validación sirve para elegir configuraciones; la prueba se lee una sola vez al final para medir sin haber ajustado nada contra ella. Un *pliegue (fold)* es una de las particiones en que se reparten las imágenes.

## 7. Qué viene de la literatura y qué aportamos

- **De la literatura:** DINOv2, SimSiam, U-Net, MoCo, CCAM y la mayoría de las métricas. Mantienen su nombre.
- **Aporte del proyecto:** la forma de adaptar DINOv2 a piel sin que olvide lo que sabía, la enseñanza de coherencia frente al tono, la comparación con un modelo de control y la evaluación con pesos limpios.
- **Cuatro palabras que no son sinónimos:** *arquitectura* (la estructura de la red), *receta de entrenamiento* (cómo aprende), *checkpoint* (los pesos guardados) y *pipeline* (la cadena completa de procesamiento).

## 8. Equivalencias con los códigos antiguos

Los nombres de archivo, carpetas, claves de los resultados y comandos conservan estos códigos para no romper la reproducción. En los textos se usan los nombres de este glosario.

| Nombre actual | Código antiguo |
|---|---|
| Proposed sin segmentación | `B71`, `B*`, `B_s42…` |
| Proposed con segmentación | `B71 + F4` |
| Modelo de control | `A71`, `A*`, `A_soup…` |
| DINOv2 público con el protocolo del modelo de control / de Proposed | `public_A71`, `public_B71` |
| Experimento de segmentación 1, 2, 3 | `student_r1`, `student_r2`, `student_r3` |
| Experimento de segmentación 4, 5 | `S4-A1`, `S4-A2` |
| Prueba de pesos limpios, adaptación, segmentación, matrices, tono | `E1`, `E2`, `E3` (y `E3b`), `E4`, `E5` |
| Imagen y recorte; recorte original; vistas de la lesión; forma y color; piel alrededor | `F1`, `F2`, `F3`, `F4`, `F5` |
| Modelo de referencia; adaptación inicial; control de entrenamiento; adaptación al nuevo conjunto | `D0`, `D1`, `D2`, `D3` |
| Predicción de anclaje; de tono; de objetivo | `P1`, `P2`, `P3` |
| Repetición 42, 43, 44 | `s42`, `s43`, `s44` |
| Lote 16; lote 64; lote 64 en precisión completa | `S0`, `S1`, `S2` (tablas del banco) |

## 9. Reglas de lectura

- Los registros fechados (predicciones y protocolos escritos antes de calcular) conservan su texto original; lo que se sabía en esa fecha no se reescribe.
- Las cifras viven en archivos JSON y las tablas se generan desde ellos; no se editan a mano.
- Una cifra solo se interpreta junto a su conjunto de datos, su partición, su protocolo, su repetición y su estado (oficial o exploratorio).
- La simplificación de nombres no elimina controles, repeticiones ni resultados desfavorables.
