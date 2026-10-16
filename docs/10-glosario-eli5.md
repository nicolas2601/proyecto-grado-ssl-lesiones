# 10 — Glosario y nomenclatura del proyecto

## Nombres oficiales de las dos versiones

Esta nomenclatura describe las entradas del sistema, no cambia los experimentos ni sus resultados. Solo hay dos nombres de presentación:

| Nombre de presentación | Explicación sencilla | Definición técnica y correspondencia histórica |
|---|---|---|
| **Proposed sin segmentación** | Clasifica la fotografía completa, sin usar una máscara para localizar la lesión. | Configuración registrada B71: adaptación SSL con consistencia de tono sobre DINOv2, extracción de características y clasificador lineal. B71 es un identificador de trazabilidad, no el nombre público. |
| **Proposed con segmentación** | Usa la fotografía completa y la región de lesión localizada por un segmentador; agrega información de su forma y color. | Variante experimental B71 + F4 de E3: combina características de imagen completa y recorte con margen, más descriptores derivados de la máscara de lesión de la configuración `student_r3`. No clasifica únicamente la máscara. |

**Estado de la evidencia:** la versión con segmentación es exploratoria. El registro de docs/33 retiró su adopción como mejora confirmada porque la ventaja no fue robusta al cómputo. El nombre de presentación no convierte ese resultado en una mejora demostrada ni reemplaza el resultado principal sin segmentación. Tampoco garantiza desempeño clínico.

**No son dos nombres nuevos para A71 y B71:** A71 no usa segmentación en su configuración registrada. Es un experimento de control de adaptación SSL sin el término de consistencia de tono; B71 sí incluye ese término. Comparar A71 contra B71 no aísla el efecto de segmentar y sus protocolos seleccionados también difieren. Para estudiar segmentación se consulta la comparación pareada de B71 + F4 contra B71 bajo el mismo protocolo de E3.

### Experimentos, no propuestas adicionales

Los códigos se conservan para localizar archivos y reproducir resultados. No son marcas de modelos ni versiones adicionales llamadas Proposed.

| Código histórico | Nombre descriptivo del experimento | Qué identifica y qué no identifica |
|---|---|---|
| A*, A71 | Control de adaptación SSL sin consistencia de tono | A* es el brazo de entrenamiento; A71 es su configuración de evaluación seleccionada en v7.1. No es la versión con segmentación. |
| B*, B71 | Adaptación SSL con consistencia de tono | B* identifica el brazo de entrenamiento y B71 su configuración seleccionada. Proposed sin segmentación corresponde a B71, no a cualquier ejecución B*. |
| A71/B71 originales y limpios | Condiciones de procedencia del preentrenamiento | Los limpios excluyen imágenes de validación ISIC del pool SSL. No son arquitecturas nuevas ni propuestas adicionales. Los originales en ISIC requieren declarar su carácter transductivo. |
| `student_r1`, `student_r2`, `student_r3` | Experimento de segmentación U-Net, rondas de aprendizaje 1, 2 y 3 | Son etapas del aprendizaje con pseudo-máscaras y re-etiquetado; no tres arquitecturas distintas. La configuración publicada es la tercera ronda. `r0` en pseudo-máscaras identifica la generación inicial, no un cuarto modelo final. |
| S4-A1 | Experimento de segmentación con pseudo-máscaras de saliencia DINOv2 | Encoder DINOv2 congelado y decoder tipo U-Net; no confundir A1 de segmentación con A1 del piloto de clasificación. |
| S4-A2 | Experimento de segmentación con pseudo-máscaras iniciales CCAM | Encoder DINOv2 congelado y decoder tipo U-Net. Es candidato de segmentación evaluado por separado; no reemplaza automáticamente `student_r3` en E3/F4. |
| F1–F5, E3 | Experimentos de integración de segmentación en clasificación | F1 combina imagen y recorte; F2 cambia la resolución de origen del recorte; F3 usa vistas guiadas por máscara; F4 agrega forma y color a F1; F5 agrega anillo perilesional a F1. Son condiciones, no cinco propuestas finales. |
| D0–D3, E2 | Experimentos de transferencia a fotografías clínicas PAD-UFES-20 | D0: referencia pública; D1: adaptación A* limpia; D2: continuación en fuente como control de cómputo; D3: continuación con imágenes fuente y destino sin etiquetas del destino. No son versiones con/sin segmentación. |
| s42, s43, s44 | Repeticiones con semillas aleatorias distintas | Evalúan variabilidad del entrenamiento. No se elige la semilla que dé el mejor resultado después de ver evaluación. |
| v6, v7, v7.1 | Etapas históricas del protocolo | No son arquitecturas; un resultado conserva su versión, partición, pesos y receta de evaluación. |
| C, S, T, E y números asociados | Identificadores locales de pruebas | Su significado depende del documento que los registra. S1 de un plan experimental no tiene por qué ser Figura S1; T5 de una figura tampoco renombra automáticamente otro experimento. Siempre indicar el contexto. |

### Qué viene de la literatura y qué aportamos

- **Referencias de la literatura (baselines):** arquitecturas y métodos publicados usados para comparar. Se mantienen sus nombres, por ejemplo DINOv2, SimSiam y U-Net. Que una técnica forme parte de Proposed no significa que la hayamos inventado.
- **Aporte del proyecto:** diseño, integración y evaluación de la adaptación SSL, la consistencia de tono y la variante que incorpora segmentación y descriptores, con los límites documentados. No presentamos como propia la arquitectura base de la literatura.
- **Arquitectura:** estructura de la red. **Receta de entrenamiento:** cómo aprende. **Checkpoint:** pesos de una ejecución. **Pipeline:** cadena completa de procesamiento, extracción y clasificación. No usar estas cuatro palabras como sinónimos.
- **Student (estudiante):** red entrenada con pseudo-etiquetas en estos experimentos. **Pseudo-máscara:** región de lesión estimada automáticamente, no anotación de especialista. **Teacher (maestro):** fuente de supervisión automática cuando el método la tenga; no asumir que todas las etapas usan una arquitectura teacher–student explícita.
- **Backbone congelado:** sus pesos no cambian al ajustar el clasificador. **Fine-tuning supervisado:** actualiza pesos con etiquetas diagnósticas. SSL describe el aprendizaje de la representación sin esas etiquetas; la sonda lineal posterior sí usa etiquetas de entrenamiento.
- **Segmentación:** estima dónde está la lesión; no determina por sí sola qué diagnóstico tiene. **Clasificación:** asigna una categoría diagnóstica. No llamar sensibilidad diagnóstica al Dice de una máscara.
- **Transformación de tono:** modifica luminosidad/apariencia en la región indicada por la máscara; no crea evidencia de un fototipo clínico distinto ni demuestra una mejora por sí sola.
- **ITA y terciles de L*:** estimaciones de color y agrupaciones de luminosidad, respectivamente. No equivalen a fototipo Fitzpatrick ni prueban sesgo causal por tono de piel.
- **Fitzpatrick:** clasificación clínica de respuesta de la piel al sol; no es una etiqueta deducible de forma fiable solo por oscurecer una fotografía.
- **Partición (split):** separación de entrenamiento, ajuste y evaluación. **Holdout:** subconjunto reservado para selección. **Transductivo:** imágenes de evaluación estuvieron disponibles en SSL sin etiquetas; debe declararse y no presentarse como evaluación completamente independiente.
- **Sensibilidad MEL, F1 macro, Dice e IoU:** mayor es mejor. **ECE y HD95:** menor es mejor. La comparación requiere la misma población y protocolo; un valor aislado no demuestra superioridad.
- **Intervalo de confianza y contraste pareado:** cuantifican incertidumbre y diferencia sobre las mismas imágenes. Que dos intervalos individuales se solapen no demuestra equivalencia; para el contraste pareado se interpreta el intervalo de la diferencia y su regla registrada.

### Regla de lectura de todo el repositorio

En textos de presentación se usan los dos nombres oficiales. En registros históricos, tablas generadas, comandos, rutas, nombres de pesos y claves de resultados se conservan los códigos originales para no romper la reproducción. Esos códigos se interpretan con las tablas anteriores, no como más propuestas. Los registros conservan lo que se sabía en su fecha; sus predicciones no se convierten en resultados por cambiar la nomenclatura.

Las tablas generadas y las matrices de confusión se regeneran con la nueva nomenclatura en la capa de presentación. Sus JSON, claves internas, cifras y nombres de archivo no se modifican. Las celdas de código de notebooks quedan intactas; las celdas Markdown sí se actualizan. Los documentos binarios y gráficos históricos sin estos códigos conservan su contenido original. Los códigos que aparecen dentro de rutas, comandos o identificadores literales siguen siendo necesarios para reproducir resultados.

## Conceptos generales e históricos

Los ejemplos siguientes pertenecen a etapas anteriores del proyecto; no sustituyen las definiciones ni los protocolos de las dos versiones anteriores.

- **SSL (aprendizaje autosupervisado):** enseñarle al modelo a entender imágenes de piel **sin
  decirle la respuesta**, poniéndolo a resolver rompecabezas con las imágenes mismas. Después,
  con pocas etiquetas, aprende a clasificar mucho más rápido.

- **Domain-adaptive:** agarrar un modelo que ya sabe "ver" imágenes normales (entrenado en ImageNet)
  y **acostumbrarlo al mundo de la piel** con fotos dermatológicas sin etiqueta.

- **SimSiam:** una receta de SSL que compara **dos versiones deformadas de la misma foto** y le pide
  al modelo que las vea parecidas. No necesita ejemplos "negativos" → funciona con una sola GPU.

- **Fine-tune (afinado):** después del SSL, entrenar el modelo con las etiquetas reales (7 clases)
  para que dé el diagnóstico de apoyo.

- **LP-FT (Linear Probe → Fine-Tune):** una forma **cuidadosa** de afinar. Primero se entrena solo el
  clasificador de arriba con el resto del modelo **congelado** (LP), y recién después se descongela
  todo con pasos chiquitos (FT). ¿Por qué? Porque afinar de golpe **borra** lo bueno que aprendió el
  SSL (Kumar 2022). LP-FT lo preserva → es una de las razones por las que el SSL termina ganando.

- **LLRD (decaimiento de learning rate por capa):** darle **pasos más grandes a las capas de arriba**
  (las que deciden la clase) y más chiquitos a las de abajo (bordes, texturas, que ya son buenas). Así
  el afinado no destruye lo que ya estaba bien.

- **Ablación:** experimento para **aislar el efecto** de una sola cosa. Aquí: entrenar dos modelos
  idénticos que solo se diferencian en el punto de partida (ImageNet vs SSL) para ver si el SSL
  realmente aporta.

- **Multi-semilla (seeds) y "fuera del ruido":** entrenar varias veces cambiando la **semilla
  aleatoria** y reportar el promedio ± la variación. Si la diferencia entre SSL e ImageNet es **mucho
  más grande que esa variación** (aquí ~5×), no es suerte: es un resultado real. Un tribunal lo exige.

- **AUC-ROC:** mide qué tan bien el modelo **ordena** los casos (pone lo sospechoso arriba), sin
  depender de un umbral fijo. 0.5 = azar, 1.0 = perfecto. El nuestro ~0.94 = muy bueno.

- **Umbral de screening:** en tamizaje de melanoma no se usa la decisión normal (argmax); se **baja el
  umbral** para no dejar pasar cánceres, aceptando más falsas alarmas. Es el trade-off correcto en
  medicina de tamizaje.

- **Linear probe:** congelar el modelo y ponerle encima un clasificador chiquito. Mide **qué tan
  buenas son las características** que aprendió, sin dejar que el afinado las tape.

- **Leakage (fuga de datos):** hacer trampa sin querer. Si la misma lesión aparece en entrenamiento
  y en prueba, el modelo "ya la vio" → el número sale inflado. Se evita partiendo por `lesion_id`.

- **macro-F1:** promedio justo del rendimiento **por clase**. Importa porque hay clases con miles de
  fotos y otras con cien. La accuracy engañaría; macro-F1 no.

- **Sensibilidad de melanoma:** de todos los melanomas reales, **cuántos detecta**. En tamizaje es
  lo más importante: no dejar pasar un cáncer.

- **Calibración / ECE:** que cuando el modelo diga "80% seguro", acierte de verdad ~80% de las veces.
  Un modelo médico sobreconfiado es peligroso.

- **Temperature scaling:** un ajuste que **calma la sobreconfianza** del modelo sin cambiar sus
  decisiones.

- **Abstención:** que el modelo diga **"no sé, mejor que lo vea un especialista"** cuando duda, en
  vez de arriesgar una respuesta.

- **Grad-CAM:** un mapa de calor que muestra **dónde está mirando** el modelo. Queremos que mire la
  lesión, no un pelo ni una regla de medir.

- **Shortcut learning:** cuando el modelo **hace trampa** aprendiendo una pista tonta (ej. "si hay
  regla, es melanoma") en vez de la lesión de verdad.

- **BN drift (BatchNorm):** si entrenás a un tamaño de imagen (128px) y evaluás a otro (224px), unas
  estadísticas internas del modelo no coinciden → los números se desajustan. Se arregla usando el
  mismo tamaño en todo.

- **Fitzpatrick:** clasificación de respuesta de la piel al sol; no es equivalente a ITA ni a luminosidad de una fotografía. No podemos prometer rendimiento equivalente entre grupos sin evaluación apropiada.

## Dónde leer las tablas con códigos históricos

Los generadores aplican los nombres visibles de este glosario; las claves y referencias a archivos siguen usando los códigos originales. Cada tabla se verifica contra su generador y sus JSON de origen:

- [Configuraciones de evaluación v7.1](../resultados/clasificacion/v71/README.md): A* v7.1 corresponde al control A71; B* v7.1 corresponde a Proposed sin segmentación (B71).
- [Banco HAM10000](../resultados/banco_completo/TABLA_HAM10000.md), [ISIC 2019](../resultados/banco_completo/TABLA_ISIC2019.md) e [ISIC 2020](../resultados/banco_completo/TABLA_ISIC2020.md): distinguir controles, referencias de literatura y procedencia de pesos; no colapsar filas diferentes bajo una cifra única.
- [Experimentos de integración y ablaciones](../resultados/banco_completo/TABLA_ABLACION.md): F4 corresponde a Proposed con segmentación; las otras condiciones siguen como experimentos, no como propuestas adicionales.
- [Resultados de segmentación](../resultados/segmentacion/TABLA_SEGMENTACION.md): los estudiantes son configuraciones del experimento de segmentación, no clasificadores diagnósticos.

La simplificación a dos nombres no elimina controles, repeticiones ni resultados desfavorables. Toda cifra debe conservar su dataset, partición, protocolo, semilla y estado exploratorio cuando corresponda.
