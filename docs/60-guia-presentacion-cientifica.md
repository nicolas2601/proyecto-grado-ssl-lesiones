# 60 — Guía de presentación científica (anotada, no aplicar todavía)

**Estado: ANOTADA, no ejecutar.** Por decisión del 2026-10-30, los diagramas y la reescritura de
la documentación **no se tocan mientras siguen las pruebas**. Este documento fija el estándar
acordado para cuando se retome la documentación del paper en
`C:/Users/mpaul/Documents/Anteproyecto`.

Fuentes del estándar, leídas el 2026-10-30:

- **RPI-Net** — Li, X., Peng, B., Zhang, J., Zhang, Z., & Xie, Z. (2026). *Reliable Multi-Source
  Contrastive Pseudo-Labels Interaction Network for unsupervised skin lesion segmentation.*
  Biomedical Signal Processing and Control, 112, 108433.
- **TiSage / LUTSeg** — Sanchez, K., Hinojosa, C., Ávila, A. A., Riano-Rojas, A. C., Romero,
  D. H., Páez, J. C., Llinás, M., & Ghanem, B. *LUTSeg: A Longitudinal Multi-Expert Dataset for
  Ulcer Tissue Segmentation.* (Formato LNCS.)
- **USL-Net** — Li et al. (2024).

Los tres se citan en APA 7 en el documento final, según la convención del proyecto.

---

## 1. El problema actual, nombrado

La documentación del proyecto tiene la información pero **no tiene hilo**. Un lector llega a una
cifra sin haber entendido por qué importa, y a una figura sin saber qué pregunta responde.
Los papers de referencia resuelven esto con una estructura fija donde cada sección responde una
sola pregunta y entrega al lector listo para la siguiente.

El objetivo no es escribir más corto —los papers son concisos por límite de página, nosotros
debemos **explicar cada punto**— sino escribir **ordenado y encadenado**.

## 2. Esqueleto de secciones (de RPI-Net, es el más completo)

| # | Sección | Pregunta que responde | Qué debe contener |
|---|---|---|---|
| 1 | Introducción | ¿Por qué existe este trabajo? | Contexto clínico → limitación concreta de lo existente → nuestra propuesta en una frase → **contribuciones numeradas** |
| 2 | Trabajo relacionado | ¿Qué hicieron los demás y por qué no alcanza? | Subdividido por **grado de supervisión**: supervisado / débilmente supervisado / no supervisado. Cada método en 1–3 frases: qué hace, y **cuál es su límite** |
| 3 | Metodología | ¿Cómo funciona lo nuestro? | 3.1 Vista general + figura de arquitectura. 3.2, 3.3… un módulo por subsección, **cada uno con sus ecuaciones numeradas** |
| 4 | Experimentos | ¿Qué medimos y qué dio? | 4.1 Datasets y métricas · 4.2 Comparación con el estado del arte · 4.3 Comparación cualitativa · 4.4 **Ablación** · 4.5+ análisis de sensibilidad · últimA: **eficiencia computacional** |
| 5 | Discusión | ¿Qué significa y dónde falla? | Fortalezas con evidencia, **limitaciones explícitas**, trabajo futuro |
| 6 | Conclusión | ¿Qué queda? | Sin cifras nuevas; recapitula lo demostrado |

**Las contribuciones numeradas al final de la introducción son obligatorias.** RPI-Net usa cuatro;
TiSage usa tres con viñetas. Es el contrato con el lector: todo lo que se promete ahí tiene que
aparecer medido en la sección 4.

## 3. Cómo encadenar (el "hilo" que falta)

Regla que siguen los dos papers y que hay que adoptar:

> Cada párrafo de método termina justificando **por qué existe el siguiente**.

Ejemplo literal de RPI-Net: genera pseudo-etiquetas → *"para refinar y manejar el ruido de las
pseudo-etiquetas, RPI-Net integra dos módulos"* → IPC y PIA → *"las pseudo-etiquetas refinadas se
usan entonces en el módulo de segmentación"* → DUL. El lector nunca encuentra un módulo sin saber
qué problema vino a resolver.

Aplicado a nosotros: no presentar SLIC, DermLIP, EMA y la banda de incertidumbre como una lista
de componentes. Presentarlos como una **cadena de problemas resueltos**.

## 4. Figuras: qué representa cada tipo

Esta es la parte más importante del encargo. **Nada de cajas genéricas.** Cada elemento del
diagrama debe *parecerse* a lo que representa.

| Si el objeto es… | Se dibuja como… | Referencia |
|---|---|---|
| Una red convolucional | **Prismas rectangulares apilados en perspectiva**, con el ancho decreciendo según el downsampling | RPI-Net Fig. 2 (encoder/decoder) |
| Un tensor o mapa de features | **Cuboide 3D** con las caras sombreadas; alto × ancho × canales legibles | RPI-Net Fig. 3 |
| Una imagen de entrada | **La dermatoscopia real**, no un rectángulo con la palabra "imagen" | RPI-Net Fig. 2, TiSage Fig. 3 |
| Una máscara o pseudo-etiqueta | **La máscara binaria real** en blanco sobre negro | RPI-Net Fig. 2 |
| Un mapa de incertidumbre | **El heatmap real** en escala de grises o térmica | RPI-Net Fig. 1(f), Fig. 2 |
| Superpíxeles SLIC | **La imagen real con las fronteras de los superpíxeles sobredibujadas**, en versión gruesa y fina para que se vea la diferencia de escala | TiSage Fig. 3(A) |
| Una operación matricial (atención) | **Las matrices Q, K, V como grillas de celdas coloreadas**, con el producto punto y el softmax marcados con su símbolo | RPI-Net Fig. 3 |
| Un encoder congelado | El bloque con un **ícono de candado**; si se entrena, ícono de llama | TiSage Fig. 3 |
| Una clasificación por píxel | **El mapa de clases con la paleta por categoría** y su leyenda de colores | TiSage Fig. 1 |

Convenciones adicionales que se observan en los dos papers:

- **Agrupación por color de fondo**: cada etapa del pipeline va dentro de un panel de color suave
  distinto, con su nombre en una esquina (RPI-Net: *Pseudo-label generation*, *Segmentation
  module*, *Interaction module*, *Uncertainty learning*; TiSage: paneles A, B, C).
- **Los paneles se referencian desde el texto** como Fig. 3(A), Fig. 3(B).
- **Las variables del diagrama son las mismas del texto.** Si en la ecuación dice `q(x)`, en la
  figura dice `q(x)`, no "prior fusionado".
- **Leyenda de símbolos dentro de la figura** cuando hay operadores (× producto punto, + suma
  vectorial).
- El pie de figura es **autocontenido**: explica qué se ve sin obligar a volver al texto.

En nuestro caso concreto, esto implica dibujar de verdad: los superpíxeles SLIC sobre una lesión
real, el encoder DermLIP con candado, la banda de incertidumbre como heatmap, el teacher EMA y el
student como dos redes con el flujo de gradiente detenido marcado explícitamente.

## 5. Figuras cualitativas: la grilla comparativa

Las dos referencias usan la misma estructura y conviene copiarla tal cual:

- **Filas** = casos (etiquetados (a), (b), (c), (d)), elegidos a propósito y **declarando el
  criterio de selección**. RPI-Net dice literalmente que eligió dos casos de sub-segmentación y
  dos de sobre-segmentación de USL-Net. Eso es honestidad: el lector sabe que no son los mejores
  casos.
- **Columnas** = `Imagen | Ground truth | Método competidor | Nuestro método`.
- **Codificación de error por color**: RPI-Net pinta en rojo y verde las diferencias contra el
  ground truth, lo que permite ver *dónde* falla sin leer una cifra.

Para nosotros: los 32 casos de Dice cero de Run 008 son material de figura cualitativa, no solo
una tabla. Y la taxonomía (vello, micro-lesión, macro-pantalla, contraste ultrabajo) da las filas.

## 6. Tablas: cuatro tipos, cada una con un trabajo

| Tabla | Qué compara | Convención |
|---|---|---|
| **Estado del arte** | Nuestro método contra ~10 métodos publicados, en **3 datasets** × 5 métricas | **Negrita** = mejor, **subrayado** = segundo mejor. Métodos ordenados por fecha de publicación |
| **Ablación** | Construcción incremental: `base` → `+módulo1` → `+módulo2` → … | Cada fila agrega un componente. Se ve crecer la métrica. Incluir la **cota superior** con ground truth |
| **Robustez** | Comportamiento bajo condición adversa (RPI-Net: ruido 10 % y 30 %) | Dos bloques de columnas, uno por nivel |
| **Eficiencia** | Tiempo, memoria GPU, FLOPs, parámetros | Entrenamiento e inferencia por separado; dataset completo y mitad, para mostrar escalabilidad |

Observación importante: **RPI-Net reporta las métricas donde pierde.** En ISIC-2017 dice que
quedó segundo, y en ISIC-2018 admite que su sensibilidad es menor que la de USL-Net. Eso **suma**
credibilidad. Nuestro proyecto tiene resultados negativos medidos y pre-registrados (la máscara no
ayuda al clasificador, el ITA está por debajo de la línea base, el SSL propio pierde en PAD) — se
reportan igual, con el mismo tono.

## 7. Métricas y su presentación

- RPI-Net: **ACC, DIC, JAC, SEN, SPE** más mIoU en tabla aparte.
- TiSage: **mIoU y Dice**, más IoU por clase.
- Nosotros ya usamos **Dice, IoU, HD95** con IC 95 % por bootstrap, que es **más riguroso que las
  dos referencias** (ninguna reporta intervalos de confianza). Eso se conserva y se declara como
  fortaleza metodológica.
- Declarar siempre la dirección: ↑ mejor o ↓ mejor.
- Resolución y preprocesamiento explícitos: RPI-Net dice "224 × 224, pelo removido con black-hat
  morfológico más inpainting". Nosotros tenemos esa información y hay que exponerla igual.

## 8. Tono de redacción

- **Voz activa en primera persona del plural**: "proponemos", "evaluamos", "observamos".
- Afirmaciones **acotadas**: "en el régimen de 1/8 etiquetas", "bajo el protocolo registrado",
  nunca "nuestro método es superior" sin condición.
- **Explicar cada punto**, a diferencia de los papers que comprimen por límite de página. El
  destinatario incluye lectores no técnicos: cada símbolo matemático se define en palabras la
  primera vez que aparece.
- Términos en inglés aceptados (*pseudo-label*, *backbone*, *superpixel*) en cursiva la primera
  vez, con su equivalente en español entre paréntesis.
- Español neutral/profesional, APA 7, sin emojis — convenciones ya vigentes del proyecto.

## 9. Lo que NO se copia de los papers

- **Su nivel de compresión.** Ellos tienen límite de página; nuestra tesis debe explicar.
- **Su ausencia de intervalos de confianza.** Nosotros los reportamos.
- **Su paleta.** Se mantiene la paleta cappuccino del Anteproyecto y Times New Roman, según la
  convención ya fijada en `CLAUDE.md`. Lo que se adopta es el **tipo de representación gráfica**,
  no los colores.

## 10. Checklist para cuando se retome

- [ ] Contribuciones numeradas en la introducción, y cada una medida en la sección 4
- [ ] Trabajo relacionado subdividido por grado de supervisión
- [ ] Cada módulo con su subsección y sus ecuaciones numeradas
- [ ] Figura de arquitectura con imágenes reales, no cajas
- [ ] Superpíxeles SLIC dibujados sobre lesión real, escalas gruesa y fina
- [ ] Encoder congelado con candado, entrenable con llama
- [ ] Matrices de atención como grillas con operadores marcados
- [ ] Grilla cualitativa con criterio de selección declarado y error por color
- [ ] Tabla de estado del arte con negrita/subrayado
- [ ] Tabla de ablación incremental con cota superior
- [ ] Tabla de eficiencia computacional
- [ ] Resultados negativos reportados con el mismo tono que los positivos
- [ ] Cada párrafo de método justifica el siguiente
