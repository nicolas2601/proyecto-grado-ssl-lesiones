# 13 — ¿Por qué estas decisiones? (arquitectura y método, explicado)

Este documento responde en cristiano **por qué elegimos lo que elegimos**: por qué ResNet-50, por qué
SimSiam, por qué no un Transformer, por qué domain-adaptive, etc. Cada decisión tiene un porqué
técnico **y** un porqué de recursos (somos estudiantes con una GPU prestada, no un laboratorio).

---

## 1. ¿Por qué ResNet-50 como backbone?

**Qué es:** ResNet-50 es una red convolucional (CNN) de 50 capas con *conexiones residuales* (He et
al., 2016). Las conexiones residuales son "atajos" que dejan pasar la señal sin degradarse, lo que
permite entrenar redes profundas sin que el gradiente se desvanezca.

**Por qué la elegimos:**

1. **Es el estándar de oro en dermatología con IA.** La enorme mayoría de papers de clasificación de
   lesiones cutáneas (incluidos los benchmarks de ISIC) usan ResNet-50 o EfficientNet. Elegirla nos
   hace **comparables con la literatura** — si reportamos macro-F1 con ResNet-50, un revisor sabe
   exactamente contra qué medirlo.
2. **Pesos pre-entrenados de ImageNet disponibles y probados.** Arrancamos de un backbone que ya
   "sabe ver" bordes, texturas y formas. No partimos de cero (ver §4).
3. **Cabe en una GPU T4 (16 GB).** Se entrena con batch razonable sin quedarnos sin memoria. Un
   Transformer grande (ViT-B) no entra cómodo y necesita muchísimos más datos para brillar.
4. **Balance perfecto para el problema.** ResNet-18/34 es demasiado chica para 7 clases con
   texturas finas; ResNet-101/152 es más cara sin ganancia clara en un dataset de ~10 k imágenes.
   **ResNet-50 es el punto dulce** tamaño/rendimiento/costo.
5. **Grad-CAM funciona directo sobre ella.** Su última capa convolucional produce mapas de calor
   interpretables — clave para nuestra evidencia de fiabilidad (¿mira la lesión o el artefacto?).

**Qué descartamos y por qué:**

| Alternativa | Por qué NO (en esta fase) |
|---|---|
| ViT-B/16 (Vision Transformer) | Necesita millones de imágenes o pre-entrenamiento masivo para superar a una CNN; con ~10 k imágenes y una T4 rinde peor y cuesta más. Queda para la fase con GPUs UNAB. |
| EfficientNet-B4/B5 | Buena opción, comparable; ResNet-50 gana en simplicidad, soporte de Grad-CAM y comparabilidad con la literatura. |
| ResNet-18 / 34 | Muy chicas: pierden capacidad para distinguir clases visualmente parecidas (mel vs nv). |
| ResNet-101 / 152 | Más caras, más lentas, más propensas a sobreajustar con pocos datos; sin ganancia clara. |

---

## 2. ¿Por qué SimSiam como método SSL?

**Qué es SSL (self-supervised learning):** entrenar la red **sin etiquetas**, inventándole una tarea
"de pretexto". En SimSiam (Chen & He, CVPR 2021) la tarea es: toma una imagen, hazle dos
recortes/aumentos distintos, y obliga a la red a producir la **misma** representación para ambos. Así
aprende qué hace que "una lesión sea esa lesión" sin que nadie le diga la clase.

**Por qué SimSiam y no otro método SSL:**

1. **Es *negative-free*.** Métodos como SimCLR necesitan comparar contra montones de ejemplos
   "negativos" en el mismo batch → exigen **batches enormes** (4096+), imposibles en una T4. SimSiam
   solo necesita las dos vistas de la misma imagen → **corre con batch 128 en una GPU**.
2. **No necesita cola de memoria ni momentum encoder** (como MoCo) → menos partes móviles, menos
   cosas que se rompan.
3. **Es robusto a batch chico**, justo nuestro caso.
4. **Truco anti-colapso simple:** el `stop-gradient` evita que la red haga trampa colapsando todo a
   un vector constante. Monitoreamos `out_std` por época para confirmar que no colapsa.

**Qué descartamos y por qué:**

| Método SSL | Por qué NO (ahora) |
|---|---|
| DINO / DINOv2 | Es el ideal del anteproyecto, pero está pensado para ViT y cómputo grande. Inviable de entrenar bien en una T4. Vía futura con GPUs UNAB. |
| SimCLR | Requiere batches gigantes por los negativos → no cabe en la T4. La Práctica 1 lo confirmó empíricamente. |
| MoCo v3 | Cola de memoria + momentum encoder; más complejo, sin ventaja clara aquí. |
| MAE / SimMIM (masked image modeling) | Reconstruyen parches enmascarados; viables en 2D y buena vía futura, pero pensados para ViT y mucho dato. |
| MambaMIM | Es 3D / tomografía / segmentación con backbone Mamba → **no encaja** con nuestro problema 2D de clasificación dermatoscópica. Se cita como estado del arte, no se usa. |

---

## 3. ¿Por qué "domain-adaptive" y no SSL desde cero?

**Domain-adaptive** = no entrenamos el SSL desde cero, sino que **continuamos** un backbone que ya
viene pre-entrenado en ImageNet, adaptándolo al dominio dermatoscópico con SimSiam sobre HAM+ISIC.

**Por qué:** en la **Práctica 1** probamos SSL **desde cero** (backbone aleatorio) con cómputo de
estudiante. Resultado: F1 0.285, **peor que un backbone totalmente aleatorio** (0.318). No es un bug
— es lo esperado: el SSL desde cero necesita **millones** de imágenes y batches enormes para
despegar. Con nuestros recursos, la jugada correcta es **partir de ImageNet y especializar**. Esta
lección está grabada: **SSL nunca desde cero** en régimen de bajos recursos.

---

## 4. ¿Por qué transfer learning (partir de ImageNet)?

Porque el modelo ya "sabe ver" (bordes, texturas, colores, formas) tras entrenar en 1.2 millones de
imágenes naturales. Nosotros solo tenemos ~10 k imágenes médicas — no alcanzan para enseñarle a ver
de cero. Reutilizar ese conocimiento previo y **especializarlo** a dermatoscopía es lo que hace
posible un buen resultado con pocos datos. Es la práctica estándar en imagen médica.

---

## 5. ¿Por qué 224 × 224 píxeles?

- Es la resolución **nativa** con la que se entrenaron los pesos ImageNet de ResNet-50.
- Igualamos SSL, fine-tune y evaluación **todos a 224** para evitar el *BN drift*: si el SSL corre a
  128 px y evalúas a 224 px, las estadísticas de BatchNorm no coinciden y el linear probe **colapsa
  aparentemente** (nos pasó: daba 0.264). Al unificar a 224 el problema desaparece (subió a 0.45).
- Subir a 384/512 px multiplica el costo de cómputo sin ganancia clara en este dataset.

---

## 6. ¿Por qué split por lesión y no por imagen?

Una misma lesión aparece en varias fotos en HAM10000. Si partimos por imagen, la **misma lesión**
puede caer en train y en validación → el modelo "ya la vio" y la métrica sale **inflada** (fue
exactamente el pecado del prototipo v1, que daba 0.873 falso). Partiendo por `lesion_id`
(StratifiedGroupKFold) garantizamos **cero solapamiento** → el número baja a ~0.74 pero es **real**.

---

## 7. ¿Por qué medir con KPIs clínicos y no con accuracy?

Porque el dataset está **desbalanceado** (muchísimos nevus, pocos dermatofibromas) y porque es
**medicina**: no da igual equivocarse en cualquier clase. Un modelo puede tener 90 % de accuracy y
aun así dejar pasar melanomas. Por eso medimos:

- **macro-F1** (promedia todas las clases por igual, no premia acertar solo la mayoritaria),
- **AUC-ROC** (capacidad de ranking, independiente del umbral),
- **sensibilidad de melanoma** (que no se nos escape el cáncer),
- **ECE / calibración** (que la confianza sea honesta),
- **abstención** (que sepa decir "no sé, deriva a especialista").

---

## 8. Resumen en una frase

**Elegimos las herramientas más fuertes que *de verdad* corren con una GPU T4 de estudiante**
(ResNet-50 + SimSiam domain-adaptive), medimos con rigor clínico, y documentamos honestamente el
techo que se alcanza así — dejando el camino trazado para escalar (ViT/DINOv2, más datos, validación
externa) cuando tengamos mejores equipos.

Ver también [04 — Decisiones técnicas (ADR)](04-decisiones-tecnicas.md) para el registro formal.
