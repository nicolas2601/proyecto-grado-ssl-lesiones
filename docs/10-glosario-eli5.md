# 10 — Glosario ELI5 (explicado sencillo)

Conceptos del proyecto explicados como si fueran para alguien sin fondo técnico.

- **SSL (aprendizaje autosupervisado):** enseñarle al modelo a entender imágenes de piel **sin
  decirle la respuesta**, poniéndolo a resolver rompecabezas con las imágenes mismas. Después,
  con pocas etiquetas, aprende a clasificar mucho más rápido.

- **Domain-adaptive:** agarrar un modelo que ya sabe "ver" imágenes normales (entrenado en ImageNet)
  y **acostumbrarlo al mundo de la piel** con fotos dermatológicas sin etiqueta.

- **SimSiam:** una receta de SSL que compara **dos versiones deformadas de la misma foto** y le pide
  al modelo que las vea parecidas. No necesita ejemplos "negativos" → funciona con una sola GPU.

- **Fine-tune (afinado):** después del SSL, entrenar el modelo con las etiquetas reales (7 clases)
  para que dé el diagnóstico de apoyo.

- **Ablación:** experimento para **aislar el efecto** de una sola cosa. Aquí: entrenar dos modelos
  idénticos que solo se diferencian en el punto de partida (ImageNet vs SSL) para ver si el SSL
  realmente aporta.

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

- **Fitzpatrick:** escala de tonos de piel (I = muy clara ... VI = muy oscura). Nuestro dataset es
  casi todo piel clara → no podemos prometer que funcione igual en piel oscura.
