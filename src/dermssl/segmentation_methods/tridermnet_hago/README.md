# TriDermNet-HAGO: variante sin diagnósticos

Fuente: PDF completo aportado, DOI 10.1016/j.bspc.2026.111536.

`model.py`: Region dilataciones1/2/4, compresión1x1 y residual; Edge Sobel, dos
convoluciones, gate sigmoid; Token patches, posiciones, atención y FFN; TER
modula spatial con gamma/beta token y agrega contexto espacial. Para preservar
tokens ante el GAP singleton se añade residual en cross-attention: modificación
explícita. Stem stride2, width32, token64 y pooling acotado son adaptaciones.

W3 no acepta clases diagnósticas ni máscaras expertas. Se reemplaza clasificación
por cobertura de lesión pseudo, y consistencia cobertura/área. Edge se supervisa
sólo en vecindarios válidos/confidentes. Las cifras supervisadas del artículo no
se trasladan a esta variante.

`hago.py` usa warmup, estadísticas por capa, gradientes de tareas y control de
entropía. Default suprime tarea dominante (prosa), contrario al signo positivo
impreso; `paper_logistic` conserva ese signo como control. Eq44 imprime el peso
clasificación dos veces; se sigue la combinación de dos pesos del algoritmo.
La combinación gstab/gbal no definida se resuelve con normalización por capa del
gradiente balanceado. Opt base AdamW; ruido apagado por defecto. Si se activa,
es acotado y reproducible, sólo en heads/decoder; no altera otras corridas.

Referencia nativa: 150 épocas, 256², masks+classes. Nuestra adaptación:224² y
pseudomáscaras; no es una reproducción de ese resultado supervisado.
