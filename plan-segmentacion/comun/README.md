# Componentes comunes a los dos caminos

Estado: pendiente, no iniciado.

Todo lo que debe ser idéntico en el camino A y en el camino B vive aquí. Así la única variable de la comparación es la fuente de pseudo-etiquetas.

| Carpeta | Contenido previsto |
|---|---|
| `datos/` | Inventario del pool, redimensionado, caché y splits |
| `estudiante/` | U-Net (y DoubleU-Net si hace falta) y el bucle de entrenamiento sin máscaras |
| `evaluacion/` | Métricas de segmentación, cálculo de ITA/FST, ΔITA y estratificación por tono |
| `configs/` | Argumentos del piloto (10 épocas) y del entrenamiento final |

Regla: nada de aquí puede leer las máscaras de ISIC 2018 durante el entrenamiento ni para elegir checkpoint o umbral. Solo `evaluacion/` puede usarlas, y únicamente para medir.
