# modelo de segmentación (U-Net) y bucle de entrenamiento

## Nomenclatura de lectura

**Proposed sin segmentación** clasifica sin usar una máscara de lesión. **Proposed con segmentación** combina imagen completa, recorte y descriptores de lesión; es una variante experimental, sin mejora robusta demostrada. El Modelo de control no es una tercera propuesta ni la versión segmentada.

Los experimentos de segmentación se identifican por arquitectura, origen de pseudo-máscaras y ronda de aprendizaje. Las rondas no son arquitecturas distintas; los identificadores antiguos se conservan solo donde permiten localizar archivos o reproducir comandos.

Ver [nombres, equivalencias y explicación para lectores técnicos y no técnicos](../../../docs/10-glosario-eli5.md). Los registros históricos conservan sus cifras, predicciones y limitaciones.

Estado: pendiente.

Alcance previsto:

- U-Net de 4 etapas (32/64/128/256 canales, cerca de 7,77 M de parámetros) como modelo de segmentación único de los dos caminos.
- DoubleU-Net solo si el resultado del piloto lo justifica.
- Entrenamiento con Adam, umbral 0,5, BCE más Dice suave, y argumentos parametrizados (épocas, rondas, resolución, batch), para que Nicolás corra el final con el mismo script.
- Inferencia segura desde un checkpoint existente: `python train_student.py --cache-dir salidas/cache224 --output-dir salidas/student_r2 --predict-only --preload`. Requiere `student_r2/best.pt`, no entrena y no reemplaza `predictions.npy` salvo con `--overwrite-predictions`.

Punto que hay que editar respecto del código actual:

- `src/dermssl/torch_unet.py` (`train_unet`) elige el mejor checkpoint por Dice de validación **con máscaras de ISIC 2018**. Eso contradice el plan. Falta definir un criterio sin máscaras, por ejemplo pérdida sobre pseudo-etiquetas, acuerdo entre maestros o estabilidad del ITA.
- `ISIC2018SegmentationDataset` en ese mismo archivo lee máscaras, así que no debe usarse para entrenar.
- La arquitectura `UNet` sí es reutilizable.
