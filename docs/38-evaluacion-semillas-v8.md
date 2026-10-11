# Evaluación anatómica exploratoria de semillas v8

Registrado antes de leer las máscaras expertas o calcular estos resultados.
No ajuste de checkpoint, umbral ni receta. S1 conserva span ≤0.01, S2 omite solo
inversión, S3 excluye pelo, S4 inpainting del input del maestro congelado.

ISIC2018 Task1 Training Input / GroundTruth ya utilizado en evaluación histórica:
no test intacto ni confirmación nueva. Emparejar todos los archivos por ID y sufijo
_segmentation, fallar si falta alguno; muestra uniforme de 100 pares, semilla 44,
sin diagnóstico ni selección por métricas. Guardar IDs y hashes antes de máscaras.
Informar coincidencias por ID canónico con pool nativo (quitar _downsampled), sin
presentarlas como independencia de imágenes ni como deduplicación por contenido.

Predicciones se terminan antes de abrir referencia experta. RGB por load_image del
evaluador histórico, resolución 224, campo existente calculado sobre original.
Maestro/cabeza/checkpoint/precisión BF16 y flags de pseudo_r0 congelados. Pares
original/inpainting en el mismo forward; guardar máscaras de etiquetas 0/1/255.
Evaluar FOREGROUND==1 contra referencia >127 con resize nearest del evaluador;
IGNORE se considera no-lesión para evaluación sobre todo el frame: no ocultar
falsos negativos quitando el área ignorada. Abstención cuenta como predicción vacía.

Métricas del evaluador existente: Dice, IoU, sensibilidad, cobertura y HD95 a 224
píxeles. HD95 indefinido por máscara vacía se registra null y se cuenta explícitamente;
Dice/IoU mantienen todos los 100 casos. No comparar con puntuación de estudiante.
Contrastes pareados de Dice: media Sx−S0, bootstrap descriptivo 4.000 réplicas,
semilla 45, IC percentil 95%; no prueba confirmatoria ni selección automática.
Mantener etapas de entrenamiento aisladas y no reintroducir cuarentena.
