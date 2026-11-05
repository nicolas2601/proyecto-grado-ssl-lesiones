# Transversal v8: comparación completa de tres encoders SSL, A0/A1

Protocolo exploratorio: MoCo-v3-style, VICReg y SimSiam, semillas SSL 42/43/44.
PAD (1.134 casos de desarrollo) y HAM (1.528) se procesan por separado. Los
manifiestos, pesos, código y metadatos quedan fijados por hash en la preparación.

A0 usa imagen original. A1 usa segmentación 026 para seleccionar piel predicha
y aplicar una perturbación fija L* de −8. No es normalización ni fototipo
clínicamente validados. Las imágenes que fallan QA técnico conservan identidad
y permanecen en el denominador. Se registran máscaras, fallbacks y cobertura
por clase. La preservación de la máscara predicha ocurre antes del resize del
encoder; no demuestra que toda lesión real quede protegida.

Cada brazo recalcula features y reajusta su lector balanced-LR con el protocolo
original: cinco folds externos, tres internos y tres semillas CV. C se selecciona
por separado en cada brazo dentro del fold de entrenamiento. A0 debe reproducir
las probabilidades OOF originales antes de aceptar A1; los folds y grupos se
cotejan por hash. Se conservan los batches de 16 y BF16 de la exportación.

Los intervalos del delta de macro-F1 usan 1.000 remuestreos pareados de los
componentes de identidad, con semilla 20261102. Son condicionales a las redes y
predicciones OOF observadas; no capturan nueva variación de entrenamiento ni
acreditan independencia clínica adicional. No se ajustan por multiplicidad: los
contrastes son exploratorios, y no se seleccionará solamente el favorable.

Dos workers usan una GPU cada uno y dos hilos de CPU por proceso. La segmentación
y el caché de imágenes se calculan una sola vez. Cada job tiene un límite de
1.800 segundos; el lote puede durar más que el ejemplo técnico. Un fallo impide
iniciar jobs nuevos y conserva los artefactos y logs de los jobs ya arrancados.
No hay entrenamiento de encoders ni reanudación automática.

Revisiones solicitadas: protocolo/fugas de datos; intervención de color; y
ejecución/procedencia. Las dos revisiones con lectura de archivos pidieron
reajuste A1, validación de grupos, cobertura y cierre con hashes; se incorporaron.
La tercera revisó fragmentos proporcionados de la intervención, sin acceso
directo al filesystem, y destacó el riesgo de falsos negativos del segmentador.

`preparation.json` acredita el inventario y las pruebas de Linux. Un recibo de
lanzamiento y el ledger remoto deben consultarse para conocer el estado real;
este documento no declara resultados ni finalización del lote.
