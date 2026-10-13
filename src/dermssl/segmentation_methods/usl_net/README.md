# USL-Net: interpretación del texto

Fuente: arXiv 2309.13289v3 aportado por usuario, DOI 10.1016/j.bspc.2023.105769.

`model.py`: fusión concatenada de SimCLR/MoCov1/MoCov2, ResNet50 con último stride1,
CCAM background 1x1 y complemento, FCN de conv2/3/4 con fusión a 256 canales.
`ccam.py` reutiliza el objetivo CCAM ya revisado en el repo; no modifica históricos.
`seeds.py` entrena sólo la cabeza usando encoders congelados con procedencia.

UM reutiliza mínima SUMA de distancias a H/2,W/2, 500 píxeles, .35/.65 e ignore .5.
Seguimos la prosa frente al signo opuesto del pseudocódigo y a la ecuación BCE
mal impresa. Vecindad4, igualdad incierta y corte de distancia sin valor inventado
son decisiones propias. La fusión de canales y proyecciones FCN se explicitan;
no se afirman los detalles no especificados de las dimensiones impresas.

El banco exige checkpoint/provenance por origen y normalización explícita.
Se puede preentrenar SSL local; su receta registrada de aumentos/presupuesto es
adaptación, no presupuesto nativo de artículos SSL. Native USL: 5x300 épocas,
no lanzadas. `common/refinement.py` convierte predicciones posteriores en nuevas
pseudomáscaras UM, preservando incertidumbre y originales.
