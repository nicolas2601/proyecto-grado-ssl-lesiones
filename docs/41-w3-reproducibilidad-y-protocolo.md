# 41. W3: revisión de reproducibilidad y preparación de comparación

Registrado el 2026-10-12 (America/Bogota), antes de ejecutar nuevos maestros.
Base de código: `59c7ae4`. Complementa los documentos 35, 36 y 39; no altera
sus restricciones ni declara ejecutados los métodos.

## Resultado de la revisión

Se mantienen los tres candidatos acordados: SLED, USL-Net y RPI-Net.
MaskCut/TokenCut permanece como alternativa del registro inicial; no reemplaza
silenciosamente a RPI-Net. Disponibilidad de un artículo no significa disponibilidad
de una réplica ni superioridad en nuestros datos.

| Candidato | Fuente revisada | Estado local | Próximo trabajo concreto |
|---|---|---|---|
| SLED | Artículo ICDM 2023 y código oficial enlazado por el artículo, revisión `3213eac` | Revisado; adaptación y entorno pendientes | Resolver uso del código o implementar desde el artículo; entrada independiente de GT y smoke CPU |
| USL-Net | Artículo completo arXiv v3, 21-07-2024; DOI editorial de 2024 | Revisado; repositorio oficial no identificado en las fuentes consultadas | Matriz de diferencias y adaptación propia explícita; no contar el UM existente como un maestro nuevo |
| RPI-Net | Vista editorial, volumen 112, artículo 108433, febrero de 2026 | Texto completo, código y pesos pendientes | Obtener especificación verificable de IPC/PIA/DUL antes de implementar; permanece bloqueado |

### SLED: observaciones estáticas sobre la revisión fijada

El artículo enlaza `SELGroup/SLED`. Este repositorio y `zengguangjie/SLED`
resuelven al mismo commit consultado. No se identificó LICENSE en sus árboles;
la API también devuelve licencia nula. No se incorpora código de terceros al
proyecto. La ausencia de licencia debe resolverse para reutilizarlo; una
implementación propia desde las ideas del artículo se documentaría como tal.

- Entorno publicado: Python 3.10.6, versiones antiguas de NumPy/scikit-learn/
  scikit-image/Numba, JPype y Java 8. No instalar estas versiones en la `.venv`
  histórica de Python 3.13. Un entorno separado necesita su propia verificación.
- `main.py` carga GT y calcula Dice dentro de las rutinas de ejemplo. No se
  observa que GT entre al núcleo segmentador; aun así, el piloto debe desacoplar
  generación de máscaras y evaluación, sin abrir anotaciones en generación.
- `seg/lesion_part_selection.py:gkern` usa un único punto por eje; su expresión
  produce un kernel 1x1 igual a 1. El broadcasting elimina la ponderación gaussiana
  espacial esperada. Corrección y comparación contra el comportamiento publicado
  deben quedar registradas antes de interpretar resultados.
- `regionprops` omite la etiqueta cero, mientras SLIC empieza en cero: revisar
  el centroide del primer superpíxel. Validar grafos sin aristas y mapas constantes
  antes de divisiones por volumen, rango o varianza mínima.
- Isolation Forest no fija semilla explícita. El multiescala abre diez procesos
  con los valores por defecto y escribe temporales en cwd. El adaptador debe
  fijar semilla, limitar concurrencia y utilizar carpetas únicas por caso.

Estas son observaciones del código, no resultados de segmentación local.
No se ejecutó el JAR ni el código de terceros en el clúster.

### USL-Net frente al código actual

La versión v3 especifica fusión de SimCLR/MoCov1/MoCov2 con ResNet50, CCAM,
UM y refinamiento con ResNet50+FCN. Nuestro `uncertainty_module.py` ya declara
inspiración en USL-Net y usa 0.35/0.65. Repetirlo con otro nombre no establece
un comparador independiente.

| Elemento | Artículo v3 | Código propio observado |
|---|---|---|
| Extracción | Fusión de tres variantes contrastivas | La réplica de esa fusión y sus pesos no están verificados |
| Cabeza CCAM | Convolución 1x1 | `CCAMHead` tiene kernel por defecto 3; verificar configuración del checkpoint, no asumir su valor efectivo |
| Centralidad | Texto: mínimo de suma de distancias de píxeles; pseudocódigo utiliza comparación opuesta | Distancia del centroide, desempate por área y límites adicionales |
| Restricciones | UM y exclusión de casos sin región fiable | Campo, artefactos, límite de área 0.6 y modo unión añaden decisiones propias |
| Refinamiento | Cinco iteraciones de 300 épocas | No está registrada una réplica completa con FCN y ese presupuesto |

La contradicción de centralidad y detalles de fusión deben explicitarse.
Un ensayo que cambie solamente centralidad se llamará ablación de UM, no
USL-Net completo. Una adaptación de arquitectura/presupuesto se publicará como
adaptación propia; no se trasladarán las cifras del artículo.

### RPI-Net

La vista editorial confirma el método y sus módulos, pero no aporta la
especificación completa suficiente para una implementación verificable. No se
identificaron código oficial ni pesos en las fuentes consultadas. No se usará
otro repositorio llamado RPI-Net de tareas RNA/proteína. No se inventarán módulos
a partir del resumen ni se llamará RPI-Net a un consenso propio.

## Comparación registrada por fases

1. **Preparación:** mismos 100 IDs elegibles de W0, mismos originales y hashes.
   Son desarrollo sin etiquetas; no constituyen evaluación anatómica independiente.
   Entrada primaria 224x224. Cualquier tamaño nativo adicional se reporta como
   contraste separado, con su propio tiempo/memoria y transformación geométrica.
2. **Smoke:** entrada sin diagnósticos/GT, determinismo, salida finita, dimensiones,
   codificación, temporales aislados, timeout y memoria. Empezar con una imagen;
   medir tiempo/RAM antes de fijar el presupuesto del lote de 100. No elegir una
   muestra sustituta cuando falla una imagen.
3. **Maestros:** comparar S0 CCAM congelado con candidatos habilitados. Las
   pseudomáscaras r2 verificadas en W0 provienen del estudiante; no sustituyen S0.
   Registrar foreground/background/ignore, cobertura, abstención, regiones,
   fracción de área, fallos por artefactos, tiempo y memoria. Acuerdo entre maestros
   no se denomina Dice anatómico ni decide cuál es correcto.
4. **Evaluación separada:** identificar antes una reserva experta sin contaminación
   ni decisiones previas. Si no existe, toda lectura será exploratoria. Umbrales,
   checkpoints y receta se fijan antes de abrir GT. Dice/IoU completos y p10 como
   primarios, más HD95, sensibilidad y precisión por píxel. No descartar fallos ni
   comparar únicamente píxeles confidentes sin publicar cobertura y rendimiento
   completo. Los mapas de semillas y las máscaras finales se evalúan por separado.
5. **Estudiante:** misma arquitectura, init, split, imágenes elegibles, pasos,
   semillas y presupuesto para contrastar maestros. Solo después de las puertas
   de los documentos 35/36; no lanzar nuevos entrenamientos mientras falte la
   evaluación anatómica independiente. Las reproducciones con presupuesto nativo
   de artículos se distinguen de adaptaciones con presupuesto común.

## Seguimiento y puertas de ejecución

El registro JSON `w3-reproducibility-v8.json` mantiene pendientes explícitos.
`scripts/preflight_w3.py` valida otra vez los hashes de los 100 originales y
exporta únicamente sus rutas/IDs/hashes, sin transformadas ni pseudomáscaras como
entradas nuevas. También inventaría versiones sin importar ni ejecutar terceros.
Su estado preparado no equivale a smoke ni a método ejecutado.

Para cada corrida efectiva: W&B con `WANDB_NETRC_FILE`, misma entidad/proyecto
del protocolo, nombre/semilla/método/config/revisión, conteos, coste y métricas
agregadas. Ninguna clave en configuración y ninguna imagen/ID clínico en los
registros remotos. Conservar el JSON local como fuente; registrar fallos de
seguimiento y no presentar una corrida desconectada como monitorizada.

No hay comparación de exactitud ni ganador en esta entrega. W4 conserva el
estado diseñado. Las puertas del registro evitan habilitar un candidato por
haber leído su artículo o por compartir nombres con componentes ya existentes.

## Fuentes y alcance

- SLED: https://arxiv.org/abs/2309.01899 y PDF del autor:
  https://penghao-bdsc.github.io/papers/Unsupervised%20Skin%20Lesion%20Segmentation%20via%20Structural%20Entropy%20Minimization%20on%20Multi-Scale%20Superpixel%20Graphs.pdf
- Código enlazado por el artículo: https://github.com/SELGroup/SLED/tree/3213eacb65577c0d52f424369158fb56c609238e
- USL-Net v3 (texto completo): https://arxiv.org/html/2309.13289v3;
  DOI https://doi.org/10.1016/j.bspc.2023.105769
- RPI-Net, vista editorial: https://doi.org/10.1016/j.bspc.2025.108433
  El acceso directo al texto editorial dio 403; no se afirma lectura completa.
- Código propio revisado en `59c7ae4`: `ccam_engine.py` y
  `uncertainty_module.py`, en `plan-segmentacion/camino-b-cnn/pseudo-mascaras/`.

## Verificación de esta preparación

Cinco pruebas pasaron en la .venv existente del clúster (0,19 s). El preflight
real verificó los hashes de los 100 originales y emitió los tres estados de
puertas pendientes. Métodos ejecutados: ninguno. Entrenamientos iniciados: ninguno.
La evidencia agregada es [preflight.json](../resultados/auditoria_v8/w3_preparation_20261013/preflight.json).
Los IDs/rutas de las imágenes permanecen en el clúster. La ejecución se hizo en
la copia de trabajo de 59c7ae4 con los hashes de scripts nuevos incluidos en el
reporte; la publicación posterior preserva esos contenidos.

El inventario detectó Python 3.13.15; pyod y JPype no están instalados.
No se instalaron paquetes ni se ejecutó software de terceros. Dos advertencias
de pytest corresponden a opciones de timeout no reconocidas, como en la
verificación previa del logger; no hubo fallos.
