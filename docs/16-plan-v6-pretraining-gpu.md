# 16. Plan v6: pre-entrenamiento SSL en GPU con trazabilidad MLflow

> **Documento histórico.** Describe una etapa anterior del proyecto y fue superado. Los resultados vigentes están en el [README](../README.md) y en los documentos 27 a 33. Se conserva para mostrar cómo se llegó al diseño actual.

Documento de decision para la fase v6 del Objetivo 2. Escrito para lectura sin
contexto previo; los terminos tecnicos estan en el glosario (documento 10) y la
auditoria que motiva esta fase es el documento 15.

## 1. Que cambia respecto a v5

v5 respondio una pregunta: con el extractor congelado y cero etiquetas en el,
las features auto-supervisadas de DINOv2 superan a las supervisadas de
ImageNet (gano 6 de 6 comparaciones en la corrida de Kaggle). v5 no entrena
ningun backbone; solo evalua.

v6 da el paso siguiente: pre-entrenar nosotros, sin etiquetas, sobre el pool
dermatoscopico, en las GPU del cluster. La confirmacion de que el hardware
disponible es exclusivamente GPU (sin TPU) simplifica el plan del documento 15:
desaparecen los riesgos de torch_xla y xFormers, y el codigo es PyTorch
estandar con DistributedDataParallel y precision bf16.

## 2. Los dos brazos de pre-entrenamiento

Ambos brazos son estrictamente libres de etiquetas. La evaluacion posterior es
la de v5, sin cambios: backbone congelado, probes kNN/prototipos/regresion
logistica, compuertas G0-G2.

1. Brazo principal, continual pretraining de DINOv2. Se parte de los pesos
   publicos de DINOv2 ViT-S/14 (entrenados sin etiquetas sobre 142M de
   imagenes) y se continua el objetivo DINO de auto-destilacion sobre el pool
   de piel. Hipotesis: adaptar el extractor a la textura de piel, bordes de
   lesion y artefactos del dermatoscopio supera al generalista congelado que
   gano en v5.

2. Brazo purista, DINO desde inicializacion aleatoria (`dino_scratch`).
   El mismo ViT-S/14 y el mismo objetivo DINO del brazo principal, pero sin
   ningun peso externo. Al compartir arquitectura y objetivo, la comparacion
   contra el brazo 1 aisla UNA sola variable: los pesos de partida. La
   diferencia medida es entonces atribuible exactamente a lo que aportan las
   142M de imagenes externas, con todo lo demas constante. Se espera que
   pierda (48-70 mil imagenes contra 142 millones); reportarlo con numeros es
   parte del aporte, porque acota que consigue el SSL con recursos y datos
   propios unicamente. Nota de decision: este brazo reemplaza al SimSiam
   ResNet-50 como purista principal a sugerencia del director. La razon
   defendible no es que ResNet "no sea SSL" (con init aleatoria es
   estrictamente libre de etiquetas); es que SimSiam-ResNet diferia del brazo
   1 en tres variables a la vez (arquitectura, objetivo e inicializacion) y
   la comparacion quedaba confundida. Riesgo asumido: un ViT desde cero con
   este pool entrena mal (el DINO original uso 1.28M de imagenes); el
   indicador de colapso en MLflow lo hace visible temprano.

3. Brazo de ablacion opcional, SimSiam desde inicializacion aleatoria.
   ResNet-50 sin pesos externos (la receta validada de Chen & He 2021 a
   pequena escala). Se corre solo si sobra presupuesto de GPU-horas; aporta
   un segundo punto purista con otra familia de arquitectura y otro objetivo.

4. Metodo propuesto de la tesis (`simsiam_dinov2`, tabla del documento 17):
   el objetivo SimSiam sobre el backbone DINOv2 ViT-S/14, partiendo de los
   pesos publicos libres de etiquetas. La apuesta: un objetivo de dos vistas
   (sin multi-crop, sin cabeza teacher) hace la adaptacion continua mas
   barata en computo que DINO completo, con la misma garantia libre de
   etiquetas de punta a punta. Comparado contra el brazo 1 a igual
   presupuesto, la pregunta es si el objetivo simple alcanza al complejo
   cuando ambos parten de los mismos pesos. Riesgo declarado: SimSiam sin
   teacher-EMA ni centering es mas propenso al colapso sobre ViT; el
   indicador de desviacion estandar del embedding en MLflow lo detecta
   temprano, y un colapso documentado tambien es resultado.

La condicion de comparacion es a igual presupuesto de computo (GPU-horas), no
a igual numero de epocas. Cada corrida guarda checkpoints del backbone en
fracciones fijas del presupuesto (10, 25, 50 y 100 por ciento), y cada
checkpoint se evalua con el pipeline v5. Eso produce la curva
"calidad en funcion del computo", que es la novedad 1 del documento 15.

Las otras dos novedades tambien viven en esta fase:

- Constancia de color como invarianza (novedad 2): shades-of-gray o gray-world
  se aplican de forma aleatoria e independiente a cada vista de un par SSL, en
  lugar del preproceso deterministico de la literatura.
- Multi-crop guiado por la lesion (novedad 3): los recortes locales de DINO se
  sesgan hacia el anillo del borde de la lesion, calculado con umbral de Otsu
  sobre los pixeles, sin usar ninguna etiqueta. Si la segmentacion resulta
  degenerada, el muestreo cae a uniforme y la tasa de fallback queda registrada
  en MLflow. El interruptor --no-lesion-guided permite la ablacion.

## 3. Unrolled networks: evaluado y descartado

Se evaluo la sugerencia de incorporar unrolled networks (algorithm unrolling).
La tecnica desenrolla las iteraciones de un algoritmo de optimizacion como
capas de una red; requiere un modelo directo de medicion (y = Ax + ruido) y por
eso su literatura vive en problemas inversos: reconstruccion de resonancia
magnetica, microscopia de super-resolucion, deblurring. La clasificacion de
dermatoscopia no tiene operador de medicion que desenrollar, y la busqueda
bibliografica no encontro ningun trabajo que combine unrolling con
clasificacion dermatologica. Incorporarlo seria una novedad sin motivacion
clinica ni matematica en este problema, y dispersaria el proyecto (el
documento 15 ya recomienda concentrarse en tres novedades coherentes). Queda
como trabajo futuro si el proyecto derivara hacia mejora o reconstruccion de
imagen, que si es territorio natural de la tecnica.

## 4. Trazabilidad con MLflow

Cada corrida de pre-entrenamiento registra en MLflow:

- Parametros: la configuracion completa (metodo, backbone, resolucion, lotes,
  semilla, presupuesto), el tamano de mundo DDP y el learning rate efectivo.
- Metricas por paso: perdida SSL, learning rate, desviacion estandar del
  embedding (indicador temprano de colapso), GPU-horas transcurridas,
  imagenes por GPU-hora y tasa de recortes guiados por lesion.
- Artefactos: los checkpoints del backbone en cada fraccion del presupuesto y
  el backbone final.

No se necesita servidor: el backend es un directorio en el filesystem
compartido del cluster. Los nodos de computo escriben ahi y la interfaz se
consulta desde el nodo de login:

    mlflow ui --backend-store-uri file:/scratch/derm/mlruns

Si mlflow no esta instalado, la corrida no falla: entrena igual y avisa en voz
alta que no esta siendo registrada. Bajo DDP solo el rango 0 registra, para no
crear una corrida duplicada por GPU. En Kaggle se mantiene el registro
CSV/JSON del notebook v5; MLflow es para el cluster.

## 5. Como se lanza

    # una GPU
    python scripts/pretrain.py --method dino_continual \
        --image-root /scratch/derm/pool --output-dir /scratch/derm/runs/dino \
        --budget-gpu-hours 8 --tracking-uri file:/scratch/derm/mlruns

    # cuatro GPU en un nodo
    torchrun --nproc_per_node=4 scripts/pretrain.py --method dino_continual ...

    # humo: valida todo el cableado en minutos
    python scripts/pretrain.py --method simsiam --smoke \
        --image-root /ruta/con/pocas/imagenes --output-dir /tmp/smoke

La validacion de configuracion corre antes de importar torch: una bandera mal
puesta falla en milisegundos, no despues de esperar la cola del cluster. La
reanudacion tras una caida conserva el tiempo ya gastado, de modo que el
presupuesto de GPU-horas no se reinicia.

## 6. Garantia de cero etiquetas

La invariante central es ejecutable, no declarativa:

- La configuracion no tiene ningun campo donde expresar un archivo de
  etiquetas, y un escaner rechaza cualquier campo o ruta con aspecto de
  supervision (assert_label_free).
- El recolector de datos solo devuelve archivos de imagen; las tablas de
  metadatos son invisibles para el pipeline por construccion.
- Los datasets entregan tensores de imagen unicamente; no existe la tupla
  (imagen, etiqueta) en ninguna firma.

## 7. Estado de verificacion

El nucleo logico de v6 (configuracion, presupuesto, schedules, constancia de
color, guia por lesion, tracking) es numpy puro y esta cubierto por la suite:
184 pruebas en verde, menos de un segundo, sin GPU. La capa torch se verifica
por compilacion en la suite y funcionalmente con la corrida de humo; esa
corrida es el primer paso obligatorio en el cluster antes de gastar el
presupuesto real.

## 8. Pendientes que dependen del laboratorio

Sin cambios respecto al documento 15, en orden de impacto:

1. Configuracion de una corrida real de DINOv2 del laboratorio (batch,
   multi-crop, schedule, temperatura del teacher). Los valores actuales son
   provisionales y estan marcados como tales en el codigo.
2. Gestor de colas, particion, walltime y GPU por nodo.
3. Entorno exacto (PyTorch, CUDA, contenedor) para fijar versiones en
   requirements-pretrain.txt.
4. Salida a internet de los nodos y espacio de scratch (30-50 GB).
5. Rotar el token de Kaggle expuesto antes de usarlo en maquinas de la
   universidad.
