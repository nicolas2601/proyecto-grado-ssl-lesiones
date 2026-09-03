# 19. Resultados de los bancos congelados y como leerlos

Cierra la etapa de comparadores del Objetivo 2: ocho backbones congelados sobre
HAM10000 e ISIC 2020, con el mismo protocolo, la misma semilla y el mismo split
anti-fuga. Las tablas viven en el README principal y se generan con
`python scripts/build_results_tables.py --bank <banco>`; este documento explica
que se puede concluir de ellas y que no.

## 1. Que se corrio

Dieciseis corridas reales en Kaggle (`SMOKE=False`, semilla 42, resolucion 224,
backbone congelado, tres probes: kNN, regresion logistica y prototipos). Los JSON
estan en `results/baselines/` y `results/baselines_isic2020/`, y traen metricas por
clase, matriz de confusion e intervalos bootstrap del 95 por ciento.

## 2. El error de medicion que aparecio, y como se detecto

Al migrar los pesos a Hugging Face se asumio que `timm.create_model(..., num_classes=0)`
devuelve el promedio global del ultimo bloque convolucional. Para VGG19 no es asi:
devuelve las pre-logits de 4096 dimensiones, o sea la salida de las capas totalmente
conectadas fc6 y fc7 de la cabeza. Los pesos eran identicos a los de torchvision,
pero el punto de medicion se habia movido dos capas hacia adentro.

Se detecto porque el pipeline es determinista. Al re-correr HAM10000, AlexNet,
EfficientNet-B0 y ResNet-50 reprodujeron sus numeros al cuarto decimal, mientras
VGG19 se movio 0.0435 y su `dim_features` salto de 512 a 4096. Un cambio aislado en
un pipeline reproducible no es ruido: es un cambio de definicion.

El arreglo lee VGG19 con `gap(m.features)` y, sobre todo, ambos ejecutores
(el generador de notebooks y el runner del cluster) ahora comprueban
`protocol.FEATURE_DIMS` antes de extraer. VGG19 volvio a dar 0.5147, exactamente el
valor previo a la migracion, que es la evidencia de que el arreglo es correcto.

**Leccion transferible**: cuando dos configuraciones deben ser equivalentes, hay que
verificar la equivalencia de la MEDICION, no solo la de los pesos. Una assertion
sobre la dimension del vector de features cuesta una linea y habria hecho fallar la
corrida en el primer segundo en lugar de producir un numero plausible pero falso.

## 3. Que se sostiene

- Ningun backbone generico congelado alcanza el ancla clinica (Gessert, 0.636 de
  balanced accuracy). El mejor de HAM10000 llega a 0.618 y su intervalo incluye el
  ancla, asi que ni la distancia esta resuelta.
- El SSL generico de ImageNet no transfiere a dermatoscopia. SimSiam ResNet-50 es
  ultimo en HAM10000 y su intervalo no toca al del lider: esa diferencia si es real.
  Es el argumento que motiva la fase v6, pre-entrenar sobre piel.
- Ningun backbone congelado sirve para tamizaje en ISIC 2020. La precision de la
  clase maligna va de 0.046 a 0.077 en los ocho, es decir unos trece falsos
  positivos por cada acierto. La balanced accuracy de 0.77 es cosmetica: viene de un
  desbalance cercano a 50 a 1.
- El probe kNN colapsa en HAM10000 (0.26 a 0.40). Con desbalance 58 a 1 el
  vecindario lo domina `nv`. La regresion logistica con pesos de clase es el probe
  informativo.

## 4. Que NO se puede afirmar

El ranking no resuelve su propio medio. En HAM10000, seis de los ocho intervalos se
solapan con el del lider; en ISIC 2020 se solapan los ocho. Las posiciones 1 a 6 no
son un orden de merito: son empates que este conjunto de validacion no separa.

En particular, **no se puede escribir que el SSL le gana al supervisado**. Lo que se
puede escribir es que la ventaja del supervisado desaparece a igual tamano: con los
pesos `augreg_in1k` (el comparador honesto, entrenado solo con ImageNet-1k), el
ViT-B/16 baja a 0.567 y queda 0.051 por debajo de DINOv2 ViT-B/14, pero los
intervalos se solapan. La inversion casi completa del orden en ISIC 2020, donde
AlexNet queda primero y DINOv2 ViT-B/14 ultimo, confirma que ahi el ranking es ruido.

## 5. Consecuencia para el proyecto

La pregunta del Objetivo 2 queda mejor planteada que antes: no se trata de ganarle a
un numero, sino de conseguir una ventaja que sobreviva a su propio intervalo de
confianza. Eso exige, en la fase v6, o bien un efecto mayor (pre-entrenamiento sobre
piel), o bien mas potencia estadistica (varias semillas y validacion cruzada), y
preferiblemente ambas cosas.
