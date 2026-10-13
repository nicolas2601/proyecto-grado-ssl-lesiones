# 42. W3: SLED corregido, USL-Net según texto y revisión completa de RPI-Net

Revisión del 2026-10-12 (America/Bogota). Base `df53048` de
`experiment/modular-audit-v8`. Este documento actualiza los pendientes del
registro 41; preserva los resultados históricos y las puertas de 35/36/39.

## Resultado concreto y alcance

Se implementaron componentes propios y un adaptador serial de SLED que requiere
una copia externa limpia y fijada del repositorio del autor. No se incorporó
código de terceros ni PDF editorial al repo. No se ejecutó el JAR, no se instalaron
paquetes en el entorno histórico, no se abrió GT, no hubo entrenamiento ni
inferencia de maestros sobre imágenes reales. Las pruebas sintéticas verifican
comportamiento de componentes, no exactitud anatómica ni reproducción completa.

| Método | Implementado en esta etapa | Pendiente antes del primer caso real |
|---|---|---|
| SLED | Gaussiano HxW, control por área, selección de componente, fusión estable, errores ante degeneraciones y adaptador a núcleo externo fijado | Uso de fuente externa, entorno aislado con pyod/JPype/JVM y regresión de intermediarios contra referencia; timeout/RAM medidos |
| USL-Net según texto | UM con mínima suma de distancias, CCAM 1x1 con complemento, decoder FCN multiescala y BCE con ignore | Pesos/origen de las tres extracciones, objetivo de CCAM, fusión e integración completa y definición de exclusión por distancia |
| RPI-Net | Revisión del PDF completo y especificación de módulos | Código/pesos oficiales no identificados; cerrar detalles faltantes antes de reimplementar |

Los tres mantienen `ready_for_pilot=false`. Una ablación de centralidad sobre S0
no es un nuevo maestro USL-Net. W4 permanece diseñado.

## SLED: modernizar sin confundir cambios del método

Fuente fijada: `SELGroup/SLED@3213eacb65577c0d52f424369158fb56c609238e`.
No es necesario imponer las versiones de 2023 a la .venv histórica. El adaptador
admite un checkout externo dentro de un proceso/entorno dedicado. Comprueba HEAD,
árbol limpio, ausencia de symlinks y JVM previa; usa ruta absoluta del JAR y cwd
temporal exclusivo. Esto verifica procedencia, no constituye un sandbox ni limita
recursos: el lanzador de piloto todavía debe aplicar timeout y presupuesto RAM.

- `gaussian_corrected`: Gaussiano normalizado HxW, sigma `0.15*max(H,W)`.
- `upstream_area`: control de selección con pesos uniformes, equivalente al efecto
  del kernel 1x1 observado en la revisión fijada para valores normales. No es
  una reproducción completa del entorno original.
- Ambas variantes usan la misma intersección círculo/rectángulo central,
  conectividad 8 y conservan la región ganadora completa. El llenado de huecos
  se restringe al campo válido: cambio adicional explícito de seguridad.
- Fusión multiescala: softmax estable de los pesos exponenciales originales;
  rechaza varianza mínima cero, mapas no finitos o geometría incompatible.
  Un fallo no se sustituye por una máscara vacía ni se elimina del denominador.
- IForest obtiene semilla explícita; las diez escalas se ejecutan en serie,
  sin el ProcessPool original. La semilla es una decisión de adaptación.
- Se conserva el comportamiento upstream del centroide de la etiqueta cero.
  Su corrección, si se prueba después, requiere otra variante registrada.
- Defaults GHT cotejados con `main.py`: nu=1e30, tau=.1, kappa=1e-30,
  omega=.9. Entrada primaria del protocolo 224x224, distinta del tamaño nativo
  560x768 del ejemplo: se reportará como adaptación geométrica.

Una migración compatible necesita comparar SLIC, grafo, particiones (invariantes
al nombre de etiquetas), scores y máscara antes de declarar equivalencia. El
código puede estar disponible sin que esa equivalencia esté demostrada.

## USL-Net: ambigüedades confirmadas en las páginas originales

PDF arXiv v3, páginas 5 y 6, inspeccionadas visualmente además de extracción.

| Punto | Documento | Decisión explícita del código propio |
|---|---|---|
| Centralidad | Texto/ecuación 2: menor suma de distancias; Algoritmo 1 usa `>` | Seguir texto: mínimo, centro H/2,W/2; no distancia de centroide |
| Regiones pequeñas | Algoritmo inicializa región 0 antes de filtrar | Filtrar todas las regiones con área menor de 500 |
| Conectividad | Flood-fill, vecindad no especificada | 4 vecinos declarados en política; 8 disponible como contraste |
| Umbrales | Fondo .35 y lesión .65 | Comparaciones estrictas; igualdad se mantiene incierta |
| Región muy distante | Texto indica excluir, no proporciona corte numérico | Corte opcional sin valor inventado; hay que fijarlo antes de piloto |
| Pérdida | Prosa: BCE ignorando inciertos; ecuación 3 repite log(1-p) | BCEWithLogits promediada sobre píxeles confidentes; inciertos sin gradiente |
| CCAM | M es fondo y M' su complemento; conv1x1 | Cabeza 1x1, sigmoid de fondo y complemento para saliencia de lesión |
| FCN | Fusión a 56x56x256, conv1x1 y bilinear | Proyecciones laterales 1x1 a 256 y suma; decisión de implementación explícita |

La ecuación 3 literal no enseña a distinguir foreground/background: en ambos
casos su término depende de log(1-p). Tampoco se reproduce silenciosamente el
factor .25 de su expresión. La implementación sigue el propósito expresado en
la prosa, no afirma resolver por autoría una errata editorial.

Nomenclatura ResNet: las etapas conv2/3/4 del diagrama corresponden a
`torchvision.layer1/2/3` (256/512/1024 canales), no a layer2/3/4 por coincidencia
de nombres. Las saliencias iniciales requieren las extracciones contrastivas
SimCLR/MoCov1/MoCov2 y su fusión; las cabezas implementadas no sustituyen sus
pesos ni entrenamiento. El presupuesto nativo cinco ciclos de 300 épocas no se
lanza desde estas pruebas.

## RPI-Net: qué aporta realmente el PDF completo

DOI 10.1016/j.bspc.2025.108433; disponible online 14-08-2025, volumen 112 (2026).
Los autores presentan continuidad explícita con USL-Net: umbrales manuales e
ignore son sustituidos por modelado aprendido de ruido, interacción y ponderación.

- P1: K-means de conv1 de MoCov3/ResNet50.
- P2: Grad-CAM++ de layer4 de MoCov3; el texto informa sustitución por CCAM
  en ISIC-2017. No trasladar una receta única sin registrar este detalle.
- P3: K-means de conv1 de MoCov2/ResNet50.
- P4: K-means de la capa final DINO/ViT, no DINOv2 por sustitución implícita.
- IPC: queries de imagen; keys/values de cada pseudomáscara. PIA: atención entre
  pseudomáscaras distintas y residual de IPC. La segmentación final proviene
  de la rama imagen; la rama de pseudomáscaras predice incertidumbres.
- DUL, ecuaciones 20–23: `BCE(mu,Pi)*exp(-sigma_i)+.5*sigma_i`, ponderada por
  inversa de promedio de `softplus(sigma_i)^2` y normalizada entre fuentes.
  Ese "entropy" es una función definida por autores, no entropía binaria usual.
- Inicial 50 épocas y fases posteriores de 30. La receta disponible no fija
  completamente optimizador, número de fases, polaridad K-means, target de CAM,
  reducciones ni convenciones de gradiente de alpha: son pendientes explícitos.

La memoria de atención espacial densa crece cuadráticamente con tokens. No
materializar atención a 224x224 sin medir; reducir tokens cambia la implementación
y debe declararse. Se necesita especificar gradientes de pesos dinámicos y probar
extremos de sigma para evitar overflow o comportamiento degenerado de la pérdida.

### Comparación publicada, no resultados del proyecto

| Dataset | SLED Dice | USL-Net Dice | RPI-Net Dice |
|---|---:|---:|---:|
| ISIC-2018 | 77.68 | 78.79 | 82.88 |
| ISIC-2017 | 73.90 | 80.45 | 75.84 |
| PH2 | 90.34 | 88.90 | 89.31 |

Valores transcritos de tabla 1 del PDF RPI-Net, en porcentaje. No son nuestros
experimentos ni garantizan validez en PAD. RPI-Net mejora ISIC-2018, pierde frente
a USL-Net en ISIC-2017 y supera ligeramente USL-Net en PH2, donde SLED obtiene
más Dice. El párrafo de resultados cita 82.85 en vez del 82.88 de tabla/resumen;
se prioriza tabla y se registra la discrepancia. No hay superioridad universal.

No hay URL oficial de implementación en el PDF revisado. Esto no demuestra
inexistencia de código. Una reimplementación futura necesita ficha de decisiones;
no se sustituye RPI-Net por otro método con nombre igual.

## Artículo adicional: TriDermNet-HAGO

DOI 10.1016/j.bspc.2026.111536, disponible online 08-10-2026; volumen 130 (2027).
Combina ramas Region/Edge/Token y segmentación/clasificación supervisadas.
Sección 5.5 describe Dice+BCE con máscaras, CE diagnóstica, edge y consistencia.
Su tabla 11 distingue métodos sin etiquetas del aprendizaje conjunto supervisado.
Por ello no entra como maestro W3 sin etiquetas; sus ideas de borde y balance
pueden inspirar una adaptación, que necesitaría etiquetas sustitutas y otra
validación. No mezclar sus cifras con las de SLED/USL/RPI.

Dos cautelas de reproducibilidad observadas: 5.1 describe división 80/10/10,
mientras 5.5 menciona splits oficiales; abstract y tablas principales reportan
cifras diferentes. La sección 5.5 promete liberar código, sin aportar URL.
No se considera disponible una implementación por esa promesa.

## Verificación y siguiente puerta

En la copia aislada del clúster, usando la .venv existente, pasaron 13 pruebas:
9 de geometría/selección/fusión/abstención y 4 de cabezas/gradientes PyTorch.
Entre otras verifican que corregir el gaussiano cambia un ganador, que suma de
distancias difiere de centroide, y que foreground/fondo generan gradientes
opuestos con gradiente cero en inciertos. Son entradas sintéticas, no pacientes.

Siguiente: resolver dependencia externa y pesos, configurar entorno separado,
verificar ejecución de una imagen con timeout/RAM y tracking, y recién entonces
los 100 originales ya registrados. El seguimiento de un piloto real debe usar
el logger común y netrc configurable, sólo métricas/config agregadas sin imágenes
ni IDs. Esta etapa tiene una corrida sintética en W&B, no una corrida de segmentación.
La evidencia se conserva en `resultados/auditoria_v8/w3_components_20261013/verification.json`.
Run verificado: https://wandb.ai/nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion/runs/150c830867ad .
Se verificó además que el círculo analítico coincide píxel a píxel con OpenCV
en 100x100, 224x224 y 560x768. Los nuevos logs W&B quedan fuera de históricos,
en una carpeta dedicada; no se modificaron ni borraron las corridas previas.

Fuentes completas aportadas por usuario (hash, no archivos editoriales publicados):

- RPI-Net: SHA256 `c9a71996c77e52a5042c78c3152d25189f342a16b36b46cc8abe2f1404b448a6`.
- USL-Net v3: SHA256 `30bd6a6ada51111027e474faa7b1a187ff16495f45927c2ea52b75d75a8907b1`.
- TriDermNet-HAGO: SHA256 `3385f298e287a3d6296ae9929b37a451fbbbbe90ba1f29ae3f3ccf0ec5ed5694`.
