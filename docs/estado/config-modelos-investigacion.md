# Sesion: config-modelos-investigacion  |  2026-10-30

## Tarea

Proponer las configuraciones de modelo que corresponden probar a continuacion, derivadas de los
resultados ya medidos. Solo analisis y propuesta: no entrenar, no ejecutar, no tocar gates.

## Contexto minimo

Via B (investigacion), no via A (documentacion). Aca si entra lo provisional, siempre marcado
como tal.

Resultados que condicionan la propuesta, todos ya medidos:

- **Clasificacion, campana v3** (protocolo PAD `target_adapt`, 1.134 imagenes, CV anidada
  agrupada por paciente, desarrollo/transductivo): baseline SSL media de 3 semillas
  **PAD 0,597**. VICReg **0,638** (+0,041) y Barlow Twins **0,633** (+0,035) cumplen el criterio
  declarado pero son **PROVISIONALES**: 3 contra 3 semillas y permutacion con p minimo 0,10.
  Nulos o peores: DINO+iBOT, MULAN, BYOL, OOB, freeze4, 336, crop 0.4, destilacion, recetas v1,
  SkinDisNet. Todas las palancas de SimSiam caen dentro de +-0,006 del baseline.
- **El encoder es la palanca, no el lector**: SigLIP2-so400m congelado 0,6668; con metadatos
  0,7027; SigLIP2-base con metadatos 0,7029. SigLIP2 es imagen-texto, **no SSL**.
- **Segmentacion**: DermLIP-SLIC Run 008 congelado, Dice medio **0,7904** [0,7824; 0,7983] sobre
  2.594 pares de ISIC 2018, con 32 casos de Dice cero taxonomizados.
- **Resultado negativo ya medido en v7**: la mascara de segmentacion **no ayudo** al clasificador
  (0 de 15 celdas mejoran, 4 empeoran) usando `student_r3` con Dice 0,7177.
- Techo honesto de literatura con split por paciente: **~0,72** macro-F1. El 0,80 no es
  reproducible para 6 clases.

## Entra

- `docs/ESTADO.md`, secciones Campana v3, Objetivo 2 y Objetivo 3
- `resultados/TABLA_CIERRE.md` (filas 1 a 4 para el resultado negativo de la mascara)
- `resultados/clasificacion/pad_ufes/README.md`
- Engram del equipo: observaciones `#2224`, `#2225`, `#2240`, `#2250`, `#2262`, `#2266`, `#2268`
- `docs/60-guia-presentacion-cientifica.md`, seccion 6, para saber que forma debe tener la
  evidencia que se busca

## Sale

Un unico archivo: `docs/estado/config-modelos-investigacion.reporte.md`, con:

1. **Lectura del espacio ya explorado**: que familias de metodo se probaron, cuales movieron la
   metrica y cuales no, y que patron explica esa diferencia.
2. **Propuesta de configuraciones**, maximo cinco, cada una con: hipotesis en una frase, cambio
   exacto respecto del baseline, criterio de exito **registrado antes del dato**, costo estimado
   en GPU-h, y que resultado previo la justifica.
3. **Diseno del pipeline segmentacion + clasificacion**: brazos, factor unico que varia,
   metrica, prueba estadistica y criterio de parada.
4. **Que NO vale la pena volver a probar**, con la evidencia que lo descarta.
5. **Orden recomendado**, condicionado a la decision pendiente sobre el entregable.

## Criterios (binarios)

1. Cada propuesta cita el resultado previo que la motiva, con su cifra e intervalo.
2. Cada propuesta declara su criterio de exito antes del dato, no despues.
3. Se distingue explicitamente que propuestas son SSL puro y cuales no, porque el objetivo
   general de la tesis exige aprendizaje autosupervisado.
4. El diseno del pipeline varia **un solo factor** entre brazos.
5. Se declara el costo en GPU-h y se respeta un tope declarado.
6. Nada se presenta como confirmado si su soporte es provisional.

## Fuera de alcance

- No entrenar, no evaluar, no lanzar nada.
- No cambiar ningun gate de ejecucion ni ningun config existente.
- No conectarse al cluster ni al servicio desplegado.
- No decidir cual es el entregable de la tesis: eso es de la sesion maestra con Nicolas y Karen.
  Si, decir que implicaciones tiene cada opcion para la propuesta.
- No tocar la documentacion del paper ni los diagramas: estan congelados.

---
## Reporte  (lo completa la sesion al terminar)

Entregado: `docs/estado/config-modelos-investigacion.reporte.md` (2026-10-31). Detalle completo
en su seccion «Reporte». Resumen:

**Verificado.** `TABLA_CIERRE.md` filas 1-9, `docs/ESTADO.md` Objetivo 2 (85-166) y Objetivo 3
(167-262), `docs/60` seccion 6, y las siete observaciones del engram del equipo traidas **por ID**
(#2224, #2225, #2240, #2250, #2262, #2266, #2268). Costos en GPU-h derivados de mediciones
reales, no estimados: 1,065-1,282 h por brazo SSL de 6.000 pasos; 3,08 h para un brazo de
segmentacion de 300 epocas.

**Patron que organiza la propuesta.** Lo que mueve la metrica es el contenido de informacion de
la representacion, no el procedimiento de adaptacion. El lector esta saturado (siete arquitecturas
convergen a 0,59-0,606), mas datos del mismo dominio no aportan (SkinDisNet +0,0003
[−0,0087; +0,0100]) y las perillas de receta caen dentro de ±0,006. Lo que si mueve: cambiar la
familia de objetivo SSL (+0,035-0,041) y cambiar el corpus de preentrenamiento (+0,0605, pero no
es SSL). Mecanismo candidato: VICReg y Barlow regularizan la covarianza del embedding, y el
baseline tiene rango 428 de 768.

**Cinco propuestas, 9,9 GPU-h en total**, cada una con hipotesis, cambio exacto, criterio de exito
registrado antes del dato, costo y el resultado previo que la motiva. Las cinco declaran si son
SSL puro.

**Correccion a este contrato.** «Segmentacion: Run 008, Dice 0,7904» esta superado: los runs
019-023 llegan a **84,16 %** de Dice medio sobre los mismos 2.594 pares.

**Error propio, corregido.** Una version previa de este bloque afirmaba que
`resultados/clasificacion/pad_ufes/README.md` no existia. **Era falso**: existe, 28.540 bytes, y
es la tabla sellada del protocolo v7 zero-shot (1.141 imagenes, 720 pacientes, 3 clases),
generada por `scripts/build_pad_ufes_table.py` con `--check` en CI. La verificacion original uso
`ls -la … | head`, que corto la salida antes de `README.md`. El contrato tenia razon al listarlo
en «Entra». Reusado ademas para acotar el alcance de P4: su linea 7 declara que el grupo III-VI de
PAD es 84 % tipo III (392/465) y que **PAD no permite concluir sobre FST V-VI**.

**Pushback registrado.** No se propone repetir «mascara como entrada del clasificador» con la
mascara mejor. El negativo medido no fue nulo sino **negativo en la condicion (b) fondo gris**
(hasta −0,0916 [−0,1409; −0,0405]): el dano viene de destruir la piel perilesional, no de un
contorno impreciso. Subir el Dice no toca ese mecanismo. Se propone en cambio el rol que nunca se
midio: la mascara como definicion del **pool de piel** para normalizar tono.

**No verificado.** Ninguna propuesta se ejecuto. El rango efectivo de VICReg/Barlow no esta
medido (es P1). No se verifico que los checkpoints de la campana v3 sigan disponibles: viven
fuera de este repo y «no conectarse al cluster» esta en Fuera de alcance.

**Bloqueos.** P2 (cerrar VICReg/Barlow con potencia) pide 7,2 GPU-h contra ~4,3 de margen bajo el
tope historico de 20; sin esa ampliacion, el unico resultado SSL positivo del proyecto queda
provisional de forma permanente (permutacion p = 0,10 con 3 contra 3). P5 (congelar el cohorte
IMA++ de 023) es prerrequisito de todo lo que toque IMA++ y de la validez de las cifras de 023.

**Alcance respetado.** No se entreno, no se evaluo, no se lanzo nada, no se cambio ningun gate ni
config, no se accedio al cluster ni al servicio desplegado, no se commiteo.
