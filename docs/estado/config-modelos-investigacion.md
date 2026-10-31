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
Verificado: <con que comando>
No verificado: <y por que>
Bloqueos:
