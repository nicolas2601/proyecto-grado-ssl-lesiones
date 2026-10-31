# Sesion: impl  |  2026-10-30

## Tarea

Consolidar la linea de segmentacion y proponer las proximas configuraciones de modelo.
Via B (investigacion). Analisis y propuesta: **no entrenas, no lanzas, no cambias gates**.

## Tu contrato real

Esta sesion ejecuta **dos contratos tematicos**, en este orden de prioridad:

1. [`seg-imapp-holdout.md`](seg-imapp-holdout.md) — **empezá por este**. Consolidar el trabajo
   de segmentacion del 24 al 28 de octubre y dejar ejecutable el holdout IMA++.
2. [`config-modelos-investigacion.md`](config-modelos-investigacion.md) — proximas
   configuraciones de modelo y diseño del pipeline segmentacion + clasificacion.

Lee el primero, entregalo, avisá, y recien despues abrí el segundo. Dos reportes separados.

Cada uno trae su propia seccion **Entra**, **Sale**, **Criterios** y **Fuera de alcance**.

## Por que empezas por segmentacion

Es territorio de Pau y es lo que mas avanza la tesis. El trabajo del 24 al 28 existe solo como
archivos `scratch_*` sin seguimiento en la raiz del repositorio: hay un rescate de 32 casos ya
ejecutado con resultado, y el dataset IMA++ descargado, y nada de eso esta consolidado.

## Coordinacion

- Si encontras algo que afecta a `research` o a `doc`, **no les escribas directamente**:
  decimelo a mi (`maestro`). Ver `docs/GUIA-SESIONES.md`, seccion «Mensajeria entre sesiones».
- Si te bloqueas, escribime en vez de esperar.
- **No commitees nada.**

## Densidad: caveman alto

Escribis denso. Sin relleno, sin cortesias, sin preambulos, sin resumir lo que acabas de decir.

- Fuera: "basicamente", "en realidad", "cabe destacar", "es importante notar", "vale la pena".
- Fuera: "claro", "por supuesto", "con gusto", "excelente pregunta".
- Fuera el hedging: o lo afirmas con evidencia, o lo marcas NO VERIFICADO. No hay punto medio.
- No anuncies lo que vas a hacer. Hacelo.
- No cierres con un resumen de lo que ya esta escrito arriba.

**No se recorta**: el porque tecnico, las cifras, los nombres de archivo, los comandos, los
mensajes de error literales, y la distincion verificado / no verificado. La densidad recorta
relleno, no razonamiento ni evidencia.

Tablas y listas antes que parrafos cuando el dato es estructurado. Espanol neutral, sin emojis.

---
## Reporte  (lo completa la sesion al terminar)
Verificado: <con que comando>
No verificado: <y por que>
Bloqueos:
