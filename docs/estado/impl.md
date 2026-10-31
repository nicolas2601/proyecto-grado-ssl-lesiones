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

Los dos contratos tematicos quedaron entregados, en el orden pedido, como dos reportes separados.

| Contrato | Reporte | Estado |
|---|---|---|
| `seg-imapp-holdout.md` | `seg-imapp-holdout.reporte.md` | entregado, commiteado en `44a1c5a` |
| `config-modelos-investigacion.md` | `config-modelos-investigacion.reporte.md` | entregado, **sin commitear** |

Cada contrato tiene su propio bloque «Reporte» completo con su verificado / no verificado /
bloqueos. Lo que sigue es solo lo que cruza los dos.

**Verificado.** Acceso a UNABIA por SSH (alias `unabia`, host `jupyter-nmoreno534`), solo lectura,
para el primer contrato; el segundo se hizo sin tocar el cluster, por su propio Fuera de alcance.
Las cifras de segmentacion se recalcularon en local desde los `cases.jsonl` crudos con delta
**0,000e+00** contra los JSON del cluster, y el invariante `dice == 2*TP/(gt+pred)` cierra con
error maximo 4,8e-12 en 12.970 casos. **023 corrio y completo** (`status: completed`,
`Epoch 300/300`, 11.085 s): la creencia de la usuaria era correcta.

**Tres hallazgos que cruzan los dos contratos.**
1. **IMA++ en 023 son 100 imagenes, no el 100 %** (0,67 % del dataset), y **ningun recibo de
   procedencia lo acredita**. Eso bloquea tanto la auditabilidad de 023 como cualquier propuesta
   de escalar IMA++. Es la propuesta P5 del segundo reporte.
2. **El copy-paste no se reproduce entre semillas** en los dos estratos mayores (el `>=10%`
   concentra el 59,3 % de los casos). Descarta una familia entera de propuestas: no se capitaliza
   el efecto, se cierra la pregunta o no se toca.
3. **Ningun run entrega un pool de piel apto para CIRCLe sin filtro por caso** (contaminacion p95
   entre 0,352 y 0,396). Pasa de ser un hallazgo de segmentacion a ser el criterio de exito
   registrado antes del dato de la propuesta P4.

**No verificado.** Lo que falta en cada linea esta en el bloque «Reporte» de su propio reporte.
Transversal: no se corrio `pytest` ni `--check` de los generadores `build_*`; esta sesion no
modifico codigo ni tablas generadas, solo escribio documentacion.

**Bloqueos.** El principal es de presupuesto: P2 pide 7,2 GPU-h contra ~4,3 de margen, y sin esa
decision el unico resultado SSL positivo del proyecto (VICReg +0,041) queda provisional de forma
permanente. El segundo es de procedencia: P5, el cohorte IMA++ de 023.

**Alcance respetado.** No se entreno, no se lanzo ninguna corrida, no se cambio ningun gate
(`execution_enabled`, `launch_ready`, `training_started`, `execution_ready`, `verified_for_use`,
`patient_independence_accredited`), no se modifico nada en UNABIA, no se renombro ni movio ningun
`scratch_*`, no se toco la linea de clasificacion de Nicolas ni la documentacion del paper.
