# Sesion: doc  |  2026-10-30

## Tarea

Preparar el terreno de la documentacion sin escribirla todavia. Via A, que esta **congelada**
mientras siguen las pruebas: no redactas capitulos, no tocas diagramas, no reescribis el paper.

Lo que si haces: dejar listo el inventario de que material esta en condiciones de documentarse y
que le falta a cada pieza.

## Entra

- `docs/60-guia-presentacion-cientifica.md` — el estandar acordado. Leelo entero primero.
- `docs/ESTADO.md` — estado por objetivo
- `resultados/TABLA_CIERRE.md` y `resultados/segmentacion/TABLA_SEGMENTACION.md`
- `resultados/clasificacion/v7/README.md` y `resultados/clasificacion/pad_ufes/README.md`
- `docs/README.md` — el indice numerado
- `C:/Users/mpaul/Documents/Anteproyecto/capitulos/` — solo lectura, para ver que hay escrito

No leas codigo fuente. No leas el cluster.

## Sale

Un unico archivo: `docs/estado/doc.reporte.md`, con:

1. **Inventario de material por seccion del paper.** Para cada seccion del esqueleto de
   `60-guia-presentacion-cientifica.md` (1 Introduccion, 2 Trabajo relacionado, 3 Metodologia,
   4 Experimentos, 5 Discusion, 6 Conclusion): que material existe hoy, donde esta, y si esta
   **cerrado** (cifra generada desde JSON, protocolo registrado antes del dato, veredicto
   escrito) o **provisional**.
2. **Lo que falta para cada tabla** de las cuatro que exige el estandar: estado del arte,
   ablacion, robustez, eficiencia. Para cada una: que datos hay, cuales faltan, de donde
   saldrian.
3. **Inventario de figuras**: que figuras existen en `assets/figuras/` y en
   `docs/figuras_analisis/`, cuales cumplen el estandar de representacion grafica de la
   seccion 4 de la guia, y cuales habria que rehacer. **Solo inventario y veredicto, no las
   rehagas.**
4. **Desajustes entre el repositorio y el anteproyecto**: afirmaciones del anteproyecto que hoy
   son falsas. Dos ya detectadas: `tab:trazabilidad-fases` fila 2 («el plan nuevo no se lanzo»)
   y fila 4 («prototipo aun no implementado»). Buscá si hay mas.
5. **Estado del indice `docs/README.md`**: que documentos numerados existen y no estan
   listados. Se sabe que llega hasta el 50 y existen hasta el 60.

## Criterios (binarios)

1. Cada pieza de material se clasifica como **cerrada** o **provisional**, con el criterio
   explicito que lo justifica.
2. Ninguna cifra se copia al reporte: se cita la ruta del archivo que la contiene.
3. El inventario de figuras emite un veredicto por figura contra el estandar de la guia.
4. Los desajustes con el anteproyecto citan el archivo `.tex` y la linea.
5. Se distingue lo verificado de lo no verificado.

## Fuera de alcance

- **No redactes ninguna seccion del paper ni de la tesis.**
- **No crees ni modifiques ninguna figura ni diagrama.** Estan congelados por decision del
  2026-10-30.
- No toques `C:/Users/mpaul/Documents/Anteproyecto`: solo lectura.
- No regeneres tablas ni corras los generadores `build_*`.
- No commitees nada.

## Coordinacion

- Si encontras algo que afecta a `impl` o a `research`, **no les escribas directamente**:
  decimelo a mi (`maestro`). Ver `docs/GUIA-SESIONES.md`, seccion «Mensajeria entre sesiones».

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
