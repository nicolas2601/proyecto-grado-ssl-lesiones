# Contratos de sesion

Canal de entrada y salida de cada sesion trabajadora, segun [SESIONES.md](../SESIONES.md).
La sesion maestra escribe el contrato; la trabajadora lee **solo su archivo** y completa el
bloque Reporte al terminar.

El formato esta definido en `SESIONES.md`. La seccion **Fuera de alcance** es obligatoria: sin
ella la sesion expande la tarea.

## Las dos vias

Ver `docs/ESTADO.md`, seccion «Las dos vias de trabajo».

- **Via A, documentacion**: produce el paper y la tesis. **Congelada** mientras siguen las
  pruebas. Solo admite material cerrado.
- **Via B, investigacion**: produce decisiones y configuraciones. Admite lo provisional, siempre
  marcado como tal. Todos los contratos de abajo son via B.

## Contratos activos

| Contrato | Via | Sesion | Estado | Reporte |
|---|---|---|---|---|
| [app-ux-producto.md](app-ux-producto.md) | B | Producto y experiencia de la app desplegada | Abierto 2026-10-30 | `app-ux-producto.reporte.md` |
| [app-tecnica-contrato.md](app-tecnica-contrato.md) | B | Contrato tecnico app ↔ servicio de inferencia | Abierto 2026-10-30 | `app-tecnica-contrato.reporte.md` |
| [config-modelos-investigacion.md](config-modelos-investigacion.md) | B | Proximas configuraciones de modelo y diseno del pipeline | Abierto 2026-10-30 | `config-modelos-investigacion.reporte.md` |
| [seg-imapp-holdout.md](seg-imapp-holdout.md) | B | Consolidar segmentacion 24-28 oct y dejar ejecutable el holdout IMA++ | Abierto 2026-10-30 | `seg-imapp-holdout.reporte.md` |

Ademas, en curso fuera de contrato: un barrido cronologico del engram del equipo que produce
`indice-vps-20261030.md`. Lo lanzo la sesion maestra.

La app desplegada esta en `https://skin.tikno.pro` y se llama **«Image study — research
prototype»** en produccion. El nombre `Perma` corresponde a un rediseño local sin commitear;
no usarlo como nombre oficial hasta que esa decision este cerrada.

## Reglas

1. Una sesion trabajadora no lee `docs/ESTADO.md` completo: lee su contrato.
2. Si el contrato es insuficiente, la sesion lo dice en Bloqueos y **se detiene**. No improvisa
   alcance.
3. El reporte distingue siempre lo verificado de lo no verificado, con el comando que lo
   demuestra. Ver la regla de verificacion en `~/.claude/CLAUDE.md`.
4. Ninguna sesion trabajadora cambia un gate de ejecucion ni publica nada.
