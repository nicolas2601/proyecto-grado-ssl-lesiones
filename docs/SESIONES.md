# Protocolo de sesiones

Como trabajan en paralelo varias sesiones de Claude Code sobre este repo.

## Principio

**El contexto completo se carga solo al maestro.** Las demas reciben su
porcion: un contrato de tarea, no el proyecto entero.

Razon: una sesion trabajadora con todo el contexto es cara y deriva — empieza
a opinar sobre cosas fuera de su tarea. Una con contrato claro entrega y no
inventa.

## Roles

| Sesion | Modelo | Hace | Carga |
| --- | --- | --- | --- |
| **maestro** | Opus 5.5 | Orquesta, decide, valida. No escribe codigo masivo | `docs/SESIONES.md`, `docs/ESTADO.md`, los 4 objetivos, decisiones abiertas |
| **implementacion** | Opus | Segmentacion, transformacion de color | Su contrato + contratos de datos + `pytorch-patterns` |
| **documentacion** | Haiku 4.5 | `docs/`, catalogo de corridas, triaje de archivos sueltos | Convenciones + el doc objetivo. Nunca el codigo entero |
| **investigacion** | Opus | Literatura, comparacion con la propuesta, justificacion de variantes | La pregunta concreta + `academic-researcher`, `scientific-thinking-literature-review` |

El maestro es el unico que decide. Las demas proponen y reportan.

## Canal entre sesiones

El repo primero, engram despues.

- **`docs/ESTADO.md`** — fuente de verdad del avance. Lo escribe el maestro.
- **`docs/estado/<sesion>.md`** — contrato de entrada y reporte de salida de
  cada sesion trabajadora.
- **engram local** (`engram`) — memoria personal de esta maquina.
- **engram del equipo** (`engram-equipo`, `https://engram-mcp.tikno.pro/mcp`) —
  memoria compartida con Nicolas. Autenticacion OAuth en el navegador.

Un archivo en git es verificable, diffeable y sobrevive si engram falla. Por
eso es el canal primario.

## Formato del contrato de tarea

```markdown
# Sesion: <nombre>  |  <fecha>

## Tarea
<una oracion>

## Entra
- <archivos o datos que puede leer>

## Sale
- <artefacto exacto que debe producir>

## Criterios (binarios)
1. <se cumple o no>
2. <se cumple o no>

## Fuera de alcance
- <lo que NO debe tocar>

---
## Reporte  (lo completa la sesion al terminar)
Verificado: <con que comando>
No verificado: <y por que>
Bloqueos:
```

"Fuera de alcance" es obligatorio. Sin esa seccion, la sesion expande la tarea.

## Apertura de sesion

1. `cd` al repo. **Nunca abrir desde `documents/`**: engram no resuelve el
   proyecto desde una carpeta con varios repos.
2. El maestro lee `docs/ESTADO.md`.
3. Las trabajadoras leen solo su `docs/estado/<sesion>.md`.
4. Retomar una sesion previa: `claude --resume`.

## Cierre de sesion

1. Completar el bloque Reporte del contrato.
2. Commit de lo producido.
3. El maestro actualiza `docs/ESTADO.md`.

## Agentes dentro de una sesion

| Agente | Para |
| --- | --- |
| `triaje` | Clasificar archivos sueltos: conservar, transformar, descartar |
| `explorador` | Ubicar algo en el repo sin llenar el contexto |
| `revisor` | Revision de un cambio escrito |
| `verificador` | Medir contra criterios |

Para decisiones de arquitectura de ML, lanzar 2-3 propuestas en paralelo con
enfoques distintos y comparar, como hace `/feature-dev` en su fase 4.

## Restriccion propia de este repo

No se descarta nada que respalde un numero reportado en el anteproyecto o la
tesis. Ante duda, el veredicto es "no evaluado", nunca "descartar".
