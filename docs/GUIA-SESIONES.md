# Guia de implementacion: trabajo multisesion

Pasos exactos para arrancar. Los flags fueron verificados con `claude --help`.

---

## Paso 0 — Una sola vez

### 0.1 Cerrar esta sesion y abrir dentro del repo

```bash
cd C:\Users\mpaul\Documents\ProyectoDeGrado
claude
```

**Critico:** siempre dentro del repo. Si abris desde `documents/`, engram no
resuelve el proyecto (hay varios repos ahi) y `engram-equipo` no aparece,
porque esta declarado en el `.mcp.json` de este repo.

### 0.2 Aprobar los servidores del proyecto

Al abrir `claude` dentro del repo por primera vez, aparece un prompt pidiendo
aprobar los servidores declarados en `.mcp.json`: `engram` y `engram-equipo`.

Aceptar. Claude Code no conecta servidores de un `.mcp.json` de repo sin
aprobacion explicita; es proteccion contra un `.mcp.json` malicioso en un clon.

Sin este paso, el siguiente no funciona: el servidor figura como
`⏸ Pending approval` y no aparece para autenticar.

Verificacion desde otra terminal:

```bash
cd C:\Users\mpaul\Documents\ProyectoDeGrado
claude mcp list
```

`engram-equipo` debe decir `Connected` o `Needs authentication`, no
`Pending approval`.

### 0.3 Autenticar engram del equipo

Ya aprobado, dentro de la sesion:

```
/mcp
```

Elegi `engram-equipo` y segui el OAuth en el navegador. Ahi entras la clave.
El token queda cifrado en Claude Code; la contrasena nunca pasa por el chat.

Verificacion: `claude mcp list` debe mostrarlo `✔ Connected`.

### 0.4 Confirmar que el resto quedo bien

Pedir en el chat:

- "¿Qué proyecto detecta engram?" → debe responder `proyecto-de-grado`.
- "¿Aparece el bloque de skills inyectadas al inicio del turno?" → debe ser no.

---

## Paso 1 — Sesion maestra

```bash
cd C:\Users\mpaul\Documents\ProyectoDeGrado
claude --model opus --name maestro
```

### Primer mensaje del maestro

Copiar tal cual:

```
Sos la sesion maestra de este proyecto.

Carga de contexto, en este orden:
1. docs/SESIONES.md — como trabajamos entre sesiones
2. anteproyecto/capitulos/metodologia.tex — los 4 objetivos especificos
3. docs/01-contexto-y-objetivos.md — OJO: esta desactualizado, dice 3
   objetivos y menciona Kaggle T4. El cluster UNABIA ya funciona.
4. engram-equipo — ultimo estado de Nicolas
5. engram local — lo mio

Tu rol: orquestar, decidir y validar. No escribis codigo masivo.

Primera tarea: auditoria del estado actual contra los 4 objetivos.
Para cada objetivo: que hay, que falta, con que evidencia del repo.
Marca "no evaluado" lo que no puedas verificar. No asumas avance.

Al terminar, escribi docs/ESTADO.md.
```

---

## Paso 2 — Sesiones trabajadoras

El maestro escribe primero el contrato en `docs/estado/<nombre>.md`. Despues
abris la sesion en **otra terminal**:

### Documentacion

```bash
cd C:\Users\mpaul\Documents\ProyectoDeGrado
claude --model haiku --name doc
```

Primer mensaje:

```
Sos la sesion de documentacion.
Lee docs/estado/doc.md: ahi esta tu contrato.
No leas codigo fuera de lo que el contrato indique.
Al terminar, completa el bloque Reporte del contrato y haces commit.
```

> Si `--model haiku` no resuelve, usar el id completo:
> `--model claude-haiku-4-5-20251001`. No verificado al escribir esta guia.

### Implementacion

```bash
claude --model opus --name impl
```

```
Sos la sesion de implementacion.
Lee docs/estado/impl.md: ahi esta tu contrato.
Cargá las skills pytorch-patterns y ml-engineer antes de tocar codigo.
Respeta la seccion "Fuera de alcance".
Al terminar, completa el Reporte y haces commit.
```

### Investigacion

```bash
claude --model opus --name research
```

```
Sos la sesion de investigacion.
Lee docs/estado/research.md.
Cargá academic-researcher y scientific-thinking-literature-review.
No modificas codigo. Producis el documento que pide el contrato.
```

---

## Paso 3 — Retomar una sesion

```bash
cd C:\Users\mpaul\Documents\ProyectoDeGrado
claude --resume
```

Abre un selector interactivo. Las sesiones aparecen con el nombre que les
diste (`maestro`, `doc`, `impl`, `research`). Por eso `--name` importa.

Variantes:

```bash
claude --resume maestro      # busca por nombre
claude --continue            # la mas reciente en este directorio
```

---

## Ciclo diario

1. Abrir `maestro` con `--resume`.
2. El maestro lee `docs/ESTADO.md` y decide las tareas del dia.
3. El maestro escribe un contrato por sesion en `docs/estado/`.
4. Abris las sesiones trabajadoras en terminales separadas.
5. Cada una completa su Reporte y commitea.
6. El maestro actualiza `docs/ESTADO.md` y cierra.

Una terminal por sesion. Corren en paralelo.

---

## Que NO hacer

- Abrir Claude desde `documents/`. Rompe la deteccion de proyecto de engram.
- Darle contexto completo a una sesion trabajadora. Deriva y se encarece.
- Dejar que una trabajadora decida alcance. Eso lo hace el maestro.
- Omitir "Fuera de alcance" en un contrato.
- Commitear desde dos sesiones al mismo archivo a la vez.

---

## Reparto vigente — 2026-10-30

Sesiones **ya abiertas**, verificadas con `/list-agents` el 2026-10-30:

| Sesion | Via | Contrato | Entrega |
|---|---|---|---|
| `maestro` | — | — | `docs/ESTADO.md`, decisiones, contratos. No escribe codigo |
| `impl` | B | [`impl.md`](estado/impl.md) → `seg-imapp-holdout.md`, luego `config-modelos-investigacion.md` | Dos reportes |
| `research` | B | [`research.md`](estado/research.md) → `app-ux-producto.md`, luego `app-tecnica-contrato.md` | Dos reportes |
| `doc` | A | [`doc.md`](estado/doc.md) | `doc.reporte.md` |

Hay ademas una sesion `documents-c6`, inactiva hace mas de un dia, abierta fuera del
repositorio. Conviene cerrarla: una sesion abierta desde `documents/` rompe la deteccion de
proyecto de engram.

Cada archivo `docs/estado/<sesion>.md` es el **punto de entrada** de esa sesion y apunta a uno o
mas contratos tematicos. Asi el nombre del archivo coincide con el nombre de la sesion y los
contratos siguen organizados por tema.

### Criterio de modelo

- **Opus**: decidir, disenar, juzgar evidencia contradictoria. `maestro`, `impl`, `research`.
- **Sonnet**: verificar contra una lista, indexar, auditar un contrato, barridos mecanicos.
- **Haiku**: transcripcion.

Se cambia con `/model` dentro de la sesion, sin reabrirla.

### Primer mensaje

```
Sos la sesion <nombre>. Lee docs/estado/<nombre>.md: ahi esta tu contrato.
No leas nada fuera de lo que el contrato indique en "Entra".
Respeta "Fuera de alcance". No commitees.
Al terminar, completa el bloque Reporte y avisame.
```

---

## Mensajeria entre sesiones

> **Correccion del 2026-10-30.** Una version anterior de esta guia afirmaba que las sesiones no
> pueden comunicarse entre si. **Es falso.** Claude Code tiene mensajeria entre sesiones desde
> v2.1.224 (v2.1.234 en Windows nativo), activada sin configurar nada. Fuente:
> <https://code.claude.com/docs/es/cross-session-messaging>.

### Como funciona

Claude descubre los destinos con `ListAgents` y envia con `SendMessage`. **Vos nunca llamas esas
herramientas**: le pedis a Claude que avise algo y el redacta y envia.

- Un mensaje es **texto plano**, nunca el historial ni los archivos del remitente.
- En la misma maquina viaja por un socket por sesion, **sin pasar por servidores de Anthropic**.
- Para ver que sesiones hay: `/list-agents` (o `/peers`). La primera linea es el nombre de
  **esta** sesion.
- Para dirigir un mensaje vos mismo: `@` mas las primeras letras del nombre, y elegis del
  autocompletado. Ejemplo: `Avisale a @impl que el holdout quedo congelado`.
- Un mensaje entrante **no aprueba permisos, no cambia configuracion y no ejecuta comandos**:
  los `/comandos` dentro del texto llegan como texto plano.
- `notify_when_idle` permite pedir aviso cuando otra sesion de esta maquina termina. Caduca a
  las 12 horas. Solo la conversacion principal puede suscribirse, y solo a sesiones locales.

### Politica de este proyecto

La mensajeria existe, pero **la usamos acotada**, y la razon no es tecnica sino de alcance: una
trabajadora que negocia con otra expande su tarea sin que nadie lo aprobara, y dos sesiones
editando el mismo archivo se pisan.

**Permitido sin preguntar:**

| Quien | A quien | Para que |
|---|---|---|
| Cualquier sesion | `maestro` | Avisar que termino, que se bloqueo, o un hallazgo que afecta a otra |
| `maestro` | Cualquiera | Entregar contexto, corregir alcance, avisar de un cambio que la afecta |
| `maestro` | Cualquiera | Pedir aviso de inactividad con `notify_when_idle` |

**Prohibido:**

- Que una trabajadora le pida trabajo a otra. Si necesita algo de otra, se lo dice al `maestro`.
- Que una trabajadora le pase a otra una decision de alcance.
- Pedirle a otra sesion una accion que en la propia fue denegada. La documentacion de Anthropic
  lo prohibe explicitamente: los limites de permiso son por sesion.

El flujo normal sigue siendo **archivo**: contrato de entrada, reporte de salida. El mensaje es
para lo **urgente o lo que llega a destiempo**, no para reemplazar el contrato. Un hallazgo que
cambia el trabajo de otra sesion a mitad de camino es exactamente el caso de uso previsto por la
documentacion; un pedido de tarea no lo es.

### Las cinco formas de paralelismo, y cual usamos

Segun <https://code.claude.com/docs/es/agents.md>:

| Enfoque | Que da | Lo usamos |
|---|---|---|
| **Subagentes** | Trabajador dentro de una sesion, contexto propio, devuelve resumen | **Si**: el `maestro` los lanza para barridos y tareas mecanicas |
| **Sesiones paralelas** + mensajeria | Sesiones que vos abris, coordinadas por archivos y mensajes | **Si**: es nuestro esquema principal |
| **Vista de agentes** (`claude agents`) | Pantalla para despachar y monitorear sesiones en segundo plano | Evaluar. Mueve cada sesion a su propio worktree antes de editar |
| **Equipos de agentes** | Lider que reparte tareas, lista compartida, mensajeria entre pares | **No**: experimental y desactivado por defecto. Ademas no aisla en worktrees |
| **Flujos dinamicos** (`/workflows`) | Script que corre muchos subagentes y verifica resultados entre si | Candidato para la auditoria completa del repositorio, no ahora |

**Worktrees**: le dan a cada sesion un checkout de git separado. Hoy no los usamos porque
ninguna sesion escribe codigo. Si alguna pasa a implementar, va en worktree.

### Verificar trabajo en curso

| Comando | Que muestra |
|---|---|
| `/list-agents` o `/peers` | Sesiones alcanzables y el nombre de esta |
| `/tasks` | Todo lo que corre en segundo plano en esta sesion, incluidos subagentes terminados |
| `/status` | Fila `Peer address`: la direccion de bandeja de entrada de esta sesion |
| `claude agents` | Vista de agentes, para sesiones en segundo plano |

### Reparto de responsabilidades con Nicolas

Registrado en la auditoria del clasificador del 2026-10-19: **«Segmentacion y transformacion de
color las lleva Pau»**. Nicolas lleva el clasificador y la app.

Consecuencia operativa: las sesiones `seg` y `config` trabajan sobre territorio de Pau. Las de
app auditan territorio de Nicolas y **solo proponen**, no cambian.

## Estado pendiente al escribir esta guia

- 445 archivos modificados sin commitear. Lo revisa la sesion de
  documentacion con el agente `triaje`.
- `docs/01-contexto-y-objetivos.md` desactualizado.
- Cluster UNABIA al 79% de disco, 20 GB libres.
- Sin verificar: el alias `haiku`, y el flujo completo de `/feature-dev`.
