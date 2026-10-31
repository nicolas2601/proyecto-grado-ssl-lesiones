# ADR 0001 - Limpieza del entorno de agentes y arquitectura de memoria

- Fecha: 2026-10-29
- Estado: Aceptada
- Ambito: Entorno de trabajo con Claude Code (global), aplicado con este repo como piloto

## Contexto

Las entregas asistidas por IA venian saliendo incompletas, con consumo alto de
tokens y desvio frecuente respecto al pedido. Antes de cambiar de modelo o de
prompt se midio el entorno real.

Datos medidos sobre `~/.claude`:

| Metrica | Valor |
| --- | --- |
| Skills instaladas | 1300 (10.4 MB de markdown) |
| Sesiones con registro en `.skillgate` | 46 |
| Skills distintas activadas alguna vez | 27 |
| Utilizacion efectiva | 2.1 % |
| Activacion maxima de una skill | 2 veces |
| Servidores MCP activos | context7, playwright, unity-mcp, 21st, ruflo (~300 tools) |
| Sistemas de memoria en paralelo | 3 (engram, context-mode, memoria de archivos) |

Hallazgos cualitativos:

- El hook `skill-router.mjs` inyectaba el cuerpo completo de skills en cada
  prompt, antes de que el agente leyera el pedido. Exigio skills en solo 16 de
  46 sesiones y con falsos positivos graves (SDKs de Azure, `flox-environments`,
  `n8n-workflow-patterns`) sobre stacks que el proyecto no usa.
- Los hooks `skill-gate-*` bloqueaban la salida exigiendo skills mal elegidas.
- `CLAUDE.md` global contenia reglas en conflicto: respuestas cortas por defecto
  frente a verificar siempre, proponer alternativas siempre y un bloque
  obligatorio de estandares de diseno web.
- Dos sistemas semanticos de memoria en paralelo guardaban el mismo hecho dos
  veces sin conocerse entre si.

## Decision

### 1. Eliminar el pre-filtro de skills, conservar la telemetria

Se eliminan de `~/.claude/settings.json`:

- `UserPromptSubmit` -> `skill-router.mjs`
- `UserPromptSubmit` -> `skill-gate-eval.mjs`
- `Stop` -> `skill-gate-stop.mjs`
- `PostToolUse` -> `skill-gate-track.mjs`
- `SessionStart` -> `context-mode-cache-heal.mjs`

Se conserva `PostToolUse` -> `skill-usage-tracker.mjs`, unica fuente de verdad
sobre uso real de skills.

Razon estructural, no solo de precision: el router decidia antes de leer el
pedido. Una decision sin contexto siempre va a ser peor que una decision
informada tomada despues de entender el pedido. El agente ya tiene las
descripciones de las skills y la herramienta para invocarlas.

### 2. Un solo sistema de memoria semantica

- `context-mode` deshabilitado en `enabledPlugins`.
- `engram` se conserva.
- `ruflo` eliminado de `~/.mcp.json` (~300 herramientas MCP).

Backup previo en `~/.claude/backups/cleanup-20261029-200223/`.

### 3. Arquitectura de memoria en cuatro capas

| Capa | Donde vive | Que guarda | Costo |
| --- | --- | --- | --- |
| 0. Verdad | El repo | El codigo. Manda siempre | 0 hasta leerlo |
| 1. Decisiones | `docs/decisions/*.md` | ADRs: que decidimos y por que | 0 hasta leerlo |
| 2. Reglas | `CLAUDE.md` del proyecto | Reglas duras, acotadas | Fijo y chico |
| 3. Episodica | engram | Que intentamos y que fallo | Solo en busqueda |

La capa 1 es la clave: las decisiones van versionadas en git, no en una base
semantica. Son diffeables, revisables, viajan con el codigo y sobreviven a la
caida de cualquier herramienta. Es la unica memoria que no puede
desincronizarse del codigo.

### 4. Memoria en grafos: descartada por ahora

Un grafo temporal de conocimiento resuelve bien dos cosas: recorrer relaciones
entre muchas entidades, y versionar hechos en el tiempo.

No se adopta por tres razones:

1. Un grafo vive o muere por la calidad de extraccion de entidades. Una
   extraccion mediocre produce un grafo confiadamente erroneo, peor que no
   tener grafo, porque se le cree.
2. El conteo de entidades del proyecto es bajo. Un grafo rinde con cientos de
   entidades relacionadas, no con decenas.
3. Para codigo, el grafo autoritativo ya existe: el grafo de imports, el
   historial de git y el repo. Un segundo grafo que parafrasea eso no agrega
   informacion, agrega deriva y una segunda fuente que puede contradecir a la
   primera.

Revisar cuando haya cinco o mas proyectos en produccion y se necesiten
consultas cruzadas entre proyectos.

### 5. Deteccion de proyecto en engram

El error `ambiguous_project` ocurria porque el servidor MCP arranca desde
`C:\Users\mpaul\documents`, que contiene varios repositorios.

Mitigacion aplicada: se crearon archivos `.engram/config.json` con
`{"project_name": ...}` en la raiz de cada repositorio.

Regla de trabajo derivada: abrir Claude Code **dentro** del repositorio, nunca
en el directorio padre que contiene varios repos.

## Consecuencias

Positivas:

- Se elimina la inyeccion de contexto irrelevante en cada prompt.
- Se elimina el bloqueo de salida por skills mal elegidas.
- Una sola fuente de memoria semantica, sin duplicacion de hechos.
- Las decisiones quedan auditables en git.

Negativas y riesgos aceptados:

- Se pierde la sugerencia automatica de skills. Se asume a cambio de que la
  seleccion pase a ser informada. La telemetria queda activa para revisar.
- Deshabilitar `context-mode` deja sin usar su indice previo. Es reversible
  volviendo a `true` en `enabledPlugins`.

## Pendiente

- Poda de skills de 1300 a aproximadamente 45, usando la telemetria de uso real
  como criterio. No ejecutada todavia.
- Reescritura del `CLAUDE.md` global para eliminar las reglas en conflicto.
- Definicion del contrato de terminado y de la estrategia de pruebas para este
  repositorio, que es de investigacion y no de producto web (ver ADR 0002).
