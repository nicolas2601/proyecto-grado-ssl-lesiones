# Catálogo de corridas SSL y operación en UNABIA

Este documento ayuda a encontrar el procedimiento correcto; no reemplaza los runbooks ni certifica el estado actual de una corrida. Primero identifica la familia, versión y etapa solicitadas. Luego verifica la rama/commit y lee los documentos y el código vigentes de esa versión. Los ejemplos de una familia no son comandos universales.

## Rutas por familia

| Familia o tarea | Referencias principales | Alcance |
|---|---|---|
| Preentrenamiento SSL V6 con SLURM | [cluster/README.md](../cluster/README.md), cluster/pretrain_vits_4gpu.sbatch, cluster/pretrain_vitb_4gpu.sbatch | Receta histórica de V6. Usar solo si la corrida solicitada corresponde a esa versión y se confirma que el entorno activo usa SLURM. |
| Clasificación final V7 | [docs/27](27-plan-v7-mejoras-ssl.md), [docs/28](28-rediseno-brazo-propuesto-v7.md), scripts/v7_final_launch.py, scripts/c1_final_eval.py | Corridas y evaluación V7. [docs/55](55-flujo-v7-para-corridas-v8-en-unabia.md) documenta prácticas y ejemplos V7; sus comandos no son una receta V8. |
| Segmentación Camino B | [plan-segmentacion/RUNBOOK_CLUSTER.md](../plan-segmentacion/RUNBOOK_CLUSTER.md), plan-segmentacion/camino-b-cnn/entrenamiento/run_pipeline.py | Procedimiento para ese pipeline/pod. Sus rutas, versiones de Python/PyTorch, GPUs y dependencias no se generalizan a otros experimentos. |
| Segmentación V8 de cuatro métodos | [docs/51](51-cierre-preparacion-y-bloqueos-lanzamiento-v8.md) y documentos V8 posteriores; odd/tasks/v8-cierre-bloqueos-lanzamiento.md; scripts/run_segmentation_controller_v8.py; scripts/run_segmentation_v8.py; plan-segmentacion/metodos-v8/launch-preparation/ | Solo el protocolo y el plan revisados para V8. El controlador tiene protocolo, métodos y gates propios. El pipeline.draft.json con execution_enabled:false y protocolo de borrador no autoriza entrenamiento ni es un plan ejecutable. Revisa también el estado más reciente de SimCLR: docs/51 remite a docs/52 y documentos posteriores. |
| Preparación, descarga/staging de pesos, auditoría o evaluación | Script y documentación de la etapa solicitada, más la carpeta de resultados/evidencia correspondiente | No confundir estas etapas con un lanzamiento de entrenamiento. Verifica entradas, efectos secundarios y salidas del script concreto. |
| Otra familia o versión | Busca en scripts/, src/, docs/, plan-segmentacion/ y cluster/ | No la asignes a la ruta más parecida. Identifica su lanzador, configuración, gates y runbook propios; si faltan, pide el dato o prepara documentación antes de operar. |

## Flujo común

1. **Resolver la versión.** Revisa la solicitud, rama/commit objetivo, implementación y documentación asociada. Usa Glob para localizar los documentos más recientes de esa familia; no dependas solo del índice numerado de docs/README.md.
2. **Leer antes de medir.** Consulta los resultados, logs, manifiestos, ledger y notas ya disponibles. No repitas cuotas, versiones, accesos ni descargas documentadas si no existe una razón concreta para una nueva observación.
3. **Validar la identidad de la corrida.** Registra versión/método/etapa, semilla, configuración, commit, datos y split, exclusiones, pesos de origen, destino, tracking y política de reanudación. Confirma que la salida no exista y que no haya una corrida equivalente en curso.
4. **Validar el entorno específico.** Revisa el runbook de esa familia para el mecanismo de acceso, transferencia, Python/PyTorch, dependencias, GPUs, almacenamiento, wall time y tracking. No supongas que una ruta o recurso medido para otra sesión sigue vigente. Nunca sustituyas el PyTorch del entorno base del pod.
5. **Inspeccionar la implementación.** Confirma flags y efectos leyendo el lanzador/configuración vigente. No construyas un comando genérico ni copies uno de una versión distinta. Respeta la semántica exacta de resume y los gates existentes.
6. **Autorizar por acción.** Antes de navegar a UNABIA o ejecutar comandos en el clúster, describe la acción exacta y espera confirmación explícita. La autorización para inspeccionar no cubre el entrenamiento; un lanzamiento requiere su propio alcance y confirmación si todavía no fue confirmado explícitamente.
7. **Verificar la salida real.** Confirma estado del proceso, código de salida, run ID, logs, checkpoints y artefactos. Un clic de lanzamiento o un mensaje de “submitted” no demuestra que el entrenamiento se completó.
8. **Guardar procedencia.** Sigue el formato de evidencia de la familia. Conserva los registros previos e incluye, según aplique, commit, configuración/manifiesto y hashes, entorno, comando sin secretos, semilla, run ID, logs, checkpoints, salidas, estado y protocolo/split de evaluación. Los resultados/tablas deben generarse desde los artefactos fuente establecidos por el repositorio.

## Reglas específicas de V8

- V8 tiene su propio controlador y protocolo. En el estado documentado por este catálogo, el controlador usa el protocolo w3-four-methods-v8, requiere etapas/métodos explícitos y separa la ejecución mediante --execute; confirma esos requisitos en el código vigente antes de operar.
- No cambies execution_enabled, launch_ready, training_started ni otros gates. No lances si el plan no fue revisado o un bloqueo sigue abierto.
- Revisa odd/tasks/v8-cierre-bloqueos-lanzamiento.md para evitar duplicar trabajo. Si la solicitud coincide con una tarea de esa lista que está llevando otra herramienta, señala el solapamiento antes de avanzar.
- Protege las descargas, checkpoints, ledgers y copias de trabajo que ya existan. No repitas una descarga o conversión por costumbre: verifica primero los recibos y hashes registrados.

## Referencias de operación del navegador

El acceso de este proyecto al clúster es JupyterLab en [UNABIA](https://unabia.unab.edu.co), mediante el servidor Playwright MCP y su perfil persistente configurado en .mcp.json. No uses SSH ni gestiones cookies o storage_state. Si aparece el login, detente y pide al usuario que inicie sesión de forma interactiva. La guía completa de autorización, manejo de sesión y secretos está en [.claude/agents/cluster-session-operator.md](../.claude/agents/cluster-session-operator.md).
