# Despliegue CCAM deshabilitado en UNABIA — 2026-10-17

## Alcance y resultado observado

Se transfirió un ZIP y se desplegaron nueve archivos exactos del commit
`6c5627d30eea56d916bdb941395cd77b4df122b1` en
`/home/jovyan/work/proyectonp/repo_v8_canonical`. Los nueve archivos se reemplazaron
y coincidieron con el payload; no hubo omisiones. El inventario de fuentes del
lanzador contiene 55 archivos. La observación remota resumida se conserva en
[deployment.snapshot.json](deployment.snapshot.json); es una captura del resumen
observado, no una copia íntegra del recibo remoto.

El HEAD Git del checkout sigue en
`95a8f602b2fb49304dbbb9a78a18edc43cc3ebba`: la transferencia fue por ZIP, sin
sincronización Git en UNABIA. No debe confundirse ese HEAD con la revisión de los
archivos desplegados.

## Artefactos y hashes

Directorio remoto de evidencia:
`/home/jovyan/work/validation_v8/usl-ccam-config-6c5627d-20261017-001/deployment/`.

| Artefacto | SHA-256 |
|---|---|
| `receipt.json`, en el directorio remoto anterior | `2e10da1c435958d9c527f4cc7bf3fe039cc6493ca0576244c503753badfa90fe` |
| `ccam.gate.json`, en el directorio remoto anterior | `a1fa4b2d4772a71b0f2d3080f2fa0212e85b2fa4709d5467c7f0b620d01ef2cf` |
| Config desplegado, bytes de `plan-segmentacion/metodos-v8/launch-preparation/configs/usl_ccam.draft.json` | `b8a4836dde5e09dee80398e0a06196758aaa62fe903f037e3c7a8e50e59a13be` |
| Config, serialización JSON canónica del entrenamiento | `94466c9d9e2d02e06b6b535676334c225f68178671e8f6c3e8c076079c42011d` |
| ZIP exterior transferido | `85fb84a48c63bb8bc5378f4e3d9fbacab3f155f716250753aadf4aa1071b41cd` |
| ZIP interior con los nueve archivos Git | `cf8ed63485d388934650d6ed17fa807507c3cef7af3b3ab7d88ffd249be3f617` |

Los archivos reemplazados tienen respaldo bajo `deployment/backup/`, conservando
sus rutas relativas. El recibo registra los hashes previos y posteriores, las
acciones realizadas y los pines inmutables comprobados. Las huellas de
`readiness.review.json` y de la evidencia histórica de smoke/recursos no cambiaron.
Se conservaron los manifiestos congelados y la corrida real de cinco épocas.

## Estado de autorización y alcance de la evidencia

El estado es `deployed_disabled_candidate`: `execution_enabled=false`,
`training_started=false`, `tests_executed=false` y `model_imported=false`.
No se ejecutó el nuevo código de modelo, recuperación o selección. El config
mantiene 30 épocas, batch 2, LR 0,001 y weight decay 0,01, con límites de
planificación de 86.400 s y 1.024 MiB; ese presupuesto sigue sin medirse para la
implementación actualizada. El criterio Pearson de 0,98 es una heurística de
estabilidad, no una validación anatómica.

El gate nuevo está ligado al config y al manifiesto, pero es deliberadamente
ineligible: `passed=false`, `smoke_passed=false` y
`resource_budget_measured=false`. Los demás indicadores genéricos del candidato
que permanecen en falso no constituyen nuevos hallazgos negativos sobre datos o
pesos: aún no existe un gate aceptado para esta revisión. La evidencia y los
pines previos se preservan; no se reabren las auditorías cerradas. El smoke y las
mediciones históricas corresponden a la implementación/presupuesto anteriores y
no se presentan como ejecución del código nuevo ni medición de 30 épocas.

## Próximo paso mínimo

Corresponde acordar y autorizar una verificación acotada en UNABIA de los cambios
de reanudación y selección, junto con la medición del código actualizado; esa
ejecución todavía no está autorizada. Después se podrá vincular evidencia válida
al gate y revisar el config antes de habilitar una corrida real. No se repetirá
la investigación de cuotas, bancos o datasets desde cero. Este despliegue no
autoriza ni inicia automáticamente un entrenamiento completo de 30 épocas.
