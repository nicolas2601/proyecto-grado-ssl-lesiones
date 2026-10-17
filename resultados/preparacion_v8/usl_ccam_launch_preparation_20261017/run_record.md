# CCAM: paquete de revisión previo al lanzamiento

Fecha: 2026-10-17. Estado observado: `disabled_prepared_awaiting_user_scope_approval`.
Fuente: [captura del recibo remoto](review.snapshot.json).

## Acción y alcance

Se transfirió por ZIP y materializó en UNABIA un paquete de revisión deshabilitado.
Se registraron 55 fuentes del CLI y 57 fuentes efectivas, los pines de evidencia
existente y rutas nuevas. No hubo entrenamiento, sesión W&B, limpieza, prueba GPU
ni repetición de auditorías de datos, pesos o cuotas en esta preparación.

La propuesta cubre únicamente CCAM de USL-Net: 30 épocas completas, inicio nuevo
con semilla 42, batch 2, resolución 224, AdamW LR 0,001 y weight decay 0,01.
El checkpoint real anterior de cinco épocas se conserva; no contiene el estado
necesario para una continuación exacta de esta implementación nueva.

Se conserva el primer candidato con estabilidad Pearson >= 0,98 sobre ocho
originales fijos de aprendizaje; no hay parada temprana. El umbral es heurístico,
no una validación anatómica. Las 30 épocas son un techo de planificación basado
en un precedente del proyecto, no convergencia medida específicamente para CCAM.
Batch 2 conserva la limitación científica previamente declarada.

Límites: 86.400 segundos, RSS 65.536 MiB, 1.024 MiB combinados para salida,
tracking y logs; presupuesto CUDA 18.432 MiB y reserva 4.096 MiB. Son límites,
no mediciones de la corrida completa. Se reutilizan las mediciones anteriores
con su alcance, incluida la recuperación diagnóstica GPU corregida.

## Rutas y hashes verificables

Carpeta remota del paquete:

```text
/home/jovyan/work/validation_v8/usl-ccam-launch-preparation-4a6832c-20261017-001/review/
```

| Archivo en esa carpeta | SHA-256 |
|---|---|
| `approval-manifest.json` | `37d1f858aaa8e1e0e2d9376f7fc2f3892cc654512abe1604270f85669d897f74` |
| `ccam.run.proposed.disabled.json` | `5f84d7ab7095b0434e5581d639b7a9acca34be2776d5d5b1ef0d3a2ab764f865` |
| `ccam.gate.proposed.json` | `c9408782749e10897837417c55ae50e65c69cff2054ed9ebc5c00120cc01291e` |

ZIP transferido: SHA-256 `9d656eb14b0b951f5b14681ca68ee8b62e147a32b2343998404f764448642158`.
Config habilitado previsto, todavía no materializado: hash de archivo
`30fd4ad9e870f86009e1b591fab05cb84226f66621297912d8349c43bed7756c`;
hash JSON canónico `a1823947f09a6ce356b6ec623b2f6b255154e260df21b30cce47b95616d17aaa`.

Salida propuesta, todavía sin entrenamiento:
`/home/jovyan/work/proyectonp/runs_v8/usl_ccam-20261017-003`.
W&B previsto: entidad `nmoreno534-universidad-autonoma-de-bucaramanga`, proyecto
`dermssl-segmentacion`; sin creación de run durante la preparación.

## Frontera de habilitación y pendientes

El config permanece `execution_enabled=false`. El gate es candidato: no contiene
un campo `passed`; el `null` de la captura no significa fallo ni aprobación.
La validación productiva `gate_report` de la futura activación sigue pendiente;
la comprobación diagnóstica anterior no la sustituye. El wrapper exige
`--execute-reviewed-ccam-30` y un recibo de limpieza antes de activar y supervisar.

Durante la corrida se mantendrá fija la copia de ejecución y su inventario de
fuentes: la identidad de recuperación incluye también archivos de otros métodos
y tests. La preparación posterior se hará localmente o en otra copia, sin
desplegar cambios sobre ese inventario mientras CCAM necesite poder reanudarse.

La limpieza tardía sigue sin ejecutarse. El inventario enumera cuatro carpetas
diagnósticas con 23 archivos; además deben retirarse del espacio activo los
cuatro fixtures `diagnostic-subset.json` y `diagnostic-not-a-production-gate.json`
en `run/` y `run-fixed/` de la comprobación, conservando hashes y recibos. No se
eliminan la corrida real anterior, el recibo fallido ni los manifiestos congelados.

Sigue revisión del alcance por el usuario, limpieza registrada, activación separada
y validación del gate ligado, antes del lanzamiento único. Después de CCAM se
revisarán curvas y candidato; faltan linaje/manifiesto/gate de pseudomáscaras y
el plan serial del estudiante. No se habilita toda la línea ni otros métodos.
