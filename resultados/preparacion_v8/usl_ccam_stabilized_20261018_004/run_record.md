# CCAM estabilizado 004: implementación y lanzamiento autorizado

## Alcance y autorización

El usuario autorizó el arreglo integral y entrenar directamente, omitiendo la
validación GPU previa. Solo corresponde a CCAM; no inicia pseudomáscaras,
estudiante, SLED, RPI-Net ni TriDerm. La nueva variante es una adaptación explícita,
no una reproducción exacta del artículo ni una solución científica ya demostrada.

Código implementado y subido a `experiment/modular-audit-v8`:
`adf12df244be0251a5c3804d1fd04e756bb75b9a`. Hubo un escritor Luna y una revisión
estática Luna con seguimiento de los hallazgos. Se corrigieron ranking, fondo
confiable, parada terminal al reanudar, reserva GPU y hash del supervisor.
No se ejecutaron tests, lint, compilación ni smoke GPU. La revisión estática no
certifica el comportamiento en ejecución.

## Variante y presupuesto

- `ccam_stabilized_v1`: RMS de cada bloque de características por imagen;
  estandarización espacial de logits sin parámetros adicionales ni estadísticas
  acumuladas; se conserva la convolución 1x1.
- Objetivo separado con diagonal, ranking desplazado y reducción por todos los
  pares; el motor histórico `ccam_engine.py` se conserva.
- Batch contrastivo efectivo 64, extracción congelada en grupos de 2; no es
  acumulación de gradientes. Último batch 32, sin descartar originales:
  544 actualizaciones por época para los 34.784 registros.
- Inicio nuevo, semilla 42, techo 30 épocas, AdamW LR 0,001 y weight decay 0,01.
  Los checkpoints 003 no se usan para reanudar esta variante.
- Selección: tres épocas consecutivas con Pearson global >=0,98 y elegibilidad
  de las ocho sondas, incluido fondo confiable. Es una heurística técnica.
- Protección de colapso persistente: parada como no-éxito, checkpoint terminal
  y recibo; una reanudación no puede saltarse esa parada. No acredita convergencia.
- Recursos de planificación: 24 h, RSS 65.536 MiB, CUDA 18.432 MiB más reserva
  de 4.096 MiB, salida/tracking/logs 1.024 MiB. No son mediciones de esta variante.
  La ejecución registra picos de la sesión, componentes de pérdida y gradientes.

El [config versionado](../../../plan-segmentacion/metodos-v8/launch-preparation/configs/usl_ccam.stabilized.draft.json)
permanece deshabilitado. El gate nuevo copiará literalmente `admission_waiver`;
`smoke_passed=false` y `resource_budget_measured=false` no se convertirán en
evidencia positiva. Los demás controles siguen exigidos; el waiver solo admite
esta variante y etapa. La independencia por paciente permanece no acreditada.

## Transporte y aislamiento

ZIP: `v8-ccam-stabilized-adf12df.zip`, 137.866.360 bytes,
SHA-256 `5d8e5b3961b7ada8c8cc462dadf227de1e850c75229e4b38674c722045258f0c`.
Contiene un Git bundle completo y un manifiesto de despliegue. Bundle SHA-256:
`b1926bf20b65118416a8ae954aae6eb52459abd31256b56e2bd975149ecfbe06`.
La transferencia no requiere acceso a GitHub desde UNABIA. El bundle completo
evita depender de que el historial local contenga el HEAD antiguo del clúster.

Destino aislado: `/home/jovyan/work/proyectonp/repo_v8_ccam_stabilized_004`.
El checkout y los resultados de 003 se preservan. Salida prevista:
`/home/jovyan/work/proyectonp/runs_v8/usl_ccam-20261018-004`.
Recibos previstos:
`/home/jovyan/work/validation_v8/usl-ccam-stabilized-launch-20261018-004/`.
Python: `/home/jovyan/work/proyectonp/.venv/bin/python`.

## Estado

El despliegue del commit `adf12df244be0251a5c3804d1fd04e756bb75b9a` sí llegó al
checkout aislado. El intento 004 se abortó durante la validación del gate, antes
de crear `launch.json`, `submission.json`, el supervisor o una corrida W&B. El
error fue `Evaluation protocol identity/hash/path mismatch`: el config aprobado
omitía `protocol_id` aunque el manifiesto y el gate ya fijaban
`v8_conservative_retrospective_20261016`. No se inició entrenamiento ni se creó
el directorio de salida 004. Se conserva el recibo fallido y el log remoto en
`/home/jovyan/work/validation_v8/usl-ccam-launch-20261017-003/stabilized-deployment-execution-004.log`.

La corrección añade el identificador protocolario al borrador y reserva un
intento nuevo 005 con recibos y salida separados. `readiness.review.json` y los
gates históricos permanecen intactos. El estado del 005 se documentará en un
registro aparte después de observar su resultado real.
