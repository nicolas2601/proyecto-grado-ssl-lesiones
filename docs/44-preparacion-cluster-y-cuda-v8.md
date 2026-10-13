# 44. Preparación del clúster y compatibilidad CUDA de RPI

2026-10-13, America/Bogota. Base registrada d55c5cd, rama
experiment/modular-audit-v8. Continúa los documentos 41–43.

## Estado comprobado

Se recuperó JupyterLab y se trabajó desde una terminal nueva. El checkout base
repo_four_methods_v8_d55c5cd_20261013 sigue limpio en d55c5cd. La copia histórica
repo_audit_v8_run_20261011 conserva sus 63 modificaciones pendientes.

La API de W&B comprobó paulasaavedra1101, credencial separada con permisos 600 y
cero corridas running en el proyecto compartido antes y después de la preparación.
No se creó una nueva corrida científica ni se subieron imágenes, IDs o credenciales.

## Todos los originales elegibles

Se autenticaron el manifiesto de transformación y las exclusiones contra el
run.json histórico final_B_s42. Se conservaron las 14.128 exclusiones por ID y las
10 adicionales por contenido. Los 40.896 originales se releeron: SHA256 igual al
source_sha256 registrado y decodificación RGB válida, cero fallos. Cuatro workers
completaron la verificación en 24,87 segundos.

| Dataset | Verificados |
|---|---:|
| ISIC2019 | 10.991 |
| ISIC2020 | 24.993 |
| HAM10000 p1 | 2.444 |
| HAM10000 p2 | 2.468 |
| Total | 40.896 |

Evidencia privada en audit_v8_sync_20261013/full_pool_d55c5cd_20261013:
summary.json, original_records.jsonl, failures.jsonl, progress.json y audit.log.
Los registros por imagen permanecen en el clúster. La verificación física no
resuelve por sí sola la procedencia de los grupos ni la reserva anatómica independiente.

## Recursos medidos

48 CPU disponibles al proceso y cuota cpu.max=max; aproximadamente 123 GiB de RAM
total y 116 GiB disponibles en la consulta. Dos RTX PRO 4000 Blackwell, 24.467 MiB
totales y 23.985 MiB libres por dispositivo, sin procesos GPU antes de los probes.
El espacio libre del filesystem compartido no demuestra una cuota individual.

Probes sintéticos: batch 2, resolución 224, AdamW, cuatro pasos (uno frío y tres
medidos). Allocator limitado a 18 GiB y timeout de 90 segundos por método.
TriDerm ejercitó HAGO después del warmup. No se guardaron checkpoints aleatorios.

| Método | Pico allocated MiB | Pico reserved MiB | Pasos medidos (s) |
|---|---:|---:|---|
| USL | 376,625 | 414 | 0,0304 / 0,0280 / 0,0288 |
| RPI corregido | 1.337,990 | 1.468 | 0,0940 / 0,0918 / 0,0926 |
| TriDerm | 147,027 | 176 | 0,0415 / 0,0373 / 0,0375 |

No extrapolar estos tiempos a preentrenamiento, CCAM, exportación o ciclos nativos.
Los configs base conservan execution_enabled=false y sus límites de prueba
(180 segundos, 4096 MiB de salida). El exportador estima 46,49 GiB para una fuente
y 103,82 GiB para cuatro fuentes sobre el pool completo a 224x224.

## Corrección de RPI

El primer probe RPI falló en adaptive_avg_pool2d_backward_cuda con determinismo
obligatorio. Se preservó ese log. adaptive_average_tokens usa AvgPool2d en ratios
enteros y los mismos bins floor/ceil con medias y stack para otros tamaños.
Se conservan solapamientos, geometría y parámetros; no se desactiva el determinismo.

Las siete pruebas RPI pasaron en el clúster (0,972 s), sin fallos, errores ni
omisiones. Incluyen equivalencia de valores/gradientes en CPU para ratios enteros,
no enteros y salida mayor que entrada, y repetibilidad del backward en CUDA.
Solo se repitió el probe RPI tras el arreglo; el modelo completo pasó los cuatro pasos.

El arreglo se probó en repo_four_methods_v8_cuda_fix_20261013. La copia base tiene
objetos Git incompletos para otros contenidos/historia, por lo que git clone local
falló. Se conservó la base y se creó una copia aislada de sus archivos existentes.
Esto no equivale a verificar todo el historial Git del clúster. La publicación
usa el árbol completo de GitHub como base.

Logs y medidas: audit_v8_sync_20261013/gpu_resources_d55c5cd_20261013.
El reporte agregado local es full_pool_d55c5cd_20261013/cluster_preparation.json.

## Pesos y puertas pendientes

Se inventariaron 40 checkpoints en ubicaciones de modelos, sin modificar los
originales. La caché contiene DINOv2 B/14 y S/14. dino_continual declara
dinov2_vits14: no sustituye DINO original en RPI. El MoCo inspeccionado procede de
un smoke de una época y usa trunk/config, diferente del formato model + JSON
de procedencia que requieren los adaptadores.

Faltan bancos revisados por origen SSL, procedencia de grupos históricos, reserva
anatómica independiente y receta/presupuesto completo. Las alertas se configuraron
antes en W&B; no se emitió una alerta de prueba ni se comprobó su entrega por correo
en esta preparación. No se declara una corrida completa ni una puerta científica
resuelta por aprobar hashes o pruebas sintéticas.
