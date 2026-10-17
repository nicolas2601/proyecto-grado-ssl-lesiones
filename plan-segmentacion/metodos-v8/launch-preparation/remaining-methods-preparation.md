# Preparación deshabilitada de SLED, RPI-Net y TriDermNet-HAGO

Fecha: 2026-10-17. Alcance: fijar referencias de datos y presupuestos operativos en los cuatro borradores restantes mientras se prepara CCAM. Esta entrega no ejecuta métodos, modelos, pruebas ni descargas y no habilita ningún config. Los cambios no modifican `readiness.review.json`, gates históricos ni el protocolo aceptado.

## Referencias de datos

Los cuatro configs incorporan la misma referencia `evaluation_protocol`:

- Archivo en UNABIA: `/home/jovyan/work/validation_v8/accepted-manifests-20261016-002/protocol.json`.
- SHA-256: `f4836405a7cbb8309f1757a3488e34ab4fa4f4f02f785b4f72bae79b9d349b03`.
- Identificador: `v8_conservative_retrospective_20261016`.

La referencia informativa a los originales de aprendizaje queda fijada en `preparation_review.learning_manifest`:

- Archivo: `/home/jovyan/work/validation_v8/accepted-manifests-20261016-002/learning.manifest.json`.
- SHA-256: `7749471410aa98a76dd978cca2ef58f9642d14404530c98e534484ba40a7a6ff`.
- Registros: 34.784.

`manifest_bound=true` expresa únicamente este pin informativo. El recibo de comprobación de correspondencia de metadatos de estos cuatro configs todavía debe emitirse en UNABIA. No sustituye al argumento `--manifest` de cada etapa, no acredita consumo autorizado y no permite pasar originales donde `fit` requiere un manifiesto de pseudo-máscaras. Los manifiestos derivados, checkpoints y sidecars futuros se fijarán con sus hashes reales cuando existan. `gate_bound=false` y `resource_caps_measured=false` permanecen explícitos.

La cobertura clínica incompleta y la exclusión conservadora pHash siguen siendo limitaciones declaradas del protocolo aceptado; estos pins no acreditan independencia paciente/lesión ni identidad de duplicados.

## Presupuestos propuestos, sin medición nueva

| Config | Dispositivo | Épocas conservadas | Timeout (s) | RSS (MiB) | Salida máxima por invocación (MiB) | VRAM / reserva (MiB) |
|---|---|---:|---:|---:|---:|---:|
| `configs/sled.draft.json` | CPU | 1, sin ajuste de pesos | 604800 | 65536 | 49152 | No aplica / 4096, campo común inactivo |
| `configs/rpi_net.draft.json` | cuda:0 | 50 iniciales | 604800 | 65536 | 102400 | 16384 / 4096 |
| `configs/rpi_later_phase.draft.json` | cuda:0 | 30, una fase posterior | 604800 | 65536 | 102400 | 16384 / 4096 |
| `configs/tridermnet_hago.draft.json` | cuda:0 | 150, warmup 10 | 1209600 | 65536 | 49152 | 16384 / 4096 |

Los timeouts son techos operativos amplios de 7 y 14 días, elegidos como supuestos de planificación ante la ausencia de una duración medida por método. No son predicciones científicas ni estimaciones acreditadas de tiempo total. Los 16 GiB de VRAM más 4 GiB de reserva reducen el presupuesto anterior de 20+4 GiB respecto de las tarjetas observadas de 24.467 MiB; esto tampoco acredita la memoria necesaria ni reserva una GPU. La cuota agregada del clúster sigue usándose solo como referencia de planificación, sin investigación adicional de cuota individual.

La salida se estima con la fórmula ya documentada en `resources.draft.json`, usando `N=34784`, `R=224` y `S` fuentes:

```text
generate_or_predict_MiB = N * (R^2 * ((S+2)*4 + (S+1)*6) + 16384) / 1048576
refine_MiB = N * (R^2*4 + 4096) / 1048576
```

Con una fuente, el sobre estimado es 40.490,75 MiB; con cuatro, 90.424,8125 MiB. Se añaden al menos 512 MiB de reserva y se redondea el límite de salida a 49.152 o 102.400 MiB, respectivamente. Son envolventes conservadoras de archivos, no mediciones. El almacenamiento acumulado deberá sumar los outputs retenidos de cada etapa, fase y ciclo, además de checkpoints y tracking; el límite por invocación no prueba capacidad para retener toda la comparación.

## Orden de dependencias y pendientes ejecutables

### SLED y TriDermNet-HAGO

1. Verificar en UNABIA la correspondencia config/protocolo/manifiesto y medir recursos CPU de SLED antes de crear su gate. Se conservan CPU, escalas, variante primaria, selección y 30 pases de refinamiento; no se agrega ningún checkpoint.
2. Autorizar por separado y ejecutar `SLED generate` sobre los originales elegibles. No se ejecuta en esta preparación.
3. Exigir un manifiesto de pseudo-máscaras con cobertura completa y exitosa de los 34.784 originales. Casos ausentes, fallidos o abstenciones bloquean la adopción del manifiesto completo; no se eliminan silenciosamente para declarar cobertura.
4. Fijar el SHA-256 real de ese manifiesto y su procedencia como insumo de TriDerm. Por ahora `pseudo_manifest_sha256=null` y `pseudo_teacher_method="sled"` permanecen intactos.
5. Medir recursos GPU de TriDerm y emitir su gate ligado a config, protocolo y manifiesto derivado. Mantener 150 épocas, warmup 10 y la adaptación sin clases diagnósticas ni máscaras expertas.

La planificación o preparación puede avanzar mientras corre CCAM; esta entrega no autoriza concurrencia de procesos CPU/GPU ni lanzamientos adicionales.

### RPI-Net

1. Verificar el banco candidato existente, su correspondencia con el protocolo y el manifiesto en UNABIA. Se conserva `/home/jovyan/work/public_ssl_v8_preparation/source-bank.rpi-net.reviewed-candidate.json`; no se inventa un hash de banco ni se redescargan pesos.
2. Tras medir recursos y aprobar gates específicos, generar los insumos iniciales P1–P4 y el manifiesto de pseudo-máscaras con su procedencia real.
3. Ajustar la fase inicial de 50 épocas y conservar su checkpoint final real con sidecar y linaje. Los campos de checkpoint continúan `null` en estos borradores porque el artefacto todavía no existe.
4. Predecir sobre los originales de aprendizaje con el estudiante de esa fase. Refinar únicamente P2 con sus probabilidades, conservando P1, P3 y P4; no sustituir este paso por un refresco CAM.
5. Inicializar la única fase posterior de 30 épocas desde el checkpoint final real de la fase inicial y su sidecar verificados. `initial_checkpoint` e `initial_checkpoint_provenance` siguen `null` hasta entonces.
6. Usar la predicción de evaluación sobre las imágenes reservadas únicamente con el modelo y linaje finales; las referencias expertas no entran al aprendizaje, a la selección de checkpoint ni al refresco P2.

Los borradores no constituyen todavía planes seriales completos del controlador. La revisión posterior en UNABIA debe fijar argumentos de entrada/salida por etapa, recursos observados y gates correspondientes antes de cualquier habilitación. No se repiten la auditoría completa de imágenes ni el smoke CCAM para este cambio de metadatos.
