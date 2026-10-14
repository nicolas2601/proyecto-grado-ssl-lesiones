> Estado más reciente: avance verificado en UNABIA del 2026-10-14, documentado en [54](54-usl-net-banco-pesos-v8.md) y en `resultados/preparacion_v8/readiness_20261014/readiness.review.json`. La validación local de [52](52-simclr-huggingface-validacion-local-v8.md) solo identificó el artefacto; la transferencia, instalación del runtime y forward sintético se confirmaron después en el clúster.

# Pendientes por método para V8

> **Estado actualizado, 2026-10-14 UTC:** el checkpoint SimCLR real se transfirió a UNABIA (102.542.786 bytes; SHA-256 `fdd96bbcc92008232f31e0f7de46d30810e7115a05005d7cff926e78e6fa37ea`); cinco módulos del runtime coinciden por hash con la revisión `eed86c892ef1c0516e21a024a66f0b8788aedd76`; y el banco USL-Net pasó un forward sintético en CPU, con salida finita `[1,9216,14,14]`, sin imágenes reales ni entrenamiento. El recibo está en el clúster y como artefacto W&B; no se versiona su archivo crudo en Git. Run: [g8us8pz9](https://wandb.ai/nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion/runs/g8us8pz9). Equivalencia numérica con TensorFlow/Google original, manifiestos y grupos paciente/lesión, evaluación independiente, cuotas y presupuestos completos, configs/manifiestos seriales, revisión de SLED y comprobación de CUDA/launcher siguen pendientes. `execution_enabled=false`, `launch_ready=false`, `training_started=false`: este avance no habilita entrenamiento.
>
> La lectura posterior del 2026-10-14 confirmó CUDA y 2 NVIDIA RTX PRO 4000 Blackwell desde `/home/jovyan/work/proyectonp/.venv/bin/python`, con 23.754 MiB libres de 23.985 MiB reportados por GPU en esa medición. Una lectura anterior de `cpu.max=100000 100000` equivale a 1 CPU en ese perfil observado; la terminal de validación mostró después `cpu.max=max 100000` y 48 CPU de afinidad. La discrepancia sigue sin reconciliar: ninguna lectura se declara cuota efectiva única de entrenamiento. El NFS compartido muestra aproximadamente 39,5 TiB disponibles; no se obtuvo una cuota individual ni una reserva GPU. El presupuesto provisional de VRAM más reserva supera en 822 MiB la memoria libre observada. Recursos, presupuestos y runtime de lanzamiento siguen pendientes; el detalle actualizado está en el inventario de abajo.
>
> El flujo histórico de Nicolás para V7 se resume en [55](55-flujo-v7-para-corridas-v8-en-unabia.md) para orientar preparación y cierre de corridas; sus datos, recetas y checkpoints no se trasladan a segmentación V8.

## Historial de preparación

Fecha: 2026-10-13. Rama: `experiment/modular-audit-v8`. Estado: preparación. Este documento no habilita ni inicia entrenamiento, auditorías o pruebas.

### Avance de inventario de recursos y runtime (2026-10-14 UTC)

Esta observación precede al despliegue aislado y la suite de validación registrados en el siguiente avance; describe el directorio activo inspeccionado en ese momento.

La lectura en vivo de UNABIA confirmó el entorno activo y sus límites visibles. Falta acreditar la cuota individual y preparar el código V8 fijado para poder validar cada método en su runtime de lanzamiento.

| Aspecto | Evidencia y pendiente |
|---|---|
| Entorno activo | `/home/jovyan/work/proyectonp/.venv/bin/python`: Python `3.13.15`, PyTorch `2.11.0+cu128`, compilación CUDA `12.8`, CUDA disponible y 2 GPU visibles. Esto no verifica todavía launcher/controlador ni su revisión. |
| Dependencias | NumPy `2.5.3`, SciPy `1.18.0`, scikit-learn `1.9.0`, scikit-image `0.26.0`, Pillow `12.3.0`, W&B `0.30.0` y torchvision `0.26.0+cu128`. Son versiones observadas, no un entorno congelado aprobado. |
| CPU, RAM y procesos | Observación anterior: afinidad 48 CPU; `cpu.max=100000 100000` (equivale a 1 CPU en ese perfil), `memory.max=max`, `pids.max=150962`. Lectura posterior en terminal de validación: `cpu.max=max 100000`, con cgroup visible raíz y 48 CPU de afinidad. Perfil CPU sin reconciliar; ninguna observación acredita la cuota efectiva única. Los límites de shell ilimitados no acreditan una asignación individual de entrenamiento. |
| GPU y presupuesto | Cada RTX PRO 4000 Blackwell reportó 23.754 MiB libres / 23.985 MiB totales. El borrador de 20.480 MiB de VRAM + 4.096 MiB de reserva = 24.576 MiB supera la memoria libre observada en 822 MiB. Sigue provisional: medir y revisar consumo por método antes de fijar o aprobar el presupuesto real. |
| Almacenamiento individual | `/home/jovyan/work` está montado por NFS desde `192.168.247.100:/compartido_serv_unab/home/nmoreno534`; `statvfs` indica cerca de 39,8 TiB totales / 39,5 TiB disponibles en el montaje compartido. El panel menciona almacenamiento QNAP personal sin mostrar una cuota. `quota`, `lfs`, `squota`, `sacctmgr`, `scontrol`, `sinfo` y `squeue` no están disponibles. La cuota individual sigue sin verificar. |
| Código para validar métodos | `/home/jovyan/work/proyectonp` no contiene un checkout Git ni `scripts/run_segmentation_v8.py` o `scripts/run_segmentation_controller_v8.py`. Los candidatos inspeccionados en el nivel superior eran bundles de auditoría. La consulta `git ls-remote` de la rama V8 pidió usuario y fue cancelada sin introducir credenciales. Transferir el proyecto V8 y verificar su revisión/hashes sigue pendiente. |

Se conservan los checksums y la correspondencia de 2.594 IDs de ISIC2018 registrados abajo. Los manifiestos científicos, la evaluación independiente, cuotas y presupuestos por etapa, validación de métodos y transferencia del código siguen abiertos. `execution_enabled=false`, `launch_ready=false` y `training_started=false`; no se ejecutaron nuevas pruebas ni entrenamiento. El gate de limpieza sintética se mantiene pendiente para inmediatamente antes de la primera corrida real.

### Revisión adicional de cuota y discrepancia CPU (2026-10-14, aproximadamente 17:45 UTC)

En la terminal Jupyter de `/home/jovyan/work/validation_v8/f1dc0d2-20261014`, `quota`, `xfs_quota`, `repquota` y `lfs` no están instalados y no se encontraron variables de entorno con límites de cuota, almacenamiento, GPU, CPU o memoria. El montaje `/home/jovyan/work` es NFS `192.168.247.100:/compartido_serv_unab/home/nmoreno534`, con opciones `rw,noatime`; `statvfs` refleja capacidad compartida. La UI UNABIA anuncia almacenamiento personal QNAP sin valor de cuota. La cuota individual continúa sin verificar.

Esta terminal mostró jerarquía cgroup visible raíz (`/`), `cpu.max=max 100000`, `memory.max=max`, `cpuset.cpus.effective=0-47` y afinidad de 48 CPU. La observación anterior del runtime activo había registrado `cpu.max=100000 100000`, equivalente a 1 CPU en ese perfil. Se conservan ambas observaciones sin atribuir ninguna a una cuota efectiva única. **Gate pendiente:** confirmar con UNABIA el entorno y runtime de lanzamiento, reconciliar el perfil CPU y acreditar cuotas individuales antes de aprobar presupuestos. Los perfiles de comprobación por método siguen provisionales y las pruebas sintéticas se conservan hasta la limpieza mediante allowlist inmediatamente anterior a la primera corrida real.

### Avance de validación aislada de métodos en UNABIA (2026-10-14 UTC)

La suite seleccionada pasó en UNABIA: `58 passed, 10 subtests passed in 5.80s`, observado en la terminal Jupyter. Se ejecutó secuencialmente con Python `3.13.15`, sin entrenamiento científico de los cuatro métodos ni lectura de imágenes reales, y sin W&B. `tests/segmentation_v8/test_common.py` incluye un ajuste sintético de fixture de una época con una Mini Conv2d en CPU; por eso el resultado no significa cero pasos de optimización. `training_started=false` se refiere al entrenamiento científico. Después se completaron los cuatro `--stage check` y cuatro dry-runs del controlador con fixtures temporales; estos resultados no acreditan una corrida completa ni validan presupuestos científicos.

La fuente del commit remoto `f1dc0d272732b6c6bbeef46f12c18d88c742763d` se exportó a `/home/jovyan/work/validation_v8/f1dc0d2-20261014`. El hash Git blob del archivo exportado, `a167d0fbc17437a1ff43f06f173304bdac6ce0a6`, coincidió entre origen y clúster. Es un export de fuente para validación aislada; no acredita metadatos Git de un checkout de lanzamiento. `pytest-timeout` faltaba y se instaló únicamente en `.validation_deps` dentro del sandbox, sin modificar el entorno de entrenamiento.

Comando ejecutado en ese directorio de UNABIA:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.validation_deps:src:plan-segmentacion/comun/nucleo:plan-segmentacion/camino-b-cnn/pseudo-mascaras python -m pytest -q --strict-config tests/segmentation_v8 tests/test_public_ssl_staging_v8.py tests/test_public_ssl_snapshot_v8.py tests/test_public_ssl_md5_prefix_v8.py
```

Los checks de `sled`, `usl_net`, `rpi_net` y `tridermnet_hago` usaron `scripts/run_segmentation_v8.py --method <method> --stage check --out /home/jovyan/work/validation_v8/f1dc0d2-20261014/stage_checks/<method>`. Cada proceso devolvió JSON válido con `execution_enabled:false`, sin error informado. Los cuatro declararon `timeout_seconds=180`, `max_rss_mb=4096`, `max_output_mb=4096`, `max_gpu_memory_mb=null` y `gpu_reserve_mb=1024`. Son perfiles de comprobación declarados: no acreditan cuota individual ni límites finales. La ausencia de límite GPU mantiene bloqueado el gate presupuestario hasta medir y revisar cada método bajo su perfil real.

El controlador se comprobó con cuatro planes JSON temporales `fixture_only`, uno por método, con un paso `stage=predict`: `python scripts/run_segmentation_controller_v8.py --plan <plan>`, sin `--execute`. Los cuatro procesos retornaron código 0 y `execution_started:false`, con el orden esperado de un paso. La inspección posterior confirmó ausencia de `ledger.json`, `run_root` y directorios de salida. Esto valida parsing y orden serial del fixture; no valida manifiestos/gates reales ni ejecución.

**Siguiente acción:** resolver IDs, hashes de contenido y grupos paciente/lesión con un conjunto independiente; acreditar la cuota individual con UNABIA; medir y aprobar límites GPU por método bajo el perfil real; y preparar manifiestos, gates y procedencia auténticos, incluida la identidad Git requerida por el launcher. Se mantienen `execution_enabled=false`, `launch_ready=false` y `training_started=false`. Antes de la primera corrida real se deben retirar archivos de prueba y outputs/run/artefactos W&B sintéticos mediante una lista explícita de rutas e identificadores permitidos, preservando primero su procedencia y recibos fuera de resultados reales.

### Avance de forwards sintéticos en UNABIA (2026-10-14 UTC)

Los forwards se realizaron exclusivamente en `/home/jovyan/work/validation_v8/f1dc0d2-20261014`, con la fuente aislada de `f1dc0d272732b6c6bbeef46f12c18d88c742763d`, en CPU a 224×224. Las redes usaron semilla 42, `eval()` y `torch.inference_mode()`. Solo se consumieron tensores/datos artificiales, sin lectura de datos clínicos, archivos de entrada o checkpoints, ni backward, optimizador, épocas, W&B o entrenamiento científico. Este avance es posterior a la suite y sus fixtures descritos arriba.

| Método | Resultado observado y alcance |
|---|---|
| RPI-Net | Pasó con `RPINet(width=256, token_grid=14)`, tensor de imagen artificial y 4 mapas pseudo sintéticos. `logits [1,1,224,224]` y `sigma [1,4,224,224]`, todos finitos. El primer extractor asumió una tupla y trató una clave string del mapping/dict devuelto como tensor: fue un error del harness, no del modelo. Se reejecutó una sola vez con el lector corregido y ese resultado pasó. |
| TriDermNet | Pasó con `TriDermNet(width=32, token_dim=64, patch_size=4, heads=4, max_token_side=28)`. `logits [1,1,224,224]`, `edge_logits [1,1,224,224]`, `coverage_logits [1,1]` y `tokens [1,784,64]`, todos finitos. HAGO no se ejecutó porque implica optimización. |
| SLED | Pasó como forward estructurado con estímulo artificial determinista de figura/contraste a 224×224 y `SLEDConfig()` completo. `mask` y `probability` de 224×224, probabilidades finitas, 10 escalas entre 200 y 650, umbral `0.5404293735189061` y 6.451 píxeles de máscara. Se conserva el intento previo con ruido independiente como abstención esperada por menos de 3 superpíxeles; ese intento no cuenta como pass. |
| USLStudent | El intento anterior llegó al forward, pero el helper falló después y no dejó un resultado final legible/verificable. Queda inconcluso y no cuenta como pass; no se repitió. Es distinto del forward histórico de fusión del banco USL-Net registrado en W&B. |

Estos resultados no aprueban perfiles o presupuestos por método, cuota individual, evaluación independiente/anatómica, manifiestos ni los demás gates de entrenamiento. Se mantienen `execution_enabled=false`, `launch_ready=false` y `training_started=false`; ninguna corrida científica queda habilitada. Las pruebas sintéticas se conservan por ahora. Inmediatamente antes de la primera corrida real se deben inventariar y retirar mediante allowlist sus rutas e identificadores exactos, incluidos los sintéticos de W&B, preservando primero recibos y procedencia fuera de los resultados reales.

### Avance de solapamiento ISIC2018 y evaluación independiente (2026-10-14 UTC)

El cruce corregido de IDs completos en UNABIA encontró 583 coincidencias entre los 2.594 casos de ISIC2018 y los 40.896 IDs del inventario. Los 2.011 casos restantes son candidatos auditados; no se congela un manifiesto ni se declara un holdout independiente o ejecutable. La cifra previa de 1.917 coincidencias se corrige explícitamente: comparaba solo el sufijo numérico de siete dígitos e incluía 1.334 falsos positivos.

La comparación usó `/home/jovyan/work/proyectonp/audit_v8_sync_20261013/full_pool_d55c5cd_20261013/original_records.jsonl` y los ZIP de imágenes y máscaras de ISIC2018 registrados abajo, desde la terminal Jupyter autenticada. El inventario contiene 40.896 filas y 40.896 IDs únicos: `isic2019` 10.991, `isic2020` 24.993 y `ham10000` 4.912. El ZIP de imágenes y la colección API coinciden exactamente en 2.594 IDs completos. El cruce correcto conserva el ID completo, sin sustituirlo por su sufijo numérico.

Se hashearon los 1.334 alias numéricos (784.381.695 bytes), con cero coincidencias SHA-256 exactas; los 677 casos sin alias previamente revisados (3.267.773.886 bytes) también tuvieron cero coincidencias. Los 2.011 candidatos no coinciden por ID completo ni por SHA-256 exacto con el inventario auditado; esto no descarta imágenes casi duplicadas.

El CSV oficial ISIC2018 aporta simultáneamente `patient_id` y `lesion_id` para 390 de los 2.011 candidatos: 350 pacientes y 390 lesiones. Esos grupos conocidos tienen cero cruce con los 583 IDs solapados y cero cruce con los grupos históricos disponibles. Es un subconjunto candidato auditado, sin independencia acreditada. La cobertura histórica de pacientes sigue incompleta: los valores `dummy_*` del CSV ISIC2019 se excluyen y la cobertura verificable de pacientes ISIC2019/HAM es insuficiente. El esquema JSONL original contiene `bytes`, `dataset`, `height`, `image_id`, `image_path`, `image_sha256` y `width`, sin campos de paciente, lesión o `case_group`.

**Próximos gates:** descartar imágenes casi duplicadas, completar y auditar agrupación paciente/lesión histórica trazable y resolver un conjunto independiente con historial de exposición adecuado. La consulta previa del benchmark ISIC2018 y la cobertura histórica incompleta impiden congelar o etiquetar estos candidatos como holdout independiente. Se conservan `execution_enabled=false`, `launch_ready=false` y `training_started=false`; este avance no aprueba cuotas ni presupuestos. Las pruebas sintéticas de UNABIA y W&B se conservan por ahora y se retirarán solo mediante allowlist inmediatamente antes de la primera corrida real, preservando su procedencia.

### Avance de índices ISIC2018 y gate de limpieza (2026-10-14 UTC)

En UNABIA, `/home/jovyan/work/isic2018_task1`, `sha256sum -c checksums.sha256.txt` confirmó ambos ZIP. La inspección de sus índices identificó 2.594 IDs de imágenes `.jpg` y 2.594 IDs de máscaras `_segmentation.png`, con intersección exacta de 2.594 y cero IDs sin pareja en ambas direcciones; los primeros ejemplos fueron `0000000`, `0000001` y `0000002`.

| Archivo | SHA-256 verificado en UNABIA |
|---|---|
| `ISIC2018_Task1-2_Training_Input.zip` | `80f98572347a2d7a376227fa9eb2e4f7459d317cb619865b8b9910c81446675f` |
| `ISIC2018_Task1_Training_GroundTruth.zip` | `99f8b2bb3c4d6af483362010715f7e7d5d122d9f6c02cac0e0d15bef77c7604c` |

Este avance acredita checksums y correspondencia de nombres dentro de los archivos. Siguen pendientes extracción y validación de contenido, grupos paciente/lesión, historial de exposición, ausencia de solapamiento con entrenamiento/desarrollo y manifiesto de evaluación independiente congelado. `execution_enabled=false`, `launch_ready=false` y `training_started=false` se mantienen.

**Gate obligatorio antes de la primera corrida real:** inventariar los artefactos sintéticos en UNABIA y W&B, conservar su procedencia y recibos de auditoría fuera del espacio de resultados reales conforme a [54](54-usl-net-banco-pesos-v8.md), y eliminar las salidas de prueba, logs sintéticos y su run/artefacto W&B antes del lanzamiento. El inventario conocido incluye el run `g8us8pz9`, el artefacto `usl-net-synthetic-forward-20261014` y el recibo `/home/jovyan/work/public_ssl_v8_preparation/usl-net-synthetic-forward-20261014.json`; la preservación del recibo debe resolverse antes de retirar ese archivo del espacio operativo. Registrar la limpieza y comprobar que ninguna ruta de corrida real consuma pruebas sintéticas. Este gate sigue pendiente; todavía no se ejecuta limpieza y no autoriza borrar datos, pesos ni resultados científicos reales.

### Actualización del clúster (2026-10-13)

Jupyter quedó autenticado y accesible para continuar. En la lectura de solo observación: 3 kernels y 25 terminales en Running; no hubo coincidencias de procesos con los patrones de entrenamiento consultados; se observó `dermssl.hub_keepalive` (PID 6397). Dos RTX PRO 4000 Blackwell estaban libres según `nvidia-smi`; RAM disponible 115 GiB y volumen compartido con aproximadamente 40 T libres. Estos datos permiten avanzar con la preparación de archivos y conversiones; **no** habilitan la corrida científica. Faltan límites cgroup/cuotas efectivas, licencias de cada artefacto exacto y las conversiones/hash/normalización/carga estricta. El checkout de referencia continúa en `d55c5cd` con dos cambios locales RPI preservados; el adaptador no está en los checkouts revisados y no se hallaron checkpoints en `local-models`. El detalle y las limitaciones de observación están en [47](47-criterio-no-supervisado-y-controlador-v8.md). No se relanzaron pruebas, auditorías o corridas; tampoco se descargaron pesos.

### Avance de preparación de pesos (2026-10-13)

Se creó en `~/work/public_ssl_v8_preparation/` una zona nueva con `source/` y `checkpoints/incoming/`, sin tocar el checkout `d55c5cd`. El adaptador remoto se recuperó de la rama con blob SHA `0b438c82323137c1ff3263f671b2eb06e3687f22` y su SHA-256 local es `7bfa34bb153e22413409fa77c8e245a401688042af0ffa56d095714d41fcf77d`; la carga del archivo al clúster quedó interrumpida por una restricción de observación del navegador antes de enviar el archivo, por lo que **el adaptador aún no está en el clúster**. Las consultas HEAD a los enlaces oficiales de MoCo v1 (95.347.404 bytes), MoCo v2 (anotación histórica errónea: 1.121.331.339; tamaño real corregido: 112.133.139), MoCo v3 R50 (anotación histórica errónea: 2.727.866.643; tamaño real corregido: 272.786.643) y DINO ViT-S/16 (86.710.517) devolvieron HTTP 200. SimCLR en el enlace publicado redirigió a autenticación de Google; el endpoint `storage.googleapis.com` devolvió 403 desde el clúster. Esas consultas solo comprobaron acceso y tamaño: **no descargaron archivos ni generaron hashes de pesos**. Pendiente recuperar la sesión de Jupyter y el adaptador; después descargar solo archivos con licencia/origen confirmados y verificar hash, normalización y carga estricta. No se instaló TensorFlow; actualmente `torch` está disponible, `tensorflow` y `safetensors` no.

### Preparador reanudable implementado (2026-10-13, America/Bogota)

Se agregó `scripts/stage_public_ssl_weights_v8.py`, con inventario de candidatos en `plan-segmentacion/metodos-v8/public-weights.plan.json`. El modo por defecto no accede a red ni escribe; las acciones `--download` y `--convert` son explícitas. Conserva parciales con identidad/ETag/rango/hash, registra bloqueos por archivo y llama al adaptador V8 fijado para conversiones CPU publicadas junto a su procedencia. Las revisiones por artefacto siguen pendientes (`approved=false`); no se certificaron licencias ni hashes de pesos reales. El bloqueo de autenticación del navegador y el 403 de SimCLR permanecen. No se accedió al clúster, descargaron pesos reales ni se ejecutaron modelos; solo se comprobó la nueva E/S con simulación local. El conversor externo de SimCLR y el adaptador V8 son distintos: no se acredita el despliegue del segundo por la presencia del primero. Los detalles y límites están en [48](48-preparacion-reanudable-pesos-v8.md), con [informe](../resultados/preparacion_v8/staging_20261013/verification.json). El checkout `d55c5cd` y los planes científicos no se modificaron.

### Revisión documental de pesos reales y checksums (2026-10-13)

Se fijaron 16 archivos oficiales de referencia por revisión/blob en `public-weights.evidence.json`. Se confirmaron familias, enlaces y normalización: ImageNet para MoCo/DINO y unit_rgb para SimCLR. MoCo v1/v2 publican solo prefijos MD5 de ocho dígitos; el preparador ahora los comprueba sin inventar el digest faltante y conserva el SHA-256 local. El README histórico de MoCo aún declara CC BY-NC, pero LICENSE y el commit de relicenciamiento indican MIT para código; el alcance sobre checkpoints no se declara cerrado. También se detectó una diferencia estática de padding del max-pooling SimCLR respecto al encoder local, que impide inferir equivalencia numérica solo por carga estricta. Se mantienen las revisiones del plan pendientes y no se descargaron archivos reales. Ver [49](49-revision-fuentes-pesos-reales-v8.md), con el informe de tres comprobaciones nuevas de checksum simulado; no se repitieron las suites anteriores.

### Preparación revisada e intento real de transferencia (2026-10-13)

Se completó un plan documental para preparación académica no comercial, con criterio de licencia, selección y normalización registrados en `public-weights.reviewed-research.plan.json`. La revisión no verifica bytes ausentes ni resuelve por sí misma redistribución o equivalencia SimCLR. Se preparó un snapshot con 210 blobs originales contrastados y un manifiesto SHA-256 fijado. La autenticación segura abrió Jupyter y permitió consultar la carpeta existente, sin checkpoints observados; al abrir el selector de carga el navegador se bloqueó por protección de credenciales. La recuperación devolvió `native credential state cannot be safely resumed`. La transferencia no está confirmada y no se lanzó descarga ni conversión real. Ver [50](50-intento-transferencia-y-revision-preparacion-v8.md) y su informe. El siguiente paso físico sigue siendo desplegar el paquete en carpeta nueva y descargar MoCo/DINO desde el pod.

V7 queda excluida por completo: no se reutilizan sus checkpoints, bancos SSL adaptados, pseudo máscaras ni resultados entrenados. Los modelos aprendibles parten de inicialización aleatoria cuando la receta lo indica; las únicas inicializaciones externas permitidas son pesos públicos originales que la adaptación realmente requiera, con procedencia, licencia, arquitectura, normalización y SHA-256 verificados.

## Estado por método

| Método | Inicialización de V8 | Ya existe | Pendiente antes de una corrida completa |
|---|---|---|---|
| SLED | No necesita pesos ni entrenamiento. | Generador Python, abstenciones y variantes documentadas; etapas `generate/predict`. | Confirmar recursos para el manifiesto completo y congelar el manifiesto/resolución/métricas del protocolo común ya descrito en [45](45-flujo-preparacion-ejecucion-cuatro-metodos-v8.md). La configuración primaria ya queda congelada en `correct_all` + `gaussian_corrected`; `upstream_zero`/`upstream_area` quedan como controles documentados, fuera de la corrida primaria. |
| USL-Net | Encoders públicos SimCLR, MoCo v1 y MoCo v2 en el banco de fuentes; módulos entrenables nuevos desde cero. Nunca usar bancos adaptados de V7. | Generación CCAM y refinamiento para producir pseudo máscaras de otra ronda; el trainer acepta una inicialización iterativa del mismo método con procedencia validada; política UM con área mínima 500 y umbrales .35/.65. | El launcher expone `refine` con gate y comprueba que las predicciones correspondan al manifiesto de entrada. El controlador serial requiere pasos/gates explícitos; no los crea ni infiere el plan científico. Las salidas parciales no `fit` requieren revisión y un directorio nuevo. El corte de distancia al centro queda `null`. MoCo v1/v2 están convertidos estrictamente; SimCLR y el banco de tres orígenes pasaron carga estricta y forward sintético en UNABIA (CPU/eval/inference_mode; seed 42; `torch.randn` float32 `[1,3,224,224]`; salida finita `[1,9216,14,14]`; cero imágenes reales). Siguen pendientes TensorFlow/Google numérico, gates científicos, presupuesto operativo y prueba del launcher/CUDA en el runtime de entrenamiento. |
| RPI-Net | MoCo v3, MoCo v2 y DINO original públicos; MoCo v3 alimenta dos fuentes. Modelo final nuevo desde cero. No sustituir DINO por DINOv2. | Generación de las cuatro fuentes P1–P4 y entrenamiento del segmentador; se documentaron adaptaciones de polaridad y objetivo Grad-CAM++. | El paper indica 50 épocas iniciales y fases sucesivas de 30, pero no cuántas fases. V8 congela una fase posterior de 30 épocas como adaptación mínima y explícita; no es un número reportado por el artículo. El controlador serial coordina fases y refrescos de P2 si cada paso y gate aparece en el plan; no genera automáticamente el manifiesto ni aprueba gates. MoCo v2/v3 y DINO original ya se descargaron y convirtieron estrictamente; el banco candidato RPI pasó la carga integrada. Completar los manifiestos/gates científicos y declarar la selección V8 sin atribuir identidad exacta con los checkpoints del paper. |
| TriDermNet-HAGO | Arquitectura segmentadora inicializada desde cero, de acuerdo con la receta del paper; no requiere checkpoint público de segmentación ni pesos V7. | Red/objetivos de pseudo-segmentación y HAGO con calentamiento de 10 épocas y métricas de balance registradas por época; entrenamiento acepta un manifiesto con pseudo máscaras, sujeto a fijar y verificar su maestro. | No hay una etapa TriDerm que genere sus propias pseudo máscaras; V8 fija SLED como maestro no supervisado. El trainer bloquea la corrida si falta el SHA-256 exacto del manifiesto y comprueba que cada fila provenga del método fijado. No se usan máscaras expertas ni diagnósticos; el hash se completa después de generar y revisar el manifiesto SLED. Cada época registra fracción de calentamiento, peso HAGO medio, entropía media y fracción de pasos con ruido. La adaptación reemplaza diagnósticos y máscaras expertas por cobertura/área pseudo; se reporta como variante no supervisada, no como reproducción de resultados supervisados. V8 fija 150 épocas a 224px y warmup de 10 como presupuesto de adaptación por pseudo máscaras. |

## Detalle y criterio de cierre

### SLED

SLED produce la máscara directamente, por lo que no debe aparecer una etapa ficticia de entrenamiento o checkpoint. Se conservan sus abstenciones en el denominador: no se eliminan casos difíciles para mejorar métricas. La verificación previa de 40.896 imágenes fue una auditoría ya terminada y no se repite aquí.

**Cierre requerido:** medir en el clúster tiempo, RAM y almacenamiento de `generate` para el manifiesto congelado; verificar espacio libre y salida estimada; fijar los hashes del manifiesto de evaluación y de las salidas. La comparación primaria ya está especificada en [45](45-flujo-preparacion-ejecucion-cuatro-metodos-v8.md): SLED directo frente a predicciones finales, informando cobertura/abstenciones por separado. No ejecutar aún ese cálculo.

### USL-Net

El protocolo del paper usa cinco ciclos de refinamiento y 300 épocas por ciclo. El config conserva esos valores. `scripts/run_segmentation_controller_v8.py` puede recorrer en orden las cinco secuencias si cada paso, manifiesto y gate se declara en el plan; el helper de refinamiento por sí solo no constituye el flujo completo. Cada ciclo necesita identidad propia y guardar como mínimo el manifiesto de entrada, checkpoint, predicciones, pseudo máscaras resultantes, hashes, cobertura y estado. Una interrupción debe poder continuar el ciclo desde un checkpoint íntegro o, si no, reabrir una salida nueva para ese ciclo; nunca mezclar productos parciales.

El paper define un umbral de área de 500 píxeles y los umbrales .35/.65. No fija el límite de distancia centroide que descarta muestras; por tanto no inventarlo a partir de resultados observados. Mantener sin corte implica documentar esa decisión y conservar los casos sin foreground identificable según las reglas existentes.

**Cierre requerido:** preparar/revisar el plan serial de cinco ciclos y gates; concretar procedencia reproducible de los tres encoders públicos; fijar tamaño de entrada y budgets de GPU/disco con lectura actual del clúster. El entrenamiento del segmentador y CCAM parte sin pesos V7.

### RPI-Net

Las fuentes son P1 KMeans de MoCo v3 conv1, P2 Grad-CAM++ de MoCo v3, P3 KMeans de MoCo v2 conv1 y P4 KMeans del DINO original. El objetivo CAM de energía SSL y la orientación por mayoría del borde son adaptaciones, no detalles especificados por el paper. No invertir la polaridad usando etiquetas expertas.

El paper describe 50 épocas iniciales y luego fases de 30 épocas, sin declarar el número de fases. V8 fija una fase posterior (50 + 30 épocas) como adaptación mínima y reproducible. No hay que interpretar el número de fuentes (cuatro) como número de fases. La actualización de P2 debe conservar P1/P3/P4 y registrar hashes por fase. El config fija `phase_count=1` e identifica que es una adaptación V8, no un valor del paper.

**Cierre requerido:** registrar la adaptación 50+30 en protocolo/config y W&B; declarar pasos/gates en el plan y verificar los checkpoints públicos elegidos y su conversión estricta. Mantener P1–P4 separados en manifiestos.

### TriDermNet-HAGO

El paper original usa supervisión experta y clasificación diagnóstica. La variante V8 elimina esas etiquetas y usa una cobertura de lesión pseudo como objetivo auxiliar. No existe una inicialización pública original del segmentador indicada por esa receta; por eso, la red debe empezar aleatoriamente y los campos de checkpoint genérico deben permanecer vacíos. No usar los pesos de clasificación de Nico/V7 como sustituto.

Esta variante sí requiere pseudo máscaras de un maestro. El paper no prescribe uno. V8 elige SLED como maestro sin etiquetas humanas para mantener el flujo totalmente no supervisado. TriDerm usa esas pseudo máscaras como objetivos de entrenamiento: es una adaptación no supervisada basada en pseudo etiquetas, pero queda condicionada por SLED y no debe presentarse como una cuarta salida independiente. Reportar siempre la pareja SLED→TriDerm y no mezclarla con una réplica supervisada del paper. No presentar las cifras supervisadas del paper como resultado esperado de esta variante.

HAGO tiene calentamiento de diez épocas. La configuración V8 ya fija 150 épocas a 224×224 como presupuesto de adaptación no supervisada; no es reproducción del objetivo supervisado original de 150 épocas a 256×256.

**Cierre requerido:** generar pseudo máscaras con SLED, revisar su procedencia/cobertura y congelar el SHA-256 del manifiesto antes de entrenar; verificar presupuesto de GPU/RAM/tiempo y que la receta completa el warmup y registra la evolución de pesos HAGO. TriDerm consume el manifiesto fijo de SLED; no genera etiquetas desde una red aleatoria.

## Compatibilidad de las inicializaciones públicas

La lista de orígenes está acotada por método. El candidato PyTorch de SimCLR (Hugging Face, revisión fijada) fue transferido a UNABIA y verificado por tamaño y SHA-256; no es el archivo TensorFlow original de Google. El banco de tres orígenes cargó estrictamente y pasó el forward sintético descrito arriba. Carga estricta y hashes no acreditan equivalencia numérica o identidad exacta con los checkpoints del paper. El formato TensorFlow original de SimCLR tiene una ruta separada (`--format simclr_tensorflow`), pero aún no se ha verificado contra los tensores oficiales. La equivalencia numérica TensorFlow/Google queda pendiente.

| Origen | Fuente pública candidata | Trabajo técnico pendiente |
|---|---|---|
| SimCLR v1 ResNet50 1x | [Google Research SimCLR](https://github.com/google-research/simclr), archivo oficial [ResNet50_1x.zip](https://storage.cloud.google.com/simclr-gcs/checkpoints/ResNet50_1x.zip); candidato PyTorch fijado en [Hugging Face](https://huggingface.co/lightly-ai/simclrv1-imagenet1k-resnet50-1x/tree/476674d2c9804bc2cd4abd8bde6121b6d6b19990) | El candidato PyTorch fue transferido a UNABIA con tamaño y SHA-256 verificados, y participó en carga estricta/forward sintético del banco USL-Net. No equivale por ello al archivo TensorFlow original. Equivalencia numérica TensorFlow/Google, identidad exacta y alcance de licencia siguen pendientes de cierre documental/técnico. |
| MoCo v1 ResNet50 | [Checkpoint oficial MoCo v1, 200 épocas](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar) | Candidato compatible con ResNet50. Confirmar que el estado real corresponde al encoder esperado, guardar SHA-256 y licencia del artefacto, normalización y carga estricta. |
| MoCo v2 ResNet50 | [Checkpoint oficial MoCo v2, 800 épocas](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar) | Candidato compatible con ResNet50 y requerido por USL-Net/RPI-Net. Confirmar licencia del artefacto, SHA-256, normalización y carga estricta. |
| MoCo v3 ResNet50 | [Checkpoint oficial ResNet50 de 1000 épocas](https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar), publicado en [CONFIG.md de MoCo v3](https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md) | Candidato para P1/P2 de RPI-Net. Confirmar que coincide con las fuentes del protocolo, licencia del artefacto, SHA-256, normalización y carga estricta. |
| DINO ViT-S/16 | [Backbone oficial `dino_deitsmall16_pretrain.pth`](https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth), publicado en [Facebook Research DINO](https://github.com/facebookresearch/dino) | El README diferencia el backbone preentrenado del checkpoint completo con cabezas teacher/student; se elige el backbone solamente. Falta revisar la estructura real del archivo, licencia del artefacto, SHA-256, normalización y carga estricta. |

Las fuentes anteriores identifican repositorios oficiales, no certifican que el paper de cada método haya usado exactamente esos checkpoints o número de épocas. Los pendientes de descarga/carga indicados en esta tabla eran históricos para MoCo/DINO y quedaron superados por los recibos y la integración del banco descritos en [51](51-cierre-preparacion-y-bloqueos-lanzamiento-v8.md). Se debe contrastar esa selección con los tres papers y guardar URL de archivo, licencia, SHA-256 y normalización en el manifiesto de procedencia. No declarar verificación sin comprobar el archivo exacto. La conversión SimCLR requiere un lector TensorFlow disponible en el entorno del clúster. La alternativa de preentrenar encoders V8 localmente sigue siendo una adaptación distinta y no sustituye en silencio los inicializadores públicos requeridos.

## Bloqueos comunes

1. Congelar manifiesto elegible y partición por paciente/lesión; resolver la ruta y la reserva de máscaras de referencia independientes para evaluación anatómica.
2. Congelar el manifiesto reservado y las reglas del protocolo común descrito en [45](45-flujo-preparacion-ejecucion-cuatro-metodos-v8.md): SLED directo frente a redes finales, TriDerm presentado como SLED→TriDerm, y cobertura/abstenciones separadas de métricas finales.
3. Los bancos públicos candidatos de USL-Net y RPI-Net ya cuentan con evidencia de preparación/carga descrita en [51](51-cierre-preparacion-y-bloqueos-lanzamiento-v8.md) y [54](54-usl-net-banco-pesos-v8.md). No reutilizar pesos adaptados de V7. SimCLR usa el candidato PyTorch fijado; equivalencia con Google TensorFlow sigue pendiente. La carga de pesos no cierra licencia, manifiestos ni gates científicos.
4. La observación inicial de recursos está en [47](47-criterio-no-supervisado-y-controlador-v8.md), pero faltan cuotas individuales, presupuestos completos por etapa/acumulados de tiempo, RAM, VRAM y disco, y mediciones de perfil. Configs y límites siguen provisionales; no habilitan corridas.
5. Preparar y revisar planes seriales de USL/RPI con configs, manifiestos y gates por ciclo/fase; verificar revisión de SLED y disponibilidad del launcher/CUDA en la revisión fijada del runtime. El controlador no genera estos planes ni autoriza gates.
6. Mantener `execution_enabled=false` hasta que gates, configuraciones, pesos, datos, evaluación y presupuesto coincidan y estén revisados.
7. Antes del primer entrenamiento real, completar y registrar el gate de limpieza de pruebas sintéticas descrito en el avance del 2026-10-14; conservar la evidencia de auditoría fuera de los resultados reales y retirar las salidas, logs, run y artefacto sintéticos de su espacio operativo.

## Orden de trabajo propuesto

1. Congelar comparación, particiones y referencia anatómica.
2. Cerrar procedencia pública de USL/RPI; TriDerm empieza sin checkpoint externo; SLED no usa pesos.
3. Aplicar las decisiones de adaptación V8 ya fijadas: congelar SLED como maestro de TriDerm y registrar explícitamente la fase posterior 50+30 de RPI como adaptación.
4. Preparar y revisar los planes reanudables de USL/RPI, más el manifiesto SLED fijo para TriDerm.
5. Confirmar budgets del clúster y actualizar configs/gates.
6. Cuando todos los cierres estén respaldados, habilitar y ejecutar los métodos serialmente, con revisión entre métodos.

Hasta completar los pasos anteriores, los cuatro métodos siguen en preparación. Este documento no afirma que las corridas estén listas.


Para el criterio no supervisado, las limitaciones del conversor y el esquema del controlador, ver [47](47-criterio-no-supervisado-y-controlador-v8.md).

