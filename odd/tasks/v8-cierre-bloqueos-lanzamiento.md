# V8 — Cierre de bloqueos de lanzamiento (punch list de docs 46/51)

## Objetivo
Completar los pendientes documentados en docs/46, docs/49 y docs/51 para habilitar
`execution_enabled=true` y lanzar los 4 métodos de segmentación (SLED, USL-Net,
RPI-Net, TriDermNet-HAGO).

## Por qué / contexto
El usuario pidió avanzar el punch list en este orden:
1. Congelar manifiesto elegible con partición paciente/lesión.
2. Resolver SimCLR (acceso TF oficial o despliegue del parche al clúster).
3. Generar y congelar manifiesto de pseudo-máscaras SLED.
4. Cerrar banco de pesos USL-Net.
5. Medir cuotas reales del clúster.
6. Escribir plan serial concreto del controlador.
7. Habilitar `execution_enabled=true` y lanzar.

Acceso al clúster UNAB autorizado explícitamente por el usuario vía Playwright MCP
(URL: https://unabia.unab.edu.co, ver memoria `infra/unab-cluster-access`). El login
lo hizo el usuario manualmente (nunca se tipeó una contraseña por el modelo).

## Estado por tarea

- [x] **Tarea 5 — medir cuotas reales del clúster.** Confirmado en vivo por terminal
  (2026-10-14): 48 CPUs, 123 GiB RAM (115 GiB libres), 2× RTX PRO 4000 Blackwell
  24 GB cada una (0% uso), cgroup CPU/memoria en `max`/`max 100000` (sin límite real
  impuesto por la plataforma), disco compartido 40T (1% usado). Coincide exactamente
  con lo ya registrado en `resultados/preparacion_v8/readiness_20261014/readiness.review.json`
  (bloque `resources`) — no hay deriva, el registro sigue vigente. Pendiente real: NO
  es volver a medir, es traducir estos números en límites concretos de timeout/RSS/VRAM
  por etapa para el plan del controlador (tarea 6).

- [~] **Tarea 1 — congelar manifiesto elegible.** Groundwork completo, construcción
  bloqueada por falta de datos (no de código). Ver memoria Engram
  `proyecto-grado-ssl-lesiones/v8-manifiesto-elegible` para el detalle completo:
  - Ubicación real de datasets confirmada: `/home/jovyan/work/datos/{ham10000,isic2019_fullres,isic2020}/`
    con sus CSV de metadata y columnas exactas.
  - Contrato exacto de `src/dermssl/segmentation_methods/common/data.py` leído byte
    a byte (ALLOWED/FORBIDDEN, eligibility block, recomputo de SHA-256 en cada carga).
  - **Bloqueo raíz real encontrado**: la "evaluación anatómica independiente" que
    exige docs/39:82 necesita máscaras expertas de referencia (tipo ISIC2017/2018
    Lesion Boundary Segmentation Challenge). No existe ningún dataset de máscaras
    expertas descargado en el clúster — solo hay máscaras generadas por U-Net para
    QA de recorte (`mask_manifest.json`, no son expertas). Esto es un problema de
    adquisición de datos externos, no de escritura de script.
  - **Decisión del usuario (2026-10-14): descargar ISIC2018 Task 1 oficial.**
    Verificado con HEAD request (sin login, HTTP 200): `ISIC2018_Task1-2_Training_Input.zip`
    y `ISIC2018_Task1_Training_GroundTruth.zip` (27.402.895 B) en
    `https://isic-challenge-data.s3.amazonaws.com/2018/`. Descarga lanzada en segundo
    plano en el clúster (`nohup`+`disown`, sobrevive aunque se cierre la sesión del
    navegador) hacia `/home/jovyan/work/isic2018_task1/` (NO en `/home/jovyan/work/datos/`,
    que es de solo lectura — dueño `nobody`, permiso denegado al escribir ahí).
    El script de descarga calcula `checksums.sha256.txt` y escribe `.done` al terminar.
    **Corrección (2026-10-14, ~13:52 UTC): el Content-Length real del Training Input es
    11.165.358.566 B (~11,2 GB), no 1,1 GB — error de lectura propio en la primera
    verificación (se perdió un dígito). A los ~45 min llevaba 2,1 GB (~19%); a ese ritmo
    son varias horas, no ~30 minutos.** Pendiente: confirmar `.done`, verificar que
    `ISIC2018_Task1_Training_GroundTruth.zip` tenga 2594 máscaras
    `ISIC_<id>_segmentation.png`, y recién ahí diseñar cómo esta reserva se integra al
    manifiesto elegible (como partición `development` separada y fuera del pool `train`,
    per docs/39:82).

- [~] **Tarea 2 — resolver SimCLR.** Parcialmente resuelto (2026-10-14):
  - Confirmado (3 vías independientes: API JSON del bucket GCS, ruta de objeto
    directa, mirror `simclr-gcs`) que el acceso anónimo al checkpoint TF oficial
    está bloqueado por IAM de Google Cloud (401 listado, 403 objeto), no por una
    URL incorrecta. Requiere credenciales GCP reales — fuera de alcance sin que
    el usuario las provea.
  - **Parche desplegado al clúster con éxito** (autorizado por el usuario):
    `v8_simclr_same_profile_eed86c8.patch.zip` reconstruido desde el checkout
    local (los 3 archivos no cambiaron desde la revisión previa `7a586059`),
    transferido por terminal en bloques base64 (el intento con una sola línea
    de 10KB se corrompió; fragmentado en 6 bloques de 2000 caracteres funcionó),
    extraído en `~/work/public_ssl_v8_preparation/runtime_simclr_same_eed86c8/`,
    verificado hash por hash contra el repo local (match exacto). Ver doc 53.
  - Sigue pendiente: `scientific_execution_authorized=false` — desplegar código
    no es lo mismo que autorizar correr algo sobre él; y la equivalencia TF
    numérica real sigue sin verificar (solo el candidato HF está validado).
- [ ] **Tarea 3 — manifiesto pseudo-máscaras SLED.** Sin avance (depende de tarea 1).
- [x] **Tarea 4 — cerrar banco de pesos USL-Net. Cerrado y verificado end-to-end
  (2026-10-14).** Una sesión paralela (mismo usuario) preparó en paralelo el
  candidato de los tres orígenes (`plan-segmentacion/metodos-v8/source-bank.usl-net.candidate.json`,
  docs/54-banco-uslnet-preparacion-v8.md) pero no pudo verificar en el clúster —
  Jupyter devolvió `502 Bad Gateway`. Esta sesión sí tuvo el clúster accesible y
  completó la verificación real: USL-Net exige exactamente `{simclr, mocov1,
  mocov2}` (`usl_net/model.py`). MoCo v1/v2 ya estaban convertidos. Para SimCLR: se
  completó el árbol de paquete `dermssl` que faltaba en el runtime desplegado
  (checkpoints.py, dino.py, data.py, transferidos por base64 fragmentado y
  verificados hash por hash), se descargó el checkpoint HF al clúster
  (tamaño/SHA-256 exactos), se corrió `prepare_public_ssl_weights_v8.py` de
  verdad — el `checkpoint_sha256` resultante coincide byte a byte con la validación
  local previa y con la de la sesión paralela (conversión determinística). Se
  construyó `source-bank.usl-net.reviewed-candidate.json` y se verificó con
  `source_bank()` real: los 3 encoders cargan con `strict=True`, 23.508.032
  parámetros cada uno. Ver doc 54 (este, el que relata la ejecución real).
  `execution_ready` sigue en `false` — cargar pesos no autoriza entrenar.
- [ ] **Tarea 6 — plan serial concreto del controlador.** `scripts/run_segmentation_controller_v8.py`
  leído y entendido (espera plan JSON con `schema_version=1`, `protocol="w3-four-methods-v8"`,
  steps con manifest/gate/config/out, límites `timeout_seconds`/`max_rss_mb` por
  config). No escrito todavía — depende de tarea 1 (necesita el manifiesto real).
- [ ] **Tarea 7 — habilitar execution_enabled y lanzar.** Bloqueada por todo lo anterior.

## Verificación aplicada
El cierre del paso 4 está registrado en el commit `c55b6b146d78d4ab9e2a7fb6a1743e72c3fb134b`
y en [docs/54-usl-net-banco-pesos-v8.md](../../docs/54-usl-net-banco-pesos-v8.md):
SimCLR HF se transfirió y convirtió en el clúster con el SHA-256 esperado, y
`source_bank()` cargó exactamente `simclr`, `mocov1` y `mocov2` con
`strict=True` (23.508.032 parámetros por encoder). No se entrenó. El recibo
`readiness.review.json` conserva `launch_ready=false` y `training_started=false`.

El recibo versionado no documenta un forward sintético de 224×224 ni la salida
`[1, 9216, 14, 14]`; por ello ese subcheck no se declara aprobado. Esta continuación
no repitió transferencias, conversiones, cargas, auditorías, pruebas ni entrenamientos.
El estado del login y de la descarga ISIC no se inspeccionó en vivo en esta sesión.

## Próximo paso
No repetir la transferencia, conversión ni carga conjunta del banco USL-Net: esa parte
de la tarea 4 ya quedó cerrada en el commit `c55b6b1`. Cuando haya acceso seguro al
clúster, comprobar primero si existe un recibo del forward sintético 224×224; solo si
falta, hacer una única pasada sin imágenes reales ni entrenamiento y registrar si la
salida es `[1, 9216, 14, 14]`.

Las tareas 1, 2, 3 y 6 siguen abiertas: terminar y revisar la reserva anatómica y el
manifiesto elegible (incluida la descarga ISIC2018 ya iniciada), mantener explícita la
falta de equivalencia numérica con el checkpoint TensorFlow original de Google,
congelar el manifiesto de pseudo-máscaras SLED, y concretar el plan serial con sus
límites por etapa. Mantener `execution_enabled=false`; no lanzar corridas hasta que
los bloqueos científicos y operativos estén cerrados y revisados.
