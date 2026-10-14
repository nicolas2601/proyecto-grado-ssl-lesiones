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
- [~] **Tarea 4 — cerrar banco de pesos USL-Net.** En curso: candidato exacto de tres orígenes y estado de validación registrados en `plan-segmentacion/metodos-v8/source-bank.usl-net.candidate.json` y docs/54. La carga conjunta del banco en el runtime aún no se verificó; el checkpoint HF de SimCLR no está transferido al clúster.
- [ ] **Tarea 6 — plan serial concreto del controlador.** `scripts/run_segmentation_controller_v8.py`
  leído y entendido (espera plan JSON con `schema_version=1`, `protocol="w3-four-methods-v8"`,
  steps con manifest/gate/config/out, límites `timeout_seconds`/`max_rss_mb` por
  config). No escrito todavía — depende de tarea 1 (necesita el manifiesto real).
- [ ] **Tarea 7 — habilitar execution_enabled y lanzar.** Bloqueada por todo lo anterior.

## Verificación aplicada
Se validó estáticamente el manifiesto candidato USL-Net: exactamente los orígenes
`simclr`, `mocov1` y `mocov2`; hashes y perfiles de normalización presentes. No se
repitieron auditorías, conversiones ni pruebas anteriores. No se hizo carga conjunta,
forward, entrenamiento ni se tocó la descarga ISIC en curso. Jupyter devolvió
`502 Bad Gateway` / `Connection refused` al intentar verificar el runtime.

## Próximo paso
Cuando Jupyter responda, comprobar hashes/rutas y cargar el banco USL-Net con los tres
orígenes; hacer solo un forward sintético 224×224 para verificar la fusión, sin usar
imágenes reales ni entrenar. Mantener ISIC descargando y continuar dejando lanzamiento
deshabilitado hasta cerrar los bloqueos científicos restantes.
