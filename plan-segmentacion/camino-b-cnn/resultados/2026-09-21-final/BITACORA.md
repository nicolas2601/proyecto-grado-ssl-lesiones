# Bitacora - pipeline de segmentacion label-free (cluster UNABIA)

## Nomenclatura de lectura

**Proposed sin segmentación** clasifica sin usar una máscara de lesión. **Proposed con segmentación** combina imagen completa, recorte y descriptores de lesión; es una variante experimental, sin mejora robusta demostrada. El Modelo de control no es una tercera propuesta ni la versión segmentada.

Los experimentos de segmentación se identifican por arquitectura, origen de pseudo-máscaras y ronda de aprendizaje. Las rondas no son arquitecturas distintas; los identificadores antiguos se conservan solo donde permiten localizar archivos o reproducir comandos.

Ver [nombres, equivalencias y explicación para lectores técnicos y no técnicos](../../../../docs/10-glosario-eli5.md). Los registros históricos conservan sus cifras, predicciones y limitaciones.

Pod: jupyter-nmoreno534 | 2x RTX PRO 4000 Blackwell 24GB | ~/work = NFS 40T

## 2026-09-21 02:20-02:46 UTC - Preparacion del entorno (sin entrenamiento)

Commit: n/a (repo todavia no clonado en el pod)
Duracion: ~25 min

Acciones:
- Verificado hardware: 2 GPUs libres (4 MiB usados, 0% util, sin procesos). Driver 580.178.04, CUDA 13.0.
- Disco: ~/work 40T, 175G usados (1%).
- Inventario de datos (jpg):
  - ~/work/datos/ham10000: 10015 lesiones unicas (HAM10000_images_part_1=5000 + part_2=5015).
    OJO: existen ademas copias en minuscula ham10000_images_part_1/2 -> 2.7 GB duplicados, uid 65534, no borrables.
  - ~/work/datos/isic2019: 25331 | isic2019_fullres: 25331 | isic2020: 33126
  - datos_isic2018/ISIC2018_Task1-2_Training_Input: 2594
  - datos_fitzpatrick17k: 16574
- Mascaras de evaluacion (SOLO EVAL, nunca entrenamiento):
  - masks_eval/isic2018/ground_truth -> symlink a datos_isic2018/ISIC2018_Task1_Training_GroundTruth (2594 png).
  - masks_eval/ham10000/ -> descargado de Dataverse doi:10.7910/DVN/DBW86T, datafile 3838943 (10808743 bytes). 10016 png.
- Entorno: .venv creado con `python -m venv --system-site-packages`. Instalados scipy, pillow, wandb, pytest.
  Verificado: torch 2.11.0+cu128, cuda True, 2 GPUs. NO se toco torch.
- Weights & Biases: login OK como nmoreno534 (entity nmoreno534-universidad-autonoma-de-bucaramanga).
  Smoke run: https://wandb.ai/nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion/runs/jg99g03f
- Upload por el file browser de JupyterLab: probado OK (archivo de 33 B, md5 identico origen/destino).

Observaciones / bloqueos:
- El repo nicolas2601/proyecto-grado-ssl-lesiones es privado y el pod no tiene credenciales de GitHub.
  Decision: NO se instala un token de GitHub en el pod (el NFS es compartido). El codigo entra por git bundle
  subido con el file browser.
- Metricas de entrenamiento: todavia ninguna. No se ha ejecutado nada del pipeline.

## 2026-09-21 03:30-03:45 UTC - Repo al pod + verificacion + smoke + lanzamiento del piloto

Commit: f19ab56 (rama feature/fitzpatrick-domain-adaptation-informe, 4 commits sobre 276a802)

- Bundle subido por el file browser de JupyterLab. sha256 cf124918cc6161c94a33ef8e28838be4de43bbd1793dd2d4ce61088aad4c5b64, 16853222 bytes, verificado en el pod.
  Gotcha: JupyterLab muestra un dialogo "Large file size warning" a partir de 16 MB; hay que confirmar Upload o el archivo nunca sube.
- git clone -b feature/fitzpatrick-domain-adaptation-informe dermssl-repo.bundle repo -> HEAD f19ab56.
- eval_ham10000_images creado con symlinks (runbook seccion 5): 10015 enlaces contra 10015 mascaras, 1:1 exacto.
- check_env.py: torch 2.11.0+cu128, cuda 12.8, bf16 y NCCL disponibles, 2x RTX PRO 4000 23.4 GB, work_dir_free_gb 40716, shm_free_mb 64.
- pytest tests/test_seg_static.py tests/test_seg_field_roi.py tests/test_seg_run_pipeline.py -> 136 passed in 0.51s.
- Smoke GPU (smoke.sh --nproc-per-node 2), todas las etapas codigo 0:
  env 3.3s, cache 0.4s, contrastive 8.3s, ccam 5.7s, pseudo_r0 0.4s, Experimento de segmentación 1 7.2s, pseudo_r1 0.4s, Experimento de segmentación 2 7.3s, synthetic_eval 0.2s, evaluate_synthetic 3.0s.
- Piloto lanzado con cluster-piloto.json, --nproc-per-node 2, workdir ~/work/proyectonp/salidas/piloto, PID 4392.
  Antes se corrio --dry-run para validar rutas. GPUs en 4 MiB / 0% al lanzar.
  W&B: grupo piloto en https://wandb.ai/nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion

Observacion del cache (confirmada por Paula):
- Pool unico 58457 = HAM 10015 (part_1 5000 + part_2 5015) + ISIC2019 sin HAM 15316 + ISIC2020 33126.
- cache_io.scan_image_roots deduplica por stem (ISIC_xxxxxxx) entre roots en orden; ISIC2019 contiene las 10015
  imagenes de HAM10000 con el mismo id, asi que gana el primer root (ham10000_p1/p2) y se descartan las copias
  de isic2019: 25331 - 10015 = 15316. No es choque de nombres de root.

### Campo del dermatoscopio (cache del piloto)

field_stats.json: 58457 imagenes, circular 10762, full_frame 47695, fallback 0,
mean_field_fraction 0.9654, mean_valid_fraction 0.9358 (black_threshold 0.06, margin_fraction 0.04).

Cruce por dataset (field.npy + manifest.csv, umbral valid_fraction >= 0.999 = full frame;
la suma de circulares reproduce exacto el circular_count del pipeline):

| dataset      | n     | mean_valid | full_frame | circular |
|--------------|-------|------------|------------|----------|
| ham10000_p1  |  5000 | 0.9689     | 4452       |  548     |
| ham10000_p2  |  5015 | 0.9561     | 4241       |  774     |
| isic2019     | 15316 | 0.8156     | 7851       | 7465     |
| isic2020     | 33126 | 0.9833     | 31151      | 1975     |

La distribucion del campo coincide con la composicion conocida de los datasets: HAM mayormente
full_frame (87%), ISIC2019 mixto porque agrega BCN20000 (que si trae el circulo), ISIC2020 casi
todo full_frame porque SIIM-ISIC 2020 son dermatoscopicas de cuadro completo sin vineta. Concuerda
con los conteos de cuarentena previos (ISIC2019 456 mascaras degeneradas de 25k vs ISIC2020 68 de 33k).
fallback_count 0: el detector nunca tuvo que rendirse.

Pendiente de verificacion visual: contact sheet con el borde del campo dibujado (script
field_contact_sheet.py llega en el bundle 2) para descartar falsos full_frame con vineta gris.

### Comunicacion

2026-09-21 03:47 UTC (22:47 hora local) - Primer mensaje de WhatsApp a Pau (chat "Pao Pau"), autorizado
explicitamente por Nicolas en la sesion. Contenido: aviso de que el piloto arranco (camino B, sin mascaras)
+ link del proyecto en W&B de la organizacion + "te voy contando". Sin credenciales ni rutas internas del pod.
Estado de entrega al momento de escribir: una palomita (Sent), no dos. El telefono de Pau no estaba conectado;
se entrega sola cuando abra WhatsApp.
Proximos mensajes: solo por hito (piloto terminado con Dice/IoU, final terminado, o problema serio).

### Chequeos de salud del piloto (2026-09-21 04:00 UTC)

**pseudo_r0/pseudo_masks_stats.json** (excluded_fraction 0.1158, zona verde < 0.3):
images 58457, excluded 6771, excluded_by_area 6244, excluded_no_region 527, excluded_no_field 0,
mean_foreground_fraction 0.1628, mean_ignore_fraction 0.2251, field_used true, source ccam,
low 0.35, high 0.65, max_area_fraction 0.6, reads_masks_or_labels false.
excluded_no_field = 0: el detector de campo nunca fue la causa de una exclusion. El 92% de las
exclusiones son del cap de area.

**Polaridad de las pseudo-mascaras** (tools/check_polarity.py, read-only, fuera del repo):
20 imagenes de HAM comparando area de pseudo-mascara contra area de la GT de Tschandl (224, NEAREST).
Valores de la pseudo-mascara: {0, 1, 255}, con 255 = ignore.
n=20, mediana de ratio 0.96, media 0.88, ninguna con ratio > 2, 20% con ratio < 0.5.
NO hay inversion de signo ni captura del campo (eso daria ratios de 3-5; el maximo observado es 1.73).
Sesgo leve a subsegmentar. Los dos ratios 0.00 son imagenes con ignore=1.0 (excluidas enteras por el
cap de area). Los tres ratios ~0.3 son lesiones con GT de 0.42-0.74 del cuadro, justo donde
max_area_fraction 0.6 recorta. Revisar esos casos en los overlays.

**Rendimiento medido**
| etapa | img/s | seg/epoca | VRAM/GPU | util |
|-------|-------|-----------|----------|------|
| moco epoca 0 | 269.11 | 212.14 | 8190 MiB | - |
| moco epocas 1-4 | ~773 | ~73.9 | 8190 MiB | alternante |
| ccam | ~2878 | ~19.9 | 3528 MiB | 25-27% |
| Experimento de segmentación 1 epoca 0 | 700.97 | 72.22 | 3906 MiB | 98% |

La epoca 0 de cada etapa carga el warmup y no sirve para proyectar. El regimen real de MoCo es 773 img/s,
no los 269 de la epoca 0.

**Proyeccion del preset final (lado entrenamiento)**: MoCo 30 epocas ~39 min + CCAM pocos minutos +
red de segmentación entrenada con pseudo-máscaras 3 rondas x 20 epocas x ~72 s ~72 min = **~1 h 55 min**. Falta estimar la etapa evaluate con
hausdorff true (2594 + 10015 imagenes); el piloto la tiene apagada, asi que no hay muestra. Esa es la
incognita del presupuesto.

### HITO: correccion de vineta validada con datos reales (pseudo_r0)

Tres evidencias independientes cierran la causa raiz del problema de vineta:
1. excluded_no_field = 0 en pseudo_masks_stats.json: el detector de campo no excluyo ni una imagen.
2. Polaridad: mediana de ratio pseudo/GT = 0.96 sobre 20 imagenes de HAM, ningun caso con ratio > 2.
   Capturar el campo daria ratios de 3-5; el maximo observado es 1.73.
3. El red de segmentación entrenada con pseudo-máscaras entrena saturando las dos GPUs (98% util) sobre esas pseudo-mascaras.

Pendientes abiertos para la corrida final (no para ahora):
- 6244 exclusiones por area (10.7%) con cap 0.6. Las lesiones grandes (GT 0.42-0.74) quedan fuera del
  entrenamiento del red de segmentación entrenada con pseudo-máscaras, asi que puede subsegmentar lesiones grandes. Al leer la evaluacion del
  piloto: revisar si los peores Dice son mayoritariamente de GT grande. Si es asi, subir el cap a 0.75
  en el final (una mascara de campo entero sigue en 0.9-1.0, asi que el cap la sigue atrapando).
- Medir HD95 antes de lanzar el final: correr evaluate_segmentation.py a mano con --hausdorff solo sobre
  ISIC2018 y cronometrarlo, sin W&B. Con ese tiempo se escala a HAM.

## 2026-09-21 04:15 UTC - PILOTO TERMINADO (commit f19ab56)

9/9 etapas en codigo 0, cero tracebacks. Total 2242.9 s (37.4 min).
env 3.0 | cache 204.9 | contrastive 528.9 | ccam 121.5 | pseudo_r0 208.9 | Experimento de segmentación 1 391.1 |
fst 421.3 | evaluate_isic2018 281.5 | evaluate_ham10000 81.8

La diferencia 281.5 s (2594 img) vs 81.8 s (10015 img) es decodificacion: los JPEG de ISIC2018 son
de 4000x3000 y los de HAM de 600x450. Sin accion.

**Resultados**
| conjunto | n | dice mean | dice median | dice p10 | iou mean | gap_dice | peor bucket |
|----------|---|-----------|-------------|----------|----------|----------|-------------|
| isic2018 | 2594 | 0.652 | 0.726 | 0.286 | 0.528 | 0.017 | 1 |
| ham10000 | 10015 | 0.759 | 0.810 | 0.509 | 0.638 | 0.071 | 6 |

**ISIC2018 se reporta en DOS filas** (para el informe): 706 ids estan dentro del pool de entrenamiento
(transductivo, dice 0.787-0.90 por bucket) y 1888 estan fuera (dice 0.601). El dedup por stem hace que
parte de ISIC2018 entre al pool via HAM/ISIC2019. El titular 0.652 es la mezcla; reportarlo solo seria
inflar el numero.

**El cap de area NO es la causa de los peores Dice** (hipotesis descartada con datos):
corr(dice, ref_coverage) = +0.52 en ISIC2018 y +0.40 en HAM. Positiva: a lesion mas grande, mejor Dice.
Las 100 peores tienen ref_coverage mediana 0.018 / 0.016 contra 0.138 / 0.222 del conjunto: son lesiones
DIMINUTAS, no grandes. Dice por banda de ref_coverage:
| ref_cov | isic2018 n / dice | ham n / dice |
|---------|-------------------|--------------|
| [0.00,0.10) | 1056 / 0.464 | 1956 / 0.556 |
| [0.10,0.25) |  685 / 0.749 | 3623 / 0.786 |
| [0.25,0.40) |  405 / 0.811 | 2203 / 0.834 |
| [0.40,0.60) |  273 / 0.806 | 1583 / 0.825 |
| [0.60,1.01) |  175 / 0.801 |  650 / 0.802 |
Decision: max_area_fraction se queda en 0.6.

**Modo de falla real** (overlays/worst.png): las 12 peores de ISIC2018 tienen dice = 0.000 exacto.
No subsegmentan: predicen en otro lado. Marcas de tinta quirurgica violeta, burbujas de gel, reglas
graduadas y brillos especulares. El red de segmentación entrenada con pseudo-máscaras rodea cada marca de tinta e ignora el nevo palido.
Aprendio "lo oscuro y contrastado del centro" y la tinta cumple ese criterio mejor que la lesion.

Fixes acordados para el final (bundle 3, todos sin etiquetas):
1. Supresion de artefactos en la pseudo-mascara: tinta violeta (B > R con saturacion alta) y
   especulares (casi blancos sin saturacion) pasan a IGNORE. Es un prior de color, no una etiqueta.
2. min_area_fraction baja de 0.01 a 0.003: hoy ninguna lesion menor al 1% del cuadro entra como
   candidata, asi que el red de segmentación entrenada con pseudo-máscaras nunca vio lesiones pequenas. Explica el dice 0.464 en la banda
   [0, 0.10), que es el 41% de ISIC2018.
3. Detector de campo v2 (esquinas por croma, 3 de 4): los falsos circulares de HAM e ISIC2020 pasan
   a full_frame y los verdaderos siguen circulares.

**REGLA DE CORRIDA (pedida por Paula)**: toda la corrida final debe salir de UN SOLO COMMIT. El driver
lanza cada etapa como un subproceso nuevo, asi que un git pull a mitad de corrida hace que las etapas
posteriores corran con codigo distinto al de las anteriores. Por eso el piloto corrio entero con
f19ab56 desde ~/work/proyectonp/repo, y el bundle 2 se aplico en un clon aparte (repo2, commit 7055266).
El final se lanzara desde un clon limpio con el commit del bundle 3.

## 2026-09-21 05:06:40 UTC - CORRIDA FINAL LANZADA

Commit: d81a492 (clon limpio ~/work/proyectonp/repo_final, bundles 1 a 5 aplicados en orden).
sha256 verificados en el pod: bundle3 aa0f16e6...cbc6b, bundle4 5a0f4bc0...50356, bundle5 0d2090d4...c53104.
Smoke previo corrido en 99344a4: 10/10 etapas codigo 0. No se repitio en d81a492 porque
`git diff --stat 99344a4 d81a492` = un solo archivo (RUNBOOK_CLUSTER.md, +20/-5), cero codigo ejecutable.

GPUs antes de lanzar: 4 MiB usados, 0% util, sin procesos de computo.

Comando (seccion 9 del runbook, con el arranque desacoplado):
  setsid nohup python plan-segmentacion/camino-b-cnn/entrenamiento/run_pipeline.py \
    --nproc-per-node 2 --config plan-segmentacion/comun/configs/cluster-final.json \
    --workdir $HOME/work/proyectonp/salidas/final \
    --root ham10000_p1=... --root ham10000_p2=... --root isic2019=... --root isic2020=... \
    --wandb-project dermssl-segmentacion \
    --eval-set isic2018=... --eval-set ham10000=... \
    < /dev/null > final.log 2>&1 & disown

Verificacion de desacople (lo que fallo con los pisos):
    PID    PGID     SID TT           ELAPSED CMD
   1747    1747    1747 ?              00:12 python ... run_pipeline.py
TT = ? y SID = PID = PGID: sesion propia, sin terminal controlante. Sobrevive a la caida de la
terminal de Jupyter, del navegador y de la sesion del agente.

Proyeccion: ~2 h de entrenamiento (MoCo 30 epocas ~39 min + red de segmentación entrenada con pseudo-máscaras 3 rondas x 20 epocas ~72 min
+ CCAM unos minutos) + ~6 min de evaluacion con hausdorff encendido (medido: HD95 no agrega costo,
el tiempo de evaluate es decodificacion de JPEG).
Grupo en W&B: "final" (nombre del workdir).

En paralelo, en CPU: los pisos no supervisados (12 shards sobre ISIC2018 desde 04:56:29). Se acepto
la contencion de CPU entre la etapa cache del final (16 hilos, ~3.5 min) y los 12 shards.
La cache se recalcula con el detector de campo v2; no se reutiliza la del piloto.

## 2026-09-21 - INCIDENTES: reinicios del contenedor del pod

Tres caidas durante la corrida final. Cronologia y diagnostico:

| hora UTC | que murio | contexto |
|----------|-----------|----------|
| ~04:55 | los 12 shards de los pisos | pisos solos, lanzados con nohup sin setsid |
| 05:16:45 | CONTENEDOR entero (final + pisos) | final en MoCo epoca 2 + 12 shards |
| 06:07:57 | CONTENEDOR entero (final + pisos) | final en MoCo epoca 14 + 10 shards |

**Causa de la primera**: los procesos vivian en el grupo de la terminal de Jupyter; al morir la terminal
se llevo el grupo. El nohup del script padre no alcanza porque los hijos eran jobs en background del
mismo script, con el mismo PTY controlante.
Fix: `setsid nohup ... < /dev/null > log 2>&1 & disown`, verificando con
`ps -o pid,pgid,sid,tty` que quede TTY=? y SID=PID.

**Causa de las dos siguientes**: NO es el idle-culler de JupyterHub. Se descarto con evidencia:
se dejo un kernel keepalive imprimiendo cada 60 s (keepalive.ipynb, kernel en Busy) y el contenedor
se reinicio igual; el mtime del notebook (06:07:31) prueba que el kernel seguia vivo 26 s antes
del reinicio (06:07:57). Hipotesis del culler retirada.

Patron observado: las tres muertes ocurrieron con el final y los pisos corriendo a la vez. Los dos
reinicios de contenedor estan separados por 51:12. Sospecha: presion de memoria del nodo. Cada shard
de segnorm decodifica JPEG de 4000x3000 y trabaja en float64 por canal (cientos de MB por imagen con
copias intermedias); 10-12 shards son varios GB de RSS ademas de los 8.8 GB de page cache del memmap
de images.npy. `memory.events` da oom_kill 0 pero el cgroup es del contenedor nuevo, asi que no
descarta un OOM del anterior. No se puede diagnosticar desde dentro del pod: hace falta que Alfredo
mire `kubectl describe pod` (razon OOMKilled / Evicted).

**Prueba de discriminacion**: el final relanzado SOLO (sin pisos) a las 06:54:17 supero los 51 min
(contenedor vivo mas de 1 h 06) y la memoria se mantuvo en 7.8 GB usados de 123 GB. Evidencia a favor
de la presion de recursos y en contra de un max_age duro del hub.

**Gotcha critico del W&Proposed sin segmentación*: `~/.netrc` vive en el overlay efimero del contenedor, NO en ~/work (NFS).
Cada reinicio lo borra y el pipeline arranca SIN tracking, sin fallar: solo imprime una linea
"aviso: seguimiento W&B deshabilitado". El procedimiento de recuperacion es de TRES pasos:
  1. `wandb login` de nuevo
  2. relanzar el pipeline
  3. verificar que el log NO tenga la linea de "seguimiento W&B deshabilitado"
No se guarda el netrc en ~/work a proposito: dejaria la clave persistente en el NFS compartido.

**Herramientas de diagnostico dejadas**: ~/work/proyectonp/memwatch.log (free + loadavg cada 30 s,
tambien con setsid).

## 2026-09-21 10:45 UTC - PISOS NO SUPERVISADOS v2 Y COMPARACION FINAL

Pisos con --max-side 1024 y 4 shards (repo2 en 03fa5a2). Lanzados 09:36:51, cerrados 10:41:57.
segnorm isic2018 1615 s, segnorm ham10000 ~2200 s, evaluaciones 11-60 s cada una.
El downscale fue mas rapido Y mas liviano: 108 img/min con 4 shards contra 79 img/min con 12 shards
a resolucion nativa.

**Tabla de Dice medio**
| run | n | dice mean | IC 95% | mediana | p10 | hd95 mean |
|-----|---|-----------|--------|---------|-----|-----------|
| final isic2018 | 2594 | 0.718 | 0.709-0.726 | 0.792 | 0.402 | 26.05 |
| rnorm isic2018 | 2594 | 0.716 | 0.707-0.724 | 0.789 | 0.348 | 33.47 |
| xnorm isic2018 | 2594 | 0.695 | 0.687-0.704 | 0.768 | 0.318 | 33.05 |
| final ham10000 | 10015 | 0.785 | 0.781-0.788 | 0.852 | 0.507 | 22.44 |
| rnorm ham10000 | 10015 | 0.765 | 0.762-0.768 | 0.819 | 0.530 | 22.67 |
| xnorm ham10000 | 10015 | 0.733 | 0.729-0.736 | 0.788 | 0.485 | 22.96 |

**Bootstrap pareado por imagen** (tools/paired_bootstrap.py, 10000 remuestreos, semilla 42;
mismas imagenes en ambos brazos, que es el test correcto y mas potente que comparar IC marginales):

Dice (mas alto es mejor):
| comparacion | n | diferencia | IC 95% | veredicto |
|-------------|---|------------|--------|-----------|
| isic2018 red de segmentación entrenada con pseudo-máscaras - rnorm | 2594 | +0.0015 | [-0.0076, +0.0107] | INCLUYE 0: empate |
| isic2018 red de segmentación entrenada con pseudo-máscaras - xnorm | 2594 | +0.0223 | [+0.0127, +0.0318] | gana el red de segmentación entrenada con pseudo-máscaras |
| ham10000 red de segmentación entrenada con pseudo-máscaras - rnorm | 10015 | +0.0197 | [+0.0164, +0.0231] | gana el red de segmentación entrenada con pseudo-máscaras |
| ham10000 red de segmentación entrenada con pseudo-máscaras - xnorm | 10015 | +0.0521 | [+0.0485, +0.0556] | gana el red de segmentación entrenada con pseudo-máscaras |

HD95 (mas bajo es mejor, negativo = gana el red de segmentación entrenada con pseudo-máscaras):
| comparacion | n | diferencia | IC 95% | veredicto |
|-------------|---|------------|--------|-----------|
| isic2018 red de segmentación entrenada con pseudo-máscaras - rnorm | 2589 | -7.37 px | [-8.60, -6.12] | gana el red de segmentación entrenada con pseudo-máscaras |
| isic2018 red de segmentación entrenada con pseudo-máscaras - xnorm | 2589 | -6.95 px | [-8.18, -5.73] | gana el red de segmentación entrenada con pseudo-máscaras |
| ham10000 red de segmentación entrenada con pseudo-máscaras - rnorm | 10007 | -0.21 px | [-0.69, +0.28] | INCLUYE 0: empate |
| ham10000 red de segmentación entrenada con pseudo-máscaras - xnorm | 10007 | -0.51 px | [-0.98, -0.02] | gana el red de segmentación entrenada con pseudo-máscaras |

**Lectura honesta**: el red de segmentación entrenada con pseudo-máscaras NO le gana a rnorm en Dice medio sobre ISIC2018 (el IC pareado
incluye el 0). Pero sobre el MISMO conjunto le gana en HD95 por 7.4 px con IC que excluye el 0, y en
p10 (0.402 contra 0.348). En HAM la relacion se invierte: gana en Dice (+0.0197, IC excluye 0) y
empata en HD95. Contra xnorm gana en las cuatro comparaciones.

Conclusion defendible: el metodo aprendido no mejora el solapamiento medio frente a una normalizacion
de color con Otsu en ISIC2018, pero si mejora la calidad del borde y reduce los fallos catastroficos;
en HAM10000 mejora el solapamiento. Afirmar "supera a los pisos" sin matizar seria incorrecto.
