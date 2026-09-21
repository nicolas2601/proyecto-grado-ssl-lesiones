# Runbook del clúster UNABIA

Procedimiento completo para correr el Camino B en el pod de JupyterHub del
clúster UNABIA, desde clonar el repositorio hasta leer los resultados. Los
comandos se pegan tal cual en una terminal del pod.

## Entorno de destino

- Usuario: `nmoreno534`.
- Dos GPU NVIDIA RTX PRO 4000 Blackwell de 24 GB, CUDA 13.0 con driver 580.
- El entorno base del pod ya trae `torch 2.11` compilado contra cu128 sobre
  Python 3.13. **Nunca** se instala `torch` con pip dentro del pod: romper esa
  instalación deja sin GPU a todos los usuarios del clúster.
- `/dev/shm` es de 67 MB. El pipeline no usa procesos de carga de datos, así que
  esa limitación ya está respetada y no hay nada que configurar.
- El home en NFS (`~/work`) es persistente entre sesiones del pod. Todo lo que
  no esté ahí desaparece cuando el pod se reinicia.
- Carpeta de trabajo: `~/work/proyectonp`. Los datos de entrenamiento ya están
  cargados ahí.

## 1. Llevar el repositorio al pod con un bundle de git

El pod no tiene credenciales de GitHub, así que el repositorio viaja como un
bundle de git que se sube por la interfaz de JupyterLab. El bundle se genera en
el portátil con `git bundle create repo.bundle feature/fitzpatrick-domain-adaptation-informe`
y se arrastra a `~/work/proyectonp` desde el explorador de archivos de Jupyter.

Con el archivo ya en el pod:

```bash
mkdir -p ~/work/proyectonp
cd ~/work/proyectonp
git clone -b feature/fitzpatrick-domain-adaptation-informe repo.bundle repo
cd repo
```

Para traer cambios posteriores se sube un bundle nuevo con un número distinto
(`repo-2.bundle`, `repo-3.bundle`) y se trae la rama desde ahí:

```bash
cd ~/work/proyectonp/repo
git pull ../repo-2.bundle feature/fitzpatrick-domain-adaptation-informe
```

Numerar los bundles evita sobrescribir uno que todavía se esté subiendo y deja
un rastro de qué versión se corrió.

## 2. Entorno virtual sobre los paquetes del sistema

El entorno virtual hereda `torch` del entorno base con
`--system-site-packages`, y encima se instalan solo las dependencias livianas.

```bash
cd ~/work/proyectonp
python -m venv --system-site-packages ~/work/proyectonp/.venv
source ~/work/proyectonp/.venv/bin/activate
pip install scipy pillow wandb pytest
```

Verificar que `torch` llegó desde el entorno base y que ve las dos GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

La salida esperada empieza con `2.11`, sigue con `True` y termina con `2`. Si
dice `False`, no se sigue adelante: el entorno virtual se creó sin
`--system-site-packages` o el pod no tiene GPU asignada.

## 3. Sesión de Weights & Biases

La clave se pega en la terminal y no se guarda en el repositorio ni en ningún
archivo versionado.

```bash
export WANDB_API_KEY=<pegar clave>
wandb login
```

La entidad es `nmoreno534-universidad-autonoma-de-bucaramanga` y el proyecto es
`dermssl-segmentacion`. El driver nunca recibe la clave por línea de comandos:
solo propaga `WANDB_PROJECT` y `WANDB_RUN_GROUP` a los procesos hijos, y la
clave viaja por el entorno heredado.

## 4. Verificación del entorno

```bash
cd ~/work/proyectonp/repo
python plan-segmentacion/comun/nucleo/check_env.py
```

Este script imprime versión de Python, versión de torch, GPU visibles y memoria
disponible. Es la misma etapa `env` que el driver corre al principio de cada
pipeline.

## 5. Datos de entrenamiento y máscaras de referencia

### Raíces de entrenamiento

Estas son las rutas medidas en la sesión del clúster. Se pasan al driver con una
bandera `--root` por carpeta, con la forma `nombre=ruta`:

```bash
--root ham10000_p1=$HOME/work/datos/ham10000/HAM10000_images_part_1 \
--root ham10000_p2=$HOME/work/datos/ham10000/HAM10000_images_part_2 \
--root isic2019=$HOME/work/datos/isic2019 \
--root isic2020=$HOME/work/datos/isic2020
```

**Nunca se pasa `$HOME/work/datos/ham10000` como raíz.** Esa carpeta contiene
además una copia en minúsculas de las mismas imágenes, así que apuntarle al
padre duplica todo HAM10000 en la caché. Por eso se listan las dos partes por
separado.

Si la distribución de los datos cambia, el listado se vuelve a levantar con:

```bash
find ~/work/proyectonp -maxdepth 3 -type d
find ~/work/datos -maxdepth 2 -type d
```

El nombre de cada raíz solo se usa como etiqueta en `manifest.csv`, así que
conviene mantener estos cuatro nombres para que los reportes sean legibles.

### Máscaras de referencia

Las máscaras se usan **solo para evaluar**, nunca para entrenar ni para elegir
un checkpoint o un umbral. Se descargan aparte:

- **ISIC 2018 Task 1**, verdad de terreno del conjunto de entrenamiento: 2594
  máscaras. Se obtienen del sitio del ISIC Challenge, o de Kaggle con el
  identificador `tschandl/isic2018-challenge-task1-data-segmentation`.
- **HAM10000**: archivo `HAM10000_segmentations_lesion_tschandl.zip`, de 10,8 MB,
  publicado en Harvard Dataverse con DOI `10.7910/DVN/DBW86T`. Los archivos se
  llaman `ISIC_xxxxxxx_segmentation.png` y se emparejan por identificador con
  las imágenes de HAM10000 que ya están en el pod.

Descarga desde Kaggle, si se usa esa vía:

```bash
pip install kaggle
mkdir -p ~/work/proyectonp/masks_eval
cd ~/work/proyectonp/masks_eval
kaggle datasets download -d tschandl/isic2018-challenge-task1-data-segmentation
unzip -q isic2018-challenge-task1-data-segmentation.zip -d isic2018
```

El script de evaluación espera el sufijo `_segmentation` y la extensión `.png`,
que es exactamente el nombre que traen los dos conjuntos. Si una descarga usa
otro formato, se ajusta con `--mask-suffix` y `--mask-extension`.

### Carpeta única de imágenes para evaluar HAM10000

Las máscaras de HAM10000 están en una sola carpeta, pero las imágenes viven en
dos. El evaluador recibe una carpeta de imágenes, así que se arma una con
enlaces simbólicos a las dos partes:

```bash
mkdir -p ~/work/proyectonp/eval_ham10000_images
ln -sf $HOME/work/datos/ham10000/HAM10000_images_part_1/*.jpg $HOME/work/datos/ham10000/HAM10000_images_part_2/*.jpg ~/work/proyectonp/eval_ham10000_images/
```

Son enlaces, no copias: no ocupan espacio en el NFS y se rehacen con el mismo
comando si algo cambia.

### Conjuntos de evaluación

Con lo anterior, las dos banderas quedan así:

```bash
--eval-set isic2018=$HOME/work/proyectonp/datos_isic2018/ISIC2018_Task1-2_Training_Input:$HOME/work/proyectonp/masks_eval/isic2018/ground_truth \
--eval-set ham10000=$HOME/work/proyectonp/eval_ham10000_images:$HOME/work/proyectonp/masks_eval/ham10000/HAM10000_segmentations_lesion_tschandl
```

## 6. Smoke en GPU

Antes de gastar horas de GPU se corre el pipeline completo con 64 imágenes
sintéticas. Tarda menos de dos minutos y no toca los datos reales.

```bash
cd ~/work/proyectonp/repo
python plan-segmentacion/camino-b-cnn/entrenamiento/run_pipeline.py --smoke
```

Si alguna etapa falla, el driver corta ahí e imprime el código de salida. El
smoke ejercita también las banderas del campo del dermatoscopio
(`--field-black-threshold`, `--field-margin-fraction`, `--field` y
`--max-area-fraction`), así que sirve para detectar un preset mal escrito.

## 7. Piloto

El piloto mide velocidad real: una ronda, cinco épocas contrastivas y cinco
épocas de estudiante. Corre en las dos GPU y se lanza en segundo plano para que
sobreviva a una desconexión del navegador.

```bash
cd ~/work/proyectonp/repo
setsid nohup python plan-segmentacion/camino-b-cnn/entrenamiento/run_pipeline.py \
  --nproc-per-node 2 \
  --config plan-segmentacion/comun/configs/cluster-piloto.json \
  --workdir $HOME/work/proyectonp/salidas/piloto \
  --root ham10000_p1=$HOME/work/datos/ham10000/HAM10000_images_part_1 \
  --root ham10000_p2=$HOME/work/datos/ham10000/HAM10000_images_part_2 \
  --root isic2019=$HOME/work/datos/isic2019 \
  --root isic2020=$HOME/work/datos/isic2020 \
  --wandb-project dermssl-segmentacion \
  --eval-set isic2018=$HOME/work/proyectonp/datos_isic2018/ISIC2018_Task1-2_Training_Input:$HOME/work/proyectonp/masks_eval/isic2018/ground_truth \
  --eval-set ham10000=$HOME/work/proyectonp/eval_ham10000_images:$HOME/work/proyectonp/masks_eval/ham10000/HAM10000_segmentations_lesion_tschandl \
  < /dev/null > pilot.log 2>&1 & disown
```

`setsid` es obligatorio, no cosmético. La terminal de JupyterHub muere sin
aviso (culling por inactividad o corte del WebSocket) y se lleva consigo a todo
el grupo de procesos de la sesión; un `nohup ... &` a secas deja al pipeline en
ese grupo y lo mata a mitad de camino. Con `setsid` el proceso obtiene su propia
sesión, sin terminal controlante. Confirmar antes de darlo por lanzado:

```bash
ps -o pid,pgid,sid,tty,etime,cmd -C python | rg run_pipeline
```

La columna `TTY` debe decir `?` y `SID` debe coincidir con el `PID` del driver.

Para seguir el avance:

```bash
tail -f ~/work/proyectonp/repo/pilot.log
```

Antes de lanzar conviene mirar `--dry-run` en lugar de `setsid nohup`: imprime todos
los comandos sin ejecutar nada, y es la forma más barata de confirmar que las
rutas quedaron bien.

Cada `--eval-set` toma la forma `NOMBRE=IMAGENES:MASCARAS` y se puede repetir.
Genera una etapa de evaluación independiente por conjunto. Las banderas
antiguas `--eval-image-dir`, `--eval-mask-dir` y `--eval-name` siguen
funcionando y equivalen a un único conjunto.

**Atención con la etapa `fst`.** Corre entre la última ronda del estudiante y
las evaluaciones, así que si falla ahí, el entrenamiento ya está hecho y no hay
que repetirlo: se relanza el mismo comando con `--stages fst evaluate`.

## 8. Leer los resultados

Dentro de la carpeta de trabajo (`~/work/proyectonp/salidas/piloto`):

- `evaluacion_<nombre>/summary.json`: métricas agregadas del conjunto, con
  percentiles, cobertura y, si se pasó un manifiesto de FST, el desglose por
  bucket de tono de piel y la brecha entre el mejor y el peor.
- `evaluacion_<nombre>/per_image.csv`: una fila por imagen, con Dice, IoU y la
  cobertura predicha y de referencia. Sirve para ordenar por Dice y encontrar
  los casos catastróficos.
- `evaluacion_<nombre>/overlays/`: rejillas PNG con imagen, referencia,
  predicción y superposición, para los peores, los mejores y una muestra al
  azar. Es la forma más rápida de ver si la máscara está agarrando el campo del
  dermatoscopio en vez de la lesión.
- `fst/fst_pseudo_manifest.json` y `fst/fst_pseudo_manifest.csv`: manifiesto de
  ITA y tipo Fitzpatrick calculado sobre las pseudo-máscaras de la última ronda,
  restringido al campo válido. El CSV (`image_id,fst_bucket`) es el que el
  driver le pasa solo a las etapas de evaluación, y se puede reemplazar por uno
  propio con `--fst-manifest`.
- `timing.json` y los `metrics.jsonl` de cada etapa: segundos por época e
  imágenes por segundo, que es lo que define el presupuesto de la corrida final.

### La estratificación por tono solo aplica a HAM10000

El manifiesto de FST se construye a partir de las pseudo-máscaras del pool de
entrenamiento, así que solo cubre los identificadores que están en
`cache/manifest.csv`. HAM10000 sí entra al entrenamiento, de modo que sus
imágenes de evaluación tienen bucket y el desglose por tono es real. Las
imágenes de ISIC 2018 no entran al pool, así que ninguna aparece en el
manifiesto y todas caen en el bucket `unavailable`.

En consecuencia: la brecha por tono se lee en `evaluacion_ham10000`, y en
`evaluacion_isic2018` se leen solo las métricas globales. No es un fallo del
reporte, es el alcance del manifiesto. Para estratificar ISIC 2018 haría falta
un manifiesto de FST calculado aparte sobre ese conjunto y pasarlo con
`--fst-manifest`.

En Weights & Biases, las corridas quedan agrupadas por el nombre de la carpeta
de trabajo (`WANDB_RUN_GROUP`) en:

```
https://wandb.ai/nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion
```

## 9. Corrida final

Con los tiempos del piloto en la mano se decide si el presupuesto de
`cluster-final.json` cabe en las horas de GPU disponibles. Tres rondas, veinte
épocas de estudiante por ronda y treinta épocas contrastivas.

```bash
cd ~/work/proyectonp/repo
setsid nohup python plan-segmentacion/camino-b-cnn/entrenamiento/run_pipeline.py \
  --nproc-per-node 2 \
  --config plan-segmentacion/comun/configs/cluster-final.json \
  --workdir $HOME/work/proyectonp/salidas/final \
  --root ham10000_p1=$HOME/work/datos/ham10000/HAM10000_images_part_1 \
  --root ham10000_p2=$HOME/work/datos/ham10000/HAM10000_images_part_2 \
  --root isic2019=$HOME/work/datos/isic2019 \
  --root isic2020=$HOME/work/datos/isic2020 \
  --wandb-project dermssl-segmentacion \
  --eval-set isic2018=$HOME/work/proyectonp/datos_isic2018/ISIC2018_Task1-2_Training_Input:$HOME/work/proyectonp/masks_eval/isic2018/ground_truth \
  --eval-set ham10000=$HOME/work/proyectonp/eval_ham10000_images:$HOME/work/proyectonp/masks_eval/ham10000/HAM10000_segmentations_lesion_tschandl \
  < /dev/null > final.log 2>&1 & disown
```

Verificar `TTY = ?` con el mismo `ps` de la sección 7 antes de cerrar la
terminal. Los pisos de la sección 9 se lanzan igual, con `setsid nohup`.

Si `cluster-final.json` no cabe, se baja el presupuesto desde la línea de
comandos sin editar el preset, con `--rounds` y `--epochs-per-round`.

### Hausdorff-95 en la corrida final

`cluster-final.json` trae `"hausdorff": true` en la sección `evaluate`, así que
el driver le pasa `--hausdorff` a cada evaluación y el reporte incluye la
distancia de Hausdorff-95 además de Dice e IoU. Es la métrica que castiga los
errores de borde, y por eso vale la pena en el resultado que se reporta.

Es opcional y es la parte más lenta de la evaluación. `cluster-piloto.json` la
deja apagada a propósito, porque el piloto mide velocidad de entrenamiento y no
calidad final. Si en el cierre del final el tiempo aprieta, se quita esa clave
del preset y la evaluación vuelve a salir solo con las métricas rápidas.

### Pisos no supervisados con el mismo evaluador

El resultado se lee contra dos pisos que no aprenden nada: SegRNorm y SegXNorm
(`scripts/run_segnorm_baseline.py`, normalización de color y umbral de Otsu
dentro del campo circular). Corren en CPU, en paralelo al entrenamiento, y se
evalúan con el mismo `evaluate_segmentation.py` que el estudiante, así que los
números son comparables fila a fila. No hay techo supervisado: por decisión del
proyecto nada del trabajo de segmentación entrena con máscaras.

Primero un manifiesto por conjunto de evaluación (columnas `image_id`,
`dataset`, `path`), después las máscaras, después la evaluación externa:

```bash
cd ~/work/proyectonp/repo
source ~/work/proyectonp/.venv/bin/activate
BASE=~/work/proyectonp/baselines
mkdir -p $BASE

for SET in isic2018 ham10000; do
  case $SET in
    isic2018) IMG=$HOME/work/proyectonp/datos_isic2018/ISIC2018_Task1-2_Training_Input
              GT=$HOME/work/proyectonp/masks_eval/isic2018/ground_truth ;;
    ham10000) IMG=$HOME/work/proyectonp/eval_ham10000_images
              GT=$HOME/work/proyectonp/masks_eval/ham10000/HAM10000_segmentations_lesion_tschandl ;;
  esac
  python - "$SET" "$IMG" "$BASE/$SET.csv" <<'EOF'
import csv, sys
from pathlib import Path
name, images, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
with open(out, "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["image_id", "dataset", "path"])
    for path in sorted(images.glob("*.jpg")):
        writer.writerow([path.stem, name, str(path.resolve())])
EOF
  python scripts/run_segnorm_baseline.py --manifest $BASE/$SET.csv \
    --output-dir $BASE/$SET --method both --roi circular
  for METHOD in rnorm xnorm; do
    python plan-segmentacion/comun/evaluacion/evaluate_segmentation.py \
      --pred-mask-dir $BASE/$SET/masks --pred-mask-suffix _$METHOD \
      --image-dir $IMG --mask-dir $GT --eval-resolution 224 \
      --eval-name ${SET}_$METHOD --output-dir $BASE/evaluacion_${SET}_$METHOD \
      --overlay-dir $BASE/evaluacion_${SET}_$METHOD/overlays --batch-size 32
  done
done
```

Las máscaras del piso quedan en `$BASE/<set>/masks/<image_id>_<método>.png`. Para
HAM10000 se puede añadir `--fst-manifest <workdir>/fst/fst_pseudo_manifest.csv`
a la evaluación externa y obtener el mismo desglose por tono que el estudiante.

### Tabla de resultados y evidencia visual

```bash
python plan-segmentacion/comun/evaluacion/build_results_table.py \
  --run estudiante_isic2018=<workdir>/evaluacion_isic2018/summary.json \
  --run estudiante_ham10000=<workdir>/evaluacion_ham10000/summary.json \
  --run rnorm_isic2018=$BASE/evaluacion_isic2018_rnorm/summary.json \
  --run xnorm_isic2018=$BASE/evaluacion_isic2018_xnorm/summary.json \
  --run rnorm_ham10000=$BASE/evaluacion_ham10000_rnorm/summary.json \
  --run xnorm_ham10000=$BASE/evaluacion_ham10000_xnorm/summary.json \
  --output <workdir>/resultados.md

python plan-segmentacion/comun/datos/field_contact_sheet.py \
  --cache-dir <workdir>/cache --output <workdir>/field_contact_sheet.png --per-group 8
```

La hoja de contacto dibuja el borde del campo detectado sobre una muestra por
dataset y por método (`circular` / `full_frame`); es la evidencia visual de que
el detector no se come lesiones ni deja pasar la viñeta. Los artefactos que se
versionan en `plan-segmentacion/camino-b-cnn/resultados/<corrida>/` son
`resultados.md`, los `summary.json`, los `per_image.csv`, los overlays y la hoja
de contacto; nunca la cache, `field.npy` ni los checkpoints.

## 10. Reanudar después de una interrupción

Las etapas de entrenamiento guardan `last.pt` en su carpeta de salida y se
reanudan solas: basta con volver a lanzar el mismo comando. El driver salta lo
que ya está hecho porque cada script detecta su checkpoint. Para forzar un
reinicio desde cero de una etapa se borra su carpeta, o se le pasa
`--no-resume` al script correspondiente.

Para repetir solo una parte del pipeline se usa `--stages`, que acepta varias
etapas y siempre las corre en el orden fijo `env cache contrastive ccam pseudo
student fst evaluate`:

```bash
cd ~/work/proyectonp/repo
python plan-segmentacion/camino-b-cnn/entrenamiento/run_pipeline.py \
  --stages evaluate \
  --config plan-segmentacion/comun/configs/cluster-piloto.json \
  --workdir $HOME/work/proyectonp/salidas/piloto \
  --wandb-project dermssl-segmentacion \
  --fst-manifest $HOME/work/proyectonp/salidas/piloto/fst/fst_pseudo_manifest.csv \
  --eval-set ham10000=$HOME/work/proyectonp/eval_ham10000_images:$HOME/work/proyectonp/masks_eval/ham10000/HAM10000_segmentations_lesion_tschandl
```

Eso reevalúa el checkpoint que ya existe sin volver a entrenar nada. El mismo
truco sirve con `--stages fst` para recalcular el manifiesto de ITA y FST.

El `--fst-manifest` de ese ejemplo no es decorativo. El driver solo asume el
manifiesto generado cuando la etapa `fst` forma parte de la corrida; si se
corre `--stages evaluate` a secas, la evaluación sale sin estratificación por
tono aunque el CSV ya exista en disco. Se apunta a mano, o se corre
`--stages fst evaluate`.

### Reinicio del contenedor del pod

El pod puede reiniciarse entero sin aviso (visto el 2026-09-21 a los diez
minutos de la corrida final, con el servidor sin actividad de Jupyter). Un
reinicio mata todos los procesos, incluso los lanzados con `setsid`, y borra
todo lo que no esté en `~/work`: en particular `~/.netrc`, donde `wandb login`
guarda la clave. El pipeline vuelve a arrancar sin quejarse y solo deja una
línea en el log: `seguimiento W&B deshabilitado`.

Dos medidas:

1. **Mantener el servidor activo mientras dure la corrida.** El culler de
   JupyterHub mide la actividad de kernels y terminales, no los procesos
   sueltos. Se abre un notebook `~/work/proyectonp/keepalive.ipynb` y se
   ejecuta una celda que imprime cada minuto:

   ```python
   import datetime
   import time

   while True:
       print(datetime.datetime.now().isoformat(), "keepalive", flush=True)
       time.sleep(60)
   ```

   La barra de estado debe decir `Kernel status: Busy` durante toda la corrida.
   Si el hub impone además una edad máxima al servidor, eso solo lo cambia el
   administrador del clúster.

2. **Recuperación en tres pasos, no en uno.** Tras un reinicio:

   ```bash
   source ~/work/proyectonp/.venv/bin/activate
   wandb login            # ~/.netrc se perdió con el contenedor
   wandb login --verify
   ```

   Luego se relanza el mismo comando de la sección 9 con `>>` en el log para
   no pisar lo anterior, y se comprueba que el log nuevo **no** contenga la
   línea `seguimiento W&B deshabilitado`. Las etapas de entrenamiento reanudan
   desde `last.pt`; la cache se recalcula (unos cuatro minutos).

## 11. Problemas conocidos

- **Borrar en NFS con archivos abiertos.** Si se hace `rm -rf` sobre una carpeta
  de salidas mientras un proceso todavía la tiene abierta, NFS deja archivos
  fantasma `.nfsXXXX` que no se pueden borrar. Primero se mata el proceso
  (`pkill -f run_pipeline.py`), después se borra.
- **Las GPU son compartidas.** Antes de lanzar con `--nproc-per-node 2` se
  revisa quién está usando el clúster con `nvidia-smi`. Pedir las dos GPU deja
  sin cupo a los demás usuarios, así que se reservan solo para entrenar y se
  liberan al terminar.
- **El error de `mask_token` con DDP no aplica acá.** Quien venga del trabajo de
  clasificación con DINOv2 va a buscar el parámetro sin gradiente que rompía
  `DistributedDataParallel`. Este pipeline no usa ese backbone, así que no hace
  falta `find_unused_parameters` ni ningún parche equivalente.
- **Directorios heredados con uid 65534.** Algunas carpetas viejas del NFS
  quedaron con dueño `nobody` y no se pueden escribir. No se arreglan desde el
  pod: hay que pedirle a Alfredo un `chown`.
- **Nunca usar `/` como destino de salidas.** El `--workdir` siempre apunta
  dentro de `~/work`. Escribir en la raíz del contenedor llena el disco del pod
  y se pierde todo al reiniciar.
- **Nunca instalar torch con pip dentro del pod.** Vale la pena repetirlo: el
  entorno virtual se crea con `--system-site-packages` justamente para heredar
  la instalación buena.
