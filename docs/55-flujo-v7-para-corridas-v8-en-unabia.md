# Flujo de corridas V7 de Nicolás: referencia operativa para V8 en UNABIA

Fuente remota auditada en solo lectura: `origin/experiment/modular-audit-v8` en `9d0147554b829e8f5b03a1a8a21f2d167f174a07`. Receta contrastada: `docs/28`, `scripts/v7_final_launch.py`, `src/dermssl/pretrain/v7_finals.py`, `scripts/c1_final_eval.py` y los resultados versionados de V7. Esta revisión explica qué prácticas sirven para preparar el cierre V8; no acredita que los datos o entornos V7 estén disponibles hoy en UNABIA.

## Ejecución y evaluación documentadas

La invocación documentada para s44 fue:

```bash
GIT_COMMIT=$(git rev-parse HEAD) CUDA_VISIBLE_DEVICES=0 python scripts/v7_final_launch.py --arm A --seed 44 --salidas ~/work/proyectonp/salidas --num-workers 4
```

La evaluación documentada fue:

```bash
python scripts/c1_final_eval.py --split val --single-evaluation --ham-root ~/work/datos/ham10000 --holdout $S/selection/selection_holdout.json --plain dinov2_vitb14:$V/final_A/simsiam_dinov2_backbone_final.pt --proposed dinov2_vitb14:$V/final_B/simsiam_dinov2_backbone_final.pt --exclude-ids-file $S/selection/pretrain_exclude_ids_finals.txt --out resultados/clasificacion/v7/final_eval.json
```

**Documentado:** la receta V7 ejecuta finales identificados por brazo/semilla, registra el commit y evalúa una sola vez con holdout y exclusiones explícitas. **Derivado:** s42/s43 usaron `--num-workers 0` según la lógica del script/receta; no se atribuye ese comando literal a un registro histórico. Los comandos de ejemplo son evidencia de cómo se organizó V7, no instrucciones para correr esa receta sobre V8.

El docstring de evaluación deja `$S` y `$V` sin asignar. **Reconstrucción derivada de las rutas en los scripts y `final_eval.json`:** para ese ejemplo, `$S=~/work/proyectonp/salidas/c1` y `$V=~/work/proyectonp/salidas/v7`. No se presenta esa expansión como una captura del entorno histórico.

## Entradas y procedencia V7

- El launcher recibe `--salidas ~/work/proyectonp/salidas`; entradas A/B: `~/work/proyectonp/salidas/tone/inputs/originals`.
- Exclusiones del preentrenamiento: `~/work/proyectonp/salidas/c1/selection/pretrain_exclude_ids_finals.txt`.
- Brazo B también usa `~/work/proyectonp/salidas/tone/shift2` y su `tone_transform_manifest.json`.
- Hash del split: `a83f927ab798b6f7a0614c3923fda77d082f887d10fdbe73416e34e9574ba71a`.
- Revisión del código de tono: `a754961bfd97bee21b4e48edd35b71314ed82b50cc6355480252b4d075e60e51`.

Estas rutas e identidades son V7. Se deben reconstruir y verificar las rutas, manifests, exclusiones, particiones paciente/lesión y hashes correspondientes a V8 en UNABIA; no asumir que los artefactos V7 existen allí ni reutilizarlos.

## Observaciones del runtime UNABIA en esta sesión

`nvidia-smi` mostró dos NVIDIA RTX PRO 4000 Blackwell, cada una con 24.467 MiB
totales y 23.985 MiB libres en el instante de observación. Desde
`/home/jovyan/work/proyectonp/.venv/bin/python`, PyTorch `2.11.0+cu128` reportó
CUDA disponible y dos dispositivos con esos nombres. Esto comprueba visibilidad de
CUDA en el runtime observado, no una reserva, disponibilidad futura ni presupuesto
aprobado para las corridas.

`/home/jovyan/work` está en NFS compartido `192.168.247.100:/compartido_serv_unab`;
`df` mostró 40 T totales, cerca de 373 G usados y cerca de 40 T libres redondeados.
La raíz overlay `/` mostró 98 G totales, 75 G usados y 19 G libres. `quota -s`
falló porque el comando no está instalado. Por ello, el espacio agregado del NFS no
es una cuota personal; la cuota individual y el presupuesto de almacenamiento siguen
sin verificar. Mantener abiertos los gates de recursos y storage.

## Seguimiento de W&B documentado

V7 registró en `dermssl-segmentacion/v7-final` los runs `41f0e8d164a3`, `8f1684ceea84`, `4672e3ff0604` y `b8e85ebc8ece`. La receta pudo absorber errores de tracking cuando operaba en modo no estricto; por eso, la práctica reutilizable es comprobar en la interfaz/servidor que el run y sus artefactos realmente aparezcan, y guardar enlaces/IDs como parte del cierre.

## Artefactos para el cierre de cada corrida

V7 produjo checkpoints intermedios por cuarto de avance (`<method>_ckpt_c025.pt`, `c050`, `c075`, `c100`), el checkpoint final `<method>_backbone_final.pt`, `run.json` y `resume.pt` para recuperación. Para V8, **derivado como disciplina de cierre**, cada método debe publicar los equivalentes que correspondan a su diseño, junto con:

- configuración serializada efectiva, manifiesto(s) y hashes, identidad/revisión del código y entorno/dependencias;
- logs completos, estado de salida y checkpoints requeridos por el protocolo del método; conservar `resume.pt` solo si se usa para recuperación;
- procedencia y SHA-256 de pesos/fuentes, IDs y enlaces verificables del run y artefactos W&B;
- métricas finales y salidas de evaluación reproducibles, con el split reservado, exclusiones y manifiestos exactos.

No copiar nombres de checkpoint, cadencias, evaluación ni presupuestos V7 mecánicamente: cada equivalente se define en la configuración y protocolo V8 del método. Antes de una corrida siguen cerrados los gates de V8 de manifiestos/IDs paciente-lesión y evaluación independiente, cuotas y presupuestos operativos completos, planes seriales/configs fijados y revisión de SLED/CUDA/launcher. Mantener `execution_enabled=false`, `launch_ready=false` y `training_started=false` hasta contar con evidencia para esos gates.

## Frontera entre V7 y V8

V7 usó SimSiam con DINOv2 ViT-B/14 para clasificación y recetas propias de objetivos, transformaciones, batch, pasos, GPU-hours y evaluación. Ninguno de esos elementos, checkpoints o resultados constituye una configuración para la segmentación V8. Lo reutilizable aquí es la disciplina de fijar commit, semilla/identidad de corrida, entradas y exclusiones, conservar configuración y logs, enlazar los artefactos W&B y producir evaluación reproducible. Este documento no inicia, autoriza ni afirma disponibilidad actual de entrenamiento en UNABIA.
