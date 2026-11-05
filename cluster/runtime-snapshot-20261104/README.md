# Instantanea del arnes de corridas de UNABIA, 2026-11-04

Este directorio existe porque **nada de lo que contiene estaba versionado**, y
es el codigo que realmente lanza los experimentos del proyecto. Si el pod de
UNABIA se reinicia o alguien limpia `~/work`, sin esta copia no hay forma de
reproducir ninguna corrida: el repositorio solo tiene la libreria, no el arnes.

Es el mismo precedente que los doce modulos de `src/dermssl/` rescatados en
`f14a63a` (ver `CLAUDE.md` seccion 8), con el agravante de que estos nunca
estuvieron ni en el working tree local.

## Procedencia

| que | de donde | sha256 del paquete |
|---|---|---|
| `scripts/` (113 archivos `.py`), `config/paths.json` | `~/work/clf1_clinical_reference_20261024/` | `aa30831582de47fc1e908ce9fd956500d1c654ea31d263a33454c2f7af7a7795` |
| `frozen-pretrain/` (4 modulos) | `~/work/clf1_ssl_phone_20261024/code/src/dermssl/pretrain/` | `375945c108612b870cd94bcf137d08c771dfd03edb03ea6356f1584e49a4654a` |

Los dos sha256 se verificaron **despues** de la transferencia y coinciden con el
origen. Copia, no fuente: si hay que cambiar algo, se cambia en el cluster y se
vuelve a copiar, porque los envoltorios sellan su propio hash en cada corrida.

## Los cuatro modulos que el repo nunca tuvo

`coverage_sampling.py`, `sampling_contract.py`, `image_input.py` y
`torch_monitor.py` estan en el arbol congelado y **no** en `src/dermssl/pretrain/`.
Ahi vive `clinical_batch_quota`, y por eso buscarlo en `src/` da cero aunque el
control haya corrido con cuota 4. La divergencia va en las dos direcciones: el
repo tiene `paired_factor_gate.py`, `tone_augment_pairs.py` y
`torch_view_audit.py`, que el congelado no.

## Como esta organizado `scripts/`

Son 113 archivos sin jerarquia. Las familias que importan:

| familia | archivos | que hace |
|---|---|---|
| **envoltorios SSL** | `overlay_ssl_runtime.py`, `phone_mix_`, `longdose_`, `one_factor_`, `projector_`, `icone_`, `mulan_`, `byol_simsiam_`, `oob_simsiam_`, `predictor_distill_`, `constant_predictor_`, `explora_`, `crop_scale_` | lanzan preentrenamiento parcheando el bucle congelado en tiempo de ejecucion, sin editar su fuente |
| **objetivos y hooks** | `*_losses.py`, `*_hooks.py`, `*_objectives.py`, `ssl_objectives.py`, `ssl_qv_lora.py` | las tareas pretexto que Nicolas probo |
| **construccion de pools** | `build_content_overlay.py`, `build_union.py`, `build_v3.py`, `build_registry*.py`, `content_candidate_edges.py` | derivan los manifiestos de rol y el overlay de exclusion por contenido |
| **auditoria** | `audit_cross_role_content.py`, `audit_near_duplicates.py`, `audit_skindisnet_content.py` | contaminacion entre roles y casi-duplicados |
| **lectura y evaluacion** | `evaluate_*.py`, `probe_*.py`, `*_readers.py`, `frozen_reader_heads.py`, `grouped_readout_bootstrap.py`, `multi_reference_bootstrap.py` | las lecturas congeladas y su bootstrap agrupado |
| **cola del 29 de octubre** | `build_followup_queue_20261029.py`, `ssl_followup_runner_20261029.py`, `followup_*_20261029.py`, `per_class_report_20261029.py` | los siete brazos que pidio Karen |
| **banco del 30 de octubre** | `bank_*_20261030.py`, `full_metrics_20261030.py` | fine-tuning de baselines |

## Como funciona `overlay_ssl_runtime.py`, que es el que nos toca

198 lineas. No edita el bucle congelado: lo **parchea con `unittest.mock.patch`**
durante la corrida (lineas 174-181). Lo que hay que entender antes de tocarlo:

1. **`image_root` no decide que imagenes se cargan.** La linea 174 parchea
   `config_mod.collect_images` para devolver exactamente las rutas del
   manifiesto. El `image_root` de la configuracion queda como etiqueta.
2. **El pool esta clavado por ruta.** Las lineas 51-64 exigen que el conjunto
   `(role, image_id, image_path, identity_component_id)` coincida exactamente
   con los manifiestos de rol aprobados, y rechazan rutas o identidades
   duplicadas. Cambiar de donde se leen las imagenes **obliga** a regenerar esos
   manifiestos.
3. **Lista blanca dura** (lineas 90-97): rechaza cualquier configuracion que no
   sea `simsiam_dinov2` + `dinov2_vitb14`, `freeze_blocks 8`,
   `anchor_distill 1.0`, `anchor_l2sp 0.0001`, `batch_size_per_gpu 16`,
   `clinical_batch_quota` en (1, 4), `max_steps` en (40, 6000), semilla en
   (42, 43, 44), `precision bf16`, `head_warmup_steps 32`, `num_workers 0`.
4. **`clinical_batch_quota` es el adaptador de dominio.** La linea 101 construye
   `domains = [int(row["role"] == "pad_adapt") for row in rows]` y lo pasa a
   `CoverageState`. Con cuota 4 sobre lote 16, el rol clinico ocupa el 25 % de
   cada paso.
5. **`wrapper_sha256` NO es el hash del envoltorio.** La linea 98 hace
   `seal = config_seal(config)`: es un sello de la **configuracion**. Que el
   archivo hashee distinto es correcto, no una procedencia rota.
6. **Sella y reverifica el codigo congelado** (lineas 112-113 y 185): hashea
   todos los `.py` de `frozen/code/src/dermssl/pretrain/` al arrancar y aborta
   si alguno cambio al terminar. Agregar un modulo ahi cambia el sello.
7. **Cobertura real, no declarada.** `AccessedDataset` (lineas 125-135) registra
   cada indice que el cargador toca de verdad, y `commit_pending` exige que los
   accesos cuadren con los pasos del optimizador. No se puede declarar COMPLETE
   con entrenamiento parcial (linea 183).

Vale decirlo: el arnes esta bien hecho. Verifica procedencia de entradas, sella
la configuracion, mide cobertura real en vez de confiar en el muestreador y se
niega a declarar exito parcial. El problema nunca fue su calidad, fue que no
estaba versionado.

## Que se le puede dar de uso ya

- **`audit_near_duplicates.py` y `audit_cross_role_content.py`** atacan el
  problema de independencia clinica del pool, que sigue declarado como "no al
  100 %". Son lectura, corren sin GPU.
- **`grouped_readout_bootstrap.py`** agrupa por identidad al hacer bootstrap, que
  es exactamente lo que necesita cualquier intervalo de confianza del contraste
  de tono dado que el 66 % del pool esta en componentes de mas de una imagen.
- **`per_class_report_20261029.py`** es la tabla de precision por clase que pidio
  Karen.

## Que NO hacer

- No ejecutar nada de aqui contra el cluster: son copias y sus rutas absolutas
  apuntan a `~/work`. Para correr, se usa el original en el cluster.
- No "arreglar" el estilo ni reformatear. Cualquier diferencia contra el original
  rompe la comparacion de hashes que justifica esta copia.
