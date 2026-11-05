# Envoltorio del contraste pareado de tono

Escrito el 2026-11-05. Especificacion de los cambios y su justificacion en
`docs/estado/envoltorio-cambios-para-contraste-tono.md`.

`tone_overlay_ssl_runtime.py` es un **hermano** de `overlay_ssl_runtime.py`, no
una edicion. Ese envoltorio produjo el brazo control `ssl_overlay_A_s42_6000_v1`
y su hash de fuente esta sellado en el recibo de esa corrida; editarlo en el
lugar volveria inverificable la procedencia del control. Copia byte a byte del
original en `cluster/runtime-snapshot-20261104/scripts/`.

## El cambio que importa, y por que falla callado sin el

`torch_loop._build` hace esto:

```python
75:  paths = config_mod.collect_images(cfg.image_root)       # el original SI parchea esto
76:  if cfg.method in config_mod.SIMSIAM_METHODS:
77:      if cfg.adaptation_mode == "original_transformed_consistency":
80:          paths = config_mod.collect_paired_images(...)   # ...y aca se sobreescribe
```

Y mas abajo:

```python
280: ... config_mod.collect_images(cfg.image_root))         # el plan de muestreo
```

Dos consecuencias, la segunda peor que la primera:

1. **El pinneo del pool queda sin efecto en el brazo tratado.** El original
   parchea `collect_images` para clavar el pool a las filas aprobadas, pero la
   linea 80 descarta ese resultado. La validacion de pool de las lineas 51-64
   seguiria corriendo y seguiria pasando: decorativa.
2. **El plan de muestreo se desalinea del dataset.** El plan se liga a la salida
   de `collect_images` (parcheada); el dataset itera la salida de
   `collect_paired_images` (sin parchear, en orden de disco). Si los ordenes
   difieren, el plan indexa cuadros distintos de los que cree, cada id
   registrado pertenece a otra imagen y la contabilidad de cobertura miente **sin
   levantar nada**.

Por eso `paired_images` reconstruye los pares **en el orden de las filas** y
**afirma la paridad de orden** en vez de confiar en ella:

```python
if [original for original, _ in pairs] != ordered:
    raise ValueError("Paired pool order diverged from the sampling-plan order")
```

## Los otros cuatro cambios

| # | que | por que |
|---|---|---|
| 2 | `validate_tone_config` extiende la lista blanca a los dos brazos | el original no menciona `adaptation_mode` ni `tone_consistency_weight`, asi que una configuracion tratada pasaria sin examen. Control: peso cero y sin insumos transformados. Tratado: peso > 0 y los cuatro campos del manifiesto declarados |
| 3 | `validate_paired_factor` corre antes de entrenar | el original sella **una** configuracion y nunca compara dos. Se compara contra los `resolved-params.json` **efectivos** del control, con su sha verificado: el JSON de lanzamiento describiria la intencion, no lo que corrio |
| 4 | `check_arm_parity` | el gate compara diccionarios. No cuenta archivos ni abre una imagen, asi que no veria ni que los brazos cubren distinta cantidad de cuadros —el control corrio 34.754 contra 33.620 del primer espejo— ni que una imagen y su par difieren de tamano. Los dos fueron modos de falla reales el 2026-11-04 |
| 5 | el arbol congelado no se toca | sus `.py` se hashean al arrancar y se re-hashean al terminar, igual que en el original, asi que agregar un modulo ahi invalidaria el sello de **toda** corrida anterior |

La paridad de tamano se mide sobre una muestra determinista de 256 pares con la
semilla de la corrida, no sobre los 34.754: abrir todos antes de entrenar
costaria mas de lo que compra, y un desajuste es una propiedad de la
materializacion, no de cuadros individuales.

## Despliegue

El envoltorio importa `paired_factor_gate`, que vive en `src/dermssl/pretrain/`
del repo y **no** en el arbol congelado. Hay que copiarlo al lado del
envoltorio en el cluster, no agregarlo al arbol sellado:

```sh
scp src/dermssl/pretrain/paired_factor_gate.py \
    unabia:/home/jovyan/work/clf1_clinical_reference_20261024/scripts/
scp cluster/tone-contrast-runtime/tone_overlay_ssl_runtime.py \
    unabia:/home/jovyan/work/clf1_clinical_reference_20261024/scripts/
```

Tiene que quedar en `scripts/` del experimento porque importa de
`run_overlay_ssl` y resuelve `ROOT` como el padre de su propio directorio.

## Campos nuevos de la configuracion de lanzamiento

Ademas de los que exige `validate_wrapper_config` (`base_config`, `manifest`,
`manifest_sha256`, `overlay_receipt`, `overlay_receipt_sha256`,
`lr_multiplier`, `training_gpu_hours_limit`), el brazo tratado necesita:

- `control_resolved_params`: ruta al `resolved-params.json` del control
- `control_resolved_params_sha256`: su sha256, verificado antes de usarlo

## Una trampa heredada que NO se arreglo

`launch-seal.json` guarda `wrapper_sha256` y **no es el hash del envoltorio**:
es `config_seal(config)`, un sello de la configuracion. El nombre se conserva
por compatibilidad con los sellos del control, pero se agrego el campo
`wrapper_sha256_is` que lo dice explicito, para que nadie vuelva a reportar una
procedencia rota que no esta rota.
