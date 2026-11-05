# Que hay que cambiar en el envoltorio para correr el contraste de tono

Estado: vigente. Escrito el 2026-11-04 tras leer
`overlay_ssl_runtime.py` completo (198 lineas) y el codigo congelado que maneja.

Copia versionada del envoltorio: `cluster/runtime-snapshot-20261104/`.

**El cambio lo hacemos nosotros.** Es nuestra linea, el envoltorio es de
Nicolas y no se toca el original: se parte de una copia con nombre propio.

---

## 1. El hallazgo que simplifica todo

**`image_root` no decide que imagenes se cargan.** La linea 174 parchea el
colector:

```python
with patch.object(config_mod, "collect_images",
                  return_value=[Path(r["image_path"]) for r in rows]), ...
```

Las imagenes salen del manifiesto, no del arbol de directorios. Eso significa
que apuntar un brazo al espejo PNG **no necesita tocar `image_root`**: necesita
que las filas del manifiesto traigan las rutas del espejo.

**Pero el pool esta clavado por ruta** (lineas 51-64): el conjunto
`(role, image_id, image_path, identity_component_id)` debe coincidir exactamente
con los manifiestos de rol aprobados, cuyo sha256 esta en el recibo del overlay.
Cambiar las rutas **obliga** a regenerar `source_fit.jsonl`, `pad_adapt.jsonl` y
su recibo. Esa es la unica razon por la que hay que regenerarlos; no es un
capricho del gate.

## 2. Los cinco cambios, en orden

### 2.1 Parchear tambien el colector pareado

El envoltorio parchea `collect_images`. Con
`adaptation_mode = original_transformed_consistency`, el bucle congelado **no
usa esa funcion**: valida el manifiesto transformado
(`torch_loop.py:78`, `config_mod.validate_transformed_manifest`) y arma pares
por otra via (`config_mod.collect_paired_images`).

**Si no se parchea, el hook de la linea 174 queda sin efecto en el brazo tratado
y toda la validacion de pool de las lineas 51-64 se vuelve decorativa.** El
brazo tratado cargaria lo que encuentre en disco y el control lo que diga el
manifiesto, y nadie se daria cuenta.

Hay que agregar el parche equivalente para la ruta pareada, devolviendo los
pares derivados de las mismas `rows` aprobadas.

**Este es el cambio critico.** Los otros cuatro son necesarios pero no
silenciosos: si faltan, algo revienta. Este falla callado.

### 2.2 Extender la lista blanca

Las lineas 90-97 no mencionan `adaptation_mode` ni `tone_consistency_weight`, asi
que una configuracion tratada **pasaria la lista blanca sin ser contemplada**.
Hay que exigir explicitamente:

- el brazo control: `adaptation_mode == "two_view_ssl"` y
  `tone_consistency_weight == 0.0`;
- el brazo tratado: `adaptation_mode == "original_transformed_consistency"`,
  `tone_consistency_weight` igual al valor calibrado, y
  `transformed_image_root` / `transformed_manifest_path` /
  `transformed_data_revision` no vacios;
- todo lo demas identico a la lista actual, sin excepcion.

### 2.3 Correr `paired_factor_gate` en el lanzamiento

El envoltorio sella una configuracion pero **nunca compara dos brazos**. Ahi es
donde entra `src/dermssl/pretrain/paired_factor_gate.py`, que el arbol congelado
no tiene: hay que enviarlo y llamar a `validate_paired_factor` con los
`resolved-params.json` **efectivos** de los dos brazos, no con los JSON de
lanzamiento.

Importante: el gate compara diccionarios. **No cuenta archivos ni compara
resolucion de imagen.** Los dos modos de falla que encontramos hoy —conteos
distintos y recorte sobre resoluciones distintas— pasan por debajo. Hay que
agregar esas dos comprobaciones a mano, en el envoltorio.

### 2.4 Regenerar los manifiestos de rol con las rutas del espejo

Consecuencia directa de la seccion 1. Incluye recalcular el recibo del overlay
y su `output_sha256`, porque la linea 54 lo verifica.

### 2.5 No tocar el arbol congelado

Las lineas 112-113 hashean todos los `.py` de
`frozen/code/src/dermssl/pretrain/` y la linea 185 aborta si alguno cambio
durante el entrenamiento. **Agregar un modulo ahi cambia el sello de toda
corrida previa.** Los cambios van en el envoltorio o en un modulo nuevo fuera de
ese arbol, importado por el envoltorio.

## 3. Lo que NO hay que extender

El codigo congelado **ya soporta el modo pareado entero**: `config.py`,
`levers.py`, `torch_adaptation.py`, `torch_loop.py` y `torch_simsiam.py` llevan
`tone_consistency_weight` y `original_transformed_consistency`. No hay que
implementar el mecanismo, solo dejarlo pasar.

## 4. La restriccion que no se arregla con un parche

```python
# frozen/.../config.py:42
SIMSIAM_METHODS = ("simsiam", "simsiam_imagenet", "simsiam_dinov2")
# linea 175
and cfg.method not in SIMSIAM_METHODS
    raise ValueError("la adaptación original/transformada solo es compatible con métodos SimSiam")
```

**El modo pareado solo corre sobre SimSiam**, y SimSiam es el peor de los tres
encoders medidos (PAD 0.5973 contra 0.6438 de MoCo-v3-style). Decidido el
2026-11-04 con Maria Paula: **se corre sobre SimSiam primero** para probar el
mecanismo, y **el contraste sobre MoCo-v3 lo hacemos nosotros** como linea
propia, con su comparacion final entre los dos.

Extenderlo a MoCo-v3 es trabajo real, no un parche: la adaptacion pareada esta
escrita contra la estructura de SimSiam (predictor + stop-gradient, dos vistas
simetricas). MoCo-v3 tiene cola de momentum y encoder/encoder-momentum, asi que
el termino de consistencia de tono tiene que decidir **contra cual rama** se
compara la vista transformada. Esa decision es de diseno y hay que justificarla,
no adivinarla.

## 5. Orden de ejecucion

1. Materializar las 1.134 de `pad_adapt` en el espejo, por el camino de copia
   identidad con registro de mascara inelegible. Sin segmentador fuera de
   dominio.
2. Calcular la revision de datos sobre **34.754**.
3. Regenerar manifiestos de rol y recibo con rutas del espejo (2.4).
4. Aplicar 2.1, 2.2, 2.3 y 2.5 sobre una copia del envoltorio.
5. Re-correr el control sobre el espejo, por el recorte a distinta resolucion
   (ver `contraste-tono-pareado-estado-20261105.md` seccion 3.5).
6. Recalibrar `tone_consistency_weight` con las palancas reales.
7. Lanzar `tono-026` sobre SimSiam.
8. Extender a MoCo-v3 y repetir 5-7.
9. Comparacion final entre los dos contrastes.
