# Prediccion registrada ANTES del resultado: la interrupcion de `ssl_plain_vitb14`

**Escrito 2026-09-22, con el brazo todavia corriendo y sin conocer el
desenlace.** Ese es el punto del documento: una regla decidida despues de
ver el numero no distingue entre explicar y racionalizar.

## Que paso

A las 19:51 (hora del pod), con el brazo al 95 % del presupuesto, los dos
rangos se trabaron en un colectivo de NCCL. El watchdog espero sus 600 s y
mato el proceso; `--max-retries` relanzo y el log dice
`Reanudado en 11.40 horas-GPU`. Nadie intervino.

## La duda

`simsiam_dinov2_ckpt_c100.pt` (el checkpoint del 100 % del presupuesto)
aparece con la misma marca de tiempo que el arranque del segundo intento.
Entre 11.40 y 12.00 horas-GPU hay 18 minutos de reloj, no uno. Dos lecturas:

1. Lo escribio el proceso viejo al morir, en su teardown.
2. El segundo intento contabilizo como computo los ~10 minutos de cuelgue,
   cruzo el presupuesto de inmediato y lo escribio.

Si es la 2, **este brazo entreno menos de 12 horas-GPU reales** y su fila no
es comparable con las otras tres en las mismas condiciones.

## El discriminante, fijado antes del dato

Leido de `resume.pt`, que es la fuente autoritativa:

```
elapsed_wall_seconds = 20521.615764966933   (= 11.40 h-GPU con 2 GPUs)
step                 = 5596
images_seen          = 716288
```

De ahi, la tasa y la prediccion:

```
tasa             = 5596 / 11.40      = 490,88 pasos por hora-GPU
esperado a 12.00 = 490,88 × 12.00    = 5891 pasos
banda del 3 %                        = 5714 - 6067
imagenes esperadas                   = 753 987
```

| `steps_executed` final | Lectura | Consecuencia |
|---|---|---|
| 5714 - 6067 | **1**: el brazo hizo el trabajo completo | La fila es comparable |
| cerca de 5596 | **2**: el cuelgue se conto como computo | Se declara; la fila lleva nota |
| en el medio | ninguna de las dos | Se declara la interrupcion sin inventar mecanismo |

**Decidido tambien antes del dato: si sale la lectura 2, el brazo NO se
repite.** Doce horas-GPU para recuperar un 5 % de presupuesto, con dos
etapas por delante, no se justifica. Se declara, igual que el `split_hash`
en ceros y el `code_revision` nulo de los brazos ViT-S.

## Prediccion secundaria, tambien falsable

`resume.pt` guarda **segundos acumulados**, no una marca de arranque. El
proceso nuevo restaura 20521,6 s y le suma su propio reloj. Para llegar a
las 12.00 horas-GPU necesita 21600 s:

```
21600 - 20521,6 = 1078,4 s = 17m58s
```

Exactamente un intervalo de checkpoint. Con arranque a las 20:02:40, el
cierre deberia caer **~20:20:38**.

Si cierra ahi, el cuelgue no se contabilizo y `c100` lo escribio el proceso
viejo: lectura 1. Si hubiera cerrado a las 20:02, era la 2.

**Estado al momento de escribir esto (20:04):** `backbone_final.pt` no
existe y los tres procesos tienen CPU avanzando. El brazo esta trabajando,
que es lo que predice la lectura 1 — pero eso es consistencia, no
confirmacion, y el documento se cierra sin declararlo resuelto.

---

## Resultado (agregado al cerrar, 20:21:09)

**Lectura 1, sin ambiguedad.** El brazo hizo el trabajo completo de 12.00
horas-GPU y el cuelgue NO se contabilizo como computo.

| | predicho | medido | error |
|---|---|---|---|
| `steps_executed` | 5891 | **5894** | +3 (0,05 %) |
| `samples_processed` | 753 987 | **754 432** | +445 (0,06 %) |
| hora de cierre | 20:20:38 | **20:21:09** | +31 s |

La banda era 5714-6067; cayo a tres pasos del centro. La prediccion
secundaria — que el reloj se restaura desde `elapsed_wall_seconds` y no
desde el arranque, de modo que faltaba exactamente un intervalo de
checkpoint — acerto con 31 segundos de error sobre un calculo hecho 18
minutos antes.

`c100` a las 20:02:40 lo escribio el proceso viejo en su teardown, como
decia la prediccion.

**Esta fila no lleva nota.** El presupuesto efectivo es verificable y es el
mismo que el de los otros brazos, asi que entra a la comparacion en igualdad
de condiciones.

Backbone final: 346 388 719 bytes, sha256 `11da3a058b5cf2f2f6e28a86...`

### Lo que se valida, mas alla del numero

La cadena de recuperacion funciono sin intervencion humana y por primera vez
de verdad: el watchdog de NCCL mato el proceso colgado, `--max-retries` lo
relanzo, y el resume retomo desde 11.40 horas-GPU. El costo real de una
falla que parecia grave fueron ~18 minutos de reloj.
