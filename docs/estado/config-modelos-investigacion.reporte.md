# Reporte: configuraciones de modelo que corresponden probar a continuacion

Sesion `impl`, contrato `docs/estado/config-modelos-investigacion.md`. Fecha: **2026-10-31**.
Solo analisis y propuesta. No se entreno, no se evaluo, no se lanzo nada, no se toco ningun gate
ni ningun config, no se accedio al cluster ni al servicio desplegado.

## Correcciones al contexto del contrato

Dos cifras del contrato estan desactualizadas o incompletas. Se usan corregidas.

| Contrato dice | Estado real | Fuente |
|---|---|---|
| «Segmentacion: Run 008 congelado, Dice 0,7904» | Superado. La linea activa son los runs 019-023. **023 alcanza Dice medio 84,16 %** sobre los mismos 2.594 pares | `docs/estado/seg-imapp-holdout.reporte.md` seccion 4 (linea 193), commit `44a1c5a` |
| «la mascara no ayudo al clasificador … usando `student_r3` con Dice 0,7177» | Correcto, pero incompleto: tambien se midio con A2 (Dice 0,7221), 0 de 15 mejoran y 2 empeoran | `TABLA_CIERRE.md` filas 2-3 |

### Error propio, corregido

Una version previa de este reporte afirmaba que `resultados/clasificacion/pad_ufes/README.md`
**no existia**. **Era falso.** El archivo existe, 28.540 bytes, fecha 2026-10-10. Es la tabla
sellada del protocolo v7 zero-shot: generada por `scripts/build_pad_ufes_table.py` desde
`pad_ufes_eval.json`, con `--check` y `tests/test_build_pad_ufes_table.py` en CI,
`code_revision` `0c9526fa6707af739a1f1683f2424cce5cd305e4`.

Causa del error: la verificacion se hizo con `ls -la … | head`, que corta en 10 lineas por
defecto. El directorio tiene 10 entradas mas `total`, `.` y `..`, y `README.md` ordena **despues**
de los `pad_ufes_*`, asi que quedo fuera de la salida. Se leyo una salida truncada como si fuera
completa. El contrato tenia razon al listarlo en «Entra».

Las otras dos verificaciones del mismo lote que usaron `head` se re-comprobaron sin truncar y
**resisten**: los sufijos usados en `resultados/preparacion_v8/` son `_001` a `_007`
(`ls -1 … | grep -oE '_[0-9]{3}$' | sort -u`), asi que la numeracion propuesta `_008`-`_012` no
colisiona; y los `scratch_*` de la raiz son 53 (`ls -1 scratch_* | wc -l`).

La primera correccion importa para la propuesta: la pregunta «¿ayuda una mascara mejor?» ahora se
puede plantear con 0,8416 en vez de 0,7177. La seccion 3 explica por que eso **no** alcanza para
justificar repetir el experimento.

---

## 1. Lectura del espacio ya explorado

Todo sobre PAD-UFES-20 `target_adapt`, 1.134 imagenes, CV anidada agrupada por paciente
(3 semillas CV x 5 outer x 3 inner), lector lineal balanceado, desarrollo/transductivo.
Referencia: **baseline SSL media de 3 semillas PAD 0,597** (DE 0,010; la semilla 42 con 0,606 era
alta). Las cifras por semilla puntual usan 0,6063.

### Las palancas, ordenadas por magnitud medida

| Familia de cambio | Mejor resultado | Delta e IC | Veredicto |
|---|---|---|---|
| **Encoder / corpus de preentrenamiento** | SigLIP2-so400m congelado 0,6668 | **+0,0605 [+0,028; +0,092]** | mueve, pero **no es SSL** (imagen-texto) |
| **Metadatos clinicos en el lector** | fusion clinica 0,654 | **+0,047 [+0,002; +0,092]** | mueve |
| **Familia de objetivo SSL** | VICReg 0,638 · Barlow Twins 0,633 | **+0,041 [+0,019; +0,063]** · +0,035 [+0,008; +0,062] | mueve, **PROVISIONAL** |
| **Capacidad del lector** | lineal balanceado 0,606 | techo: nada le gana | **no es el cuello de botella** |
| **Perillas de receta de SimSiam** | lr x4 +0,003 | todas dentro de **±0,006** | nulo |
| **Mas datos no etiquetados del mismo dominio** | SkinDisNet 0,5940 | **+0,0003 [−0,0087; +0,0100]** | nulo, y pierde −0,0141 en HAM |

Detalle de lectores probados, todos por debajo o iguales al lineal: kNN k20 0,40/0,43 · small-MLP
0,494 · SVM 0,549 · dos encoders concatenados 0,594 · cache de prototipos 0,606 · ajuste de
logits sin ganancia · interpolacion de pesos alpha=1 sin ganancia.

Detalle de perillas y brazos nulos o peores: lr x4 **+0,003** · menos repeticiones de celular
**−0,009** · misma cuota sin SkinDisNet **−0,013** · LR constante en el predictor
**−0,0057 [−0,0152; +0,0033]** · crop-scale 0,4 **−0,0152 [−0,0277; −0,0043]** · destilacion CLS
fija **−0,0323 [−0,0617; −0,0062]** · dosis larga 12k **+0,003 [−0,006; +0,012]** · DINO+iBOT,
MULAN, BYOL, OOB, freeze4, 336, recetas v1, SkinDisNet: nulos.

### El patron que explica la diferencia

**Lo que mueve la metrica es el contenido de informacion de la representacion, no el procedimiento
de adaptacion.** Tres evidencias convergentes:

1. El lector esta saturado. Siete arquitecturas distintas convergen a 0,59-0,606. Agregar
   capacidad no agrega nada: el cuello esta aguas arriba.
2. Agregar imagenes no etiquetadas del mismo dominio no agrega informacion util. SkinDisNet
   aporta 1.633 imagenes y da **+0,0003**, con perdida en HAM. No es cantidad de datos.
3. Cambiar de donde viene la representacion si mueve: cambiar el corpus y objetivo de
   preentrenamiento (SigLIP2) da +0,0605; cambiar la **familia** de objetivo SSL da +0,035-0,041.
   Cambiar la receta dentro de la misma familia da ±0,006.

**Y hay un mecanismo candidato, no solo una correlacion.** Los brazos que movieron la metrica
(VICReg, Barlow Twins) comparten algo que los nulos no tienen: **regularizan explicitamente la
covarianza del embedding** para evitar colapso dimensional. Los nulos (perillas de SimSiam, BYOL,
DINO+iBOT, MoCo v3) evitan el colapso por asimetria o momentum, sin termino explicito. Y el
diagnostico del baseline ya registra **rango 428 → 420 sobre 768 dimensiones**: no hay colapso
total, pero cerca del 45 % de las dimensiones estan muertas. Eso es exactamente lo que
VICReg y Barlow atacan por construccion.

Esa hipotesis es **testeable y casi gratis**, y es la propuesta P1. Convierte un +0,04 provisional
en un resultado con mecanismo, que es lo que hace reportable una tabla de ablacion.

---

## 2. Propuesta de configuraciones

Cinco, ordenadas por valor por GPU-h. Costos derivados de los medidos: un brazo SSL de 6.000
pasos cuesta **1,065-1,282 GPU-h** (mediciones de `plus-skindisnet`, `constant-predictor-lr`
y los dos brazos de la reactivacion A/B); se usa **1,2 GPU-h** como estimador. Un brazo de
segmentacion de 300 epocas cuesta **3,08 GPU-h** (023: 11.085 s en una GPU).

**Tope declarado para este bloque: 12 GPU-h.** Ver la advertencia de presupuesto al final de esta
seccion: **excede el techo historico y requiere autorizacion nueva.**

### P1 — Diagnostico de rango efectivo sobre los checkpoints que ya existen

| | |
|---|---|
| **Hipotesis** | La ganancia de VICReg y Barlow Twins sobre SimSiam viene de recuperar dimensiones muertas del embedding, no de un mejor ajuste al dominio. |
| **Cambio exacto** | Ninguno en el entrenamiento. Se extraen embeddings de los checkpoints ya existentes (baseline SimSiam, VICReg, Barlow, 3 semillas cada uno) sobre `target_adapt` con los OOF fijos, y se calcula el rango efectivo (razon de participacion del espectro de la covarianza) por brazo y semilla. |
| **Criterio de exito, registrado antes del dato** | **(a)** rango efectivo de VICReg >= 1,15 x el del baseline, y **(b)** las 3 semillas coinciden en signo: donde sube el rango, sube el macro-F1 (3/3). Si (a) se cumple y (b) no, el mecanismo queda **refutado** y VICReg pasa a ser un hallazgo sin explicacion. |
| **Costo** | **~0,1 GPU-h.** Solo inferencia, sin entrenamiento. |
| **Lo justifica** | Rango 428 → 420 sobre 768 registrado en el diagnostico del baseline (#2224), contra el +0,041 [+0,019; +0,063] de VICReg (#2266). |
| **SSL puro** | Si. Es un diagnostico, no cambia el objetivo. |

Primera por una razon: es el unico experimento del lote que puede **refutar** una hipotesis en vez
de medir un delta, y cuesta casi cero. Si se confirma, da el criterio para elegir el proximo
objetivo SSL sin probar uno por uno.

### P2 — Cerrar la pregunta de VICReg y Barlow Twins con potencia suficiente

| | |
|---|---|
| **Hipotesis** | La ventaja de VICReg sobre el baseline es real y no un artefacto de 3 semillas. |
| **Cambio exacto** | Repetir VICReg y Barlow Twins con semillas 45, 46 y 47, receta identica, un solo factor respecto del baseline. Nada mas. |
| **Criterio de exito, registrado antes del dato** | Con 6 semillas por brazo: **Delta PAD media >= +0,02 y prueba de permutacion p < 0,05**, mas **Delta HAM >= −0,01**. Si p queda entre 0,05 y 0,10 se reporta como **no concluyente y no se amplia mas**: se cierra con ese veredicto. |
| **Costo** | **6 corridas x 1,2 = 7,2 GPU-h.** |
| **Lo justifica** | VICReg +0,041 [+0,019; +0,063] y Barlow +0,035 [+0,008; +0,062] cumplen el criterio declarado pero la **permutacion da p minimo 0,10** con 3 contra 3 (#2266). Hoy no se puede afirmar. |
| **SSL puro** | Si. |

**Esto contradice una decision previa y hay que decirlo:** Nicolas decidio **no** ampliar a las
semillas 45/46 (#2266). Esa decision es coherente con un criterio de guardia operativa, pero deja
el unico resultado SSL positivo de la campana en estado provisional **de forma permanente**. Con
p = 0,10 no se puede escribir en la tesis que VICReg le gana al baseline. O se paga la potencia, o
se reporta «provisional» en la tabla final y se acepta que el brazo SSL propio no tiene una mejora
acreditada. Es una decision de Nicolas, no mia; la propuesta esta para que la decida con el costo
a la vista.

### P3 — Metadatos clinicos sobre el mejor encoder SSL, no sobre el viejo

| | |
|---|---|
| **Hipotesis** | La ganancia de los metadatos clinicos (+0,047) se suma a la del objetivo SSL nuevo, en vez de solaparse con ella. |
| **Cambio exacto** | Lector de fusion clinica (metadatos de imagen y paciente, **excluyendo** diagnostico, biopsia, IDs e historia de cancer) sobre los embeddings de VICReg en vez de los de SimSiam adaptado. Un solo factor: el encoder debajo del mismo lector de fusion. |
| **Criterio de exito, registrado antes del dato** | **Delta(VICReg+metadatos − VICReg) >= +0,02 con IC 95 % pareado por paciente que excluya el cero.** Si el IC incluye el cero, se concluye que metadatos y objetivo SSL capturan la misma informacion y **no se combina mas**. |
| **Costo** | **~0,2 GPU-h.** Reusa los checkpoints de VICReg; solo extraccion de embeddings y ajuste del lector. |
| **Lo justifica** | Metadatos sobre el encoder viejo: +0,047 [+0,002; +0,092] (#2224). Sobre SigLIP2: **+0,0359 [−0,002; +0,076], no concluyente** (#2225) — o sea que la ganancia de metadatos **ya se mostro solapable** con un encoder mas fuerte. Nunca se midio sobre VICReg. |
| **SSL puro** | **El encoder si; el sistema completo no.** Los metadatos no son etiquetas diagnosticas ni hay fuga de la etiqueta de evaluacion, asi que el aprendizaje de representacion sigue siendo autosupervisado. Pero el clasificador final deja de ser «solo imagen» y eso hay que declararlo en la tabla, no esconderlo en una nota. |

El valor de P3 no es la ganancia esperada, es el **riesgo que elimina**: si metadatos y VICReg se
solapan, sumarlos en la tesis seria contar dos veces la misma mejora.

### P4 — Piloto de CIRCLe con filtro por caso, un solo factor por brazo

| | |
|---|---|
| **Hipotesis** | La mascara de segmentacion sirve al clasificador **como definicion del pool de piel para normalizar tono**, no como recorte de la lesion; y el filtro por caso de contaminacion CIELAB es lo que decide si funciona. |
| **Cambio exacto** | Tres brazos, un factor por paso. **A0**: VICReg sobre imagen cruda, sin mascara ni normalizacion de tono (referencia). **A1**: A0 + normalizacion CIRCLe con pool de piel derivado de la mascara del run 021, **sin** filtro. **A2**: A1 + filtro por caso que descarta la imagen del pool si su contaminacion CIELAB supera `tau`. |
| **Criterio de exito, registrado antes del dato** | Dos condiciones, las dos necesarias. **(1)** macro-F1 A1 − A0 con IC pareado por paciente que **no empeore** (limite inferior >= −0,01). **(2)** la **brecha de macro-F1 entre terciles de luminancia se reduce** en A1 contra A0, con IC que excluya el cero. Si (2) falla, **se detiene y no se corre A2**: si CIRCLe no cierra la brecha de tono, el filtro no tiene nada que rescatar. |
| **`tau` propuesto** | `tau = 0,20` de contaminacion de lesion en el pool de piel por caso. Justificacion: el p95 medido esta entre **0,352 y 0,396** en los cinco runs 019-023, y la mediana de lesion no protegida entre 0,118 y 0,181. Un `tau` de 0,20 corta la cola sin descartar el cuerpo de la distribucion. **Registrado antes del dato y no se mueve despues.** |
| **Costo** | **2 brazos x 1,2 = 2,4 GPU-h** como piloto de una semilla, con freno explicito en (2). Si pasa, escalar a 3 semillas cuesta 7,2 GPU-h mas y **es una decision aparte**. |
| **Lo justifica** | La brecha de tono esta medida y las mitigaciones probadas no la cerraron: Dice en el tercil oscuro −0,0583 [−0,0652; −0,0515] para A2 y −0,0616 [−0,0688; −0,0542] para `student_r3`; T6 (umbral por tercil) y T7 (reentrenamiento con oscurecimiento) **no cerraron la brecha**. El codigo del pipeline CIRCLe+DINOv2 ya existe commiteado (`b5d44d6`) **sin resultado consolidado**. |
| **SSL puro** | Si en el encoder. CIRCLe es una normalizacion de entrada, no supervision. |

**Limite de alcance de P4, y hay que declararlo antes de correrlo.** La metrica secundaria son
**terciles de luminancia (L\*)**, que por construccion parten el dato en tres y siempre son
calculables. Lo que **no** son es un proxy de equidad en piel oscura:
`resultados/clasificacion/pad_ufes/README.md` linea 7 declara que en PAD-UFES-20 el grupo III-VI
es **84 % tipo III (392/465)**, con IV 62, V 10 y VI 1, y concluye literalmente que *«PAD-UFES-20
no permite concluir sobre FST V-VI»*. El mismo README muestra el sintoma: en el estrato III-VI hay
celdas con `nan no calculable (1000/1000)`, o sea no estimables en ninguna de las 1.000
remuestras.

Consecuencia para P4: si cierra la brecha entre terciles, eso se reporta como **mejora de
equidad entre terciles de luminancia de PAD**, no como mejora en piel oscura. Afirmar lo segundo
con estos datos seria sobreinterpretar. Si el objetivo es FST V-VI, PAD no es el dataset y hay que
decirlo antes de gastar las GPU-h, no despues.

**Por que el pool de piel y no el recorte de la lesion:** ver seccion 3.

### P5 — Congelar el cohorte IMA++ de 023 antes de proponer escalarlo

| | |
|---|---|
| **Hipotesis** | Ninguna. Es un paso de acreditacion, no un experimento. |
| **Cambio exacto** | Identificar en `scripts/train_dermlip_semi_023.py` de donde sale el «IMA++ audited cohort (100 images)», congelar ese listado con manifiesto, sha256 y recibo, y cruzarlo contra `preflight/isic2018-task1-eval-2594-v2/manifest.json` para medir solapamiento con el split de evaluacion. |
| **Criterio de exito, registrado antes del dato** | Manifiesto con hash existente **y** solapamiento con el split de evaluacion = 0 casos. Si el solapamiento es > 0, las cifras de 023 sobre los 2.594 quedan invalidadas como evaluacion y hay que recortar el cohorte antes de cualquier escalado. |
| **Costo** | **0 GPU-h.** |
| **Lo justifica** | Ningun recibo de procedencia de 023 acredita IMA++: `run_identity.json`, `summary.json`, `training_started.json` y el config (sha256 `e86d2232…`) no listan ningun artefacto de IMA++, y el `pool_provenance` declara 34.330 = ISIC2019 14.411 + ISIC2020 19.919. Detalle en `seg-imapp-holdout.reporte.md` seccion 4. |
| **SSL puro** | No aplica. |

**Es prerrequisito de cualquier propuesta que toque IMA++.** No se puede proponer escalar de 100 a
N imagenes sin saber cuales son las 100 actuales ni si contaminan la evaluacion. Si el solapamiento
resulta > 0, el Dice 0,8416 de 023 no es una cifra de evaluacion.

### Advertencia de presupuesto

| Concepto | GPU-h |
|---|---|
| P1 | 0,1 |
| P2 | 7,2 |
| P3 | 0,2 |
| P4 (piloto, con freno) | 2,4 |
| P5 | 0 |
| **Total propuesto** | **9,9** |

El tope historico declarado de la campana del clasificador es **20 GPU-h**, con
**15,71 GPU-h** como maximo comprometido al cierre de la reactivacion A/B. Eso deja
**~4,3 GPU-h de margen**. Las cinco propuestas juntas piden 9,9 y **no entran**. Entran P1 + P3 +
P5 + P4-piloto (**2,7 GPU-h**). **P2 sola (7,2) excede el margen y requiere ampliar el tope.**
Eso es decision de Nicolas, no se asume aqui.

---

## 3. Diseno del pipeline segmentacion + clasificacion

### Por que NO repetir «mascara como entrada del clasificador»

Es la propuesta obvia y hay que descartarla explicitamente. El resultado negativo ya esta medido y
pre-registrado dos veces: con `student_r3` (Dice 0,7177) **0 de 15 celdas mejoran y 4 empeoran**;
con A2 (Dice 0,7221) **0 de 15 mejoran y 2 empeoran**. `TABLA_CIERRE.md` filas 2-3.

El argumento para repetirlo seria «la mascara era mala, ahora tenemos 0,8416». No alcanza, por
dos razones:

1. **El efecto medido no fue nulo, fue negativo en una direccion especifica.** Las celdas que
   empeoran son casi todas la condicion **(b) fondo gris**: publico −0,0916 [−0,1409; −0,0405],
   A* s42 −0,0519 [−0,0950; −0,0114], A* s43 −0,0579 [−0,1014; −0,0180]. Borrar el fondo es lo que
   hace dano. El dano viene de **destruir la piel perilesional**, que carga senal diagnostica, no de
   que el contorno este mal trazado. Subir el Dice de 0,72 a 0,84 mejora el contorno; no devuelve el
   contexto que el recorte elimina.
2. **El mecanismo de dano es independiente de la calidad de la mascara.** Una mascara perfecta
   (Dice 1,0) borraria el fondo **mejor**, o sea haria mas dano por esa via, no menos.

Repetir un negativo pre-registrado con un insumo mejor y sin cambiar el mecanismo es pesca. Si
alguien lo quiere correr igual, que sea con criterio registrado antes y declarado como replicacion,
no como propuesta nueva.

### El uso de la mascara que si esta sin probar

La mascara tiene un segundo rol que v7 **nunca** midio: **definir el pool de piel sana desde el
cual estimar y normalizar el tono**. Mecanismo distinto al del recorte — no quita contexto al
clasificador, solo selecciona pixeles para estimar un estadistico de color. Y ahi la calidad de la
mascara si entra de forma directa, porque un pixel de lesion que se cuela en el pool contamina la
estimacion de tono.

Ese es el diseno de P4. Brazos, factor unico por paso:

| Brazo | Encoder | Mascara | Normalizacion de tono | Filtro por caso | Factor que cambia respecto del anterior |
|---|---|---|---|---|---|
| **A0** | VICReg | no | no | no | — (referencia) |
| **A1** | VICReg | run 021 | CIRCLe sobre pool de piel | **no** | se agrega la normalizacion de tono |
| **A2** | VICReg | run 021 | CIRCLe sobre pool de piel | **si, `tau` = 0,20** | se agrega el filtro por caso |

- **Metrica primaria:** macro-F1 en PAD `target_adapt`, CV anidada agrupada por paciente, 3
  semillas CV. **Metrica secundaria y razon de ser del brazo:** brecha de macro-F1 entre terciles
  de luminancia (L\* de la piel, **no** ITA ni fototipo).
- **Prueba estadistica:** bootstrap pareado por paciente, 500 remuestreos, OOF fijo, sin refit y
  sin correccion por multiplicidad — la misma convencion ya usada en la campana, para que los
  numeros sean comparables con lo medido.
- **Criterio de parada:** si en A1 contra A0 la brecha entre terciles **no** se reduce con IC que
  excluya el cero, **se detiene antes de correr A2**. CIRCLe existe para cerrar brecha de tono; si
  no la cierra, afinar el pool con un filtro no tiene nada que rescatar.
- **Por que la mascara del 021 y no la del 023:** 021 tiene **1 mascara degenerada** sobre 2.594
  contra las **5** de 023, y la mejor fuga de lesion entre los cuatro runs pareados
  (0,1499). 023 gana en contaminacion p95 (0,3520) pero su cohorte IMA++ no esta acreditado (P5),
  asi que usarla ataria el pipeline a un insumo sin procedencia.

**Restriccion que condiciona todo el diseno.** La contaminacion p95 del pool de piel esta entre
**0,352 y 0,396 en los cinco runs**. En el 5 % peor de los casos, cerca del 35 % de la masa del
pool es lesion no detectada. **Ningun run entrega un pool apto sin filtro por caso.** Por eso el
filtro no es un refinamiento opcional del brazo A2: es la condicion para que CIRCLe no estime tono
sobre lesion en la cola de la distribucion. Y por eso `tau` va registrado antes del dato.

---

## 4. Que NO vale la pena volver a probar

| Descartado | Evidencia que lo descarta |
|---|---|
| **Mas capacidad en el lector** | Siete arquitecturas convergen a 0,59-0,606: kNN 0,40/0,43, small-MLP 0,494, SVM 0,549, dos encoders 0,594, cache de prototipos 0,606, ajuste de logits sin ganancia, interpolacion de pesos sin ganancia. El lector lineal es el techo. |
| **Perillas de receta dentro de SimSiam** | Todas dentro de ±0,006 del baseline. lr x4 +0,003; LR constante en predictor −0,0057 [−0,0152; +0,0033]; dosis larga 12k +0,003 [−0,006; +0,012]; menos repeticiones −0,009. |
| **Mas imagenes no etiquetadas del mismo dominio** | SkinDisNet (1.633 imagenes): PAD **+0,0003 [−0,0087; +0,0100]**, HAM **−0,0141 [−0,0244; −0,0035]**. No es cantidad de datos. |
| **crop-scale 0,4** | −0,0152 [−0,0277; −0,0043]. Medido dos veces (campana v3 y reactivacion A/B). |
| **Destilacion CLS con ancla fija** | −0,0323 [−0,0617; −0,0062] en PAD y −0,0520 [−0,0802; −0,0220] en HAM. Ademas fallo dos veces por crecimiento de memoria propio; el bugfix funciono y **la metrica siguio peor**. |
| **DINO+iBOT, MULAN, BYOL, OOB, freeze4, 336, recetas v1** | Nulos en la campana v3. DINO+iBOT ya se amplio a 3 semillas por pedido de Karen y sigue nulo. |
| **ViT-L como encoder** | **NO-GO tecnico**, no cientifico: el proyector, el exportador (768 dims) y el runner estan atados a ViT-B. Cambiarlo es un refactor, no un experimento. |
| **Mascara como entrada del clasificador, sin cambiar el mecanismo** | 0/15 mejoran y 4 empeoran con `student_r3`; 0/15 y 2 empeoran con A2. Seccion 3. |
| **El estimador ITA como fototipo** | Rinde **por debajo de la baseline trivial**: 22,9 % contra 51,3 % de mayoria en PAD-UFES-20; 28,5-30,8 % contra 32,9 % en Fitzpatrick17k-C. No usarlo como variable de estratificacion clinica. Para terciles usar L\* y declararlo. |
| **Variantes nuevas de copy-paste en segmentacion** | El efecto **no se reproduce entre semillas** en los dos estratos mas grandes: `5-10%` +0,49 (s42) contra −0,14 (s43); `>=10%` **−1,50 contra +0,06**, y ahi esta el 59,3 % de los casos. Ademas triplica los ceros de solapamiento (5 → 15 en s42). No capitalizar el efecto. Si se toca copy-paste, que sea **para cerrar la pregunta** con semillas, no para construir sobre el. |
| **El delta del grupo 2 sintetico como evidencia** | Circular: benchmark construido con copy-paste midiendo una ablacion de copy-paste, +34,7 puntos contra +0,24 en fotos reales. Y su n efectivo es **25**, no 75, por declaracion del propio archivo. |

---

## 5. Orden recomendado

**Bloque inmediato, 2,7 GPU-h, entra en el margen existente:**

1. **P5** (0 GPU-h) — primero, porque si el cohorte IMA++ solapa con la evaluacion, el 0,8416 de
   023 no es una cifra de evaluacion y eso cambia la seccion 3.
2. **P1** (0,1 GPU-h) — puede refutar una hipotesis por casi nada y da el criterio para elegir el
   proximo objetivo SSL.
3. **P3** (0,2 GPU-h) — elimina el riesgo de contar dos veces la misma mejora.
4. **P4 piloto** (2,4 GPU-h) — con el freno en la brecha de terciles.

**Requiere decision de presupuesto:**

5. **P2** (7,2 GPU-h) — excede el margen de ~4,3 GPU-h bajo el tope historico de 20.

### Condicionado a la decision pendiente sobre el entregable

La decision de cual algoritmo se entrega **no se toma aqui** (`docs/61-decision-entregable-algoritmo.md`,
sesion maestra con Nicolas y Karen). Implicaciones por opcion:

| Si el entregable es… | Entonces el orden cambia asi | Y hay que aceptar |
|---|---|---|
| **SSL puro** (compromiso con el objetivo general de la tesis) | **P2 pasa a ser obligatoria, no opcional.** Es el unico resultado SSL positivo del proyecto y hoy tiene p = 0,10. Sin potencia, la tesis no puede afirmar que su SSL mejora nada. | Techo ~0,638. SigLIP2 llega a 0,667 / 0,703 y queda como **comparador de cota superior**, declarado como imagen-texto, no SSL. |
| **El mejor disponible** (deja de ser SSL puro) | P2 baja de prioridad: VICReg deja de ser el entregable y pasa a ser una ablacion. Sube SigLIP2 + metadatos (0,7029 con `siglip2-base`, que iguala a `so400m` y es mas barato). | El objetivo general de la tesis hay que reescribirlo. No es un ajuste de redaccion. |
| **Segmentacion como contribucion principal** | Suben P5 y P4. La linea de segmentacion tiene el resultado mas fuerte del proyecto (Dice 0,8416) y el unico beneficio reproducible medido hoy (reduccion de FP sobre artefacto de *inpainting*: 25/25 → 5/25). | Que la mascara **no ayuda al clasificador** sigue siendo un negativo medido, y hay que reportarlo igual. |

En las tres opciones, **P5 y P1 convienen igual**: uno acredita procedencia y el otro cuesta 0,1
GPU-h.

Nota de presentacion, de `docs/60` seccion 6: RPI-Net reporta las metricas donde pierde y eso
**suma** credibilidad. Este proyecto tiene negativos medidos y pre-registrados — la mascara no
ayuda al clasificador, el ITA esta por debajo de la baseline trivial, el SSL propio pierde en PAD
zero-shot, el copy-paste no reproduce. Se reportan con el mismo tono que los positivos. Los cinco
criterios de exito de arriba estan registrados **antes** del dato por esa razon.

---

## Reporte

**Verificado:**

- `resultados/TABLA_CIERRE.md`: filas 1-9 leidas directo (`head -30`). Filas 2-3 confirman
  0/15 mejoran con `student_r3` y 4 empeoran; 0/15 con A2 y 2 empeoran, con los IC de cada celda
  que empeora.
- `docs/ESTADO.md`: Objetivo 2 (lineas 85-166) y Objetivo 3 (lineas 167-262) leidos completos.
  De ahi salen la tabla v7 (A\* 0,6464 / B\* 0,6606 / publico 0,6158), la tabla de segmentacion,
  la brecha por tercil de luminancia, el estado del ITA y la lista de pendientes.
- `docs/60-guia-presentacion-cientifica.md` seccion 6: extraida con `awk` entre los
  encabezados 6 y 7.
- Engram del equipo, siete observaciones traidas **por ID** con `mem_get_observation`:
  **#2224** (estado del clasificador, lectores, metadatos, SigLIP2, brazos SSL, tope de 20 GPU-h),
  **#2225** (bootstrap por paciente de SigLIP2), **#2240** (SkinDisNet), **#2250** (LR constante y
  fallo de memoria de la destilacion), **#2262** (reactivacion A/B final, costos GPU-h por brazo),
  **#2266** (cierre de la campana v3, decision de no ampliar semillas), **#2268** (runner de
  seguimiento, NO-GO de ViT-L, fuga transductiva de los brazos CB).
- Cifras de segmentacion: de `docs/estado/seg-imapp-holdout.reporte.md`, entregado por esta misma
  sesion y commiteado en `44a1c5a`. Todas recalculadas en local desde los `cases.jsonl` crudos con
  delta 0,000e+00 contra los JSON del cluster.
- `resultados/clasificacion/pad_ufes/README.md`: **existe** (28.540 bytes, 2026-10-10). Leido
  directo. Protocolo de la lectura primaria: 1.141 imagenes, 720 pacientes, 3 clases
  (`bcc`, `mel`, `nv`), probe `logreg`, bootstrap pareado agrupado por paciente con semilla
  20260923 y 1.000 remuestras. PAD completo: 2.298 imagenes, 1.373 pacientes. Caveat de fototipo
  en la linea 7. Una version previa de este reporte lo declaro inexistente por leer un `ls`
  truncado con `head`; ver «Error propio, corregido» arriba.
- Costos GPU-h: medidos, no estimados a ojo. 1,065 h (`plus-skindisnet`), 1,282 h
  (`constant-predictor-lr`), 1,1355 y 1,1328 h (reactivacion A/B); 3,08 h para 023
  (11.085 s del log de consola).

**No verificado:**

- **Ninguna de las cinco propuestas se ejecuto.** Son propuestas; sus criterios de exito estan
  registrados antes del dato, que es justamente el punto.
- **El rango efectivo de VICReg y Barlow Twins.** El 428 → 420 del baseline viene de #2224; los de
  los brazos nuevos no estan medidos. Es P1.
- **Que los checkpoints de VICReg y Barlow sigan disponibles** para P1 y P3. Viven en
  `~/Documentos/clasificador1-auditoria/…/campaign-20261027-v3/` y en el cluster
  `/home/jovyan/work/clf1_clinical_reference_20261024`, los dos fuera de este repo y **fuera del
  alcance de este contrato** («no conectarse al cluster»). Si no existen, P1 y P3 cuestan un
  reentrenamiento y el presupuesto cambia.
- **El margen real de GPU-h.** Los 15,71 h son el maximo comprometido declarado en #2262, no una
  lectura del consumo actual. Se comprueba con el ledger de GPU de la campana.
- **El codigo del pipeline CIRCLe de `b5d44d6`**: existe commiteado segun `docs/ESTADO.md` linea
  125, pero no se leyo ni se verifico que acepte un pool de piel externo ni un filtro por caso.
  P4 asume que se puede parametrizar; si no, hay costo de implementacion sin estimar.
- **El solapamiento del cohorte IMA++ con el split de evaluacion.** Es el objeto de P5.
- **La discrepancia D1 cross-dominio** (`TABLA_CIERRE.md` fila 10 contra las tablas generadas de
  ISIC 2019/2020). Declarada sin resolver en `docs/ESTADO.md`; no se toco.
- No se corrio `pytest` ni `--check` de los generadores `build_*`. Esta sesion no modifico codigo
  ni tablas generadas.

**Bloqueos:**

1. **P2 no entra en el presupuesto.** Pide 7,2 GPU-h contra ~4,3 de margen bajo el tope historico
   de 20. Requiere que Nicolas amplie el tope o acepte que el unico resultado SSL positivo del
   proyecto queda **provisional de forma permanente** (p = 0,10 con 3 contra 3). Es una decision
   suya; la propuesta la deja explicita con el costo a la vista.
2. **P1 y P3 dependen de checkpoints que viven fuera de este repo y del alcance del contrato.** No
   se pudo verificar que existan.
3. **P5 es prerrequisito de todo lo que toque IMA++**, y de la validez de las cifras de 023 sobre
   los 2.594 si el solapamiento resulta > 0.
4. **El orden final depende de la decision pendiente del entregable**, que es de la sesion maestra
   con Nicolas y Karen. La seccion 5 da las implicaciones de cada opcion sin decidir ninguna.
5. **Una imprecision del contrato corregida** (segmentacion declarada en Run 008 / Dice 0,7904,
   superada por los runs 019-023). Si otro documento copio esa referencia, hay que arreglarlo ahi
   tambien.
6. **Un error propio corregido**, registrado para que no se repita el metodo que lo produjo:
   verificar existencia de archivos con `ls … | head` da falsos negativos. Afecto a
   `resultados/clasificacion/pad_ufes/README.md`, que **si existe**. Las otras dos verificaciones
   del mismo lote se re-comprobaron sin truncar y resisten. Detalle en «Error propio, corregido».
7. **Ningun gate ni config fue modificado.** No se entreno, no se evaluo, no se accedio al cluster
   ni al servicio desplegado. No se commiteo nada.
