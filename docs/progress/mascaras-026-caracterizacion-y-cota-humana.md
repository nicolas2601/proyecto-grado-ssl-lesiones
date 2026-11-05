# Máscaras del teacher 026: caracterización fuera de dominio y cota humana

## 2026-11-05 — Medición por proxy, refutación del post-proceso, cota de IMA++

Todo lo de abajo es solo lectura e inferencia en CPU. Ningún gate de ejecución
se movió, ningún entrenamiento se lanzó. Artefactos en UNABIA bajo
`/home/jovyan/work/mask-lcc-probe-20261105/` y
`/home/jovyan/work/mask_quality_026_20261105.json`.

### 1. El eje desplazado en PAD es la topología, no el área

36.282 máscaras medidas, 0 archivos faltantes: 1.134 `pad_adapt`, 1.528
`ham_dev`, 33.620 `source_fit`. El comparador en dominio es `source_fit`, que es
el insumo real de preentrenamiento SSL (ISIC2019 14.072 + ISIC2020 19.548, cero
PAD/HAM).

| dataset | n | p25 | mediana | p75 | KS vs dominio |
|---|---|---|---|---|---|
| `pad_adapt` | 1.134 | 0,081 | 0,141 | 0,267 | **0,051** |
| `ham_dev` | 1.528 | 0,107 | 0,188 | 0,310 | **0,148** |
| `source_fit` | 33.620 | 0,072 | 0,142 | 0,275 | — |

El área de PAD calca su dominio de entrenamiento cuantil por cuantil. El
desplazado en área es HAM. Donde PAD sí se despega es en fragmentación
(componentes significativos, 224×224 canónico): **PAD 41,0 %**, `source_fit`
12,6 %, `ham_dev` 9,3 %. PAD = 3,3x su dominio; HAM queda por debajo.

**No lo explica el pelo.** PAD tiene 4,6x más inpaint (mediana 0,208 vs 0,044 y
0,045). Condicionando a pelo < 5 %: PAD 31,7 % contra 10,4 % del pool, sigue
3,0x arriba. El exceso es intrínseco al dominio clínico de teléfono.

**Severidad igual, tasa distinta.** Dentro del subconjunto fragmentado, el
`largest_component_share` medio es 0,823 (PAD), 0,818 (HAM), 0,808 (pool). PAD no
fragmenta peor: fragmenta más seguido. Por eso no existe un post-proceso
PAD-específico — cualquier regla toca los tres dominios igual.

### 2. El post-proceso está refutado con verdad de campo

Medido contra las 2.594 máscaras expertas de ISIC2018 Task 1. Chequeo de
reproducción contra `ita-ladder-20261104/isic2018_arm3_rows.jsonl`: máx |Δ|
1,4e-3, media 0,831938 contra 0,8320 declarado.

| variante | Dice | Δ vs cruda | IC95 del Δ |
|---|---|---|---|
| cruda | 0,8319 | — | — |
| componente mayor (LCC) | 0,8263 | **−0,0056** | [−0,0090; −0,0023] |
| despeckling (< 0,5 %) | 0,8319 | +0,0000 | [−0,0000; +0,0001] |

El IC del LCC excluye cero. Sobre los fragmentados, −0,0334. Desglosado por
`largest_component_share`: por debajo de 0,80 el daño va de −0,056 a −0,076,
porque ahí los componentes secundarios son lesión real. LCC dirigido (solo si
share ≥ 0,80) da +0,0003 con IC que cruza cero.

La abstención es lo que funciona, y está implementada aparte — ver
[abstención por fragmentación](abstencion-fragmentacion.md).

**PH2 no sirve para esta pregunta:** solo 3 de 200 predicciones fragmentan, sin
potencia. Su media da 0,968 y no 0,9375 porque `ph2_validation_cases` es 39: el
resto entró al entrenamiento (el 026 es `ph2cp`).

### 3. IMA++: 1.263 casos y la cota humana correcta

IMA++ tiene 14.967 imágenes y 22.472 máscaras, 18 anotadores, 4 niveles de
pericia, pero solo **2.394 son multi-anotador** (4-7 máscaras, con consenso
STAPLE y voto mayoritario). Exclusiones sobre esas 2.394: ISIC2018 eval quita
999, inventario histórico 774, `source_fit` 42, pool v2 42, pool v3 0. Quedan
**1.341**, y **1.263** tras deduplicar por lesión y paciente.

| cota | valor | IC95 |
|---|---|---|
| Dice pareado humano-humano (archivo IAA) | 0,7948 | [0,7853; 0,8040] |
| **Dice humano vs consenso STAPLE** | **0,8932** | [0,8881; 0,8982] |

Brecha de 9,8 pp. El 026 marca 0,8319: queda **por encima** de la pareada y
**por debajo** de la cota correcta. Reportar la pareada afirmaría que el modelo
iguala o supera a los anotadores humanos, que es falso. La cota del informe es
la que vive en la misma escala que el puntaje del modelo: **0,8932**.

Calculada sin descargas y sin GPU: las 22.472 máscaras ya están en disco. Filas
en `ima_human_ceiling_rows.jsonl`, 2,13 anotadores por imagen, 0 desajustes de
forma, 0 sin consenso.

### 4. IMA++ no cubre el agujero de PAD

`image_type` es `dermoscopic` en **14.967 de 14.967**, y en 1.341 de 1.341 del
banco usable. Cero imágenes clínicas. PAD-UFES-20 es fotografía clínica de
teléfono, así que un Dice sobre IMA++ mide el mismo dominio donde ya hay
ISIC2018 en 0,832. Lo que IMA++ agrega es la cota humana y una referencia de
consenso: mejora la calidad de la afirmación en dominio, no la cobertura de
dominios.

**El agujero de PAD no se cierra con datos existentes.** Toda la verdad de campo
de segmentación disponible es dermatoscópica (ISIC2018, PH2, IMA++). PAD-UFES-20
trae solo `images/` y `metadata.csv`; DDI trae 656 PNG y su CSV, sin máscaras
(busqueda `mask|seg|truth` en las 657 entradas del zip: 0 coincidencias).
Cerrarlo requiere anotar. Presupuesto con la sd por imagen de ISIC2018 (0,1924):
n=50 da ±5,3 pp, n=150 da ±3,1 pp, n=200 da ±2,7 pp.

### 5. Procedencia: `34.115` no describe las máscaras del pool

`generate_masks_026_pool.py` registra `selection_manifest_sha256` del pool v3
(`0bb8be1d…`, 34.115 registros) pero `read_rows()` no lee ese archivo: lee
`config/ssl-overlay-v1/mixed-fit.jsonl` y filtra `role == "source_fit"`. El
manifest v3 no tiene campo `role`. Cruzando IDs: 709 del manifest sin máscara y
214 máscaras fuera del manifest. Las máscaras del pool son **33.620**.

### Cifras retiradas hoy

| cifra | por qué muere |
|---|---|
| LCC o morfología como arreglo de la fragmentación | empeora el Dice, IC95 excluye cero |
| 0,7948 como cota humana | es pareada humano-humano, no comparable con un puntaje de modelo |
| «IMA++ cubre el agujero de PAD» | IMA++ es 100 % dermatoscópico; la afirmación se originó en este hilo y se retira |
| 34.115 máscaras del pool | son 33.620; el campo del manifest no es la fuente de las filas |
| «el área de la máscara está desplazada en PAD» | KS 0,051 contra su dominio; el desplazado en área es HAM (0,148) |

### No revisado, declarado

- El recibo de construcción del pool v3, por lo que el 0 de solapamiento con
  IMA++ **no** se declara disjunción por diseño.
- Licencias caso por caso de las 1.340 imágenes de IMA++ que habría que
  descargar (la columna `copyright_license` es CC-0 en su mayoría).
- El gate de solapamiento de ISIC2018 (583 coincidencias, 2.011 candidatos sin
  congelar) sigue abierto; no se tocó.
