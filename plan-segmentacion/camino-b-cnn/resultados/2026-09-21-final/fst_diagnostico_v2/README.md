# Control del gap por tono — estimador corregido

Segunda tanda de diagnóstico, posterior al fix del estimador de fototipo
(`c272bdb` guard de b*, `bbd9bab` terciles de luminosidad).

## Qué responde

Con el estimador v1 la "brecha por fototipo" resultó ser prevalencia de
enfermedad, no tono. Con el estimador corregido vuelve a aparecer un gap por
tono en HAM10000 (`gap_dice` 0,1114). La pregunta obvia era si el gap nuevo es
otra vez composición.

**No lo es.** El Dice por tercil dentro de cada diagnóstico mantiene el orden
claro > medio > oscuro en seis de las siete clases, y en las cuatro
mayoritarias el gap es *mayor* que el global.

## Contenido

| Archivo | Qué es |
|---|---|
| `diag_dx_por_tercil.py` / `.txt` | HAM10000: Dice por tercil dentro de cada diagnóstico |
| `diag_isic2018_tercil_tamano.py` / `.txt` | ISIC2018: Dice por tercil dentro de cada banda de tamaño |

Los dos scripts son de solo lectura.

## ISIC2018: el resultado es distinto, y hay que decirlo así

En ISIC2018 no hay diagnóstico disponible, así que el único confusor cruzable
es el tamaño de la lesión (`ref_coverage`). El resultado NO replica lo de HAM:

| banda | light | medium | dark | gap |
|---|---|---|---|---|
| [0,00 – 0,10) | 0,7950 (33) | 0,7526 (44) | 0,7934 (30) | +0,042, no monótono |
| [0,10 – 0,25) | 0,8742 (97) | 0,8647 (112) | 0,8533 (57) | +0,021 |
| [0,25 – 0,60) | 0,8166 (113) | 0,8026 (122) | 0,7143 (45) | +0,102 |
| [0,60 – 1,01] | n<30 | n<30 | n<30 | omitido |

Y sí hay diferencia de composición: el tercil oscuro tiene más lesiones grandes
(16,5 % en la banda ≥0,60 contra 6,5 % del claro) y también más pequeñas
(19,0 % contra 12,7 %).

Lectura honesta: en ISIC2018 el gap de 0,0554 se concentra en una sola banda de
tamaño, desaparece casi por completo en las dos primeras, y el tercil oscuro
tiene una distribución de tamaños distinta. Con n=722 y sin poder controlar por
diagnóstico, **el gap de ISIC2018 no está limpio de composición**. El de
HAM10000 sí lo está. Por eso HAM es el conjunto principal de equidad e ISIC2018
solo secundario.

## Diferencia con `fst_diagnostico/` (la tanda anterior)

Aquel paquete diagnosticaba el estimador **v1** y lee el manifiesto viejo
(respaldado en `salidas/final/fst.v1_signo_b/`). Este lee el manifiesto
**nuevo** (`salidas/final/fst/fst_pseudo_manifest.csv`, schema 1.1, con la
columna `tone_tercile`), que es el que produjo el rerun del 2026-09-21 17:09.

Los dos son reproducibles hoy, cada uno contra su manifiesto. No se pueden
intercambiar.

## Contexto de los cortes

Los terciles se calculan sobre el pool completo (n=54.978), no sobre HAM.
Cortes: L* 67,25 y 78,59. Dentro de HAM eso da:

| tercil | n | L* media | rango |
|---|---|---|---|
| light | 864 | 82,77 | 78,59 – 96,37 |
| medium | 4876 | 71,99 | 67,25 – 78,58 |
| dark | 4082 | 62,27 | 42,27 – 67,24 |

Nota para el informe: "dark" es el tercio inferior de **este** pool, con L*
medio 62. No es piel oscura en sentido absoluto, y HAM no se reparte en tercios
iguales porque los cortes salen del pool.

## Lectura recomendada

El mecanismo más probable del gap es físico, no un sesgo aprendido: a menor
luminosidad de la piel el contraste lesión/fondo baja, y un método sin
etiquetas que se apoya en saliencia se degrada más. Conviene reportarlo como
limitación del enfoque, no como acusación al modelo.

## Entorno

Pod UNABIA `jupyter-nmoreno534`, 2026-09-21, repo en `bbd9bab`.
