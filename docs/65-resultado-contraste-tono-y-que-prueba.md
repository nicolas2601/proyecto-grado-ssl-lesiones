# 65. Resultado del contraste de tono: qué salió, qué prueba, y qué quedó sin probar

Estado: vigente. Cerrado el 2026-11-05.

Leer antes: `63-instrumento-ita-alcance-y-limites.md`,
`64-aporte-medibilidad-ita-y-desplazamiento-logrado.md`, y
`estado/contraste-tono-pareado-estado-20261105.md` secciones 3.10 a 3.12.

---

## 1. El resultado, sin adornos

Contraste pareado, semilla 42, 6.000 pasos, un solo factor verificado por
`paired_factor_gate`. Lectura en el orden predeclarado **antes** de que
existieran las cifras (sección 3.11 del documento de estado).

| | n | control | tratado | delta | IC95 bootstrap | % negativo |
|---|---:|---:|---:|---:|---|---:|
| **PAD completo** (desenlace) | 1.134 | 0,606331 | 0,593403 | **−0,012928** | **[−0,0249 ; −0,0015]** | 98,6 % |
| PAD, máscaras ≥0,90 | 874 | 0,604240 | 0,595610 | −0,008630 | [−0,0203 ; +0,0029] | 93,2 % |
| **HAM completo** (desenlace) | 1.528 | 0,513984 | 0,502782 | **−0,011202** | [−0,0315 ; +0,0064] | 89,5 % |
| HAM, máscaras ≥0,90 | 1.454 | 0,504933 | 0,493998 | −0,010936 | [−0,0314 ; +0,0064] | 88,7 % |

Bootstrap de 2.000 remuestreos **agrupando por `identity_component_id`**, no por
imagen: el 66 % del pool está en componentes de duplicados de más de una imagen.

**Aplicar la transformación tonal como término de consistencia en el
preentrenamiento SSL empeora el macro-F1 en torno a 1,1-1,3 puntos.** En PAD el
intervalo excluye el cero.

### La estratificación se comporta como predice cada dataset

| | fragmentación de máscara | efecto de quitar las fragmentadas |
|---|---:|---|
| PAD | 22,9 % | atenúa el daño **un tercio** (−1,29 → −0,86 pp) |
| HAM | 4,8 % | **no cambia nada** (−1,12 → −1,09 pp) |

Eso es coherencia interna fuerte, no una coincidencia: el dataset con cola de
fragmentación mejora al quitarla y el que no la tiene, no. Si el daño fuera todo
de la máscara, HAM no debería empeorar, y empeora igual.

**Veredicto según la tabla predeclarada:** completo negativo con estratificado
también negativo significa que **el fallo no se explica por la fragmentación de
la máscara**. La hipótesis pasa al método.

### Cuidado de lectura que se impone

No se lee "el subconjunto perdió significancia" como "el efecto desapareció":
el subconjunto de PAD tiene 23 % menos casos, así que parte del ensanchamiento
del intervalo viene del tamaño. Lo comparable es el punto estimado.

---

## 2. Por qué el experimento es válido, que es lo que permite concluir algo

Un negativo solo sirve si se puede atribuir. Las cuatro verificaciones:

1. **Un solo factor, por código.** `paired_factor_gate` pasó con los campos que
   difieren siendo exactamente los cinco del conjunto de tratamiento:
   `adaptation_mode`, `tone_consistency_weight`, `transformed_data_revision`,
   `transformed_image_root`, `transformed_manifest_path`. Ni uno más.
2. **El control reprodujo el original BIT A BIT.** Mismo `backbone_sha256`
   (`09c15345...`) y mismo `coverage.counts` —el conteo de accesos de cada una
   de las 34.754 imágenes— a pesar de leer PNG donde el viejo leía JPG y de
   correr bajo otro envoltorio. Eso prueba de una sola vez que el espejo es
   pixel-idéntico sobre 96.000 extracciones, que el envoltorio nuevo no altera
   nada, y que el determinismo se sostiene.
3. **Sin colapso de representación.** Al mismo paso, `embedding_std` 0,022555
   (control) contra 0,022542 (tratado): difieren en la quinta decimal. Era el
   riesgo real, porque en SimSiam una pérdida que baja con la representación
   colapsada se ve igual que una que baja por aprender.
4. **Dilución declarada:** 34.315 de 34.754 transformadas (98,74 %). Los 439
   restantes separados por causa: 421 `degenerate_area` de `source_fit` y 18 de
   `pad_adapt`.

---

## 3. Qué prueba el resultado, y qué NO prueba

### Lo que se probó

Que **el término de consistencia tonal, aplicado en el preentrenamiento SSL con
peso 1.0, no ayuda y hace un daño pequeño pero medible.**

### Lo que NO se probó, y es la parte importante

**No se probó la hipótesis de CIRCLe.** Tres razones, todas estructurales:

1. **Falta el contrapeso supervisado.** La ecuación 1 de CIRCLe es
   `L_cls + lambda*L_reg`, y `L_cls` es entropía cruzada **supervisada**. Las dos
   partes tiran en direcciones opuestas: el regularizador le saca el tono a la
   representación y el término supervisado la obliga a conservar lo que hace
   falta para clasificar. **En SSL no hay etiquetas, así que nada protege la
   señal diagnóstica.** Y SimSiam ya es de por sí un objetivo de invariancia: se
   apiló una segunda sin contrapeso.
2. **El objetivo medido es el equivocado para la hipótesis.** El delta es sobre
   macro-F1 global. La hipótesis de la línea tonal es sobre **equidad entre
   tonos de piel**, no sobre exactitud agregada. Un método puede costar algo de
   exactitud global y reducir la brecha; eso sería positivo y **no se midió**.
3. **Se midió sobre el encoder más débil.** El modo pareado solo corre sobre
   SimSiam (`config.SIMSIAM_METHODS`), que es el peor de los tres medidos: PAD
   0,5973 contra 0,6438 de MoCo-v3-style.

### Y la referencia más cercana de la literatura es negativa

**Benčević 2024** (CMPB 245:108044) reporta que la conversión a CIELAB **falló**
como mitigación de sesgo. Nuestra vía no es la suya —transformamos dentro de Lab
en vez de convertir el espacio de entrada— pero este resultado **va en la misma
dirección que la única evidencia publicada comparable**. No es una sorpresa; lo
novedoso habría sido un positivo.

**No se compara contra magnitudes de CIRCLe**, porque no son transferibles: cuatro
diferencias medibles (coseno contra L2 al cuadrado, `.detach()` contra ecuación
simétrica, SSL contra supervisado, CIELAB contra StarGAN entrenado).

---

## 4. CIRCLe fiel a nivel de cabeza: delta cero, y el diagnóstico que lo explica

Corrido el 2026-11-05 con `src/dermssl/finetune/circle_torch.py`, el módulo del
propio proyecto que **nunca se había usado**. Mismo protocolo que el lector
publicado: rasgos normalizados por fila, pesos de clase balanceados,
`StratifiedGroupKFold` de 5 pliegues agrupado por `identity_component_id`, las
mismas tres semillas. Único factor: `reg_weight`. La cabeza arranca del mismo
estado en los dos brazos de cada pliegue.

| `reg_weight` | macro-F1 | sd |
|---|---:|---:|
| 0,0 | 0,458610 | 0,008625 |
| 0,1 (CIRCLe, valor por defecto del repo oficial) | 0,458357 | 0,008904 |
| **delta** | **−0,000253** | |

**Con el contrapeso supervisado el regularizador no cuesta nada. Tampoco gana
nada.** Y el motivo está medido:

```
cosine original vs tono, mediana: 0,990
569 de 1.134 por encima de 0,99
```

**El encoder congelado ya es casi invariante a la transformación.** La imagen se
desplaza 12,5° de ITA y la representación se mueve un 2,4 % en coseno. La mitad
de las imágenes no se mueve. Y como el backbone está congelado, lo único
entrenable es la cabeza lineal: sobre rasgos ya invariantes no tiene nada que
hacer.

**Advertencia de comparabilidad:** el 0,4586 absoluto **no es comparable** con el
0,6063 publicado. Ese usa `balancedLR` con grilla de C y selección anidada; este
usa una cabeza lineal con AdamW y 200 épocas. Solo el delta interno es válido.

---

## 5. Una hipótesis propia que los datos refutaron

Se propuso que el daño venía de **colores implausibles**: el código no acota
`b*` (`out_lab[..., 2] = lab[..., 2] + delta_b`, sin `clip`), el 44 % del pool
tiene `b*` de piel cerca de cero, y con dosis típica el desplazamiento de ±6,25
cruza el eje y deja piel azulada.

**Refutado**, midiendo el desplazamiento de rasgos contra cada candidato:

| correlación con el desplazamiento de rasgos | Spearman | p |
|---|---:|---:|
| **magnitud del desplazamiento ITA** | **−0,8391** | **6,9e-297** |
| píxeles con `b*` invertido | −0,1221 | 4,3e-05 |
| píxeles recortados por gama | −0,1644 | 3,3e-08 |

El encoder responde **proporcionalmente al tamaño del desplazamiento**, como uno
esperaría, y nada indica patología por color implausible. El cuartil que más
mueve la representación tiene `|ita|` mediana 19,5 contra 3,0 del que menos.

**Consecuencia:** acotar `b*` sigue siendo correcto por rigor del método —el
código debería acotarlo y no lo hace— pero **no es el arreglo del negativo**, y
presentarlo como tal sería faltar a la verdad.

---

## 6. Lo que falta medir, en orden

1. **La brecha por tono, sobre DDI.** Es la hipótesis real de la línea y no se
   midió. DDI es el único dataset con **fototipo asignado por consenso de dos
   dermatólogos**, no estimado, así que permite evaluar equidad sin depender del
   instrumento ITA que este proyecto ya descartó. Verificado e indexado en el
   clúster el 2026-11-05: 656 imágenes, hash contra el valor pineado, y los tres
   conteos por fototipo coincidiendo con los publicados (208 / 241 / 207).
2. **Curva dosis-respuesta.** Brazo de peso 0,1 corriendo. Convierte "empeoró"
   en "empeoró proporcionalmente a cuánto se empujó", que es una afirmación
   mucho más fuerte que un punto solo.
3. **CIRCLe con el backbone entrenable y supervisión.** La prueba real: ni el
   contraste SSL (sin etiquetas) ni el probe congelado (representación fija) la
   hacen. Requiere afinar el encoder con etiquetas más el regularizador.

### Trampa a evitar en la evaluación de DDI, ya medida

**Mapear DDI a las 7 clases de HAM reintroduce el mismo sesgo que `docs/64`
documenta para el ITA:**

| fototipo | mapean a HAM | cobertura | tasa maligna |
|---|---:|---:|---:|
| FST I-II | 127 / 208 | 61 % | 28,3 % |
| FST III-IV | 168 / 241 | 70 % | 31,0 % |
| **FST V-VI** | **73 / 207** | **35 %** | **13,7 %** |

El grupo más oscuro pierde dos tercios de sus casos y su tasa de malignidad cae
de 23,2 % a 13,7 %. La causa está en las no mapeadas: `mycosis-fungoides` (32),
`squamous-cell-carcinoma` (25), `subcutaneous-t-cell-lymphoma` — **malignidades
que la ontología de HAM no contiene**, más frecuentes en piel oscura.

**Por lo tanto la evaluación corre sobre las 656 con la lectura binaria
maligno/benigno**, que conserva el balance, y el mapeo a HAM queda como
secundario. `dermssl.ddi` ya lo anticipaba: su docstring declara la lectura
maligno/benigno como **primaria**.

**Y esto generaliza el aporte de `docs/64`.** Ya no es un hallazgo sobre el ITA:
son **dos filtros independientes** —medibilidad del ITA y mapeabilidad a HAM—
que sesgan la población en direcciones distintas. La regla es que **cualquier
filtro previo a un análisis de equidad debe auditarse por su interacción con el
atributo protegido y con la etiqueta**. Dos instancias en un día no es
coincidencia.
