# 61 — Decisión pendiente: qué se entrega como algoritmo de la tesis

**Estado: decisión abierta.** No la toma una sesión de trabajo. Corresponde a Nicolás Moreno,
María Paula Saavedra, el director Andrés Jerez y la asesora Karen Sánchez.

Documento preparado el 2026-10-31 para llevar a esa conversación. Reúne lo medido, lo que cada
opción implica y lo que falta saber antes de decidir. No decide por nadie.

---

## 1. El problema, en una frase

**El mejor clasificador que el proyecto ha medido no es autosupervisado, y el objetivo general
de la tesis exige aprendizaje autosupervisado.**

Objetivo general, texto literal de `Anteproyecto/objetivos.tex`:

> Desarrollar un algoritmo de inteligencia artificial **basado en aprendizaje autosupervisado**
> para clasificar lesiones cutáneas a partir de imágenes dermatológicas, orientado al apoyo del
> análisis clínico en el contexto del departamento de Santander.

No es un problema de ingeniería. Es un problema de alcance declarado, y por eso no lo resuelve
otra corrida.

---

## 2. Lo que está medido

### 2.1 Advertencia metodológica que condiciona todo lo demás

Existen **dos protocolos distintos y no comparables entre sí**. Mezclar sus cifras en una misma
tabla sería un error que invalidaría la comparación.

| | Protocolo v7 | Protocolo campaña v3 |
|---|---|---|
| Conjunto | PAD-UFES-20 completo | PAD-UFES-20 `target_adapt` |
| n | 1.141 imágenes / 720 pacientes | 1.134 imágenes |
| Clases | 3 (`bcc`, `mel`, `nv`) | 6 (diagnóstico PAD) |
| Lectura | Zero-shot, sin adaptación | CV anidada agrupada por paciente (3 semillas CV x 5 outer x 3 inner) |
| Naturaleza | Evaluación reservada | **Desarrollo / transductivo** |
| Fuente | `resultados/clasificacion/pad_ufes/README.md` | Engram del equipo `#2224`, `#2225`, `#2266` |

Ninguno de los dos es validación clínica.

### 2.2 Protocolo v7 — el SSL propio pierde en fotografía de celular

Bootstrap pareado agrupado por paciente. VERIFICADO en
`resultados/clasificacion/pad_ufes/README.md`.

| Contraste | Δ Macro-F1 | IC 95 % pareado |
|---|---:|---|
| B* s42 − DINOv2 público | **−0,1070** | [−0,1350; −0,0789] |
| B* s43 − DINOv2 público | **−0,1312** | [−0,1628; −0,0984] |
| A* s42 − DINOv2 público | −0,0859 | [−0,1112; −0,0572] |
| A* s43 − DINOv2 público | −0,1085 | [−0,1351; −0,0825] |

El ECE también favorece al público (+0,1128 y +0,0978, IC excluye el cero).

En HAM10000 (dermatoscopia, n = 2.004) la relación se invierte: A* 0,6464, B* 0,6606, público
0,6158, con P1 y P3 cumplidas en las tres semillas. **La ventaja del SSL propio existe en
dermatoscopia y no transfiere a fotografía clínica.**

### 2.3 Campaña v3 — el encoder es la palanca

NO VERIFICADO por esta sesión: cifras registradas en el engram del equipo, no recalculadas aquí.

| Configuración | Macro-F1 PAD | ¿SSL puro? |
|---|---:|---|
| Baseline SSL, media de 3 semillas | 0,597 (DE 0,010) | Sí |
| **Barlow Twins** | **0,633** (+0,035 [+0,008; +0,062]) | Sí |
| **VICReg** | **0,638** (+0,041 Welch [+0,019; +0,063]) | Sí |
| DINOv2-giant | 0,613 | No (pesos públicos) |
| OpenCLIP L/14 | 0,613 | No (imagen-texto) |
| **SigLIP2-so400m, solo imagen** | **0,6668** | **No (imagen-texto)** |
| **SigLIP2-so400m + metadatos** | **0,7027** | **No** |
| SigLIP2-base + metadatos | 0,7029 | No |
| Clásico, no profundo | 0,31 | No |

Criterio declarado antes del dato: media de 3 semillas, ΔPAD ≥ +0,02 y ΔHAM ≥ −0,01. VICReg y
Barlow Twins **lo cumplen**.

**Pero son provisionales**, y la palabra es del propio registro: 3 semillas contra 3, y prueba de
permutación con **p mínimo = 0,10**. No hay significancia por permutación. Nicolás decidió no
ampliar a las semillas 45 y 46.

Brazos nulos o peores: DINO+iBOT, MULAN, BYOL, OOB, freeze4, 336, crop 0.4, destilación (empeora),
recetas v1, SkinDisNet. Conclusión registrada: todas las palancas de SimSiam probadas caen dentro
de ±0,006 del baseline.

### 2.4 El techo honesto

Literatura con split por paciente: **~0,72 macro-F1**. El 0,80 que circula **no es reproducible**
para 6 clases. Fuente: `#2224`, `#2230`.

SigLIP2 + metadatos (0,7027) está prácticamente en ese techo. VICReg (0,638) está a 0,08.

### 2.5 Qué sirve hoy la aplicación

El servicio desplegado en `derm.nicolasmoreno.site`, que consume la app publicada en
`skin.tikno.pro`, responde con `release_arm: B71`. VERIFICADO con `curl` el 2026-10-30.

**B71 es el brazo que pierde en fotografía de celular.** El servicio también expone A71, así que
cambiarlo es configuración, no desarrollo.

---

## 3. Las opciones

### Opción A — Mantener el entregable SSL actual

Se entrega A*/B* de v7, o el baseline de la campaña v3. Rinde alrededor de 0,60–0,61 en PAD.
SigLIP2 se documenta como **comparador de techo**, no como entregable.

| | |
|---|---|
| **Objetivo general** | Se cumple sin discusión |
| **Objetivo 2** | Cerrado con v7, que ya tiene protocolo registrado y veredictos escritos |
| **Objetivo 3** | El resultado negativo de PAD se reporta como limitación central |
| **Objetivo 4** | La app debe servir A71, y declarar explícitamente el alcance: dermatoscopia, no celular |
| **Trabajo restante** | Mínimo. Ya está casi todo medido |
| **Riesgo** | Que el jurado pregunte por qué no se usó el mejor método disponible, habiéndolo medido |

Argumento defendible: el objetivo era **demostrar el aporte del SSL**, no alcanzar el máximo
rendimiento posible. En dermatoscopia el aporte está demostrado con IC que excluye el cero.

### Opción B — VICReg o Barlow Twins como entregable SSL

Se entrega el mejor brazo SSL de la campaña v3 (0,638 o 0,633), con SigLIP2 como comparador.

| | |
|---|---|
| **Objetivo general** | Se cumple |
| **Objetivo 2** | Mejora medible sobre el baseline SSL, con criterio pre-registrado |
| **Objetivo 3** | **Requiere cerrar la confirmación**: hoy p = 0,10 por permutación |
| **Objetivo 4** | Exige exportar el nuevo encoder y rehacer el bundle de inferencia |
| **Trabajo restante** | Semillas 45 y 46, o una prueba de confirmación con más potencia. Luego export y despliegue |
| **Riesgo** | Que al ampliar semillas el efecto se diluya y haya que volver a la opción A |

Es la única opción que mejora el resultado **sin salir del alcance declarado**.

### Opción C — Cambiar el entregable a SigLIP2 + metadatos

Se entrega el mejor resultado medido (~0,70).

| | |
|---|---|
| **Objetivo general** | **No se cumple como está escrito.** SigLIP2 es imagen-texto, no SSL |
| **Objetivo 2** | Habría que reformular: el algoritmo ya no sería autosupervisado |
| **Objetivo 3** | El SSL pasaría a ser un brazo comparado, no el propuesto |
| **Objetivo 4** | Export nuevo, y los metadatos exigen que la app los capture |
| **Trabajo restante** | Alto: cambiar el encoder, rehacer el pipeline de inferencia, rediseñar la captura de la app |
| **Riesgo** | Alterar el objetivo general a esta altura requiere aprobación del director, y deja sin sustento meses de trabajo en SSL |

**Caveat adicional que no se puede omitir:** la ganancia de los metadatos (+0,047 a +0,049)
depende de datos clínicos que la aplicación hoy no captura. Un usuario con el teléfono no aporta
edad, localización anatómica ni antecedentes. El 0,70 **no es el rendimiento esperable de la app**.

### Opción D — Reformular el objetivo general

Ampliar el objetivo a "aprendizaje con supervisión limitada" o equivalente, lo que admitiría
tanto SSL como modelos imagen-texto congelados.

| | |
|---|---|
| **Coste** | Modificar el anteproyecto ya aprobado. Requiere aval del director y de la asesora |
| **Beneficio** | Honestidad: el hallazgo real del proyecto es que el encoder importa más que la receta SSL |
| **Riesgo** | Puede leerse como ajustar el objetivo al resultado, que es exactamente lo que no se debe hacer |

Se incluye porque existe, no porque se recomiende sin discusión previa.

---

## 4. Lo que falta saber antes de decidir

Ordenado por cuánto cambia la decisión:

1. **El resultado de la cola del 2026-10-29.** Siete brazos quedaron corriendo desatendidos:
   `dino-ibot-s43`, `dino-ibot-s44`, `dino-ibot-cb-s42`, `moco-v3-s42`, `simsiam-cb-s42`,
   `masked-pixel-s42`, `rotation-s42`. Si DINO+iBOT con tres semillas cambia el cuadro, cambia la
   decisión. **No hay observación posterior en el engram del equipo.**
2. **Si VICReg y Barlow sobreviven a más semillas.** Define si la opción B es viable.
3. **Qué opina Karen.** Pidió explícitamente DINO+iBOT con 3 semillas, tabla de precisión por
   clase y muestreo balanceado. Su criterio sobre qué constituye un entregable SSL válido es
   determinante.
4. **Si la cohorte externa es exigible.** DDI requiere que Nicolás acepte el RUA. Sin cohorte
   externa independiente, ninguna opción puede afirmar generalización.

---

## 5. Recomendación

**Opción B, condicionada; y si no se confirma, opción A.**

Razones:

1. **Es la única que mejora el resultado sin salir del alcance declarado.** No obliga a tocar el
   objetivo general ni a explicar por qué se abandonó el SSL después de meses.
2. **La confirmación es barata comparada con lo que está en juego.** Dos semillas más sobre un
   brazo ya implementado, contra rehacer el encoder, el bundle de inferencia y la captura de la
   app.
3. **El resultado negativo de PAD no desaparece en ninguna opción**, así que conviene elegir la
   que mantiene la coherencia del relato: el SSL aporta en dermatoscopia, no transfiere a
   fotografía clínica, y eso se reporta como hallazgo.

Independientemente de la opción elegida, **tres cosas hay que hacer igual**:

- **Cambiar el brazo que sirve la app**, o justificar por escrito por qué sirve B71 pese a perder
  en fotografía de celular. Es lo que hoy ve un usuario real.
- **Documentar SigLIP2 como comparador de techo.** Medirlo y reportarlo es un resultado valioso
  del proyecto, se entregue o no. Omitirlo sería ocultar evidencia propia.
- **Declarar que las cifras provisionales lo son.** VICReg con p = 0,10 no se presenta como
  confirmado.

---

## 6. Lo que este documento no hace

- No recalcula ninguna cifra. Las de la campaña v3 provienen del engram del equipo y están
  marcadas como NO VERIFICADAS por esta sesión.
- No cambia ningún gate de ejecución.
- No decide. Reúne lo que hay para que la decisión se tome con la evidencia a la vista.
