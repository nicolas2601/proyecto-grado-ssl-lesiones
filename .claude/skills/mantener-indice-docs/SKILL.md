---
name: mantener-indice-docs
description: Usar para revisar o actualizar el índice de `docs/README.md` frente a los documentos numerados reales (`docs/00-*.md` a `docs/NN-*.md`), detectando documentos nuevos sin indexar y colisiones de numeración. Invocar con frases como "actualiza el índice de docs", "revisa si el índice de documentación está al día", o cada vez que se vaya a agregar (o se acabe de agregar) un documento numerado nuevo bajo `docs/`.
---

Flujo de verificación con una sola edición puntual y confirmada al final — no
un rediseño del índice ni una reclasificación silenciosa de "vigente" vs
"superado". El índice (`docs/README.md`) es una tabla manual: una fila por
documento, en orden numérico, con una columna Estado. Esta skill la mantiene
sincronizada con los archivos reales de `docs/`, pero nunca decide por su
cuenta qué significa un documento nuevo.

## Por qué existe esta skill

El índice ya quedó desactualizado una vez de forma real en este proyecto: al
escribir esta skill, `docs/README.md` solo documenta hasta el doc `30` y
luego salta a `35`, `48`, `49`, `50` — y le faltan **21 archivos numerados que
sí existen** en `docs/` (`29`, `31`, `33`, `34`, `36` a `47`, `51` a `55`),
confirmado comparando `Glob("docs/[0-9]*.md")` contra los enlaces reales de
la tabla. Además hay una colisión real sin resolver: `docs/54-banco-uslnet-
preparacion-v8.md` y `docs/54-usl-net-banco-pesos-v8.md` comparten el prefijo
`54` y tratan temas distintos (banco de datos de preparación vs. banco de
pesos), y ninguno de los dos se referencia a sí mismo como vigente/superado
ni menciona al otro (confirmado con `Grep` sobre ambos archivos — no hay
ningún texto de "superado por" ni "vigente" en ninguno). Esta cifra de 21
documentos faltantes y la colisión de `54` son el estado verificado al
escribir esta skill, no un hecho permanente: **siempre volvé a derivarlo con
`Glob`/lectura real del `README.md` en cada ejecución**, nunca asumas que
esta lista sigue vigente.

Nota aparte, no es un problema a corregir aquí: el número `20` también tiene
tres archivos (`20-esquema-fst-manifest.md`, `20-falsear-los-guards.md`,
`20-validacion-externa-fst-fitzpatrick17k.md`), pero ese caso **ya está
resuelto** en el índice — la tabla los lista como filas distintas `20a`,
`20b`, `20c`, cada una enlazando a su archivo real. Tres archivos con el
mismo número no es automáticamente una colisión sin resolver: primero
revisá si el índice ya los distingue con sufijos de letra antes de tratarlo
como el mismo caso que `54`.

## Pasos

1. **Listar los documentos numerados reales.**
   `Glob("docs/[0-9]*.md")` y extraé el prefijo numérico de cada nombre de
   archivo (la parte antes del primer guion, ej. `51` de
   `51-cierre-preparacion-y-bloqueos-lanzamiento-v8.md`). Agrupá por número:
   la mayoría de los números tienen un solo archivo; algunos (como `20`)
   tienen varios por diseño ya resuelto.

2. **Leer el índice real, no asumir su forma.** `Read("docs/README.md")`
   completo. La tabla principal tiene columnas `Doc | Tema | Estado`, con el
   número de doc como link markdown al archivo (`[51](51-...md)`) o, para
   casos especiales, como texto plano con el link embebido en la celda Tema
   (ver fila `17`, que es un `.docx` sin `.md` correspondiente, y fila `25`,
   que documenta explícitamente "No existe: el número quedó sin usar."). No
   inventes una forma distinta de tabla: calcá la que ya está ahí.

3. **Cruzar Glob contra el índice.** Para cada número agrupado en el paso 1,
   confirmá si al menos uno de sus archivos aparece como link target en
   alguna fila de la tabla (buscá el nombre de archivo exacto entre
   paréntesis). Si ningún archivo de ese número aparece, es un **doc faltante
   del índice**. Si el número tiene más de un archivo, revisá también si el
   índice ya los distingue por separado (como `20a/20b/20c`) antes de pasar
   al paso 4.

4. **Detectar colisiones de numeración sin resolver.** Un número con más de
   un archivo que *no* está ya resuelto con sufijos de letra en el índice
   (como el caso real `54-banco-uslnet-preparacion-v8.md` /
   `54-usl-net-banco-pesos-v8.md`) es una colisión. **Nunca decidas cuál es
   "el 54 real" ni renumbres ningún archivo** — otros documentos, commits o
   referencias cruzadas pueden apuntar al número actual, y renumerar es una
   decisión humana. Reportá la colisión con ambas rutas completas y detenete
   ahí: no la edites en el índice como si tuviera una sola fila correcta.

5. **Para cada doc faltante (no colisión), buscar una pista de clasificación
   antes de preguntar.** `Grep` dentro del propio archivo por patrones como
   `superad[oa] por`, `reemplazad[oa] por`, o un encabezado que diga
   explícitamente "vigente". Si encontrás una referencia cruzada clara (por
   ejemplo, un doc que dice en su propio texto "esto superó a docs/NN"),
   proponé la clasificación citando la línea exacta donde la viste. Si no
   encontrás nada —como pasó al escribir esta skill con los dos docs `54`,
   donde `Grep` no encontró ninguna mención de "superado"/"vigente" en
   ninguno de los dos— **no inventes una clasificación**: marcá ese doc como
   "sin clasificar, necesita confirmación humana o de la sesión que lo creó"
   y seguí con el siguiente.

6. **Reportar antes de editar.** Mostrá una lista con tres grupos: (a) docs
   faltantes con clasificación sugerida y su evidencia textual citada, (b)
   docs faltantes sin pista suficiente, (c) colisiones de numeración sin
   resolver. Para (c), el reporte es el final del trabajo de esa skill en esa
   corrida — no se edita nada para esas filas.

7. **Editar solo lo confirmado.** Agregá filas a la tabla de
   `docs/README.md` únicamente para los docs del grupo (a) que el usuario o
   la sesión invocadora ya confirmó explícitamente como "Vigente" o "Superado
   por <doc>" (ya sea porque lo confirmaron ahora en respuesta al reporte, o
   porque la clasificación sugerida en el paso 5 tenía evidencia textual
   directa y fue aceptada). Insertá la fila en su posición numérica correcta
   dentro de la tabla, respetando el formato exacto de columnas y de link
   (`| [NN](NN-slug.md) | <tema en una línea> | <Estado> |`). No reescribas
   ni reordenes filas existentes, no toques la sección "Otros documentos" ni
   la "Regla de actualización" salvo que el usuario lo pida explícitamente.

## Qué NO hacer

- No renumeres ni muevas ningún archivo de `docs/` para resolver una
  colisión — eso es una decisión humana, nunca autónoma de esta skill.
- No le asignes "Vigente" o "Superado por X" a un doc faltante basándote en
  su fecha de nombre, su posición en el repo, o una suposición sobre el
  contenido — solo evidencia textual explícita (paso 5) o confirmación
  humana directa.
- No edites `docs/README.md` para las filas en estado de colisión sin
  resolver (paso 4): esas se reportan, no se escriben.
- No asumas que la lista de 21 docs faltantes y la colisión del `54` citadas
  arriba siguen vigentes sin volver a correr `Glob` y releer el `README.md`
  real en el momento de la ejecución.
