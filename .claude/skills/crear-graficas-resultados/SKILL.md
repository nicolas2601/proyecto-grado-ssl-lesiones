---
name: crear-graficas-resultados
description: "Usar para crear o revisar gráficas cuantitativas a partir de resultados del proyecto SSL: comparar métodos, métricas, semillas, particiones o cambios entre corridas. No usar para diagramas de arquitectura o para regenerar tablas."
---

# Crear gráficas de resultados

Produce gráficas reproducibles y fieles a los datos. Una gráfica comunica una
pregunta concreta; no sustituye la fuente tabular ni el registro de corrida.

## Flujo

1. **Define la pregunta y el uso.** Identifica si la gráfica es para análisis
   exploratorio, tesis, informe o artículo. Lee el manuscrito y las figuras
   existentes para aplicar el estilo correspondiente; no impongas el perfil
   visual de una entrega a otra.
2. **Verifica los datos.** Localiza JSON/CSV autoritativos, su versión, commit
   y procedencia. Inspecciona el script de evaluación y la definición de la
   métrica. Registra método, partición, unidad, denominador, número de casos y
   semillas. No deduzcas datos individuales a partir de agregados.
3. **Especifica la gráfica antes de dibujarla.** Anota propósito, variables,
   series/grupos, escala, unidades, fuente, tratamiento de valores faltantes,
   incertidumbre disponible, tamaño final y leyenda/caption. Usa intervalos
   solo si están calculados y presentes en la fuente o se pueden derivar
   válidamente de observaciones disponibles.
4. **Usa el flujo reproducible del repo.** Busca scripts de gráficas y
   generadores vigentes antes de crear otro. Si una gráfica existente se
   genera desde JSON, ejecuta o amplía ese script; no edites la imagen exportada
   a mano. Mantén el código, parámetros y rutas de salida versionables.
5. **Representa los datos honestamente.** Etiqueta partición, unidad y escala;
   evita recortes o transformaciones que exageren diferencias. No inventes
   valores, barras de error, intervalos, muestras, clases ni resultados.
   Distingue explícitamente resultados observados de esquemas ilustrativos.
6. **Revisa la salida.** Renderiza SVG/PDF o el formato solicitado y verifica
   legibilidad al tamaño final, ejes, marcas, leyenda, colores, fuente y
   correspondencia con la fuente de datos. Contrasta los valores graficados
   con el JSON/CSV y cita la ruta del script y de los datos.

## Estilo y control visual

- Para el Anteproyecto o la tesis SSL, aplica el perfil visual de `CLAUDE.md` §9. Mantén fondo blanco, Times New Roman, texto sin negrita general y la paleta cappuccino/tonos claros; no uses Purple Teal como paleta del proyecto. Reserva Fitzpatrick a gráficas de tono/fototipo.
- Para otros artículos, adopta la guía visual de la revista o manuscrito de destino. No mezcles perfiles entre entregas.
- Define el tamaño físico según el ancho de columna o página final. Ajusta tipografía, etiquetas, marcas, grosores y marcadores para que sean legibles al tamaño impreso.
- Elige rangos, marcas y aspecto de ejes que representen los datos sin exagerar diferencias. Usa aspecto igual solo cuando las unidades y la pregunta visual lo justifiquen.
- Mantén colores y codificación redundante consistentes entre figuras; evita depender únicamente del color para distinguir grupos.
- Mantén cuadrículas discretas, leyendas compactas y sin solapamientos. Conserva los ajustes de composición en el script reproducible.
- Exporta PDF/SVG vectorial cuando el destino lo admita; para raster, fija resolución suficiente. Revisa e itera con la gráfica insertada en la plantilla real.

Usa [BeautifulFigures](https://github.com/AndreyChurkin/BeautifulFigures/blob/main/01_main_simple_example/Python/README.md) y su [script de Matplotlib](https://github.com/AndreyChurkin/BeautifulFigures/blob/main/01_main_simple_example/Python/scripts/beautiful_figure_example.py) como referencia de composición y calidad de publicación. Para el Anteproyecto, adapta esos parámetros a su perfil de `CLAUDE.md` §9; no copies la paleta del ejemplo ni retoques una exportación fuera del script.
## Límites

- Las tablas numéricas se regeneran con
  `.claude/skills/actualizar-tablas-resultados/SKILL.md`.
- Los diagramas de método, arquitectura y flujo corresponden a
  `.claude/skills/crear-figuras-cientificas/SKILL.md`.
- No registres ni interpretes la corrida en esta skill; usa
  `.claude/agents/documentador-corridas.md`.

## Entrega

Resume la pregunta visual, datos y partición usados, archivo de generación,
formatos exportados, verificación del render y limitaciones de interpretación.
