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

- Sigue el estilo del documento de destino. La skill `figuras-tesis-ssl` solo guía la tesis SSL; no traslades su paleta, tipografía o perfil a otros documentos sin evidencia.
- Define el tamaño físico según el espacio final (por ejemplo, una columna o página completa). Ajusta fuentes, etiquetas, marcas, líneas y marcadores para que sigan legibles al insertar la gráfica.
- Elige rangos, marcas y aspecto de ejes que representen los datos sin exagerar diferencias. Usa aspecto igual solo cuando las unidades y la pregunta visual lo justifiquen.
- Mantén una paleta sobria y consistente dentro del mismo manuscrito: asigna el mismo color a cada grupo en todas las figuras. Combina color con forma o estilo de línea; considera daltonismo.
- Mantén las cuadrículas discretas y detrás de los datos. Ubica la leyenda sin tapar observaciones y con márgenes compactos.
- Conserva los ajustes de diseño en el script reproducible; no retoques manualmente la exportación de forma que el resultado diverja de su fuente.
- Exporta PDF o SVG vectorial cuando el destino lo admita. Para raster, fija resolución suficiente para el tamaño final.
- Revisa e itera con la gráfica colocada en la plantilla real, no solo en la ventana del entorno de desarrollo.

Los criterios de tamaño, tipografía, ejes, cuadrícula, leyenda, paleta coherente y revisión en plantilla se inspiran en el [README de BeautifulFigures](https://github.com/AndreyChurkin/BeautifulFigures/blob/main/01_main_simple_example/Python/README.md) y su [ejemplo de Matplotlib](https://github.com/AndreyChurkin/BeautifulFigures/blob/main/01_main_simple_example/Python/scripts/beautiful_figure_example.py). La paleta Purple Teal y sus parámetros concretos son ejemplos, no una convención automática para este proyecto.

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
