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

## Estilo

- Sigue el estilo ya definido en la tesis, paper o informe que recibirá la
  gráfica. Si se aplica la skill de proyecto `figuras-tesis-ssl`, úsala como
  guía para la tesis; no traslades su paleta, tipografía ni perfil a otros
  documentos sin evidencia de que corresponde.
- Elige colores distinguibles y aptos para daltonismo; no dependas solo del
  color para separar series.
- Fija el tamaño físico de salida y comprueba el tamaño real de letra y la
  fuente disponible. Prioriza formato vectorial para publicación cuando sea
  compatible con la entrega.
- No uses efectos decorativos que dificulten la comparación o sugieran
  precisión que los datos no tienen.

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
