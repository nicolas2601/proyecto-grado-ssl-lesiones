---
name: crear-figuras-cientificas
description: "Usar para diseñar, revisar o integrar figuras científicas no tabulares: diagramas de métodos, arquitectura, flujo de datos, protocolo o comparación conceptual. Para gráficas cuantitativas usa crear-graficas-resultados."
---

# Crear figuras científicas

Diseña figuras que expliquen el método o el flujo de trabajo del proyecto.
Esta skill cubre diagramas y esquemas; no dibuja resultados cuantitativos ni
sustituye tablas.

## Flujo

1. **Identifica destino y propósito.** Define documento, audiencia, tamaño
   final y afirmación que debe aclarar. Revisa el texto, implementación y
   figuras existentes antes de diagramar.
2. **Verifica cada componente.** Contrasta nombres de módulos, entradas,
   salidas, orden, pérdidas, particiones y conexiones con el código y el
   protocolo vigentes. No agregues operaciones, componentes o conclusiones
   porque sean habituales en otros métodos.
3. **Prepara una ficha de figura.** Especifica propósito, fuente y versión,
   componentes que aparecerán, flujo de izquierda a derecha o por etapas,
   etiquetas/unidades necesarias, perfil visual, tamaño, formato y caption.
   Distingue lo observado de lo conceptual.
4. **Construye una figura editable y reproducible.** Usa el formato vectorial
   o fuente declarativa del proyecto (por ejemplo SVG o Mermaid, según el
   entregable). Guarda la fuente junto a las demás figuras del documento,
   siguiendo su estructura actual. Evita dejar como única fuente un PNG
   rasterizado.
5. **Comprueba integridad científica.** Etiqueta como “esquema” lo que sea
   conceptual o ilustrativo; no lo presentes como imagen, arquitectura
   ejecutada o resultado experimental real. Conserva la procedencia de
   imágenes y recursos visuales. No inventes muestras, valores, pacientes,
   datos clínicos ni observaciones.
6. **Renderiza e integra.** Revisa el archivo exportado al tamaño final; comprueba
   texto, flechas, alineación, contraste, tipografía, enlaces y caption.
   Confirma que la figura coincide con la descripción del método y cita sus
   fuentes.

## Estilo y alcance

- Sigue el perfil del manuscrito o entrega actual. La skill
  `figuras-tesis-ssl` es una guía de referencia para figuras de tesis SSL;
  sus convenciones visuales específicas solo aplican cuando la figura va a
  esa tesis y el repo las confirma.
- Mantén etiquetas concisas, jerarquía visual clara y contraste suficiente.
  No uses color como único medio para distinguir categorías.
- Usa fotografías o imágenes clínicas solo si son datos reales, están
  autorizadas para ese uso y se conserva su procedencia. No generes imágenes
  sintéticas que puedan parecer casos clínicos reales.
- Una gráfica basada en métricas debe seguir
  `.claude/skills/crear-graficas-resultados/SKILL.md`; una tabla numérica debe
  seguir `.claude/skills/actualizar-tablas-resultados/SKILL.md`.

## Entrega

Indica el propósito, fuentes verificadas, formato editable/exportado, lugar de
integración, prueba visual al tamaño final y qué partes son esquemáticas.
