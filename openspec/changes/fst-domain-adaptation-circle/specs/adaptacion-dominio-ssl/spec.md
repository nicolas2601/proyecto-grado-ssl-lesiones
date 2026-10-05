# Especificación de adaptación de dominio SSL

## Propósito

Adaptación SSL comparable y sin filtración de etiquetas.

## Requisitos

### Requirement: Transformación preservadora de lesión

El sistema MUST evaluar transformaciones de tono acotadas con máscara trazable y registrar su preservación de lesión según el criterio configurado.

#### Scenario: Transformación rechazada

- GIVEN una transformación que no preserva la lesión
- WHEN se valida
- THEN se excluye de la comparación elegible

### Requirement: Adaptación SSL comparable

El sistema MUST comparar DINOv2 SSL como referencia y ResNet50 ImageNet como control en base, adaptación y adaptación más CIRCLe, con divisiones comparables. El preentrenamiento SSL MUST NOT consumir etiquetas diagnósticas ni de tono.

#### Scenario: Comparación inválida

- GIVEN una variante con etiquetas sensibles o división distinta
- WHEN se prepara la comparación
- THEN no se declara resultado válido
