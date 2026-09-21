# Especificación de evidencia FST

## Propósito

Evidencia FST provisional, trazable y no clínica.

## Requisitos

### Requirement: Ingesta y cobertura versionadas

El sistema MUST versionar un manifiesto de 68.472 imágenes: HAM10000 (10.015), ISIC2019 resized (25.331) e ISIC2020 resized (33.126), con identificador, procedencia, división y cobertura.

#### Scenario: Cobertura auditada

- GIVEN los conjuntos contienen imágenes sin metadatos o máscara válida
- WHEN se genera el manifiesto
- THEN registra el total, la procedencia y la cobertura incompleta

### Requirement: FST y máscaras provisionales trazables

El sistema MUST trazar productor, versión y calidad de cada máscara automática o pseudo-máscara; MUST distinguir FST de imagen completa y enmascarada como provisional. ISIC2018 MAY emplearse solo ante cobertura o calidad insuficiente, para CNN auxiliar de segmentación; MUST NOT ser dato principal, clasificador FST ni sustituto de CIRCLe.

#### Scenario: Máscara insuficiente

- GIVEN una máscara bajo el criterio de calidad y cobertura insuficiente
- WHEN se procesa FST
- THEN la salida no se valida y el fallback queda limitado a segmentación
