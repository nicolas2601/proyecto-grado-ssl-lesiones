# Especificación de mitigación CIRCLe

## Propósito

Mitigar disparidad sin convertir CIRCLe en estimador FST.

## Requisitos

### Requirement: Comparación escalonada de CIRCLe

Ante una puerta de equidad fallida, el sistema MUST comparar referencia, solo transformación y transformación con consistencia CIRCLe; CIRCLe MUST NOT identificar ni validar FST. Solo SHALL promoverse si supera las puertas de éxito.

#### Scenario: Promoción bloqueada

- GIVEN una puerta fallida o una salida CIRCLe
- WHEN se informa o decide su promoción
- THEN se conservan las tres variantes y no se afirma FST ni promoción
