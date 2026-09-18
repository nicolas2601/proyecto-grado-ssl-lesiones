# Especificación de evaluación de equidad por tono

## Propósito

Medir desempeño, incertidumbre y equidad de forma reproducible.

## Requisitos

### Requirement: Métricas comparables e incertidumbre

El sistema MUST informar accuracy, balanced accuracy, macro-F1, macro-precision, macro-recall, AUC, ECE y matriz de confusión globales; F1, recall, accuracy, AUC por tono, EOD, NAR, brecha máxima e intervalos bootstrap.

#### Scenario: Grupo limitado

- GIVEN un grupo bajo el tamaño mínimo configurado
- WHEN finaliza la evaluación
- THEN informa sus métricas e incertidumbre como limitadas

### Requirement: Puerta de éxito de equidad

Una variante solo SHALL declararse exitosa si reduce la brecha de equidad y su caída absoluta frente a la base comparable no supera 0,02 en macro-F1 global ni AUC global. MUST emitir en JSON manifiestos, métricas, intervalos, puertas, probe y semillas; CSV MAY acompañar auditoría.

#### Scenario: Mejora válida

- GIVEN brecha reducida, caídas dentro de 0,02 y evidencia JSON completa
- WHEN se evalúa la puerta
- THEN puede declararse exitosa y reproducible

#### Scenario: Regresión global

- GIVEN caída mayor a 0,02 o JSON incompleto
- WHEN se evalúa la puerta
- THEN no se declara éxito
