# W3: fuentes, diferencias metodológicas y preparación segura

## Problema

Los tres candidatos acordados no son intercambiables con ejecutables listos.
SLED depende de un entorno antiguo y tiene problemas observables; el pipeline
actual ya utiliza decisiones inspiradas en USL-Net; RPI-Net no tiene una
implementación verificada en las fuentes consultadas.

## Cambio

Registrar fuentes y revisiones, riesgos y diferencias, y fases de comparación
en docs/41 y un JSON de puertas. Añadir un preflight que verifica el conjunto
fijo de originales y reporta preparación sin ejecutar métodos ni entrenamientos.

## Alcance y validación

Preservar los tres candidatos; no sustituir RPI-Net ni afirmar réplica exacta.
Pruebas de cambios de entradas, aislamiento de anotaciones, duplicados y puertas.
Ejecutar la preparación con el manifiesto W0 verificado en la copia aislada.
Los resultados de Dice y exactitud requieren evaluación separada posterior.
