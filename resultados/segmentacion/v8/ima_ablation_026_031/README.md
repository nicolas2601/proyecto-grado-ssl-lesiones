# Ablación IMA++ v8: incertidumbre y decisión para el piloto

Reporte generado desde los casos congelados; análisis exploratorio post hoc.

Delta = IMA++ − control. IC 95 % por bootstrap pareado de imágenes,
condicional a los seis modelos. No acredita independencia por paciente.

| Estrato | N | Δ s42 (pp) | Δ s43 (pp) | Δ s44 (pp) | Δ medio [IC 95 %] (pp) |
|---|---:|---:|---:|---:|---:|
| overall | 2594 | +1.19 | +1.87 | +0.39 | +1.15 [+0.75; +1.56] |
| <1% | 48 | -1.76 | -7.90 | -5.76 | -5.14 [-8.23; -2.07] |
| 1-2% | 149 | -4.05 | -8.84 | -8.91 | -7.27 [-8.83; -5.70] |
| 2-5% | 427 | -0.54 | -1.10 | -3.74 | -1.80 [-3.19; -0.27] |
| 5-10% | 432 | +1.42 | +1.75 | -0.87 | +0.77 [-0.35; +1.99] |
| >=10% | 1538 | +2.21 | +4.08 | +2.99 | +3.09 [+2.79; +3.40] |

Los IC por semilla están en `analysis.json`. Se muestran todos los estratos;
los intervalos no están ajustados por multiplicidad. El promedio de semillas
no representa una estimación de variabilidad de futuros entrenamientos.

La sensibilidad por cambio de signo entre semillas tiene solo ocho asignaciones; p bilateral global = 0.25. Se informa como sensibilidad exploratoria, no como prueba de superioridad.

La mejora global convive con una caída del Dice en los tres estratos menores
al 5 % en las tres semillas. IMA++ reduce contaminación del pool de piel, pero
la mejora de ese objetivo no implica mejor delimitación de lesiones pequeñas.

## Controles de máscara y CIELAB

| Run | Dice ISIC | Dice PH2 val | Degeneradas | Retención offline | Contaminación p95 antes/después |
|---|---:|---:|---:|---:|---:|
| 026 | 0.8320 | 0.9375 | 6 | 88.8% | 0.3829 / 0.1398 |
| 027 | 0.8439 | 0.9336 | 4 | 90.1% | 0.3555 / 0.1328 |
| 028 | 0.8180 | 0.9348 | 9 | 88.4% | 0.3940 / 0.1397 |
| 029 | 0.8367 | 0.9348 | 1 | 90.8% | 0.3426 / 0.1227 |
| 030 | 0.8293 | 0.9345 | 5 | 88.6% | 0.3854 / 0.1382 |
| 031 | 0.8332 | 0.9327 | 5 | 90.3% | 0.3487 / 0.1254 |

## Decisión operativa

Se fija 026 (control, semilla 42) como referencia para preparar el primer ejemplo
técnico. No se declara un ganador científico ni una superioridad general de IMA++.
La selección es una decisión de ingeniería conservadora, no una validación
independiente con PH2 ni una elección optimizada sobre macro-F1 de PAD.

Esta regla operativa se fija después de observar la ablación y antes del
nuevo piloto; no se presenta como criterio científico preregistrado.

El piloto sigue deshabilitado hasta resolver QA sin referencia experta en PAD,
compatibilidad de adaptadores y una configuración de ejecución revisada.

## Límites

- Los intervalos post hoc no están ajustados por comparaciones múltiples.
- El bootstrap de imágenes no modela correlación por paciente/lesión ni nuevas semillas de entrenamiento.
- Solo hay tres semillas; la sensibilidad por cambio de signo tiene ocho asignaciones y muy baja resolución.
- La contaminación CIELAB usa referencia experta: es auditoría offline, no un filtro operativo para PAD.
- La evaluación ISIC es retrospectiva; separar cohortes de origen no acredita independencia por paciente.
- Cuatro logs operativos fueron truncados; los recibos de identidad no reconstruyen esos logs.
