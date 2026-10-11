# S4: inferencia pareada del mismo maestro

Registrado antes de inferencia S4. Sin entrenamiento ni optimizador.
Usar los mismos 200 IDs W01/W2; cabeza CCAM y backbone MoCo históricos congelados.
Comparar maestro sobre RGB original y RGB con TELEA + detector elongado existente
(kernel 9, umbral 10, longitud 15, ancho 5, radio 3). Solo cambia input del maestro.
Campo y reglas de artefactos calculados sobre RGB original, mismos flags S0.
ITA se mantiene en RGB original y los resultados W1 no se recalculan con inpainting.

Inferencia en eval/no_grad, BF16 en backbone, cabeza FP32, interpolación bilineal,
cuantización float16 idénticas al exportador histórico. Lote pareado alterna original
/limpia en el mismo forward. Reproducibilidad numérica respecto a histórico: registrar
MAE y máximo por mapa; puerta previa MAE ≤0.005 y máximo ≤0.03 en cada caso.
Son tolerancias de diagnóstico ante kernel/lote/hardware, no umbrales científicos.
Si no cumple, marcar bloqueado, no interpretar diferencias S4 vs S0 histórico.
Comparar adicionalmente S4 contra S0 reinferido en el mismo lote para aislar efecto.
No determinar superioridad anatómica por acuerdo entre maestros/estudiante.
