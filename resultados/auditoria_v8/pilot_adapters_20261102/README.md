# Transversal v8: validación técnica de adaptadores del piloto integrado

Esta carpeta contiene evidencia de una validación **sintética, solo en CPU**.
No contiene un resultado clínico ni acredita la ejecución del piloto sobre PAD.
No se entrenó ningún modelo ni se ejecutó el lector de clasificación.

## Entradas y comprobaciones

`inputs.json` fija los dos checkpoints y el código offline de DINOv2. Sus pesos
son el teacher ResNet34 de 026 y el encoder MoCo-v3-style de Nicolás, semilla 42.
Se verifican hashes antes de cargar, `weights_only=True`, identidad del teacher,
arquitecturas y estados estrictos. El código de ResNet34 debe coincidir con el
registrado durante el entrenamiento; no basta que las formas de tensores encajen.

`receipt.json` conserva las comprobaciones de la versión final de los adaptadores:
paridad de preprocesamiento frente a torchvision, paridad del CLS frente al
forward de referencia, features finitas y preservación de los píxeles de lesión
predicha y fuera del campo válido. `execution.json` registra la ejecución aislada.
Los hashes de los scripts están incluidos en el recibo; la entrada sintética se
reconstruye con la semilla registrada.

La imagen sintética no permite juzgar la calidad de las máscaras ni la pureza
real de piel. El estado de QA favorable solo indica que pasó las heurísticas de
geometría y disponibilidad definidas. `clinical_skin_purity_verified=false`,
`real_patient_images_used=false`, `reader_evaluated=false` y `pilot_started=false`
conservan explícitamente esos límites.

## Reproducción

Usar una copia aislada con los pesos y el checkout offline fijados en
`inputs.json`, y con el código de ResNet34 que coincide con la identidad de 026.
El punto de entrada es:

```powershell
python scripts/validate_pilot_adapters.py --config resultados/auditoria_v8/pilot_adapters_20261102/inputs.json --output-json <nueva-ruta-del-recibo.json>
```

El programa se niega a sobrescribir evidencia anterior. Las rutas de pesos del
recibo corresponden a UNABIA; no se descargaron pesos al repositorio.

## Pendiente para el piloto completo

Definir y revisar QA de piel aplicable a PAD sin referencia experta, congelar el
manifiesto del ejemplo y reconstruir el lector OOF con las mismas particiones
por paciente sobre las nuevas features. Reutilizar probabilidades obtenidas con
imágenes originales no evaluaría la intervención CIELAB.
