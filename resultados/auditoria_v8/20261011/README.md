# Ejecución modular v8 del 2026-10-11

Código y decisiones: [registro completo](../../../docs/39-resultados-y-puertas-v8.md). Agregados: W0/W1/W2, inferencia S4 y evaluación experta exploratoria. No incluye imágenes clínicas ni registros individuales.

Fuente de ejecución: f8b1790f8f6049415eccc92a066d696a48171e4e, snapshot verificado sobre base local 9c23daf; no checkout completo de esa revisión. Copia aislada en /home/jovyan/work/proyectonp/repo_audit_v8_run_20261011. Original y pendientes conservados.

Salidas grandes bajo /home/jovyan/work/proyectonp/salidas/auditoria_v8. Índice final: w01_w2_s42_20261011/progress.html. Protocolos registrados antes de respectivos resultados: docs/36, docs/37 y docs/38.

Verificación ejecutada: ocho pruebas W0, cinco pruebas de pilotos, reporte --check de W0 y W01/S4; 200 IDs, 800 máscaras y 1400 archivos S4; 100 pares expertos, 500 etiquetas, hashes de input/reference, métricas binarias y bootstrap recalculados. HD95 se obtuvo con el evaluador existente; no se recalculó en el verificador posterior.

```bash
PYTHONPATH=src python scripts/verify_seed_evaluation.py --run /home/jovyan/work/proyectonp/salidas/auditoria_v8/w2_eval_isic2018_s44_20261011
python scripts/build_audit_progress.py --root /home/jovyan/work/proyectonp/salidas/auditoria_v8
```

Todos los 100 pares expertos coinciden por ID canónico con el pool nativo. No hay evidencia externa intacta ni nuevo entrenamiento de estudiante/SSL. S0 sigue control; cuarentena excluida. W3 tiene dependencias y procedencia/licencia pendientes; W4 permanece diseñado detrás de las puertas registradas.
