# Bancos históricos y salidas de runners

`results/` conserva los artefactos de los bancos v4/v5, fine-tuning, learning
rate probes y scripts de generación. Se mantiene en su ubicación porque varios
builders, notebooks y tests lo referencian directamente.

No es la fuente narrativa del resultado final del pipeline propuesto. Para esa
lectura usa [`../resultados/README.md`](../resultados/README.md).

## Índice

| Área | Contenido |
|---|---|
| [`baselines/`](baselines/) | Banco congelado principal de HAM10000. |
| [`baselines_isic2019/`](baselines_isic2019/) | Banco multiclase comparable con el ancla de ISIC 2019. |
| [`baselines_isic2020/`](baselines_isic2020/) | Banco binario agrupado por paciente. |
| [`finetune/`](finetune/) | Salidas de fine-tuning por dataset y backbone. |
| [`finetune_lrprobe/`](finetune_lrprobe/) | Pruebas de learning rate del backbone. |
| `generar_figuras*.py` | Generadores de figuras de etapas anteriores. Revisar sus entradas antes de ejecutar. |

## Relación con la evidencia final

- [Bancos congelados y cómo leerlos](../docs/19-resultados-bancos-congelados.md).
- [Auditoría SSL puro y plan v5](../docs/15-auditoria-ssl-puro-y-plan-v5.md).
- [Plan v6](../docs/16-plan-v6-pretraining-gpu.md).
- [Resultados finales del pipeline propuesto](../resultados/README.md).

Los checkpoints grandes, caches de MLflow y datos crudos siguen excluidos por
`.gitignore`. Un JSON versionado no implica que el dataset completo o el
checkpoint correspondiente estén dentro del repositorio.
