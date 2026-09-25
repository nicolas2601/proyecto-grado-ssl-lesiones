# Notebooks

Los notebooks son artefactos ejecutables para Kaggle. Las versiones generadas
se conservan para reproducibilidad, pero el resultado publicado debe leerse en
el JSON/README que cada banco produjo.

## Índice

| Carpeta o archivo | Uso |
|---|---|
| [`baselines/`](baselines/README.md) | Ocho backbones congelados sobre HAM10000. |
| [`baselines_isic2019/`](baselines_isic2019/) | Banco multiclase ISIC 2019. |
| [`baselines_isic2020/`](baselines_isic2020/) | Banco binario ISIC 2020. |
| [`finetune/`](finetune/README.md) | Fine-tuning principal. |
| [`finetune_isic2019/`](finetune_isic2019/) | Fine-tuning sobre ISIC 2019. |
| [`finetune_isic2020/`](finetune_isic2020/) | Fine-tuning sobre ISIC 2020. |
| [`finetune_lrprobe/`](finetune_lrprobe/) | Probe de learning rate. |
| `proyecto_ssl_lesiones_v5.ipynb` | Notebook integrado de evaluación v5. |

## Generación

- Baselines: `scripts/build_baseline_tests.py`.
- Fine-tuning: `scripts/build_finetune_tests.py`.
- Tablas y resultados: `scripts/build_results_tables.py`.

No editar notebooks generados a mano. Cambiar el builder, regenerar, ejecutar
los tests de frescura y luego actualizar la documentación enlazada.
