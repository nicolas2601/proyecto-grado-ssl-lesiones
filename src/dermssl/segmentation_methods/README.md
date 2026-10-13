# Adaptaciones de segmentación v8

| Carpeta | Contenido |
|---|---|
| `sled/` | Grafo, solver H2 propio, SLIC, Isolation Forest, GHT y selector corregido |
| `usl_net/` | Fusión de tres orígenes SSL, CCAM y FCN con ignore según texto |
| `rpi_net/` | Cuatro fuentes, IPC/PIA y DUL con convenciones declaradas |
| `tridermnet_hago/` | Region/Edge/Token, TER, cobertura pseudo y HAGO adaptado |
| `common/` | Datos, hashes, procedencia, backbones/SSL, pérdidas, ciclos y overlays locales |

Cada carpeta tiene `README.md` y `config.json`. Pruebas en
`tests/segmentation_v8/test_<metodo>.py`; contratos en `test_common.py`.
Guía completa: [documento 43](../../../docs/43-cuatro-metodos-segmentacion-v8.md).

Desde la raíz del repo, consultar configuración sin ejecutar:

```bash
python scripts/run_segmentation_v8.py --method sled --stage check
python scripts/run_segmentation_v8.py --method usl_net --stage check
python scripts/run_segmentation_v8.py --method rpi_net --stage check
python scripts/run_segmentation_v8.py --method tridermnet_hago --stage check
```

Para comprobación sintética con W&B (una carpeta nueva por ejecución):

```bash
python scripts/run_segmentation_v8.py --method sled --stage synthetic --include-common \
  --out salidas/v8/sled/synthetic-01 --wandb-project dermssl-segmentacion
```

`WANDB_NETRC_FILE` apunta al archivo privado existente; no pegar credenciales ni
ponerlas en configs. Repetir cambiando `--method` y `--out`. Etapas reales requieren
`execution_enabled=true`, manifiesto congelado, evidencia de puertas y presupuesto.
No descargar pesos ni iniciar GPU por importar módulos.
