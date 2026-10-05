# 34. Entorno del clúster y datos

Generado por `scripts/build_cluster_env.py` desde `resultados/entorno/`; no editar a mano. Sin hostnames internos, IPs ni tokens.

## Hardware

- GPU: 2 × NVIDIA RTX PRO 4000 Blackwell, 24467 MiB cada una; driver 580.178.04; CUDA 12.8 (build de torch); cuDNN 9.19.0 (según torch: 91900); NCCL 2.28.9.
- CPU: Intel(R) Xeon(R) w7-2495X, 48 CPU lógicas (1 socket, 2 hilos por núcleo).
- RAM: 123 GiB. `/dev/shm`: 64M (los DataLoader usan 0 workers).
- Disco de datos: volumen de red compartido, 40T (318G usados).
- SO: Ubuntu 24.04.4 LTS, kernel 7.0.0-31-generic. Contenedor: sesión JupyterHub de un solo usuario (no Docker; cgroup v2).

## Software

- Python 3.13.15, torch 2.11.0+cu128. scikit-learn por defecto 1.9.0; para reconstruir splits existentes se usa 1.7.2 (aislado; StratifiedGroupKFold(shuffle=True) cambió después de 1.7.2).
- `timm` en el venv: no; para el fine-tuning: 1.0.30 (instalado con --no-deps --target en un directorio aislado de work; PYTHONPATH solo para el job de fine-tuning; el venv no se modificó). Revisión de los re-preentrenamientos: `f1c2053d7e9e3f3eae4e1665d6942243b02e3872`.
- Hilos: probe `PROBE_BLAS_THREADS` = 4 (fijado por dermssl.probe_threads); pretrain OMP/MKL/OPENBLAS = 12.
- `pip freeze` completo: `resultados/entorno/pip_freeze.txt` (284 líneas).

## Datos

| Dataset | Imágenes | Split de evaluación | val n | sha256 de la lista de val |
|---|---|---|---|---|
| ISIC 2019 | 25331 | estratificado simple SIN grupo de paciente (el release no lo trae) | 5067 | `8cde8a71ccf5bf2e477ffdab445e8f1afbc37a164ee0cd1aba93ef785cc0b57f` |
| ISIC 2020 | 33126 | StratifiedGroupKFold por patient_id (0 grupos compartidos) | 6660 | `ec3f5408881aa4623493e95e2840f360bab3de300c6fc022a4b88e55d5a0567e` |
| HAM10000 | 10.015 | v7 val n = 2.004, agrupado por lesion_id | 2.004 | ver docs/28 |

## Pool de preentrenamiento SSL

- Pool v7: 55034 imágenes; exclusión v7: 3851 ids (sha256 `95d21013a42a5c99a6024ef6c8ae73aada8bdc1193ae512665035bb59b737d87`); pool efectivo 51256.
- Pool limpio: exclusión 14842 ids (sha256 `3b7e883392ba5ae0cb290166fa2d4e2dfbb325d4bce0bca3e96562f8c39dfc1d`); pool efectivo 40906.
