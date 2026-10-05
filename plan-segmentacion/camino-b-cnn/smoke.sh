#!/usr/bin/env bash
# Verifica la instalacion: corre todas las etapas del camino B con 64 imagenes
# sinteticas (2 rondas de re-etiquetado, evaluacion sintetica). Deberia tardar
# menos de dos minutos en una GPU. Uso: bash plan-segmentacion/camino-b-cnn/smoke.sh
# Con dos GPU: bash plan-segmentacion/camino-b-cnn/smoke.sh --nproc-per-node 2
set -euo pipefail
cd "$(dirname "$0")/../.."
python plan-segmentacion/camino-b-cnn/entrenamiento/run_pipeline.py --smoke "$@"
