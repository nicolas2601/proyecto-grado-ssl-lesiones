#!/usr/bin/env bash
# Publish the trained backbones as a GitHub release, with their evidence.
#
# A .pt on its own is not citable: without run.json nobody knows which data,
# which budget or which split produced it. This script refuses to publish a
# checkpoint whose run.json is missing, and writes the checksums into the
# release notes so a download can be verified.
#
#   scripts/publish_model_release.sh v1.0-tone /ruta/a/salidas/tone
#
# Expects the gh CLI authenticated against the repository.
set -euo pipefail

TAG="${1:?uso: publish_model_release.sh <tag> <directorio de salidas/tone>}"
TONE_DIR="${2:?falta el directorio tone}"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

require_file() {
  [ -f "$1" ] || { echo "falta $1" >&2; exit 1; }
}

collect_arm() {
  local arm="$1"
  local src="$TONE_DIR/$arm"
  require_file "$src/simsiam_dinov2_backbone_final.pt"
  require_file "$src/run.json"
  cp "$src/simsiam_dinov2_backbone_final.pt" "$STAGING/${arm}_backbone.pt"
  cp "$src/run.json" "$STAGING/${arm}_run.json"
}

collect_arm ssl_plain
collect_arm ssl_proposed

for extra in bank/bank.json bank/strata.json bank/bank.md; do
  [ -f "$TONE_DIR/$extra" ] && cp "$TONE_DIR/$extra" "$STAGING/$(basename "$extra")"
done

( cd "$STAGING" && sha256sum ./* > SHA256SUMS.txt )

NOTES="$STAGING/NOTES.md"
{
  echo "# Backbones adaptados al tono — $TAG"
  echo
  echo "Extractores DINOv2 ViT-S/14 congelados, preentrenados sin etiquetas"
  echo "sobre 55.034 dermatoscopias. Cómo cargarlos y qué preprocesamiento usar:"
  echo "\`docs/23-modelo-descargable-y-uso.md\`."
  echo
  echo "| Archivo | Qué es |"
  echo "|---|---|"
  echo "| \`ssl_proposed_backbone.pt\` | Método propuesto: entrenado con pares de tono |"
  echo "| \`ssl_plain_backbone.pt\` | Control: sin pares de tono |"
  echo "| \`*_run.json\` | Evidencia de cada corrida: presupuesto, semilla, hash de partición |"
  echo "| \`bank.json\`, \`strata.json\`, \`bank.md\` | Resultados del banco congelado |"
  echo
  echo "Comprobación rápida de que el archivo descargado sirve:"
  echo
  echo '```bash'
  echo "python scripts/try_model.py --backbone ssl_proposed_backbone.pt --image lesion.jpg"
  echo '```'
  echo
  echo "NO es un dispositivo de diagnóstico: sin validación clínica, entrenado"
  echo "sobre un conjunto mayoritariamente de piel clara y con peor desempeño"
  echo "en el tercil de piel menos luminosa."
  echo
  echo "## Checksums"
  echo
  echo '```'
  cat "$STAGING/SHA256SUMS.txt"
  echo '```'
} > "$NOTES"

echo "listo para publicar desde $STAGING:"
ls -la "$STAGING"
echo
echo "publicando release $TAG"
gh release create "$TAG" "$STAGING"/* --title "Backbones adaptados al tono — $TAG" --notes-file "$NOTES"
