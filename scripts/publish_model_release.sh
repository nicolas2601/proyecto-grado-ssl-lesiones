#!/usr/bin/env bash
# Publish the trained backbones as a GitHub release, with their evidence.
#
# A .pt on its own is not citable: without run.json nobody knows which data,
# which budget or which split produced it. The notes are generated from those
# manifests (scripts/build_release_notes.py), so they describe the models the
# release actually contains instead of whatever the last edit claimed.
#
#   scripts/publish_model_release.sh v1.0-tone <directorio>
#   scripts/publish_model_release.sh v1.0-tone <directorio> esperado.sha256
#
# The directory holds either the downloaded artifact layout
# (<arm>_backbone.pt + <arm>_run.json) or the training layout (<arm>/run.json).
# The optional third argument is the checksum list produced where the weights
# were trained: the release is refused if a file does not match, because a
# checkpoint corrupted in transit fails in the user's hands, not in ours.
#
# Expects the gh CLI authenticated against the repository.
set -euo pipefail

TAG="${1:?uso: publish_model_release.sh <tag> <directorio> [esperado.sha256]}"
SRC_DIR="${2:?falta el directorio con los pesos y los run.json}"
EXPECTED="${3:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
STAGING="$WORK/assets"
NOTES="$WORK/NOTES.md"
mkdir -p "$STAGING"
trap 'rm -rf "$WORK"' EXIT

[ -d "$SRC_DIR" ] || { echo "no existe el directorio $SRC_DIR" >&2; exit 1; }

if [ -n "$EXPECTED" ]; then
  EXPECTED_ABS="$(realpath "$EXPECTED")"
  echo "verificando los checksums de origen"
  if ! ( cd "$SRC_DIR" && sha256sum --check --strict "$EXPECTED_ABS" ); then
    echo "ABORTADO: un archivo no coincide con el checksum de origen." >&2
    echo "Se corrompio en la transferencia; volve a bajarlo antes de publicar." >&2
    exit 1
  fi
fi

python "$REPO_ROOT/scripts/stage_release_assets.py" --src "$SRC_DIR" --staging "$STAGING"

( cd "$STAGING" && sha256sum ./* > SHA256SUMS.txt )

python "$REPO_ROOT/scripts/build_release_notes.py" \
  --tag "$TAG" --dir "$STAGING" --checksums "$STAGING/SHA256SUMS.txt" --out "$NOTES"

echo "listo para publicar:"
eza -l "$STAGING" 2>/dev/null || ls -la "$STAGING"
echo
echo "publicando release $TAG"
gh release create "$TAG" "$STAGING"/* \
  --title "Backbones adaptados al tono - $TAG" --notes-file "$NOTES"
