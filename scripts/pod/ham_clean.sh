#!/usr/bin/env bash
# HAM10000 clean-weights sensitivity (docs/33): B71-limpio and A71-limpio on the C1 val, or its rehearsal.
#
#   scripts/pod/ham_clean.sh rehearsal <commit>   # same code on the holdout; never opens val, no lock
#   scripts/pod/ham_clean.sh val <commit>         # the ONE read of val (own out-dir, own val_read.lock)
#
# <commit> is the commit that holds scripts/ham_clean_run.py: it must be an ancestor of HEAD and the clone must
# be clean. Every step logs to $OUT/logs/<step>.log and its REAL exit code is kept (no pipe in the way);
# the marker "<name>.marker" with "rc=0" is written only when every step exited 0 AND the artifacts exist.
# Env: P (default $HOME/work/proyectonp), REPO, PYTHON, OUT_ROOT (default $P/salidas/bench/ham_clean).
# Copy afterwards (by hand, reviewed): ham_clean_val.json, .probas.npz and .environment.json to
# resultados/banco_completo/ham_clean/ and run scripts/build_bench_all.py.
set -euo pipefail

MODE=${1:?usage: ham_clean.sh <rehearsal|val> <commit>}
REQUIRED=${2:?usage: ham_clean.sh <rehearsal|val> <commit>}
P=${P:-$HOME/work/proyectonp}
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PYTHON:-python}
OUT_ROOT=${OUT_ROOT:-$P/salidas/bench/ham_clean}
export TORCH_HOME=${TORCH_HOME:-$HOME/work/.torch} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

abort() { echo "ABORT: $*" >&2; exit 2; }

require_commit() {
  git -C "$REPO" merge-base --is-ancestor "$1" HEAD || abort "$1 is not an ancestor of HEAD in $REPO"
  [ -z "$(git -C "$REPO" status --porcelain)" ] || abort "the clone $REPO is not clean"
}

# run_step <name> <log-dir> <command...>: exit code of the command, kept as is.
run_step() {
  local name=$1 logs=$2 rc=0
  shift 2
  mkdir -p "$logs"
  "$@" >"$logs/$name.log" 2>&1 || rc=$?
  echo "[$name] rc=$rc (log: $logs/$name.log)"
  return "$rc"
}

# finish <marker> <missing-count>: the rc=0 marker exists only with every artifact in place.
finish() {
  local marker=$1 missing=$2
  [ "$missing" -eq 0 ] || { echo "FAILED: $missing artifact(s) missing, no rc=0 marker" >&2; exit 1; }
  echo "$(basename "$marker" .marker) commit=$(git -C "$REPO" rev-parse HEAD) at=$(date -u +%FT%TZ) rc=0" >"$marker"
  echo "OK: $marker"
}

need() {
  local missing=0 f
  for f in "$@"; do [ -s "$f" ] || { echo "missing artifact: $f" >&2; missing=$((missing + 1)); }; done
  echo "$missing"
}

C1=$P/salidas/c1
COMMON=(--ham-root "$HOME/work/datos/ham10000" --image-dir "$P/eval_ham10000_images"
        --split-index "$C1/c1_split_index.json" --holdout "$C1/selection/selection_holdout.json")
CLEAN_WEIGHTS=$P/salidas/bench/v7clean

require_commit "$REQUIRED"

case "$MODE" in
val)
  OUT=$OUT_ROOT/val
  for f in val_read.lock ham_clean_val.json ham_clean_extract.json; do
    [ ! -e "$OUT/$f" ] || abort "$OUT/$f exists: the val is read once; use a new OUT_ROOT"
  done
  run_step extract "$OUT/logs" "$PY" "$REPO/scripts/ham_clean_run.py" "${COMMON[@]}" --weights-dir "$CLEAN_WEIGHTS" \
    --out-dir "$OUT" --stage extract --single-evaluation --probe-jobs 10 --decode-workers 16
  run_step val "$OUT/logs" "$PY" "$REPO/scripts/ham_clean_run.py" "${COMMON[@]}" --weights-dir "$CLEAN_WEIGHTS" \
    --out-dir "$OUT" --stage val --single-evaluation
  finish "$OUT/ham_clean_val.marker" "$(need "$OUT/ham_clean_val.json" "$OUT/ham_clean_val.probas.npz" \
    "$OUT/ham_clean_val.environment.json" "$OUT/ham_clean_extract.json" "$OUT/val_read.lock")"
  ;;
rehearsal)
  REF=$OUT_ROOT/rehearsal_reference
  OUT=$OUT_ROOT/ham_clean_rehearsal
  mkdir -p "$REF"
  cp "$REPO/resultados/clasificacion/v71/selection.json" "$REPO/resultados/clasificacion/v71/holdout_grid.json" "$REF/"
  # the controls of the rehearsal: the ORIGINAL v7.1 val stage run on the holdout (never on val), in a scratch dir
  run_step reference "$OUT/logs" "$PY" "$REPO/scripts/v71_run.py" "${COMMON[@]}" --weights-dir "$P/salidas/v7" \
    --out-dir "$REF" --stage val --split holdout --ci-resamples 20 --probe-jobs 10 --decode-workers 16
  run_step extract "$OUT/logs" "$PY" "$REPO/scripts/ham_clean_run.py" "${COMMON[@]}" --weights-dir "$CLEAN_WEIGHTS" \
    --out-dir "$OUT" --reference-dir "$REF" --stage extract --rehearsal --probe-jobs 10 --decode-workers 16
  run_step val "$OUT/logs" "$PY" "$REPO/scripts/ham_clean_run.py" "${COMMON[@]}" --weights-dir "$CLEAN_WEIGHTS" \
    --out-dir "$OUT" --reference-dir "$REF" --stage val --rehearsal
  missing=$(need "$OUT/ham_clean_holdout_rehearsal.json" "$OUT/ham_clean_holdout_rehearsal.probas.npz" \
    "$OUT/ham_clean_holdout_rehearsal.environment.json" "$OUT/ham_clean_extract.json")
  if [ -e "$OUT/val_read.lock" ] || compgen -G "$OUT/inputs/val_*" >/dev/null; then
    echo "the rehearsal touched val (lock or val_* inputs)" >&2
    missing=$((missing + 1))
  fi
  finish "$OUT/ham_clean_rehearsal.marker" "$missing"
  ;;
*) abort "mode must be rehearsal or val" ;;
esac
