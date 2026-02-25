#!/usr/bin/env bash

set -euo pipefail

log()  { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[ERR ]\033[0m %s\n" "$*" >&2; }

UV="/home/seth/.local/bin/uv"

# --- Config ---
CHUNK_SIZES=(5 10 15 25 50 75 100 500 1000 1)

# MODEL_VARIANTS=("svm" "dt" "knn" "rf" "xgb" "feedforward")
MODEL_VARIANTS=("feedforward")

CE_TYPES=("none" "ice" "approx_cce" "cce" "approx_tce")

SUCCEEDED=()

# --- Helpers ---
print_successes() {
  if ((${#SUCCEEDED[@]} == 0)); then
    log "No successful runs recorded."
    return 0
  fi

  log "Successfully completed runs so far:"
  for success in "${SUCCEEDED[@]}"; do
    log "  - $success"
  done
}

run_one() {
  local label="$1"
  shift

  log "Running: $label"

  if "$@"; then
    SUCCEEDED+=("$label")
    return 0
  fi

  err "Failed: $label"
  print_successes
  return 1
}

# --- Preflight ---
if [[ ! -x "$UV" ]]; then
  err "uv not found or not executable at: $UV"
  exit 127
fi

$UV --version || true

# --- Working directory ---
# Prefer SLURM's submit dir / chdir, and fall back to repo discovery.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
fi

# If we're inside a git repo, jump to its root (robust even under SLURM).
if command -v git >/dev/null 2>&1; then
  GIT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
  if [[ -n "$GIT_ROOT" ]]; then
    cd "$GIT_ROOT"
  fi
fi

log "PWD: $(pwd)"
log "Using uv: $UV"

# --- Main sweep ---
for cs in "${CHUNK_SIZES[@]}"; do
  for model in "${MODEL_VARIANTS[@]}"; do
    for ce in "${CE_TYPES[@]}"; do

      # --- Chunk-size runs ---
      run_one "DFAIR chunk model=$model ce=$ce chunk=$cs" \
        env PYTHONPATH=. $UV run src/core/ce_simulation.py \
          datasets/CETrain/combined_data.csv \
          datasets/CEFlows2/CEFlows2_merged.csv \
          --log2File \
          --modelVariant "$model" \
          --ceType "$ce" \
          --max_rows 100000 \
          --useCircularLogger \
          --debug \
          --useMLP \
          --chunk_size "$cs" \
        || exit 1

      run_one "UNSW chunk model=$model ce=$ce chunk=$cs" \
        env PYTHONPATH=. $UV run src/core/ce_simulation.py \
          datasets/UNSW_NB15/NF-UNSW-NB15-v3.csv \
          datasets/CEFlows2/CEFlows2_merged.csv \
          --log2File \
          --modelVariant "$model" \
          --ceType "$ce" \
          --max_rows 100000 \
          --useCircularLogger \
          --debug \
          --useMLP \
          --chunk_size "$cs" \
          --unsw \
        || exit 1

      run_one "CIC chunk model=$model ce=$ce chunk=$cs" \
        env PYTHONPATH=. $UV run src/core/ce_simulation.py \
          datasets/CIC_UNSW/NF-CICIDS2018-v3.csv \
          datasets/CEFlows2/CEFlows2_merged.csv \
          --log2File \
          --modelVariant "$model" \
          --ceType "$ce" \
          --max_rows 100000 \
          --useCircularLogger \
          --debug \
          --useMLP \
          --chunk_size "$cs" \
          --unsw \
        || exit 1

      # --- Adaptive chunking runs ---
      run_one "DFAIR AC model=$model ce=$ce" \
        env PYTHONPATH=. $UV run src/core/ce_simulation.py \
          datasets/CETrain/combined_data.csv \
          datasets/CEFlows2/CEFlows2_merged.csv \
          --log2File \
          --modelVariant "$model" \
          --ceType "$ce" \
          --max_rows 100000 \
          --useCircularLogger \
          --debug \
          --useMLP \
          --useAC \
        || exit 1

      run_one "UNSW AC model=$model ce=$ce" \
        env PYTHONPATH=. $UV run src/core/ce_simulation.py \
          datasets/UNSW_NB15/NF-UNSW-NB15-v3.csv \
          datasets/CEFlows2/CEFlows2_merged.csv \
          --log2File \
          --modelVariant "$model" \
          --ceType "$ce" \
          --max_rows 100000 \
          --useCircularLogger \
          --debug \
          --useMLP \
          --useAC \
          --unsw \
        || exit 1

      run_one "CIC AC model=$model ce=$ce" \
        env PYTHONPATH=. $UV run src/core/ce_simulation.py \
          datasets/CIC_UNSW/NF-CICIDS2018-v3.csv \
          datasets/CEFlows2/CEFlows2_merged.csv \
          --log2File \
          --modelVariant "$model" \
          --ceType "$ce" \
          --max_rows 100000 \
          --useCircularLogger \
          --debug \
          --useMLP \
          --useAC \
          --unsw \
        || exit 1

    done
  done
done

log "All runs completed successfully."
print_successes
