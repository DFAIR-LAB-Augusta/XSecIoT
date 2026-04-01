#!/usr/bin/env bash

set -euo pipefail

log()  { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[ERR ]\033[0m %s\n" "$*" >&2; }

if [[ -n "${UV:-}" ]]; then
  UV="$UV"
elif command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
else
  UV="$HOME/.local/bin/uv"
fi

# --- Config ---
CHUNK_SIZES=(5 10 15 25 50 75 100 500 1000 1)

RUNS=(0 1 2 3 4)
SEEDS=(17 42 67 92 117 )


# MODEL_VARIANTS=("svm" "dt" "knn" "rf" "xgb" "feedforward")
MODEL_VARIANTS=("feedforward")

CE_TYPES=("none" "ice" "approx_cce" "cce" "approx_tce")

CADE_DIMS=(76 512 128 32) # First num is # of features, 76 for dfair, 3x for UNSW
CADE_UNSW_DIMS=(21 512 128 32) # First num is # of features, 76 for dfair, 3x for UNSW
CADE_MARGIN=10.0
CADE_MAD_THRESHOLD=3.5
CADE_MIN_DRIFT_RATIO=0.05
CADE_MIN_DRIFT_COUNT=1
CADE_BATCH_SIZE=64
CADE_EPOCHS=50
CADE_LR=0.001
CADE_LAMBDA_1=0.1
CADE_SIMILAR_RATIO=0.25
CADE_DISPLAY_INTERVAL=10
CADE_DEVICE="/GPU:0"

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
# for cs in "${CHUNK_SIZES[@]}"; do Chunk size proven to be solid alr
for run in "${RUNS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for model in "${MODEL_VARIANTS[@]}"; do
            for ce in "${CE_TYPES[@]}"; do

            # --- Chunk-size runs --- Dont need chunk
            #   run_one "DFAIR chunk model=$model ce=$ce chunk=$cs" \
            #     env PYTHONPATH=. $UV run src/core/ce_simulation.py \
            #       datasets/CETrain/combined_data.csv \
            #       datasets/CEFlows2/CEFlows2_merged.csv \
            #       --log2File \
            #       --modelVariant "$model" \
            #       --ceType "$ce" \
            #       --max_rows 100000 \
            #       --useCircularLogger \
            #       --debug \
            #       --useMLP \
            #       --chunk_size "$cs" \
            #     || exit 1

            #   run_one "UNSW chunk model=$model ce=$ce chunk=$cs" \
            #     env PYTHONPATH=. $UV run src/core/ce_simulation.py \
            #       datasets/UNSW_NB15/NF-UNSW-NB15-v3.csv \
            #       datasets/CEFlows2/CEFlows2_merged.csv \
            #       --log2File \
            #       --modelVariant "$model" \
            #       --ceType "$ce" \
            #       --max_rows 100000 \
            #       --useCircularLogger \
            #       --debug \
            #       --useMLP \
            #       --chunk_size "$cs" \
            #       --unsw \
            #     || exit 1

            #   run_one "CIC chunk model=$model ce=$ce chunk=$cs" \
            #     env PYTHONPATH=. $UV run src/core/ce_simulation.py \
            #       datasets/CIC_UNSW/NF-CICIDS2018-v3.csv \
            #       datasets/CEFlows2/CEFlows2_merged.csv \
            #       --log2File \
            #       --modelVariant "$model" \
            #       --ceType "$ce" \
            #       --max_rows 100000 \
            #       --useCircularLogger \
            #       --debug \
            #       --useMLP \
            #       --chunk_size "$cs" \
            #       --unsw \
            #     || exit 1

              # --- Adaptive chunking runs ---
            run_one "DFAIR AC model=$model ce=$ce" \
                env PYTHONPATH=. $UV run src/core/ce_simulation.py \
                datasets/CETrain/combined_data.csv \
                datasets/CEFlows2/CEFlows2_merged.csv \
                --log2File \
                --modelVariant "$model" \
                --monitorType ce \
                --ceType "$ce" \
                --max_rows 100000 \
                --useCircularLogger \
                --debug \
                --useMLP \
                --useAC \
                --seed "$seed" \
                --runNum "$run" \
                || exit 1

            run_one "UNSW AC model=$model ce=$ce" \
                env PYTHONPATH=. $UV run src/core/ce_simulation.py \
                datasets/UNSW_NB15/NF-UNSW-NB15-v3.csv \
                datasets/CEFlows2/CEFlows2_merged.csv \
                --log2File \
                --modelVariant "$model" \
                --monitorType ce \
                --ceType "$ce" \
                --max_rows 100000 \
                --useCircularLogger \
                --debug \
                --useMLP \
                --useAC \
                --unsw \
                --seed "$seed" \
                --runNum "$run" \
                || exit 1

            run_one "CIC AC model=$model ce=$ce" \
                env PYTHONPATH=. $UV run src/core/ce_simulation.py \
                datasets/CIC_UNSW/NF-CICIDS2018-v3.csv \
                datasets/CEFlows2/CEFlows2_merged.csv \
                --log2File \
                --modelVariant "$model" \
                --monitorType ce \
                --ceType "$ce" \
                --max_rows 100000 \
                --useCircularLogger \
                --debug \
                --useMLP \
                --useAC \
                --unsw \
                --seed "$seed" \
                --runNum "$run" \
                || exit 1


            done
            #   # --- CADE Adaptive chunking runs ---
            # run_one "DFAIR AC model=$model monitor=cade" \
            #     env PYTHONPATH=. XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit \
            #     "$UV" run src/core/ce_simulation.py \
            #     datasets/CETrain/combined_data.csv \
            #     datasets/CEFlows2/CEFlows2_merged.csv \
            #     --log2File \
            #     --modelVariant "$model" \
            #     --monitorType cade \
            #     --max_rows 100000 \
            #     --useCircularLogger \
            #     --debug \
            #     --useMLP \
            #     --useAC \
            #     --seed "$seed" \
            #     --runNum "$run" \
            #     --cadeDims "${CADE_DIMS[@]}" \
            #     --cadeMargin "$CADE_MARGIN" \
            #     --cadeMadThreshold "$CADE_MAD_THRESHOLD" \
            #     --cadeMinDriftRatio "$CADE_MIN_DRIFT_RATIO" \
            #     --cadeMinDriftCount "$CADE_MIN_DRIFT_COUNT" \
            #     --cadeBatchSize "$CADE_BATCH_SIZE" \
            #     --cadeEpochs "$CADE_EPOCHS" \
            #     --cadeLr "$CADE_LR" \
            #     --cadeLambda1 "$CADE_LAMBDA_1" \
            #     --cadeSimilarRatio "$CADE_SIMILAR_RATIO" \
            #     --cadeDisplayInterval "$CADE_DISPLAY_INTERVAL" \
            #     --cadeForceRetrain \
            #     --cadeDevice "$CADE_DEVICE" \
            #     || exit 1

            # run_one "UNSW AC model=$model monitor=cade" \
            #     env PYTHONPATH=. XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit \
            #     "$UV" run src/core/ce_simulation.py \
            #     datasets/UNSW_NB15/NF-UNSW-NB15-v3.csv \
            #     datasets/CEFlows2/CEFlows2_merged.csv \
            #     --log2File \
            #     --modelVariant "$model" \
            #     --monitorType cade \
            #     --max_rows 100000 \
            #     --useCircularLogger \
            #     --debug \
            #     --useMLP \
            #     --useAC \
            #     --unsw \
            #     --seed "$seed" \
            #     --runNum "$run" \
            #     --cadeDims "${CADE_UNSW_DIMS[@]}" \
            #     --cadeMargin "$CADE_MARGIN" \
            #     --cadeMadThreshold "$CADE_MAD_THRESHOLD" \
            #     --cadeMinDriftRatio "$CADE_MIN_DRIFT_RATIO" \
            #     --cadeMinDriftCount "$CADE_MIN_DRIFT_COUNT" \
            #     --cadeBatchSize "$CADE_BATCH_SIZE" \
            #     --cadeEpochs "$CADE_EPOCHS" \
            #     --cadeLr "$CADE_LR" \
            #     --cadeLambda1 "$CADE_LAMBDA_1" \
            #     --cadeSimilarRatio "$CADE_SIMILAR_RATIO" \
            #     --adeDisplayInterval "$CADE_DISPLAY_INTERVAL" \
            #     --cadeForceRetrain \
            #     --cadeDevice "$CADE_DEVICE" \
            #     || exit 1

            # run_one "CIC AC model=$model monitor=cade" \
            #     env PYTHONPATH=. XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit \
            #     "$UV" run src/core/ce_simulation.py \
            #     datasets/CIC_UNSW/NF-CICIDS2018-v3.csv \
            #     datasets/CEFlows2/CEFlows2_merged.csv \
            #     --log2File \
            #     --modelVariant "$model" \
            #     --monitorType cade \
            #     --max_rows 100000 \
            #     --useCircularLogger \
            #     --debug \
            #     --useMLP \
            #     --useAC \
            #     --unsw \
            #     --seed "$seed" \
            #     --runNum "$run" \
            #     --cadeDims "${CADE_UNSW_DIMS[@]}" \
            #     --cadeMargin "$CADE_MARGIN" \
            #     --cadeMadThreshold "$CADE_MAD_THRESHOLD" \
            #     --cadeMinDriftRatio "$CADE_MIN_DRIFT_RATIO" \
            #     --cadeMinDriftCount "$CADE_MIN_DRIFT_COUNT" \
            #     --cadeBatchSize "$CADE_BATCH_SIZE" \
            #     --cadeEpochs "$CADE_EPOCHS" \
            #     --cadeLr "$CADE_LR" \
            #     --cadeLambda1 "$CADE_LAMBDA_1" \
            #     --cadeSimilarRatio "$CADE_SIMILAR_RATIO" \
            #     --adeDisplayInterval "$CADE_DISPLAY_INTERVAL" \
            #     --cadeForceRetrain \
            #     --cadeDevice "$CADE_DEVICE" \
            #     || exit 1
        done
    done
done

log "All runs completed successfully."
print_successes
