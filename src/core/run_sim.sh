#!/usr/bin/env bash

set -euo pipefail
log() { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m[ERR ]\033[0m %s\n" "$*" >&2; }

CHUNK_SIZES=(5 10 15 25 50 75 100 500 1000 1)
˜

# TO REDO:

# MODEL_VARIANTS=("svm" "dt" "knn" "rf" "xgb" "feedforward")
MODEL_VARIANTS=("feedforward")


CE_TYPES=("none" "ice" "approx_cce" "cce"  "approx_tce")
˜

SUCCEEDED=()

for cs in "${CHUNK_SIZES[@]}"; do
  for model in "${MODEL_VARIANTS[@]}"; do
    for ce in "${CE_TYPES[@]}"; do

      log "Running simulation: modelVariant=$model, ceType=$ce, chunk_size=$cs"

      # --- MLP CE Chunk Size ---
      COMMAND="PYTHONPATH='.' caffeinate uv run src/core/ce_simulation.py datasets/CETrain/combined_data.csv datasets/CEFlows2/CEFlows2_merged.csv --log2File --modelVariant $model --ceType $ce --max_rows 100000 --useCircularLogger --debug --useMLP --chunk_size $cs " # DFAIR

      # COMMAND="PYTHONPATH='.' caffeinate uv run src/core/ce_simulation.py datasets/UNSW_NB15/NF-UNSW-NB15-v3.csv datasets/CEFlows2/CEFlows2_merged.csv --log2File --modelVariant $model --ceType $ce --max_rows 100000 --useCircularLogger --debug --useMLP --chunk_size $cs --unsw" # UNSW_NB15

      # COMMAND="PYTHONPATH='.' caffeinate uv run src/core/ce_simulation.py datasets/CIC_UNSW/NF-CICIDS2018-v3.csv datasets/CEFlows2/CEFlows2_merged.csv --log2File --modelVariant $model --ceType $ce --max_rows 100000 --useCircularLogger --debug --useMLP --chunk_size $cs --unsw" # CICIDS2018

      if eval "$COMMAND"; then
        SUCCEEDED+=("$model + $ce + $cs")
      else
        err "Failed command: modelVariant=$model, ceType=$ce, $cs"
        log "Successfully completed combinations:"
        for success in "${SUCCEEDED[@]}"; do
          log "  - $success"
        done
        exit 1
      fi


      # --- MLP CE ACC ---
      # log "Running simulation: modelVariant=$model, ceType=$ce, Adaptive Chunking"

      # COMMAND="PYTHONPATH='.' caffeinate uv run src/core/ce_simulation.py datasets/CETrain/combined_data.csv datasets/CEFlows2/CEFlows2_merged.csv --log2File --modelVariant $model --ceType $ce --max_rows 100000 --useCircularLogger --debug --useMLP --useAC " # DFAIR

      # COMMAND="PYTHONPATH='.' caffeinate uv run src/core/ce_simulation.py datasets/UNSW_NB15/NF-UNSW-NB15-v3.csv datasets/CEFlows2/CEFlows2_merged.csv --log2File --modelVariant $model --ceType $ce --max_rows 100000 --useCircularLogger --debug --useMLP --useAC --unsw" # UNSW_NB15

      # COMMAND="PYTHONPATH='.' caffeinate uv run src/core/ce_simulation.py datasets/CIC_UNSW/NF-CICIDS2018-v3.csv datasets/CEFlows2/CEFlows2_merged.csv --log2File --modelVariant $model --ceType $ce --max_rows 100000 --useCircularLogger --debug --useMLP --useAC --unsw" # CICIDS2018

      # --- ACC --- 
      # if eval "$COMMAND"; then
      #   SUCCEEDED+=("$model + $ce + Adaptive Chunking")
      # else
      #   err "Failed command: modelVariant=$model, ceType=$ce, Adaptive Chunking"
      #   log "Successfully completed combinations:"
      #   for success in "${SUCCEEDED[@]}"; do
      #     log "  - $success"
      #   done
      #   exit 1
      # fi
    done
  done
done

log "All combinations completed successfully:"
for success in "${SUCCEEDED[@]}"; do
  log "  - $success"
done
