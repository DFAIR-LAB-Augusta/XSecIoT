#!/usr/bin/env bash
# Study A: DFAIR, HTTP_Flood held out, 5 seeds, ICE, FNN classifier + multiclass MLP conformal model.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${D3_OUT:-$(mktemp -d)}"
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT
cd "$REPO"

start=$SECONDS
set +e
uv run --no-sync python -m firce.experiments --preset dfair --seeds 0 1 2 3 4 \
  --modelVariant feedforward --useMLP --ceType ice --out "$OUT" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e

if [ "$status" -ne 0 ]; then
  echo "d3_study_a: driver exited $status" >&2
  exit "$status"
fi

RUN_DIR="$(tail -n 1 "$LOG")"
if [ ! -d "$RUN_DIR" ]; then
  echo "d3_study_a: last stdout line is not a run dir: $RUN_DIR" >&2
  exit 1
fi

if [ ! -f "$RUN_DIR/summary.csv" ]; then
  echo "d3_study_a: missing $RUN_DIR/summary.csv" >&2
  exit 1
fi
rows=$(( $(wc -l < "$RUN_DIR/summary.csv") - 1 ))
if [ "$rows" -ne 9 ]; then
  echo "d3_study_a: summary.csv has $rows data rows, expected 9" >&2
  exit 1
fi

for k in 0 1 2 3 4; do
  if [ ! -f "$RUN_DIR/seed_$k/metrics.json" ]; then
    echo "d3_study_a: missing $RUN_DIR/seed_$k/metrics.json" >&2
    exit 1
  fi
done

echo "d3_study_a: run dir $RUN_DIR"
echo "d3_study_a: wall time $((SECONDS - start)) s"
echo 'd3_study_a: OK'
