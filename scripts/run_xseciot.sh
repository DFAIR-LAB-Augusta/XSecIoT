#!/usr/bin/env bash
#
# run_xseciot.sh — run cicflowmeter & the streaming_pipeline via UV concurrently

set -euo pipefail

# ensure src/ is on the path via .env or explicitly
export PYTHONPATH=src

echo "[INFO] Requesting sudo access to run cicflowmeter later..."
sudo -v

cleanup() {
    echo "Terminating processes..."
    kill "${CICFLOW_PID:-}" 2>/dev/null || true
    kill "${STREAM_PID:-}"  2>/dev/null || true
    wait
    exit 0
}

trap cleanup SIGINT SIGTERM

# 1) Start cicflowmeter sniffer in background
(
  cd /home/pooltab/Desktop/CIC_Testing/cicflowmeter/src || {
    echo "Failed to cd into cicflowmeter"; exit 1
  }
  sudo /home/pooltab/Desktop/CIC_Testing/CICTestVenv/bin/python -m cicflowmeter.sniffer \
       -i wlo1 \
       -u http://127.0.0.1:2048
) &
CICFLOW_PID=$!
echo "cicflowmeter sniffer started (pid=$CICFLOW_PID)"

# 2) Start streaming_pipeline via UV in background
(
  cd "$(dirname "$(realpath "$0")")/../.." || {
    echo "Could not cd to project root"
    exit 1
  }

  echo "Running from $(pwd)"

  PYTHONPATH="." uv run --project . src/core/streaming_pipeline.py

) &
STREAM_PID=$!
echo "streaming_pipeline started under UV (pid=$STREAM_PID)"

# wait for either to exit
wait -n

cleanup