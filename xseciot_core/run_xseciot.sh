#!/bin/bash
# This script runs cicflowmeter and listenCSV concurrently.
# When you press Ctrl+C (or a termination signal is sent), both processes are killed.

cleanup() {
    echo "Terminating processes..."
    kill "$CICFLOW_PID" 2>/dev/null
    kill "$LISTEN_PID" 2>/dev/null
    wait
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start cicflowmeter sniffer in the background
(
  cd /home/pooltab/Desktop/CIC_Testing/cicflowmeter/src || { echo "Failed to change directory"; exit 1; }
  sudo "$(which python3)" -m cicflowmeter.sniffer -i wlo1 -u http://127.0.0.1:2048
) &
CICFLOW_PID=$!

# Start listenCSV.py in the background
(
  cd "$HOME/Desktop/CIC_Testing/testFlows" || { echo "Failed to change directory"; exit 1; }
  sudo "$(which python3)" listenCSV.py
) &
LISTEN_PID=$!

wait -n

cleanup
