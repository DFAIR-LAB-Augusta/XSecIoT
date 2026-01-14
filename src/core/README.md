# 🔍 `core/`

This folder contains the **FIRCE** streaming runtime for XSecIoT: live flow ingestion, ML inference, **Conformal Evaluation** (ICE/CCE/Approx-CCE/TCE) for drift detection, and rolling logs for adaptive retraining.

---

## 🗂️ Directory Layout

```text
core/
├── adaptive_chunking.py        # Adaptive chunk/window sizing for streaming & CE
├── ce_model_training.py        # Train CE-side classifiers for simulations/replay
├── ce_simulation.py            # CE simulation/replay harness (batch/stream)
├── circular_logger.py          # In-memory circular buffer logger
├── config.py                   # Centralized config & paths for the runtime
├── conformalEval/              # Conformal evaluators + config
│   ├── adaptive_significance_controller.py    # Adaptive significance (threshold) controller
│   ├── approx_cce.py           # Approximate CCE variant
│   ├── cce.py                  # Cross Conformal Evaluation
│   ├── conformal_config.toml   # CE settings (evaluator, windows, thresholds)
│   ├── conformal_evaluators.py # Unified CE interfaces & wrappers
│   ├── ice.py                  # Inductive Conformal Evaluation
│   ├── README.md
│   ├── tce.py                  # Transductive Conformal Evaluation
│   └── utils.py                # CE helpers (calibration buffers, p-values, etc.)
├── listener.py                 # HTTP listener for CSV flow batches (POST)
├── models/
│   ├── feedforward_binary.py   # Binary classifier(s) used by CE
│   ├── mlp_ce.py               # MLP model for CE pipelines
│   └── torch_device.py         # Device selection & torch helpers (CPU/GPU)
├── perf_stats.py               # Runtime/perf metrics aggregation
├── README.md
├── rolling_csv.py              # Size-bounded CSV logger (gzip) for streaming data
├── run_sim.sh                  # One-liner launcher for FIRCE simulation (UV)
├── run_xseciot.sh              # Convenience script to run full streaming stack
└── streaming_pipeline.py       # Ingest → preprocess → (scale/PCA) → classify → CE → log → (retrain)
```

---

## 🔧 Components

| File/Dir                    | Purpose                                                                                                        |
| --------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **`streaming_pipeline.py`** | Main runtime orchestration for live flows (ingest → preprocess → predict → CE drift → log → optional retrain). |
| **`listener.py`**           | Lightweight HTTP endpoint to receive CICFlowMeter-style CSV batches.                                           |
| **`rolling_csv.py`**        | Append-only, size-capped CSV logging (gzipped) for labeled streaming data.                                     |
| **`circular_logger.py`**    | In-memory circular buffer alternative for high-throughput logging.                                             |
| **`conformalEval/`**        | CE implementations (ICE/CCE/TCE/Approx-CCE), config, and adaptive significance control.                        |
| **`adaptive_chunking.py`**  | Adjusts processing chunk sizes based on drift/runtime signals.                                                 |
| **`ce_simulation.py`**      | Offline replay/simulation of CE behavior for ablations and benchmarking.                                       |
| **`ce_model_training.py`**  | Trains CE-side models used during simulations.                                                                 |
| **`models/`**               | Model definitions (MLP/FFN) and device helpers.                                                                |
| **`perf_stats.py`**         | Collects/aggregates performance stats (accuracy/F1/runtime, etc.).                                             |
| **`config.py`**             | Centralized configuration (paths, flags, thresholds).                                                          |
| **`run_sim.sh`**            | UV-compatible launcher for FIRCE simulation using local datasets.                                              |
| **`run_xseciot.sh`**        | Starts the full streaming stack (e.g., cicflowmeter + pipeline).                                               |

---

## ⚡ Runtime Flow

1. **Ingest** — CICFlowMeter (or equivalent) emits CSV flow batches → `listener.py` receives via HTTP POST.
2. **Preprocess** — Normalize columns, select numeric features, apply scaler/PCA if configured.
3. **Classify** — CE-backed model predicts class labels on the batch.
4. **Drift Detect** — CE computes p-values; if below threshold, mark drift and (optionally) trigger retraining/recalibration.
5. **Log** — Persist original rows + predictions + CE metadata via `rolling_csv.py` (or in-memory via `circular_logger.py`).

---

## 🚀 Quick Start

From the **repo root** (expects inputs in `datasets/`, writes artifacts to `logging/`, and models to `binary_models/` / `multiclass_models/`):

```bash
uv sync
./src/core/run_sim.sh
```

To run the end-to-end streaming stack (e.g., with CICFlowMeter):

```bash
bash ./src/core/run_xseciot.sh
```

--- 

## 📢 Contact

Seth Barrett | [GitHub](https://github.com/sethbarrett50) | [sebarrett@augusta.edu](mailto:sebarrett@augusta.edu)
Bradley Boswell | [GitHub](https://github.com/bradleyboswell) | [brboswell@augusta.edu](mailto:brboswell@augusta.edu)
Swarnamugi Rajaganapathy, PhD | [GitHub](https://github.com/swarna6384) | [swarnamugi@dfairlab.com](mailto:swarnamugi@dfairlab.com)
Lin Li, PhD | [GitHub](https://github.com/linli786) | [lli1@augusta.edu](mailto:lli1@augusta.edu)
Gokila Dorai, PhD | [GitHub](https://github.com/gdorai) | [gdorai@augusta.edu](mailto:gdorai@augusta.edu)
