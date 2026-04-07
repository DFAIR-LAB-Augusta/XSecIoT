# `firce/`

This folder contains the **FIRCE runtime** for XSecIoT: a modular streaming and simulation framework for IoT intrusion detection using **Conformal Evaluation (CE)** for drift detection and adaptive retraining.

FIRCE supports:
- Real-time and simulated data pipelines
- Multiple CE strategies (ICE, CCE, Approx-CCE, TCE)
- Adaptive chunking and significance control
- Drift-aware monitoring (including CADE integration)
- Flexible logging and performance tracking

---

## Directory Layout

```text
firce/
├── adaptive_chunking.py        # Adaptive chunk/window sizing logic
├── ce_model_training.py        # Train CE-side models (simulation/replay)
├── cli.py                      # CLI entrypoint for FIRCE pipelines
│
├── conformalEval/              # Conformal evaluation implementations
│   ├── adaptive_sig_ctlr.py    # Adaptive significance controller
│   ├── approx_cce.py           # Approximate Cross-Conformal Evaluation
│   ├── cce.py                  # Cross-Conformal Evaluation
│   ├── conformal_config.toml   # CE configuration
│   ├── conformal_evaluators.py # Unified CE interface
│   ├── ice.py                  # Inductive Conformal Evaluation
│   ├── tce.py                  # Transductive Conformal Evaluation
│   ├── utils.py                # CE utilities
│   └── README.md
│
├── drift_monitor/              # Drift detection abstraction layer
│   ├── base.py                 # Base monitor interface
│   ├── cade_config.py          # CADE-specific configuration
│   ├── cade_monitor.py         # CADE-based drift detection
│   ├── conformal_monitor.py    # CE-based drift detection
│   └── factory.py              # Monitor factory (CADE vs CE)
│
├── models/                     # ML models used in FIRCE
│   ├── feedforward_binary.py   # Binary classifier(s)
│   ├── mlp_ce.py               # MLP used in CE pipelines
│   └── torch_device.py         # Device selection (CPU/GPU)
│
├── pipelines/                  # Execution pipelines
│   ├── simulation_pipeline.py  # Offline CE simulation / replay
│   └── streaming_pipeline.py   # Real-time streaming pipeline
│
├── runtime/                    # Core runtime orchestration
│   ├── bootstrap.py            # Runtime initialization
│   ├── constants.py            # Global constants
│   ├── inference.py            # Model inference logic
│   ├── monitoring.py           # Drift monitoring orchestration
│   ├── retraining.py           # Retraining + recalibration logic
│   └── sim_types.py            # Simulation type definitions
│
├── utils/                      # Shared utilities
│   ├── arg_parser.py           # CLI argument parsing
│   ├── circular_logger.py      # In-memory logging (deque-based)
│   ├── config.py               # Central configuration
│   ├── listener.py             # HTTP ingestion endpoint
│   ├── logger.py               # Logging utilities
│   ├── perf_stats.py           # Performance tracking
│   ├── plotter.py              # Metrics visualization
│   └── rolling_csv.py          # Disk-based rolling CSV logger
│
└── README.md
```

---

## Architecture Overview

FIRCE is structured into **four major layers**:

### 1. Pipelines

* `streaming_pipeline.py`: real-time processing
* `simulation_pipeline.py`: offline replay / experimentation

These define the **end-to-end execution flow**.

---

### 2. Runtime

Handles execution logic independent of pipeline type:

* inference
* monitoring
* retraining
* bootstrap/init

This layer ensures **consistent behavior across streaming and simulation**.

---

### 3. Drift Monitoring

Located in `drift_monitor/`, this layer abstracts drift detection:

* **Conformal-based**: `conformal_monitor.py`
* **CADE-based**: `cade_monitor.py`
* Unified via `factory.py`

This allows FIRCE to **swap drift detection strategies cleanly**.

---

### 4. Conformal Evaluation

Located in `conformalEval/`:

* ICE (Inductive)
* CCE (Cross)
* Approx-CCE (FIRCE contribution)
* TCE (Transductive)

Includes:

* calibration logic
* p-value computation
* adaptive significance control

---

## Runtime Flow

### Streaming Pipeline

1. **Ingest**

   * CSV batches received via `utils/listener.py`

2. **Preprocess**

   * Feature selection, normalization, optional PCA

3. **Inference**

   * Model prediction (`runtime/inference.py`)

4. **Drift Detection**

   * CE or CADE monitor evaluates drift

5. **Adaptation**

   * Adaptive chunking (`adaptive_chunking.py`)
   * Optional retraining (`runtime/retraining.py`)

6. **Logging**

   * Disk (`rolling_csv.py`) or memory (`circular_logger.py`)

---

### Simulation Pipeline

Used for:

* benchmarking CE methods
* evaluating drift response
* ablation studies

Flow mirrors streaming but operates on static datasets.

---

## Key Features

### Conformal Evaluation for Drift Detection

* Uses statistical guarantees instead of heuristics
* Supports multiple CE variants
* Enables principled drift detection via p-values

### Approx-CCE (FIRCE Contribution)

* Retains structure of CCE
* Avoids k-model training overhead
* Enables efficient recalibration in streaming settings

### Adaptive Significance Control

* Dynamically adjusts drift threshold (α)
* Prevents over-sensitive drift triggering after retraining

### Adaptive Chunking

* Adjusts processing window size based on drift frequency
* Balances responsiveness and computational cost

### Modular Drift Monitoring

* Easily switch between:

  * CE-based detection
  * CADE-based detection

---

## Usage

From repo root:

### Run simulation

```bash
uv run firce \
    datasets/CETrain/combined_data.csv \
    datasets/CEFlows/flows.csv \
    --pipeline simulation
```

### Run streaming pipeline

```bash
uv run firce \
    --pipeline streaming
```

(See `cli.py` and `utils/arg_parser.py` for full options.)

---

## Notes on CADE Integration

This project includes a **modernized integration of CADE**:

* Original: [https://github.com/whyisyoung/CADE](https://github.com/whyisyoung/CADE)
* Integrated via `drift_monitor/cade_monitor.py`

CADE is used as an **alternative drift detector**, enabling comparison against conformal methods within the same pipeline.

---

## Contact

Seth Barrett
[https://github.com/sethbarrett50](https://github.com/sethbarrett50)
[sebarrett@augusta.edu](mailto:sebarrett@augusta.edu)

Bradley Boswell
[https://github.com/bradleyboswell](https://github.com/bradleyboswell)
[brboswell@augusta.edu](mailto:brboswell@augusta.edu)

Swarnamugi Rajaganapathy, PhD
[https://github.com/swarna6384](https://github.com/swarna6384)
[swarnamugi@dfairlab.com](mailto:swarnamugi@dfairlab.com)

Lin Li, PhD
[https://github.com/linli786](https://github.com/linli786)
[lli1@augusta.edu](mailto:lli1@augusta.edu)

Gokila Dorai, PhD
[https://github.com/gdorai](https://github.com/gdorai)
[gdorai@augusta.edu](mailto:gdorai@augusta.edu)
