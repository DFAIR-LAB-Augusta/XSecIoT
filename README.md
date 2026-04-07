# FIRCE and FIRE for IoT Intrusion Detection

`xseciot` provides a unified framework for **machine learning-based intrusion detection in IoT environments**, combining:

- **FIRCE** — a streaming, drift-aware evaluation framework using conformal prediction
- **FIRE** — a batch-oriented baseline for preprocessing, training, and offline experimentation

For most users, **FIRCE is the primary entrypoint**.

---

## Overview

### FIRCE (Primary System)

FIRCE is a **streaming and simulation framework** that supports:

- real-time and replay-based intrusion detection
- conformal evaluation for drift detection (ICE, CCE, Approx-CCE, TCE)
- adaptive retraining and recalibration
- adaptive chunking and significance control
- modular drift monitoring (CE and CADE)
- structured logging and performance tracking

---

### FIRE (Baseline System)

FIRE provides:

- dataset preprocessing
- model training
- batch simulation and evaluation

It is used for:
- baseline comparisons
- controlled experiments
- supporting FIRCE model development

---

## Key Features

### FIRCE

- streaming + simulation pipelines
- conformal drift detection with statistical guarantees
- Approx-CCE (efficient cross-conformal evaluation)
- adaptive chunking (dynamic window sizing)
- adaptive significance control (dynamic α)
- modular drift monitoring (CE + CADE)
- binary and multiclass workflows

### FIRE

- offline preprocessing and feature engineering
- supervised model training
- batch evaluation pipelines

---

## Repository Layout

```text
.
├── scripts/                  # Data processing, orchestration, and utilities
├── src/
│   ├── firce/                # Streaming + conformal evaluation framework
│   │   ├── cli.py
│   │   ├── adaptive_chunking.py
│   │   ├── conformalEval/
│   │   ├── drift_monitor/
│   │   ├── models/
│   │   ├── pipelines/
│   │   ├── runtime/
│   │   └── utils/
│   └── fire/                 # Offline preprocessing, modeling, and simulation
├── scripts/                  # Labeling, merging, stats, scanning, SLURM
├── Makefile                  # Reproducible workflows (recommended entrypoints)
├── pyproject.toml            # Project metadata and dependencies
├── uv.lock                   # Locked dependency graph
└── README.md
```

---

## Requirements

* Python **3.11+**
* `uv`

---

## Installation

```bash
uv sync
```

Optional (GPU / ML stack depending on config):

```bash
uv sync --extra tf
```

---

## Quick Start

### Run FIRCE simulation

```bash
make bin.test
```

UNSW example:

```bash
make bin.unsw
```

---

### Run manually via CLI

```bash
uv run firce \
  datasets/CETrain/combined_data.csv \
  datasets/CEFlows2/CEFlows2_merged.csv \
  --pipeline simulation
```

---

### Run full pipeline

```bash
bash scripts/run_xseciot.sh
```

---

### Scan network devices (data collection)

```bash
make scan TARGET_IP=192.168.1.0/24
```

---

## Configuration

Key FIRCE components:

* `src/firce/conformalEval/conformal_config.toml` — CE configuration
* `src/firce/adaptive_chunking.py` — adaptive window sizing
* `src/firce/drift_monitor/` — drift detection implementations (CE + CADE)
* `src/firce/models/torch_device.py` — device selection (CPU/GPU)
* `src/firce/pipelines/` — simulation and streaming orchestration

---

## Utility Scripts

Examples:

```bash
# Label dataset
make label UNLABELED_DATASET_PATH=datasets/CEFlows/unlabeled

# Merge datasets
make merge LABELED_DATASET_PATH=datasets/CEFlows/mc_labeled

# Aggregate performance
make overall.perf LOG_DIR=logging

# Scrape structured stats
make overall.scrape LOG_DIR=logging
```

Or directly:

```bash
uv run scripts/labeling.py --dataset_path ...
```

---

## Development

Run tests:

```bash
make test
```

Lint + format:

```bash
make lint
```

Check dependencies:

```bash
make deps.check
```

Build artifacts:

```bash
make build
```

Pre-publish validation:

```bash
make preflight
```

---

## Research Context

This repository contains **research-oriented code** for IoT intrusion detection.

* Results depend heavily on dataset assumptions and preprocessing
* Models and pipelines are intended for experimentation and evaluation
* Validate behavior before deploying in operational environments

---

## Attribution

This project includes components derived from:

* CADE: [https://github.com/whyisyoung/CADE](https://github.com/whyisyoung/CADE)

The CADE implementation has been modernized and integrated into FIRCE for drift detection research.

---

## Authors

Seth Barrett
Bradley Boswell
Swarnamugi Rajaganapathy
Lin Li
Gokila Dorai
