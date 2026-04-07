# `scripts/` — Data Processing & Experiment Utilities

This directory contains **standalone utilities and orchestration scripts** for working with datasets, logs, and experiment outputs produced by **FIRCE** and **FIRE**.

These scripts are **not part of the core library API**. They are designed for:
- research workflows
- dataset preparation
- experiment automation
- post-processing and analysis

All scripts are intended to be run from the **repository root** using `uv`.

---

## Overview

The scripts support common workflows:

- labeling and preparing datasets
- merging experiment outputs
- aggregating and scraping performance metrics
- running simulations and pipelines
- scanning network devices (for data collection)
- executing jobs on HPC systems (SLURM)

---

## Directory Layout

```text
scripts/
├── labeling.py               # Apply labels to unlabeled datasets
├── merge.py                  # Merge multiple CSV datasets
├── overall_perf_stats.py     # Aggregate performance metrics
├── overall_stats_scraper.py  # Extract structured stats from logs
├── scanner.py                # Network/device scanning utility
├── run_sim.sh                # FIRCE simulation wrapper
├── run_xseciot.sh            # End-to-end pipeline launcher
├── slurm.sh                  # SLURM job script for HPC execution
└── README.md
```

---

## Components

| Script                         | Purpose                                                                         |
| ------------------------------ | ------------------------------------------------------------------------------- |
| **`labeling.py`**              | Assigns ground-truth labels to unlabeled flow datasets                          |
| **`merge.py`**                 | Combines multiple CSV datasets into a single file                               |
| **`overall_perf_stats.py`**    | Aggregates metrics (accuracy, precision, recall, F1) across runs                |
| **`overall_stats_scraper.py`** | Parses logs into structured/tabular outputs                                     |
| **`scanner.py`**               | Scans network devices (e.g., ports, OS detection) for data collection workflows |
| **`run_sim.sh`**               | Convenience wrapper for FIRCE simulation runs                                   |
| **`run_xseciot.sh`**           | Launches full XSecIoT pipeline (data → FIRCE)                                   |
| **`slurm.sh`**                 | Example SLURM script for running experiments on HPC clusters                    |

---

## Requirements

* Python **3.11+**
* `uv`

Install dependencies from the repository root:

```bash
uv sync
```

---

## Usage

All scripts are run from the repository root.

### General pattern

```bash
uv run scripts/<script_name>.py [arguments]
```

View CLI options:

```bash
uv run scripts/<script_name>.py --help
```

---

## Common Workflows

### Label a dataset

```bash
uv run scripts/labeling.py \
  --dataset_path datasets/CEFlows/unlabeled
```

---

### Merge labeled datasets

```bash
uv run scripts/merge.py \
  --dataset_path datasets/CEFlows/mc_labeled
```

---

### Aggregate performance statistics

```bash
uv run scripts/overall_perf_stats.py
```

With explicit log directory:

```bash
uv run scripts/overall_perf_stats.py \
  --log_dir logging
```

---

### Extract structured stats from logs

```bash
uv run scripts/overall_stats_scraper.py
```

```bash
uv run scripts/overall_stats_scraper.py \
  --log_dir logging
```

---

### Run FIRCE simulation (wrapper)

```bash
bash scripts/run_sim.sh
```

---

### Run full pipeline

```bash
bash scripts/run_xseciot.sh
```

---

### Scan network devices

```bash
uv run scripts/scanner.py \
  --targets 192.168.1.0/24 \
  --output-dir logging/scans
```

---

## Integration with Makefile

Many of these scripts are exposed via `Makefile` targets:

```bash
make bin-label
make merge
make overall-perf
make overall-scrape
```

Using the Makefile is recommended for reproducibility and consistency.

---

## Data & Output Conventions

* **Input datasets**: `datasets/`
* **Logs / outputs**: `logging/`
* **Models**: `binary_models/`, `multiclass_models/`

Scripts assume schemas produced by FIRCE pipelines.

---

## HPC Usage (SLURM)

The `slurm.sh` script provides a template for running experiments on SLURM-managed systems.

You should:

* adjust resource requests (CPU/GPU/memory)
* update dataset paths
* ensure `uv` is available in the environment

---

## Notes

* These scripts are intentionally lightweight and loosely coupled
* They may assume specific dataset formats used in FIRCE experiments
* Errors are often due to schema mismatches — verify upstream pipeline outputs

---

## Authors

Seth Barrett
Bradley Boswell
Swarnamugi Rajaganapathy
Lin Li
Gokila Dorai

