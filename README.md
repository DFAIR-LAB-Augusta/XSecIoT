# FIRCE and FIRE for IoT Intrusion Detection

`xseciot` provides two related components for machine learning-based intrusion detection in IoT environments.

`FIRCE` is the streaming and evaluation side of the project. It supports flow classification, concept drift detection through conformal evaluation, adaptive retraining, and runtime monitoring.

`FIRE` is the offline experimentation side of the project. It contains preprocessing, model development, and simulation-oriented code used to support the broader research workflow.

For most users of this repository, the primary entrypoint is `FIRCE`.

## Features

`FIRCE` includes support for:

- streaming or replay-style intrusion detection workflows
- conformal evaluators including ICE, CCE, Approx-CCE, and Approx-TCE
- adaptive chunking
- adaptive significance control
- rolling logging and optional circular logging
- binary and multiclass model workflows

`FIRE` includes support for:

- offline preprocessing
- model training support
- simulation utilities
- supporting research workflows for IoT IDS experiments

## Repository Layout

```text
.
├── scripts/                      # Helper scripts for labeling, merging, and summary stats
├── src/
│   ├── firce/                    # Streaming and conformal-evaluation pipeline
│   │   ├── cli.py
│   │   ├── ce_simulation.py
│   │   ├── adaptive_chunking.py
│   │   ├── conformalEval/
│   │   ├── drift_monitor/
│   │   ├── models/
│   │   ├── pipelines/
│   │   ├── runtime/
│   │   └── utils/
│   └── fire/                     # Offline preprocessing, modeling, and simulation support
├── README.md
├── pyproject.toml
└── uv.lock
```

## Requirements

* Python 3.11 or newer
* `uv`

## Installation

Clone the repository and sync the environment:

```bash
uv sync
```

To verify the package builds correctly:

```bash
uv build
```

## FIRCE Usage

The `firce` CLI is the main interface for the streaming framework.

### 1. Offline simulation from saved flow CSV files

This mode is intended for replaying previously generated flow records.

Example:

```bash
uv run firce \
  datasets/CETrain/combined_data.csv \
  datasets/CEFlows2/CEFlows2_merged.csv \
  --log2File \
  --modelVariant feedforward \
  --ceType approx_tce \
  --max_rows 100000 \
  --useCircularLogger \
  --debug \
  --useMLP \
  --useAC \
  --pipeline simulation
```

UNSW example:

```bash
uv run firce \
  datasets/UNSW_NB15/NF-UNSW-NB15-v3.csv \
  datasets/CEFlows2/CEFlows2_merged.csv \
  --log2File \
  --modelVariant feedforward \
  --ceType approx_tce \
  --max_rows 100000 \
  --useCircularLogger \
  --debug \
  --useMLP \
  --useAC \
  --unsw \
  --pipeline simulation
```

### 2. Live usage with `cicflowmeter`

This mode is intended for near-real-time operation, where packet captures are converted into flow features and then consumed by the FIRCE pipeline.

The general workflow is:

1. capture traffic
2. generate flow records through `cicflowmeter`
3. feed those flow records into the FIRCE runtime pipeline

Because live deployment details depend on your interface, traffic source, and how `cicflowmeter` is launched in your environment, this repository should document the exact operational command you want users to run once your live workflow is finalized.

At minimum, the README should state that FIRCE supports operation alongside `cicflowmeter` for live flow generation.

## Configuration

Important FIRCE components include:

* `src/firce/conformalEval/conformal_config.toml` for conformal-evaluation settings
* `src/firce/adaptive_chunking.py` for adaptive chunk-size behavior
* `src/firce/models/torch_device.py` for runtime device selection
* `src/firce/drift_monitor/` for drift-monitoring implementations
* `src/firce/pipelines/` for pipeline orchestration

## Utility Scripts

The repository also includes helper scripts for common data-processing tasks.

Label a dataset:

```bash
uv run scripts/labeling.py --dataset_path datasets/CEFlows/unlabeled
```

Merge labeled data:

```bash
uv run scripts/merge.py --dataset_path datasets/CEFlows/mc_labeled
```

Aggregate performance statistics:

```bash
uv run scripts/overall_perf_stats.py
```

Scrape summary statistics from logs:

```bash
uv run scripts/overall_stats_scraper.py
```

## Development

Run tests:

```bash
uv run pytest -q
```

Run formatting and lint fixes:

```bash
uv run ruff format .
uv run ruff check . --fix
```

Run dependency checks:

```bash
uv run deptry .
```

Run a build and metadata check before publishing:

```bash
uv build
uv tool run twine check dist/*
```

## Research Use Notice

This repository contains research-oriented code intended for experimentation and evaluation. Validate behavior, datasets, and model assumptions carefully before using it in operational environments.

## Authors

* Seth Barrett
* Bradley Boswell
* Swarnamugi Rajaganapathy
* Lin Li
* Gokila Dorai