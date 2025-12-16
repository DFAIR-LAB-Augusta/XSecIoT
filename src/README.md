# 📂 `src/`

This directory contains the **source code** for XSecIoT. It is organized into three branches:

* **`core/`** — FIRCE: the real-time streaming IDS + Conformal Evaluation engine
* **`FIRE/`** — offline preprocessing, modeling, and simulation framework
* **`utils/`** — Python utilities for processing/logging experiment outputs

---

## 🔍 `core/` — FIRCE (Streaming Runtime)

The **production-style** pipeline that ingests CICFlowMeter-style flows, classifies in real time, detects concept drift via Conformal Evaluators (ICE/CCE/Approx-CCE/TCE), and logs for adaptive retraining.

**Key modules & scripts**

* **`streaming_pipeline.py`** — end-to-end runtime: ingest → preprocess → (PCA/scale) → classify → CE drift detection → log → optional retrain
* **`ce_simulation.py`** — batch/stream simulation harness for CE (replay logs, ablations, metrics)
* **`ce_model_training.py`** — trains CE-side classifiers used during simulation
* **`adaptive_chunking.py`** — adjusts chunk/window sizes in response to runtime conditions
* **`circular_logger.py` / `rolling_csv.py`** — high-throughput, size-bounded flow logging
* **`perf_stats.py`** — runtime metrics & performance aggregation
* **`config.py`** — central configuration for paths/flags used by the core pipeline
* **`run_sim.sh`** — one-liner launcher for FIRCE simulation (UV compatible)
* **`run_xseciot.sh`** — convenience script to run the end-to-end streaming stack (e.g., with cicflowmeter)

**Models & devices**

* **`models/`**

  * `mlp_ce.py`, `feedforward_binary.py` — CE classifier definitions
  * `torch_device.py` — CPU/GPU device selection helpers

**Conformal Evaluation**

* **`conformalEval/`**

  * `ice.py`, `cce.py`, `tce.py`, `approx_cce.py` — CE variants
  * `conformal_evaluators.py` — unified interfaces/wrappers
  * `adaptive_significance_controller.py` — adaptive significance controller (threshold adaptation)
  * `utils.py` — CE helpers (calibration windows, p-values, etc.)
  * `conformal_config.toml` — CE configuration (e.g., evaluator, windows, thresholds)

> **I/O expectations:**
>
> * **Input:** CSV flows in `datasets/`
> * **Outputs:** models in `binary_models/` or `multiclass_models/`, logs & metrics in `logging/`

**Quick start (from repo root)**

```bash
uv sync
./src/core/run_sim.sh
```

---

## 🔬 `FIRE/` — Offline Research Framework

The **research-grade** pipeline for dataset preparation, model training, and controlled simulations.

**Key entrypoints**

* **`main.py`** — unified CLI for preprocess → train → simulate
* **`preprocessing.py`** — cleaning, sessionization, sliding-window aggregation
* **`models.py`** — binary & multiclass training; hooks for SHAP/LIME
* **`simulations.py`** — sequential/continuous/parallel simulation modes
* **`JuypterNotebooks/`** — exploratory analysis (latency, features, model comparisons)

**Example**

```bash
uv run --project . src/FIRE/main.py ./datasets/DFAIR/combined_data_with_okpVacc_modified.csv
```

---

## 🧰 `utils/` — Log & Metrics Utilities

Helpers for working with run artifacts and logs produced by FIRCE/FIRE.

* **`labeling.py`** — programmatic labeling support for flows/segments
* **`merge.py`** — safe merges of log shards and intermediate CSVs
* **`overall_perf_stats.py`** — compute aggregate metrics across runs (accuracy/F1/runtime, etc.)
* **`overall_stats_scraper.py`** — scrape/normalize “Full performance stats:” blocks from logs

---

## 📢 Contact

Seth Barrett | [GitHub](https://github.com/sethbarrett50) | [sebarrett@augusta.edu](mailto:sebarrett@augusta.edu)
Bradley Boswell | [GitHub](https://github.com/bradleyboswell) | [brboswell@augusta.edu](mailto:brboswell@augusta.edu)
Swarnamugi Rajaganapathy, PhD | [GitHub](https://github.com/swarna6384) | [swarnamugi@dfairlab.com](mailto:swarnamugi@dfairlab.com)
Lin Li, PhD | [GitHub](https://github.com/linli786) | [lli1@augusta.edu](mailto:lli1@augusta.edu)
Gokila Dorai, PhD | [GitHub](https://github.com/gdorai) | [gdorai@augusta.edu](mailto:gdorai@augusta.edu)
